import logging
import os

import comfy.utils
import folder_paths
import torch
import torch.nn.functional as F
from comfy.ldm.flux.layers import timestep_embedding
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from insightface.app import FaceAnalysis
from torch import Tensor, nn
from torchvision import transforms
from torchvision.transforms import functional

from .encoders_flux import IDFormer, PerceiverAttentionCA
from .eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

INSIGHTFACE_DIR = os.path.join(folder_paths.models_dir, "insightface")

MODELS_DIR = os.path.join(folder_paths.models_dir, "pulid")
if "pulid" not in folder_paths.folder_names_and_paths:
    current_paths = [MODELS_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["pulid"]
folder_paths.folder_names_and_paths["pulid"] = (
    current_paths,
    folder_paths.supported_pt_extensions,
)

from .online_train2 import online_train


class PulidFluxModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.double_interval = 2
        self.single_interval = 4

        # Init encoder
        self.pulid_encoder = IDFormer()

        # Init attention
        num_ca = 19 // self.double_interval + 38 // self.single_interval
        if 19 % self.double_interval != 0:
            num_ca += 1
        if 38 % self.single_interval != 0:
            num_ca += 1
        self.pulid_ca = nn.ModuleList([PerceiverAttentionCA() for _ in range(num_ca)])

    def from_pretrained(self, path: str):
        state_dict = comfy.utils.load_torch_file(path, safe_load=True)
        state_dict_dict = {}
        for k, v in state_dict.items():
            module = k.split(".")[0]
            state_dict_dict.setdefault(module, {})
            new_k = k[len(module) + 1 :]
            state_dict_dict[module][new_k] = v

        for module in state_dict_dict:
            getattr(self, module).load_state_dict(state_dict_dict[module], strict=True)

        del state_dict
        del state_dict_dict

    def get_embeds(self, face_embed, clip_embeds):
        return self.pulid_encoder(face_embed, clip_embeds)


def forward_orig(
    self,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Tensor = None,
    control=None,
    transformer_options={},
) -> Tensor:
    patches_replace = transformer_options.get("patches_replace", {})

    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    # running on sequences img
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
    if self.params.guidance_embed:
        if guidance is None:
            raise ValueError(
                "Didn't get guidance strength for guidance distilled model."
            )
        vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

    vec = vec + self.vector_in(y)
    txt = self.txt_in(txt)

    ids = torch.cat((txt_ids, img_ids), dim=1)
    pe = self.pe_embedder(ids)

    ca_idx = 0
    blocks_replace = patches_replace.get("dit", {})

    for i, block in enumerate(self.double_blocks):
        if ("double_block", i) in blocks_replace:

            def block_wrap(args):
                out = {}
                out["img"], out["txt"] = block(
                    img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"]
                )
                return out

            out = blocks_replace[("double_block", i)](
                {"img": img, "txt": txt, "vec": vec, "pe": pe},
                {"original_block": block_wrap},
            )
            txt = out["txt"]
            img = out["img"]
        else:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        if control is not None:  # Controlnet
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    img += add

        # PuLID attention
        if self.pulid_data:
            if i % self.pulid_double_interval == 0:
                # Will calculate influence of all pulid nodes at once
                for _, node_data in self.pulid_data.items():
                    condition_start = node_data["sigma_start"] >= timesteps
                    condition_end = timesteps >= node_data["sigma_end"]
                    condition = torch.logical_and(condition_start, condition_end).all()

                    if condition:
                        img = img + node_data["weight"] * self.pulid_ca[ca_idx](
                            node_data["embedding"], img
                        )
                ca_idx += 1

    img = torch.cat((txt, img), 1)

    for i, block in enumerate(self.single_blocks):
        img = block(img, vec=vec, pe=pe)

        if control is not None:  # Controlnet
            control_o = control.get("output")
            if i < len(control_o):
                add = control_o[i]
                if add is not None:
                    img[:, txt.shape[1] :, ...] += add

        # PuLID attention
        if self.pulid_data:
            real_img, txt = img[:, txt.shape[1] :, ...], img[:, : txt.shape[1], ...]
            if i % self.pulid_single_interval == 0:
                # Will calculate influence of all nodes at once
                for _, node_data in self.pulid_data.items():
                    condition_start = node_data["sigma_start"] >= timesteps
                    condition_end = timesteps >= node_data["sigma_end"]

                    # Combine conditions and reduce to a single boolean
                    condition = torch.logical_and(condition_start, condition_end).all()

                    if condition:
                        real_img = real_img + node_data["weight"] * self.pulid_ca[
                            ca_idx
                        ](node_data["embedding"], real_img)
                ca_idx += 1
            img = torch.cat((txt, real_img), 1)

    img = img[:, txt.shape[1] :, ...]

    img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
    return img


def tensor_to_image(tensor):
    image = tensor.mul(255).clamp(0, 255).byte().cpu()
    image = image[..., [2, 1, 0]].numpy()
    return image


def image_to_tensor(image):
    tensor = torch.clamp(torch.from_numpy(image).float() / 255.0, 0, 1)
    tensor = tensor[..., [2, 1, 0]]
    return tensor


def resize_with_pad(img, target_size):  # image: 1, h, w, 3
    img = img.permute(0, 3, 1, 2)
    H, W = target_size

    h, w = img.shape[2], img.shape[3]
    scale_h = H / h
    scale_w = W / w
    scale = min(scale_h, scale_w)

    new_h = int(min(h * scale, H))
    new_w = int(min(w * scale, W))
    new_size = (new_h, new_w)

    img = F.interpolate(img, size=new_size, mode="bicubic", align_corners=False)

    pad_top = (H - new_h) // 2
    pad_bottom = (H - new_h) - pad_top
    pad_left = (W - new_w) // 2
    pad_right = (W - new_w) - pad_left
    img = F.pad(
        img, pad=(pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0
    )

    return img.permute(0, 2, 3, 1)


def to_gray(img):
    x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    x = x.repeat(1, 3, 1, 1)
    return x


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


class PulidFluxModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"pulid_file": (folder_paths.get_filename_list("pulid"),)}}

    RETURN_TYPES = ("PULIDFLUX",)
    FUNCTION = "load_model"
    CATEGORY = "pulid"

    def load_model(self, pulid_file):
        model_path = folder_paths.get_full_path("pulid", pulid_file)

        # Also initialize the model, takes longer to load but then it doesn't have to be done every time you change parameters in the apply node
        model = PulidFluxModel()

        logging.info("Loading PuLID-Flux model.")
        model.from_pretrained(path=model_path)

        return (model,)


class PulidFluxInsightFaceLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "provider": (["CUDA", "CPU", "ROCM"],),  # Default to CUDA first
            },
        }

    RETURN_TYPES = ("FACEANALYSIS",)
    FUNCTION = "load_insightface"
    CATEGORY = "pulid"

    def load_insightface(self, provider):
        import onnxruntime

        # Force specific providers and their order
        providers = []
        if provider == "CUDA":
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "gpu_mem_limit": int(1024 * 1024 * 1024 * 1.7),  # 1.7 GB
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    },
                ),
                "CPUExecutionProvider",
            ]
        elif provider == "ROCM":
            providers = [
                ("ROCMExecutionProvider", {"device_id": 0}),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]

        # Set global ONNX providers
        onnxruntime.set_default_logger_severity(3)  # Set to warning level
        current_providers = onnxruntime.get_available_providers()
        logging.info(f"Available ONNX providers: {current_providers}")

        # Initialize FaceAnalysis with specific providers
        model = FaceAnalysis(
            name="antelopev2", root=INSIGHTFACE_DIR, providers=providers
        )
        model.prepare(ctx_id=0, det_size=(640, 640))

        return (model,)


class PulidFluxEvaClipLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
        }

    RETURN_TYPES = ("EVA_CLIP",)
    FUNCTION = "load_eva_clip"
    CATEGORY = "pulid"

    def load_eva_clip(self):
        from .eva_clip.factory import create_model_and_transforms

        model, _, _ = create_model_and_transforms(
            "EVA02-CLIP-L-14-336", "eva_clip", force_custom_clip=True
        )

        model = model.visual

        eva_transform_mean = getattr(model, "image_mean", OPENAI_DATASET_MEAN)
        eva_transform_std = getattr(model, "image_std", OPENAI_DATASET_STD)
        if not isinstance(eva_transform_mean, (list, tuple)):
            model["image_mean"] = (eva_transform_mean,) * 3
        if not isinstance(eva_transform_std, (list, tuple)):
            model["image_std"] = (eva_transform_std,) * 3

        return (model,)


class ApplyPulidFlux:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "pulid_flux": ("PULIDFLUX",),
                "eva_clip": ("EVA_CLIP",),
                "face_analysis": ("FACEANALYSIS",),
                "image": ("IMAGE",),
                "weight": (
                    "FLOAT",
                    {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05},
                ),
                "start_at": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "end_at": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "fusion": (
                    [
                        "mean",
                        "concat",
                        "max",
                        "norm_id",
                        "max_token",
                        "auto_weight",
                        "train_weight",
                    ],
                ),
                "fusion_weight_max": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "fusion_weight_min": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "train_step": (
                    "INT",
                    {"default": 1000, "min": 0, "max": 20000, "step": 1},
                ),
                "use_gray": (
                    "BOOLEAN",
                    {"default": True, "label_on": "enabled", "label_off": "disabled"},
                ),
            },
            "optional": {
                "attn_mask": ("MASK",),
                "prior_image": ("IMAGE",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_pulid_flux"
    CATEGORY = "pulid"

    def __init__(self):
        self.pulid_data_dict = None

    def apply_pulid_flux(
        self,
        model,
        pulid_flux,
        eva_clip,
        face_analysis,
        image,
        weight,
        start_at,
        end_at,
        prior_image=None,
        fusion="mean",
        fusion_weight_max=1.0,
        fusion_weight_min=0.0,
        train_step=1000,
        use_gray=True,
        attn_mask=None,
        unique_id=None,
    ):
        device = comfy.model_management.get_torch_device()
        model_dtype = model.model.diffusion_model.dtype

        def convert_dtype_for_operations(tensor, target_dtype):
            if not torch.is_tensor(tensor):
                return tensor
            if target_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                return tensor.to(torch.bfloat16)
            return tensor.to(target_dtype)

        def convert_model_dtype(model, target_dtype):
            if target_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                target_dtype = torch.bfloat16

            # Convert all model parameters including buffers
            for param in model.parameters():
                param.data = param.data.to(target_dtype)

            # Convert all buffers (important for batch norm, etc)
            for buffer in model.buffers():
                buffer.data = buffer.data.to(target_dtype)

            model.to(dtype=target_dtype)
            return model

        # Convert EVA CLIP model parameters
        eva_clip = convert_model_dtype(eva_clip, model_dtype)
        eva_clip.to(device)

        # Convert PuLID Flux model parameters
        pulid_flux = convert_model_dtype(pulid_flux, model_dtype)
        pulid_flux.to(device)

        # Rest of the code remains the same as before...
        if attn_mask is not None:
            if attn_mask.dim() > 3:
                attn_mask = attn_mask.squeeze(-1)
            elif attn_mask.dim() < 3:
                attn_mask = attn_mask.unsqueeze(0)
            attn_mask = convert_dtype_for_operations(attn_mask, model_dtype)

        if prior_image is not None:
            prior_image = resize_with_pad(
                prior_image.to(image.device, dtype=image.dtype),
                target_size=(image.shape[1], image.shape[2]),
            )
            image = torch.cat((prior_image, image), dim=0)
        image = tensor_to_image(image)

        face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            device=device,
        )

        face_helper.face_parse = None
        face_helper.face_parse = init_parsing_model(model_name="bisenet", device=device)
        face_helper.face_parse = convert_model_dtype(
            face_helper.face_parse, model_dtype
        )

        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        cond = []

        for i in range(image.shape[0]):
            iface_embeds = None
            for size in [(size, size) for size in range(640, 256, -64)]:
                face_analysis.det_model.input_size = size
                face_info = face_analysis.get(image[i])
                if face_info:
                    face_info = sorted(
                        face_info,
                        key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                    )[-1]
                    iface_embeds = convert_dtype_for_operations(
                        torch.from_numpy(face_info.embedding).unsqueeze(0), model_dtype
                    ).to(device)
                    break
            else:
                logging.warning(f"Warning: No face detected in image {str(i)}")
                continue

            face_helper.clean_all()
            face_helper.read_image(image[i])
            face_helper.get_face_landmarks_5(only_center_face=True)
            face_helper.align_warp_face()

            if len(face_helper.cropped_faces) == 0:
                continue

            align_face = face_helper.cropped_faces[0]
            align_face = (
                image_to_tensor(align_face).unsqueeze(0).permute(0, 3, 1, 2).to(device)
            )
            align_face = convert_dtype_for_operations(align_face, model_dtype)

            normalized_face = functional.normalize(
                align_face, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            )
            normalized_face = convert_dtype_for_operations(normalized_face, model_dtype)

            parsing_out = face_helper.face_parse(normalized_face)[0]
            parsing_out = parsing_out.argmax(dim=1, keepdim=True)
            bg = sum(parsing_out == i for i in bg_label).bool()
            white_image = torch.ones_like(align_face)

            if use_gray:
                _align_face = to_gray(align_face)
            else:
                _align_face = align_face
            face_features_image = torch.where(bg, white_image, _align_face)

            face_features_image = functional.resize(
                face_features_image,
                eva_clip.image_size,
                (
                    transforms.InterpolationMode.BICUBIC
                    if "cuda" in device.type
                    else transforms.InterpolationMode.NEAREST
                ),
            ).to(device)
            face_features_image = convert_dtype_for_operations(
                face_features_image, model_dtype
            )
            face_features_image = functional.normalize(
                face_features_image, eva_clip.image_mean, eva_clip.image_std
            )

            id_cond_vit, id_vit_hidden = eva_clip(
                face_features_image,
                return_all_features=False,
                return_hidden=True,
                shuffle=False,
            )
            id_cond_vit = convert_dtype_for_operations(id_cond_vit, model_dtype)

            for idx in range(len(id_vit_hidden)):
                id_vit_hidden[idx] = convert_dtype_for_operations(
                    id_vit_hidden[idx], model_dtype
                )

            id_cond_vit = torch.div(id_cond_vit, torch.norm(id_cond_vit, 2, 1, True))
            id_cond = torch.cat([iface_embeds, id_cond_vit], dim=-1)
            cond.append(pulid_flux.get_embeds(id_cond, id_vit_hidden))

        if not cond:
            logging.warning(
                "PuLID warning: No faces detected in any of the given images, returning unmodified model."
            )
            return (model,)

        # Handle fusion with type conversion
        if fusion == "mean":
            cond = torch.cat(cond).to(device)
            cond = convert_dtype_for_operations(cond, model_dtype)
            if cond.shape[0] > 1:
                cond = torch.mean(cond, dim=0, keepdim=True)
        elif fusion == "concat":
            cond = torch.cat(cond, dim=1).to(device)
            cond = convert_dtype_for_operations(cond, model_dtype)
        elif fusion == "max":
            cond = torch.cat(cond).to(device)
            cond = convert_dtype_for_operations(cond, model_dtype)
            if cond.shape[0] > 1:
                cond = torch.max(cond, dim=0, keepdim=True)[0]
        elif fusion == "norm_id":
            cond = torch.cat(cond).to(device)
            cond = convert_dtype_for_operations(cond, model_dtype)
            if cond.shape[0] > 1:
                norm = torch.norm(cond, dim=(1, 2))
                norm = norm / torch.sum(norm)
                cond = torch.einsum("wij,w->ij", cond, norm).unsqueeze(0)
        elif fusion == "max_token":
            cond = torch.cat(cond).to(device)
            cond = convert_dtype_for_operations(cond, model_dtype)
            if cond.shape[0] > 1:
                norm = torch.norm(cond, dim=2)
                _, idx = torch.max(norm, dim=0)
                cond = torch.stack([cond[j, i] for i, j in enumerate(idx)]).unsqueeze(0)
        elif fusion == "auto_weight":
            cond = torch.cat(cond).to(device)
            cond = convert_dtype_for_operations(cond, model_dtype)
            if cond.shape[0] > 1:
                norm = torch.norm(cond, dim=2)
                order = torch.argsort(norm, descending=False, dim=0)
                regular_weight = convert_dtype_for_operations(
                    torch.linspace(
                        fusion_weight_min,
                        fusion_weight_max,
                        norm.shape[0],
                        device=device,
                    ),
                    model_dtype,
                )

                _cond = []
                for i in range(cond.shape[1]):
                    o = order[:, i]
                    _cond.append(
                        torch.einsum("ij,i->j", cond[:, i, :], regular_weight[o])
                    )
                cond = torch.stack(_cond, dim=0).unsqueeze(0)
        elif fusion == "train_weight":
            cond = torch.cat(cond).to(device)
            cond = convert_dtype_for_operations(cond, model_dtype)
            if cond.shape[0] > 1:
                if train_step > 0:
                    with torch.inference_mode(False):
                        cond = online_train(cond, device=cond.device, step=train_step)
                else:
                    cond = torch.mean(cond, dim=0, keepdim=True)

        sigma_start = model.get_model_object("model_sampling").percent_to_sigma(
            start_at
        )
        sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_at)

        flux_model = model.model.diffusion_model
        if not hasattr(flux_model, "pulid_ca"):
            flux_model.pulid_ca = pulid_flux.pulid_ca
            flux_model.pulid_double_interval = pulid_flux.double_interval
            flux_model.pulid_single_interval = pulid_flux.single_interval
            flux_model.pulid_data = {}
            new_method = forward_orig.__get__(flux_model, flux_model.__class__)
            setattr(flux_model, "forward_orig", new_method)

        flux_model.pulid_data[unique_id] = {
            "weight": weight,
            "embedding": cond,
            "sigma_start": sigma_start,
            "sigma_end": sigma_end,
        }

        self.pulid_data_dict = {"data": flux_model.pulid_data, "unique_id": unique_id}

        return (model,)

    def __del__(self):
        if self.pulid_data_dict:
            del self.pulid_data_dict["data"][self.pulid_data_dict["unique_id"]]
            del self.pulid_data_dict


NODE_CLASS_MAPPINGS = {
    "PulidFluxModelLoader": PulidFluxModelLoader,
    "PulidFluxInsightFaceLoader": PulidFluxInsightFaceLoader,
    "PulidFluxEvaClipLoader": PulidFluxEvaClipLoader,
    "ApplyPulidFlux": ApplyPulidFlux,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PulidFluxModelLoader": "Load PuLID Flux Model",
    "PulidFluxInsightFaceLoader": "Load InsightFace (PuLID Flux)",
    "PulidFluxEvaClipLoader": "Load Eva Clip (PuLID Flux)",
    "ApplyPulidFlux": "Apply PuLID Flux",
}

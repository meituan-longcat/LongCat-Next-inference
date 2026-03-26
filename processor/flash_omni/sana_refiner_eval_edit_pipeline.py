# sana_refiner_edit_pipeline.py
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image

from diffusers import DiffusionPipeline
from diffusers.models import AutoencoderDC, SanaTransformer2DModel
from diffusers.schedulers import (
    FlowMatchEulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.utils.torch_utils import randn_tensor, get_device, is_torch_version
from tqdm import tqdm
from diffusers.image_processor import PixArtImageProcessor

@dataclass
class SanaRefinerEditOutput:
    images: List[Image.Image]  # or when output_type='pt' is returned Tensor[B,3,H,W]


def _retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        if "timesteps" not in set(scheduler.set_timesteps.__code__.co_varnames):
            raise ValueError("This scheduler doesn't support custom `timesteps`.")
        scheduler.set_timesteps(timesteps=timesteps, device=device)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        if "sigmas" not in set(scheduler.set_timesteps.__code__.co_varnames):
            raise ValueError("This scheduler doesn't support custom `sigmas`.")
        scheduler.set_timesteps(sigmas=sigmas, device=device)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class SanaRefinerEditEvalPipeline(DiffusionPipeline):
    """
    edit-style refiner inference pipeline:
    - condition tokens: encoder_hidden_states = [B, L_total, D_img] (from ret.last_latent)
    - reference image: image: encode to VAE latent as **extra channel** and concatenate with noisy latent for channel dimension
    - transformer needs to support `in_channels = latent_ch + ref_latent_ch` (see the init extension function below):
    """

    def __init__(
        self,
        vae: AutoencoderDC,
        transformer: SanaTransformer2DModel,
        scheduler: Union[FlowMatchEulerDiscreteScheduler, DPMSolverMultistepScheduler],
        cond_resampler: torch.nn.Module,  # tokens -> [B, H_down*W_down, D_joint] support (src_hw)->(tgt_hw)
        cond_proj: torch.nn.Module,       # D_joint -> transformer required hidden dim
        condition_type=None,
    ):
        super().__init__()
        self.condition_type = condition_type
        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            cond_resampler=cond_resampler,
            cond_proj=cond_proj,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.encoder_block_out_channels) - 1)
            if hasattr(self.vae.config, "encoder_block_out_channels")
            else 32
        )
        self.image_processor = PixArtImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def _split_tokens(self, encoder_hidden_states: torch.Tensor, grid_thw_list: List[Tuple[int, int, int]], ):
        # split the total sequence into tokens for each image → list of Tensor[B, h_i*w_i, D]
        if self.condition_type == "quants":
            splits = [int(h) * int(w) // 4 for (_, h, w) in grid_thw_list]
        else:
            splits = [int(h) * int(w) for (_, h, w) in grid_thw_list]
        return list(torch.split(encoder_hidden_states, splits, dim=1))

    def _vae_hw_from_grid(self, h: int, w: int, patch_size: int):
        # (grid_h, grid_w, patch_size) -> VAE latent grid
        Hp = int(h) * int(patch_size)
        Wp = int(w) * int(patch_size)
        H_down = max(1, math.ceil(Hp / self.vae_scale_factor))
        W_down = max(1, math.ceil(Wp / self.vae_scale_factor))
        return H_down, W_down

    def _prepare_latents(self, batch: int, channels: int, H_down: int, W_down: int, device, dtype, generator=None):
        shape = (batch, channels, H_down, W_down)
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    @torch.no_grad()
    def _encode_image(
        self,
        ref_img: Union[Image.Image, torch.Tensor],
        H_down: int,
        W_down: int,
        device,
        dtype,
    ) -> torch.Tensor:
        """
        encode the reference image to VAE latent, align the size to (H_down*sf, W_down*sf) before encoding.
        返回：[1, C_latent, H_down, W_down]，值域与 latent 一致（乘 scaling_factor）
        """
        sf = self.vae_scale_factor
        target_h, target_w = H_down * sf, W_down * sf

        if isinstance(ref_img, Image.Image):
            img = self.image_processor.preprocess(ref_img)  # (H,W,3) uint8
        elif isinstance(ref_img, torch.Tensor):
            img = ref_img
            if img.ndim == 3:
                img = img.unsqueeze(0)  # [1,3,H,W]
            if img.dtype != torch.float32 and img.dtype != torch.float16 and img.dtype != torch.bfloat16:
                img = img.float()
        else:
            raise TypeError("Unsupported image type. Use PIL.Image or torch.Tensor.")

        img = F.interpolate(img, size=(target_h, target_w), mode="bilinear", align_corners=False)
        img = img.to(device=device, dtype=self.vae.dtype)

        lat = self.vae.encode(img).latent
        # cast to transformer dtype (involved in concat and subsequent calculations)
        return lat.to(device=device, dtype=dtype)

    @torch.no_grad()
    def _denoise_once(
        self,
        cond_tokens: torch.Tensor,        # [B, h*w, D_in]
        src_hw: Tuple[int, int],          # (h, w)
        tgt_hw: Tuple[int, int],          # (H_down, W_down)
        num_inference_steps: int = 20,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        ref_img: Optional[torch.Tensor] = None,             # 期望 [-1,1] Bx3xHpxWp，若传入则会经 VAE 编码
        image_guidance_scale: float = 1.5,                # i >= 1 启用 image guidance
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
    ):

        device = cond_tokens.device
        weight_dtype = self.transformer.dtype

        B, L, _ = cond_tokens.shape
        h, w = src_hw
        H_down, W_down = tgt_hw

        # 1) 条件重采样到 VAE latent 网格（+ 维度投影），与原逻辑一致
        cond_tokens = cond_tokens.to(device=device, dtype=weight_dtype)
        cond_aligned = self.cond_resampler(cond_tokens, src_hw=src_hw, tgt_hw=tgt_hw)    # [B, H_down*W_down, D_joint?]
        cond_feats = self.cond_proj(cond_aligned)                                       # [B, H'*W', D_cond]

        # 1b) 负向条件对齐到同一空间（如未提供，退化为全零——等价于空 negative prompt）
        if guidance_scale > 1.0:
            uncond_feats = torch.zeros_like(cond_feats).to(device=device, dtype=weight_dtype)
        else:
            uncond_feats = None

        # 2) 准备时间步
        timesteps, _ = _retrieve_timesteps(
            self.scheduler, num_inference_steps=num_inference_steps, device=device,
            timesteps=timesteps, sigmas=sigmas
        )

        # 3) 初始噪声 latents
        num_channels_latents = self.transformer.config.in_channels  # 注意：若开启 image guidance，会检查通道数
        latents = self._prepare_latents(
            batch=B, channels= self.vae.config.latent_channels,
            H_down=H_down, W_down=W_down,
            device=device, dtype=weight_dtype, generator=generator
        )
        # 3b) 准备 image_latents（若需要 image guidance）
        if ref_img is not None:
            image_latents = self._encode_image(ref_img, H_down, W_down, device, weight_dtype)
            num_channels_image = image_latents.shape[1]
            # important: in_channels must be equal to the sum of noise channels and image channels (consistent with instruct-pix2pix)
            expected_in_channels = self.vae.config.latent_channels + num_channels_image
            if self.transformer.config.in_channels != expected_in_channels:
                raise ValueError(
                    f"in_channels mismatch: transformer.config.in_channels={self.transformer.config.in_channels}，"
                    f"but image-guidance expected {expected_in_channels} (= latent {self.vae.config.latent_channels}"
                    f" + image {num_channels_image}). please adjust the model configuration or close image_guidance."
                )
        do_image_cfg = (guidance_scale > 1.0) and (image_guidance_scale >= 1.0) and ref_img is not None


        # 4) denoise loop
        for i, t in enumerate(timesteps):
            # a) construct the input for three-way batch
            if do_image_cfg:
                # basic noise (B, Clat, H, W) -> three parts
                latent_model_input = torch.cat([latents] * 3, dim=0).to(dtype=weight_dtype)

                # scheduler input scaling (for each part)
                if hasattr(self.scheduler, "scale_model_input"):
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # concatenate image branch channels: [img, img, zeros]
                img_lat_B = image_latents
                zeros_img = torch.zeros_like(img_lat_B)
                img3 = torch.cat([img_lat_B, img_lat_B, zeros_img], dim=0)  # (3B, Cimg, H, W)

                model_input = torch.cat([latent_model_input, img3], dim=1)  # (3B, Clat+Cimg, H, W)

                # three conditions: [text, image, uncond]
                cond3 = torch.cat([cond_feats, uncond_feats, uncond_feats], dim=0)  # (3B, Lc, D)

                # timestep repeat by batch
                t_in = t.expand(model_input.shape[0]).to(device=device, dtype=weight_dtype)
                if hasattr(self.transformer.config, "timestep_scale"):
                    t_in = t_in * self.transformer.config.timestep_scale

                # forward
                noise_pred = self.transformer(
                    hidden_states=model_input,
                    encoder_hidden_states=cond3,
                    encoder_attention_mask=None,
                    timestep=t_in,
                    return_dict=False,
                )[0].float()
                # split by batch dim
                noise_text, noise_image, noise_uncond = noise_pred.chunk(3, dim=0)

                noise_pred = noise_uncond + guidance_scale * (noise_text - noise_image) \
                            + image_guidance_scale * (noise_image - noise_uncond)

            else:
                # regular (without image guidance)
                latent_model_input = torch.cat([latents, image_latents], dim=1).to(dtype=weight_dtype)
                
                if hasattr(self.scheduler, "scale_model_input"):
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                t_in = t.expand(latent_model_input.shape[0]).to(device=device, dtype=weight_dtype)
                if hasattr(self.transformer.config, "timestep_scale"):
                    t_in = t_in * self.transformer.config.timestep_scale

                # single condition
                cond = cond_feats

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=cond,
                    encoder_attention_mask=None,
                    timestep=t_in,
                    return_dict=False,
                )[0].float()

            # b) scheduler step
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # 5) decode to image
        latents_vae = latents.to(self.vae.dtype) / self.vae.config.scaling_factor
        image_out = self.vae.decode(latents_vae, return_dict=False)[0]  # [-1,1]
        return image_out  



    @torch.no_grad()
    def __call__(
        self,
        *,
        encoder_hidden_states: torch.Tensor,           # [B, L_total, D_img]
        grid_thw_list: List[Tuple[int, int, int]],     # [(t,h,w), ...] match L_total splitting
        image: Union[Image.Image, torch.Tensor, List[Union[Image.Image, torch.Tensor]]],
        patch_size: int = 14,
        num_inference_steps: int = 20,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 1.1,
        image_guidance_scale: float = 1.1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",                      # 'pil' or 'pt'
        return_dict: bool = True,
        **kwargs,
    ) -> Union[SanaRefinerEditOutput, List[Image.Image], torch.Tensor]:

        # split [B, L_total, D] into tokens for each image (here assume B==1 or each image is denoised separately)
        token_chunks = self._split_tokens(encoder_hidden_states, grid_thw_list)

        # reference image expanded to list
        if isinstance(image, (Image.Image, torch.Tensor)):
            ref_list = [image] * len(token_chunks)
        else:
            ref_list = image
            assert len(ref_list) == len(token_chunks), "len(ref_list) must match number of images"

        imgs_out = []
        for tok, (_, h, w), img_any in tqdm(
            iterable=zip(token_chunks, grid_thw_list, ref_list),
            total=len(token_chunks),
            desc="Editing images",
        ):
            H_down, W_down = self._vae_hw_from_grid(h, w, patch_size=patch_size)
            imgs = self._denoise_once(
                cond_tokens=tok,
                src_hw=(int(h), int(w)),
                tgt_hw=(H_down, W_down),
                ref_img=img_any,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                sigmas=sigmas,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                generator=generator,
                output_type=output_type,
            )
            if output_type == "pil":
                imgs_out += self.image_processor.postprocess(imgs, output_type=output_type)
            else:
                imgs_out = imgs if len(imgs_out) == 0 else torch.cat([imgs_out, imgs], dim=0)

        if not return_dict:
            return imgs_out
        return SanaRefinerEditOutput(images=imgs_out)

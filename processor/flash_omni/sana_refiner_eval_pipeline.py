import math
import copy
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image

from diffusers import DiffusionPipeline
from diffusers.models import AutoencoderDC, SanaTransformer2DModel
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor, get_device, is_torch_version
from tqdm import tqdm

@dataclass
class SanaRefinerOutput:
    images: List[Image.Image]  # 或者当 output_type='pt' 时，images 是 [B,3,H,W] 的 tensor


def _retrieve_timesteps(scheduler, num_inference_steps: Optional[int] = None,
                        device: Optional[Union[str, torch.device]] = None,
                        timesteps: Optional[List[int]] = None,
                        sigmas: Optional[List[float]] = None):
    # 兼容 diffusers 的 set_timesteps 行为
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


class SanaRefinerEvalPipeline(DiffusionPipeline):
    """
    轻量 refiner 推理管线：
    - 输入：encoder_hidden_states = [B, L_total, D_img]（来自 ret.last_latent）
           grid_thw_list = [(t,h,w), ...]（与上面的序列一一对应）
    - 模块：vae (AutoencoderDC), transformer (SanaTransformer2DModel), scheduler, cond_resampler（对齐 L 和 VAE 网格）
    - 输出：每个图一张 PIL（或 tensor）
    """

    def __init__(
        self,
        vae: AutoencoderDC,
        transformer: SanaTransformer2DModel,
        scheduler: Union[FlowMatchEulerDiscreteScheduler, DPMSolverMultistepScheduler],
        cond_resampler: torch.nn.Module,  # 你自己实现/已有：tokens->[B,H*W,D_cond]
        cond_proj: torch.nn.Module,
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
        # VAE 下采样因子：AutoencoderDC 一般是 32
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.encoder_block_out_channels) - 1)
            if hasattr(self.vae.config, "encoder_block_out_channels")
            else 32
        )

    @torch.no_grad()
    def _postprocess(self, images: torch.Tensor, output_type: str):
        """
        images: [B,3,H,W] in [-1,1] or [0,1]
        """
        if output_type == "pt":
            # 归一化到 [0,1]
            return images.clamp(0, 1)

        images = (images * 255.0).clamp(0, 255).round().to(torch.uint8)  # [B,3,H,W]
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        pil_images = []
        for i in range(images.shape[0]):
            pil_images.append(Image.fromarray(images[i]))
        return pil_images

    def _split_tokens(self, encoder_hidden_states: torch.Tensor, grid_thw_list: List[Tuple[int, int, int]]):
        """
        把总序列按 (h*w) 切回每张图的 token
        返回：list of Tensor[B, h_i*w_i, D]
        """
        if self.condition_type == "quants":
            splits = [int(h) * int(w) // 4 for (_, h, w) in grid_thw_list]
        else:
            splits = [int(h) * int(w) for (_, h, w) in grid_thw_list]
        return list(torch.split(encoder_hidden_states, splits, dim=1))

    def _vae_hw_from_grid(self, h: int, w: int, patch_size: int):
        """
        把 QwenVL 的 (h,w) grid（patch_size=14）映射到 VAE 的 latent 网格 (H_down, W_down)
        采用策略#2：先把条件映射到 VAE 网格，避免模型学“超分”。
        """
        Hp = int(h) * int(patch_size)
        Wp = int(w) * int(patch_size)
        H_down = max(1, math.ceil(Hp / self.vae_scale_factor))
        W_down = max(1, math.ceil(Wp / self.vae_scale_factor))
        return H_down, W_down

    def _prepare_latents(self, batch: int, channels: int, H_down: int, W_down: int,
                         device, dtype, generator=None):
        shape = (batch, channels, H_down, W_down)
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    @torch.no_grad()
    def _denoise_once(
        self,
        cond_tokens: torch.Tensor,        # [B, h*w, D_in]
        src_hw: Tuple[int, int],          # (h, w)
        tgt_hw: Tuple[int, int],          # (H_down, W_down)
        num_inference_steps: int = 20,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 1.0,      # 这里只有 cond，没有负支路；>1.0 等价于不用
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
    ):
        device = cond_tokens.device
        weight_dtype = self.transformer.dtype

        B, L, _ = cond_tokens.shape
        h, w = src_hw
        H_down, W_down = tgt_hw

        # 1) 条件重采样到 VAE latent 网格（+ 维度投影）
        cond_tokens = cond_tokens.to(device=device, dtype=weight_dtype)
        if self.cond_resampler is not None:
            cond_aligned = self.cond_resampler(cond_tokens, src_hw=src_hw, tgt_hw=tgt_hw)  # [B, H_down*W_down, D_joint]
        else:
            cond_aligned = cond_tokens

        # 2) 准备时间步
        timesteps, _ = _retrieve_timesteps(
            self.scheduler, num_inference_steps=num_inference_steps, device=device,
            timesteps=timesteps, sigmas=sigmas
        )

        # 3) 采样初始高斯 latent
        in_ch = self.transformer.config.in_channels
        latents = self._prepare_latents(
            batch=B, channels=in_ch, H_down=H_down, W_down=W_down,
            device=device, dtype=weight_dtype, generator=generator
        )

        # 4) 去噪循环
        for i, t in enumerate(timesteps):
            latent_model_input = latents  # 无 CFG
            # 兼容 Sana 的 timestep 缩放
            t_in = t.expand(latent_model_input.shape[0])
            if hasattr(self.transformer.config, "timestep_scale"):
                t_in = t_in * self.transformer.config.timestep_scale
            cond = self.cond_proj(cond_aligned.to(dtype=weight_dtype))
            noise_pred = self.transformer(
                hidden_states=latent_model_input.to(dtype=weight_dtype),
                encoder_hidden_states=cond,
                encoder_attention_mask=None,
                timestep=t_in,
                return_dict=False,
            )[0].float()

            # learned sigma 兼容：按标准 Sana/PixArt-Sigma 处理
            if self.transformer.config.out_channels // 2 == in_ch:
                noise_pred = noise_pred.chunk(2, dim=1)[0]

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # 5) 解码到图像
        latents_vae = latents.to(self.vae.dtype) / self.vae.config.scaling_factor
        # shift_factor（如果有）由 AutoencoderDC 内部处理 decode 前后的一致方式
        image = self.vae.decode(latents_vae, return_dict=False)[0]  # [-1,1]
        image = (image / 2 + 0.5).clamp(0, 1)                       # [0,1]

        return self._postprocess(image, output_type=output_type)

    @torch.no_grad()
    def __call__(
        self,
        *,
        encoder_hidden_states: torch.Tensor,           # [B, L_total, D_img]
        grid_thw_list: List[Tuple[int, int, int]],     # [(t,h,w), ...]
        patch_size: int = 14,
        num_inference_steps: int = 20,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",                      # 'pil' 或 'pt'
        return_dict: bool = True,
        config = None,
        **kwargs,
    ) -> Union[SanaRefinerOutput, Tuple[List[Image.Image]]]:

        self._interrupt = False
        if config is not None and not config.visual_decoder_config.enable:
            merged_grid_thw_list = [ (b , int(h)// config.visual_config.spatial_merge_size, int(w)// config.visual_config.spatial_merge_size) for (b, h, w) in grid_thw_list] # since the visual encoder will go through a token merger
            token_chunks = self._split_tokens(encoder_hidden_states, merged_grid_thw_list)
        else:
            token_chunks = self._split_tokens(encoder_hidden_states, grid_thw_list)
            
        # 把 [B, L_total, D] 切成每张图的 token

        imgs_out: List[Image.Image] = []
        for tok, (_, h, w) in tqdm(zip(token_chunks, grid_thw_list), total=len(token_chunks), desc="Processing images"):
            H_down, W_down = self._vae_hw_from_grid(h, w, patch_size=patch_size)
            imgs = self._denoise_once(
                cond_tokens=tok,
                src_hw=(int(h), int(w)),
                tgt_hw=(H_down, W_down),
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                sigmas=sigmas,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type=output_type,
            )
            # _denoise_once 对单图返回 List[PIL] 或 Tensor[B,3,H,W]，这里 B 肯定是 1
            if output_type == "pil":
                imgs_out += imgs
            else:
                # 'pt'：拼接 batch 维
                if len(imgs_out) == 0:
                    imgs_out = imgs  # type: ignore
                else:
                    imgs_out = torch.cat([imgs_out, imgs], dim=0)  # type: ignore

        if not return_dict:
            return imgs_out

        return SanaRefinerOutput(images=imgs_out)

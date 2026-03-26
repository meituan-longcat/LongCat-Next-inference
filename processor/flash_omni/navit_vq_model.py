from typing import List, Optional, Tuple, Union, Any
import torch, math
import torch.distributed
import torch.utils.checkpoint
from torch import nn
import transformers
from flash_attn import flash_attn_varlen_func
from transformers.activations import ACT2FN
from torch.nn import functional as F
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput, BaseModelOutputWithPast
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    # Qwen2_5_VisionPatchEmbed,
    # QWEN2_5_VL_ATTENTION_CLASSES,
    # Qwen2_5_VisionRotaryEmbedding,
    # Qwen2_5_VLVisionBlock,
    # Qwen2_5_VisionPatchEmbed,
    Qwen2RMSNorm
)
from .configuration_omni import OmniConfig
from .visual_vector_quantize import RQBottleneck

class Qwen2_5_VisionRotaryEmbedding_Modified(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        # self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int, device: torch.device) -> torch.Tensor:
        self.inv_freq = self.inv_freq.to(device)
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs
    
# modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
class OmniVisualEncoder(transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel):
    def __init__(self, config):
        config._attn_implementation = 'flash_attention_2'
        super().__init__(config)
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding_Modified(config.hidden_size // config.num_heads // 2)
        self.gradient_checkpointing = True  
        self._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint
        self.merge_size = config.merge_size if hasattr(config, 'merge_size') else 2
        del self.merger # register visual.merger in visual_bridge_model
    
    def get_dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.down_proj.weight.dtype

    def get_device(self) -> torch.device:
        return self.blocks[0].mlp.down_proj.weight.device
    
    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size, device=grid_thw.device)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(
        self,
        pixel_values: torch.Tensor, 
        grid_thw: torch.Tensor,
        require_window_index: bool = False,
    ):
        '''
        pixel_values.shape=[NumOfPatches, 1176]
        grid_thw.shape=[NumOfSamples, 3]. [grid_t,grid_h,grid_w]
        '''
        hidden_states = pixel_values.to(torch.bfloat16)
        grid_thw = grid_thw.to(pixel_values.device)
        
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings)
            else:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens_now,
                    position_embeddings=position_embeddings,
                )
        # hidden_states = self.merger(hidden_states)
        # reverse_indices = torch.argsort(window_index)
        # hidden_states = hidden_states[reverse_indices, :]
        
        # hidden_states.shape=[24292,1280],  window_index.shape=[6073,]
        if require_window_index:
            return hidden_states, window_index
        return hidden_states
    
    @torch.no_grad()
    def fake_input(self, device):
        merge_size = max(self.merge_size, self.config.spatial_merge_size)
        fake_image = torch.zeros([
            1,
            self.config.temporal_patch_size,
            3,
            merge_size // self.config.spatial_merge_size,
            self.config.spatial_merge_size,
            self.config.patch_size,
            merge_size // self.config.spatial_merge_size,
            self.config.spatial_merge_size,
            self.config.patch_size,
        ], dtype=torch.float32, device=device)
        patches = fake_image.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            merge_size * merge_size, 3 * self.config.temporal_patch_size * self.config.patch_size * self.config.patch_size
        )
        return [flatten_patches], [(1, merge_size, merge_size)], [1]

class OmniVisualBridge(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.merge_size = self.config.merge_size if hasattr(self.config, 'merge_size') else 2
        self.hidden_size = self.config.hidden_size * (self.merge_size**2)
        self.window_index = self.config.window_size
        # self.ln_q = nn.LayerNorm(self.config.hidden_size, eps=1e-6)
        self.ln_q = Qwen2RMSNorm(self.config.hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.config.out_hidden_size),
        )
        
    def forward(self, x: torch.Tensor, window_index) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        reverse_indices = torch.argsort(window_index)
        x = x[reverse_indices, :]

        return x

@dataclass
class VisualQuantizerOutput(ModelOutput):
    indices: Optional[torch.LongTensor] = None # 离散indices
    quants: Optional[torch.FloatTensor] = None # 查codebook查出来的离散化特征 
    projected_embeds: Optional[torch.FloatTensor] = None # 准备接入LLM的特征

    min_encodings: Optional[torch.LongTensor] = None
    perplexity: Optional[Union[torch.Tensor, float]] = None

    vq_loss: Optional[Union[torch.Tensor, float]] = None
    cmt_loss: Optional[Union[torch.Tensor, float]] = None
    entropy_loss: Optional[Union[torch.Tensor, float]] = None
    codebook_usage: Optional[Union[List, float]] = None

    recon_feat_loss: Optional[Union[torch.Tensor, float]] = None

def set_0_if_not_None(x):
    if x is not None:
        return x * 0
    else:
        return x

class VisualQuantizer(nn.Module):
    def __init__(self, quantizer_config):
        super().__init__()

        self.config = quantizer_config
        self.depth = self.config.depth
        self.decay = self.config.decay
        self.codebook_size = self.config.codebook_size
        self.codebook_dim = self.config.codebook_dim
        self.shared_codebook = self.config.shared_codebook
        self.restart_unused_codes = self.config.restart_unused_codes
        self.in_channels = self.config.in_channels

        self.vq_loss_ratio                = self.config.vq_loss_ratio
        self.entropy_loss_ratio           = self.config.entropy_loss_ratio
        self.commit_loss_ratio            = self.config.commit_loss_ratio
        self.feature_reconstruction_ratio = self.config.feature_reconstruction_ratio

        code_h_w = int(448 / 14) # 随便填一个，不影响
        latent_shape = [code_h_w, code_h_w, self.codebook_dim] # 256
        code_shape = [code_h_w, code_h_w, self.depth]

        self.quantize = RQBottleneck(
            latent_shape=latent_shape, # rvq默认 latent_shape: [ 8, 8, 256 ]  # could be inferred: H=W=resolution / (2 ** num_down), D=embed_dim
            code_shape=code_shape, # rvq默认 code_shape: [ 8, 8, 4 ]
            n_embed=self.codebook_size, # rvq默认 n_embed: 16384
            decay=self.decay, # rvq默认 decay: 0.99
            shared_codebook=self.shared_codebook, # rvq默认 shared_codebook: true
            restart_unused_codes=self.restart_unused_codes, # rvq默认 restart_unused_codes: true
        )

        if self.config.quant_conv:
            self.quant_conv = nn.Sequential(
                nn.LayerNorm(self.in_channels),
                nn.Linear(self.in_channels, self.in_channels),
                nn.GELU(),
                nn.Linear(self.in_channels, self.codebook_dim)
            )
        else:
            self.quant_conv = None
    
    def encode(self, x, input_is_fake=False):
        L, D = x.shape
        to_qnt_feat = x.clone()
        to_qnt_feat = to_qnt_feat.unsqueeze(0) # [L, D] -> [1, L, D]
        N = 1
        
        if self.quant_conv is not None:
            to_qnt_feat = self.quant_conv(to_qnt_feat) # 实际上是一个MLP

        # quantizer要求输入是nchw格式的,因此为了兼容后面的流程, 这里由N,L,d -> N,1,L,d -> N,d,1,L
        to_qnt_feat = to_qnt_feat.reshape(N, 1, L, self.codebook_dim).permute(0,3,1,2)
        if self.config.quantizer_type == "rq":
            to_qnt_feat = to_qnt_feat.permute(0, 2, 3, 1).contiguous() # N,d,1,L -> N,1,L,d
            quant, emb_loss, info = self.quantize(to_qnt_feat, input_is_fake)
            info = info.reshape(-1, info.shape[-1]) # n,h,w,lv -> n*h*w,lv
            info = [None, None, info] # 兼容其他量化
            quant = quant.permute(0, 3, 1, 2).contiguous() # rvq需要  # N,1,L,d -> N,d,1,L
        else:
            quant, emb_loss, info = self.quantize(to_qnt_feat, input_is_fake)
        return quant, emb_loss, info, x.detach() # align_feature不更新

    def forward(self, x,grid_thw=None, fake_input=False,):        

        quant, (vq_loss, commit_loss, entropy_loss, codebook_usage), (perplexity, min_encodings, min_encoding_indices), align_feature = \
            self.encode(x, fake_input)

        if fake_input:
            vq_loss         = set_0_if_not_None(vq_loss)
            commit_loss     = set_0_if_not_None(commit_loss)
            entropy_loss    = set_0_if_not_None(entropy_loss)

        return VisualQuantizerOutput(
            indices = min_encoding_indices,
            quants  = quant,

            vq_loss = vq_loss*self.vq_loss_ratio if vq_loss is not None else None,
            cmt_loss = commit_loss*self.commit_loss_ratio if vq_loss is not None else None,
            entropy_loss = entropy_loss*self.entropy_loss_ratio if vq_loss is not None else None,

            codebook_usage = codebook_usage,
            min_encodings = min_encodings,
            perplexity = perplexity
            )
import torch
from torch import nn
from torch.nn import functional as F
from flash_attn import flash_attn_func, flash_attn_varlen_func

# TP related imports
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
import re

# TP weight sharding patterns - Only Attention weights are sharded
# FFN weights are NOT sharded (kept as full weights for einsum computation)
_TP_COLUMN_PATTERNS = [
    r"transformer_layers\.\d+\.self_attention\.q_proj\.weight",
    r"transformer_layers\.\d+\.self_attention\.k_proj\.weight",
    r"transformer_layers\.\d+\.self_attention\.v_proj\.weight",
    r"transformer_layers\.\d+\.self_attention\.fused_qkv_proj\.weight",
    r"transformer_layers\.\d+\.self_attention\.q_proj\.bias",
    r"transformer_layers\.\d+\.self_attention\.v_proj\.bias",
    r"transformer_layers\.\d+\.self_attention\.fused_qkv_proj\.bias",
]
_TP_ROW_PATTERNS = [
    r"transformer_layers\.\d+\.self_attention\.out_proj\.weight",
]


def load_tp_state_dict(model, state_dict, tp_rank, tp_size):
    """Load state dict with TP-aware weight sharding.
    
    For TP layers, weights need to be sharded:
    - ColumnParallelLinear: shard on output_dim (dim=0)
    - RowParallelLinear: shard on input_dim (dim=1)
    
    Args:
        model: The model to load weights into
        state_dict: The state dict to load from
        tp_rank: Current TP rank
        tp_size: Total TP size
    """
    model_params = dict(model.named_parameters())
    
    new_state_dict = {}
    for name, param in state_dict.items():
        if name not in model_params:
            print(f"Warning: {name} not found in model parameters, skipping")
            continue
        
        model_param = model_params[name]
        
        is_column_parallel = any(re.fullmatch(p, name) for p in _TP_COLUMN_PATTERNS)
        is_row_parallel = any(re.fullmatch(p, name) for p in _TP_ROW_PATTERNS)
        
        if param.shape == model_param.shape:
            new_state_dict[name] = param
        elif is_column_parallel and param.shape[0] == model_param.shape[0] * tp_size:
            shard_size = param.shape[0] // tp_size
            start_idx = tp_rank * shard_size
            end_idx = start_idx + shard_size
            new_state_dict[name] = param[start_idx:end_idx].clone()
        elif is_row_parallel and param.shape[1] == model_param.shape[1] * tp_size:
            shard_size = param.shape[1] // tp_size
            start_idx = tp_rank * shard_size
            end_idx = start_idx + shard_size
            new_state_dict[name] = param[:, start_idx:end_idx].clone()
        else:
            new_state_dict[name] = param
    
    model.load_state_dict(new_state_dict, strict=True)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states

class MeituanWhisperAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, causal=False, window_size=(-1,-1)):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.causal = causal
        self.window_size = window_size

    def forward(self, hidden_states: torch.Tensor, seq_len: torch.Tensor):
        bsz, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, self.num_heads, self.head_dim)

        cu_len = F.pad(torch.cumsum(seq_len, dim=0), (1, 0), "constant", 0).to(torch.int32)
        max_seqlen = torch.max(seq_len).to(torch.int32).detach()
        attn_output = flash_attn_varlen_func(query_states, key_states, value_states, cu_len, cu_len, max_seqlen,
                                             max_seqlen, causal=self.causal, window_size=self.window_size)  # (bsz * qlen, nheads, headdim)
        attn_output = attn_output.reshape(bsz, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output

class FlashVarLenAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, causal=False, window_size=(-1,-1), use_fused_qkv=False, enable_tp=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_fused_qkv = use_fused_qkv
        self.enable_tp = enable_tp and get_tensor_model_parallel_world_size() > 1
        
        # Get TP info
        if self.enable_tp:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
            self.num_heads_per_partition = num_heads // self.tp_size
        else:
            self.tp_rank = 0
            self.tp_size = 1
            self.num_heads_per_partition = num_heads

        if self.use_fused_qkv:
            if self.enable_tp:
                # TP: Column parallel for fused QKV
                self.fused_qkv_proj = ColumnParallelLinear(
                    embed_dim, embed_dim * 3,
                    bias=True, gather_output=False,
                    tp_rank=self.tp_rank, tp_size=self.tp_size,
                )
            else:
                self.fused_qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
            self._register_load_state_dict_pre_hook(self._convert_state_dict_hook)
        else:
            if self.enable_tp:
                # TP: Column parallel for Q, K, V
                self.q_proj = ColumnParallelLinear(
                    embed_dim, embed_dim,
                    bias=True, gather_output=False,
                    tp_rank=self.tp_rank, tp_size=self.tp_size,
                )
                self.k_proj = ColumnParallelLinear(
                    embed_dim, embed_dim,
                    bias=False, gather_output=False,
                    tp_rank=self.tp_rank, tp_size=self.tp_size,
                )
                self.v_proj = ColumnParallelLinear(
                    embed_dim, embed_dim,
                    bias=True, gather_output=False,
                    tp_rank=self.tp_rank, tp_size=self.tp_size,
                )
            else:
                self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
                self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
                self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        
        # Output projection: Row parallel (all-reduce inside)
        if self.enable_tp:
            self.out_proj = RowParallelLinear(
                embed_dim, embed_dim,
                bias=True, input_is_parallel=True, reduce_results=True,
                tp_rank=self.tp_rank, tp_size=self.tp_size,
            )
        else:
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.causal = causal
        self.window_size = window_size
    
    def _convert_state_dict_hook(self, state_dict, prefix, local_metadata, strict, 
                                missing_keys, unexpected_keys, error_msgs):
        """在加载state_dict前转换权重格式"""
        # print(f"prefix: {prefix}", flush=True)
        # 查找需要融合的权重
        fused_qkv_proj_prefix = f"{prefix}fused_qkv_proj."
        
        # 检查原始QKV权重是否存在
        q_key = f"{prefix}q_proj.weight"
        k_key = f"{prefix}k_proj.weight" 
        v_key = f"{prefix}v_proj.weight"
        
        if all(key in state_dict for key in [q_key, k_key, v_key]):
            # print(f"Converting QKV weights for {fused_qkv_proj_prefix}")
            
            # 提取并移除原始权重
            q_weight = state_dict.pop(q_key)
            k_weight = state_dict.pop(k_key)
            v_weight = state_dict.pop(v_key)
            
            # 创建融合权重
            fused_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            state_dict[f"{fused_qkv_proj_prefix}weight"] = fused_weight
            
            # 处理bias
            q_bias_key = f"{prefix}q_proj.bias"
            k_bias_key = f"{prefix}k_proj.bias"
            v_bias_key = f"{prefix}v_proj.bias"
            
            if q_bias_key in state_dict:
                q_bias = state_dict.pop(q_bias_key)
            else:
                q_bias = torch.zeros(self.embed_dim, device=q_weight.device, dtype=q_weight.dtype)
                print(f"Warning: {q_bias_key} not found, using zeros")
            
            if k_bias_key in state_dict:
                k_bias = state_dict.pop(k_bias_key)
                print(f"Warning: Found {k_bias_key}, but k_proj should not have bias")
            else:
                k_bias = torch.zeros(self.embed_dim, device=k_weight.device, dtype=k_weight.dtype)
                print(f"Using zeros for k_bias as expected")
            
            if v_bias_key in state_dict:
                v_bias = state_dict.pop(v_bias_key)
            else:
                v_bias = torch.zeros(self.embed_dim, device=v_weight.device, dtype=v_weight.dtype)
                print(f"Warning: {v_bias_key} not found, using zeros")
            
            fused_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
            state_dict[f"{fused_qkv_proj_prefix}bias"] = fused_bias

    def forward(self, hidden_states: torch.Tensor):
        bsz, _ = hidden_states.size()

        if self.use_fused_qkv:
            qkv, _ = self.fused_qkv_proj(hidden_states) if self.enable_tp else (self.fused_qkv_proj(hidden_states), None)
            query_states, key_states, value_states = qkv.chunk(3, dim=-1)
        else:
            if self.enable_tp:
                query_states, _ = self.q_proj(hidden_states)
                key_states, _ = self.k_proj(hidden_states)
                value_states, _ = self.v_proj(hidden_states)
            else:
                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)
        
        # In TP mode, use num_heads_per_partition
        num_heads_for_attn = self.num_heads_per_partition if self.enable_tp else self.num_heads
        
        query_states = query_states.view(-1, 8, num_heads_for_attn, self.head_dim).contiguous()
        key_states = key_states.view(-1, 8, num_heads_for_attn, self.head_dim).contiguous()
        value_states = value_states.view(-1, 8, num_heads_for_attn, self.head_dim).contiguous()

        attn_output = flash_attn_func(query_states, key_states, value_states, causal=self.causal, window_size=self.window_size)
        
        # In TP mode, the output is sharded (embed_dim // tp_size)
        embed_dim_for_output = self.embed_dim // self.tp_size if self.enable_tp else self.embed_dim
        attn_output = attn_output.view(-1, embed_dim_for_output)
        
        if self.enable_tp:
            attn_output, _ = self.out_proj(attn_output)
        else:
            attn_output = self.out_proj(attn_output)
        
        return attn_output

class CasualDepthTransformerLayer(nn.Module):
    def __init__(self, hidden_size, image_head_transformer_dims, image_head_transformer_ffn_scale, depth, enable_tp=False):
        super().__init__()
        self.depth = depth
        self.llm_hidden_size = hidden_size
        self.num_heads = image_head_transformer_dims // 128
        self.transformer_ffn_scale = image_head_transformer_ffn_scale
        self.transformer_dims = image_head_transformer_dims
        # TP只用于Attention，FFN不切分（因为需要全量权重进行einsum计算）
        self.enable_tp = enable_tp and get_tensor_model_parallel_world_size() > 1
        self.enable_tp_attn = self.enable_tp  # Attention使用TP
        # FFN不切分，保持全量权重
        
        # Get TP info for attention
        if self.enable_tp:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
        else:
            self.tp_rank = 0
            self.tp_size = 1
        
        assert self.transformer_dims % 128 == 0
        assert self.transformer_dims % depth == 0

        # Attention使用TP
        self.self_attention = FlashVarLenAttention(self.transformer_dims, self.num_heads, causal=True, use_fused_qkv=False, enable_tp=self.enable_tp_attn)
        self.layernorm1 = RMSNorm(self.transformer_dims)
        self.layernorm2 = RMSNorm(self.transformer_dims)
        
        # FFN不切分TP，直接使用nn.Linear
        if self.transformer_ffn_scale <= 0:  # 兼容baichuan-omni-1d5老逻辑
            self.linear1 = nn.Linear(self.transformer_dims * self.depth, 2 * self.transformer_dims)
            self.linear2 = nn.Linear(2 * self.transformer_dims * self.depth, self.transformer_dims)
        else:
            ffn_intermediate_size = self.transformer_ffn_scale * self.transformer_dims
            # FFN直接使用普通Linear，不切分
            self.linear1 = nn.Linear(
                self.transformer_dims, 
                ffn_intermediate_size,
                bias=True,
            )
            self.linear2 = nn.Linear(
                ffn_intermediate_size,
                self.transformer_dims,
                bias=True,
            )
    
    def forward(self, x, bsz):
        bsz = x.shape[0]
        res = x
        x = self.layernorm1(x)

        # seqlens = self.depth * torch.ones((bsz,), dtype=torch.int32, device=x.device)
        # In TP mode, transformer_dims remains the same (output from RowParallel is complete)
        _x = self.self_attention(x.view(-1, self.transformer_dims))
        
        _x = _x.view(bsz, self.depth, self.transformer_dims).contiguous()

        # seq_len = x.size(1)
        # src_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        # # _x, _ = self.self_attention(x, x, x,  is_causal=True, attn_mask=src_mask)
        # _x, _ = self.self_attention(x, x, x, attn_mask=src_mask)

        
        # if _x.isnan().any():
        #     raise NotImplementedError(f"{seqlens}, 0.2")
        
        _res = _x + res  # (bs, sl, d)
        res = self.layernorm2(_res)

        if self.transformer_ffn_scale <= 0:  # 兼容baichuan-omni-1d5老逻辑 (不支持 TP)
            x = torch.einsum('bld,tld->blt', res, torch.reshape(self.linear1.weight, (2 * self.llm_hidden_size, -1, self.llm_hidden_size)))
            x = torch.nn.functional.gelu(x)
            x = torch.einsum('blt,dlt->bld', x, torch.reshape(self.linear2.weight, (self.llm_hidden_size, -1, 2 * self.llm_hidden_size)))
        else:
            # FFN不切分TP，直接使用全量权重
            if self.depth > 1:
                # einsum path for depth > 1
                x = torch.einsum('bld,tld->blt', 
                    res, 
                    torch.reshape(self.linear1.weight, 
                        (self.transformer_ffn_scale * self.transformer_dims // self.depth,
                        self.depth,
                        self.transformer_dims)
                    ))
                x = torch.nn.functional.gelu(x)
                x = torch.einsum('blt,dlt->bld',
                    x, 
                    torch.reshape(self.linear2.weight, 
                        (self.transformer_dims,
                        self.depth,
                        self.transformer_ffn_scale * self.transformer_dims // self.depth)
                    ))
            else:
                # Standard linear path for depth == 1
                x = self.linear1(res)
                x = torch.nn.functional.gelu(x)
                x = self.linear2(x)
        
        return _res + x

class OmniImageHead(nn.Module):
    def __init__(self,
                 hidden_size,
                 codebook_sizes,
                 image_head_transformer_ffn_scale,
                 image_head_transformer_dims,
                 image_head_transformer_layers,
                 image_head_enable,
                 enable_tp=False):
        super().__init__()
        self.llm_hidden_size = hidden_size
        self.codebook_sizes = codebook_sizes # list
        self.transformer_ffn_scale = image_head_transformer_ffn_scale
        self.transformer_dims = image_head_transformer_dims
        self.transformer_layers = image_head_transformer_layers
        self.enable_tp = enable_tp and get_tensor_model_parallel_world_size() > 1
        
        # Get TP info
        if self.enable_tp:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
        else:
            self.tp_rank = 0
            self.tp_size = 1

        if self.transformer_ffn_scale > 0:
            self.hidden_norm = RMSNorm(self.llm_hidden_size)
            # hidden_proj: Keep as ReplicatedLinear (nn.Linear) for simplicity
            # The transformer layers expect complete input, and TP is applied inside them
            self.hidden_proj = nn.Linear(self.llm_hidden_size, self.transformer_dims, bias=False)
        self.transformer_layers = nn.ModuleList([
            CasualDepthTransformerLayer(hidden_size,
                                        image_head_transformer_dims,
                                        image_head_transformer_ffn_scale,
                                        len(self.codebook_sizes),
                                        enable_tp=self.enable_tp) for _ in range(self.transformer_layers)])
        # self.head_proj = nn.Linear(self.transformer_dims, self.llm_hidden_size, bias=False)
        # self.headnorm = RMSNorm(self.llm_hidden_size) 
        self.headnorm = RMSNorm( self.transformer_dims) 
        # heads: Use ReplicatedLinear so each rank has complete logits for sampling
        # After sampling, results can be broadcast if needed
        self.heads = nn.ModuleList([ nn.Linear(self.transformer_dims, vq_size+1) for vq_size in self.codebook_sizes])
        self.gradient_checkpointing = True
        if not image_head_enable:
            for param in self.parameters():  # 修复SFT阶段可能错误加载导致OOM的问题
                param.requires_grad = False
    
    def load_tp_state_dict(self, state_dict):
        """Load state dict with TP-aware weight sharding."""
        load_tp_state_dict(self, state_dict, self.tp_rank, self.tp_size)

    
    def forward(self, x, visual_tokens, visual_emb_layers, batch_size, codebook_id):
        cumsum_visual_embed = torch.stack([
            visual_emb_layers[i](visual_tokens[..., i]) 
            for i, vq_size in enumerate(self.codebook_sizes[:-1])
            ], dim=1)
        cumsum_visual_embed = torch.cumsum(cumsum_visual_embed, dim=1)  # (bs, depth-1, d)
        hidden_states = torch.concat([x.reshape(-1, 1, self.llm_hidden_size), cumsum_visual_embed], dim=1)  # (bs, depth, d)
        assert hidden_states.size(1) == len(self.codebook_sizes)

        if self.transformer_ffn_scale > 0:
            hidden_states = self.hidden_norm(hidden_states)
            hidden_states = self.hidden_proj(hidden_states)

        for i, tlayer in enumerate(self.transformer_layers):
            hidden_states = tlayer(
                hidden_states,
                batch_size,
            )
        
        # hidden_states = self.head_proj(hidden_states)
        hidden_states = self.headnorm(hidden_states)
        
        
        logits = self.heads[codebook_id](hidden_states[:,codebook_id])
        return logits

class OmniAudioHead(nn.Module):
    def __init__(self, hidden_size, codebook_sizes, audio_head_transformer_ffn_scale, audio_head_transformer_dims, audio_head_transformer_layers, audio_head_enable, enable_tp=False):
        super().__init__()
        self.llm_hidden_size = hidden_size
        self.codebook_sizes = codebook_sizes # list
        self.transformer_ffn_scale = audio_head_transformer_ffn_scale
        self.transformer_dims = audio_head_transformer_dims
        self.transformer_layer_num = audio_head_transformer_layers
        self.enable_tp = enable_tp and get_tensor_model_parallel_world_size() > 1
        
        # Get TP info
        if self.enable_tp:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
        else:
            self.tp_rank = 0
            self.tp_size = 1

        if self.transformer_ffn_scale > 0:
            self.hidden_norm = RMSNorm(self.llm_hidden_size)
            # hidden_proj: Keep as ReplicatedLinear (nn.Linear) for simplicity
            self.hidden_proj = nn.Linear(self.llm_hidden_size, self.transformer_dims, bias=False)

        self.transformer_layers = nn.ModuleList([
            CasualDepthTransformerLayer(self.llm_hidden_size,
                                        self.transformer_dims,
                                        self.transformer_ffn_scale,
                                        len(self.codebook_sizes),
                                        enable_tp=self.enable_tp) for _ in range(self.transformer_layer_num)])
        self.headnorm = RMSNorm(self.transformer_dims) 
        # heads: Use ReplicatedLinear so each rank has complete logits for sampling
        self.heads = nn.ModuleList([
            nn.Linear(self.transformer_dims, vq_size+1)
            for vq_size in self.codebook_sizes
        ])
        self.gradient_checkpointing = True
        if not audio_head_enable:
            for param in self.parameters():  # 修复SFT阶段可能错误加载导致OOM的问题
                param.requires_grad = False

    def load_tp_state_dict(self, state_dict):
        """Load state dict with TP-aware weight sharding."""
        load_tp_state_dict(self, state_dict, self.tp_rank, self.tp_size)

    def forward(self, x, audios_tokens, audio_emb_layers, batch_size, codebook_id):
        cumsum_audio_embed = torch.stack([
            audio_emb_layers[i](audios_tokens[..., i]) 
            for i, vq_size in enumerate(self.codebook_sizes[:-1])
            ], dim=1)
        cumsum_audio_embed = torch.cumsum(cumsum_audio_embed, dim=1)  # (bs, depth-1, d)
        hidden_states = torch.concat([x.reshape(-1, 1, self.llm_hidden_size), cumsum_audio_embed], dim=1)  # (bs, depth, d)
        assert hidden_states.size(1) == len(self.codebook_sizes)

        if self.transformer_ffn_scale > 0:
            hidden_states = self.hidden_norm(hidden_states)
            hidden_states = self.hidden_proj(hidden_states)

        for i, tlayer in enumerate(self.transformer_layers):
            hidden_states  = tlayer(
                hidden_states,
                batch_size,
            )
        hidden_states = self.headnorm(hidden_states)
        logits = self.heads[codebook_id](hidden_states[:,codebook_id])
        return logits    

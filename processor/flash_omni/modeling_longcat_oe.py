from typing import List, Callable, Optional, Union, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
# from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
# from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from .configuration_longcat import LongcatConfig
from .visual_modeling_omni import (
    DiscreteQwenVitEncoder, 
    VisualEmbeddingBridge)
from .audio_modeling_omni import (
    RMSNorm,
    OmniAudioEncoder, 
    OmniAudioDecoder,
    OmniAudioVQBridgeTokenizer, 
    OmniAudioFlowMatchingDecoder)
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))
from utils.model_utils import load_weights_from_safetensors_helper
@use_kernel_forward_from_hub("RMSNorm")
class LongcatRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LongcatRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LongcatRotaryEmbedding(nn.Module):
    def __init__(self, config: LongcatConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LongcatMLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.ffn_hidden_size if intermediate_size is None else intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class LongcatTopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.moe_topk
        self.n_routed_experts = (
            config.n_routed_experts
            if config.zero_expert_num is None
            else config.n_routed_experts + config.zero_expert_num
        )
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.router_bias = config.router_bias

        self.classifier = nn.Linear(config.hidden_size, self.n_routed_experts, bias=self.router_bias)
        self.register_buffer("e_score_correction_bias", torch.zeros((self.n_routed_experts)))

    @torch.no_grad()
    def get_topk_indices(self, scores):
        scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(
            -1, self.config.hidden_size
        )  # hidden_states: [batchsize*seq_len, hidden_size]
        router_logits = F.linear(hidden_states.type(torch.float32), self.classifier.weight.type(torch.float32))
        scores = router_logits.softmax(dim=-1)
        topk_indices = self.get_topk_indices(scores)
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


class LongcatMoE(nn.Module):
    """
    longcat moe module.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [
                LongcatMLP(config, intermediate_size=config.expert_ffn_hidden_size)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.router = LongcatTopkRouter(config)
        self.zero_expert_num = config.zero_expert_num
        self.zero_expert_type = config.zero_expert_type

        self._parameters_organized = False
        self.moe_impl = config.moe_impl
        self.moe_switch_token_num = config.moe_switch_token_num

    def moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        total_experts = len(self.experts) if self.zero_expert_num is None else len(self.experts) + self.zero_expert_num

        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=total_experts)
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx in range(total_experts):
            expert = self.experts[expert_idx] if expert_idx < len(self.experts) else None
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)

            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]

                if self.zero_expert_num is None or expert_idx < len(self.experts):
                    expert_output = expert(expert_input)
                elif self.zero_expert_type == "drop":
                    expert_output = 0 * expert_input
                elif self.zero_expert_type == "identity":
                    expert_output = expert_input
                else:
                    raise ValueError("Unknown condition")

                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(0, token_indices, weighted_output)

        return final_hidden_states.type(hidden_states.dtype)

    def reorganize_parameters(self):
        self._parameters_organized = True

        # 获取专家数量和参数维度
        n_experts = len(self.experts)
        if n_experts == 0:
            return

        gate_shape = (n_experts, *self.experts[0].gate_proj.weight.shape)
        up_shape = (n_experts, *self.experts[0].up_proj.weight.shape)
        down_shape = (n_experts, *self.experts[0].down_proj.weight.shape)

        device = self.experts[0].gate_proj.weight.device

        self.gate_proj = torch.empty(gate_shape, dtype=self.experts[0].gate_proj.weight.dtype, device=device)
        self.up_proj = torch.empty(up_shape, dtype=self.experts[0].up_proj.weight.dtype, device=device)
        self.down_proj = torch.empty(down_shape, dtype=self.experts[0].down_proj.weight.dtype, device=device)

        self.act_fn = self.experts[0].act_fn

        # 复制参数到统一的张量中，但不修改原始专家的权重
        for i, expert in enumerate(self.experts):
            self.gate_proj[i].copy_(expert.gate_proj.weight)
            self.up_proj[i].copy_(expert.up_proj.weight)
            self.down_proj[i].copy_(expert.down_proj.weight)

            # 修改原始专家的指向 - 使用 weight.data 赋值
            expert.gate_proj.weight.data = self.gate_proj[i]
            expert.up_proj.weight.data = self.up_proj[i]
            expert.down_proj.weight.data = self.down_proj[i]

        torch.cuda.empty_cache()

    def moe_opt_v1(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        total_experts = len(self.experts) if self.zero_expert_num is None else len(self.experts) + self.zero_expert_num

        # 创建专家掩码 [total_experts, batch*seq_len, top_k]
        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=total_experts)
        expert_mask = expert_mask.permute(2, 0, 1)

        # 计算实际专家的输出
        # 将hidden_states扩展到[total_experts, batch*seq_len, hidden_size]以进行批处理
        hidden_states_expanded = hidden_states.unsqueeze(0).expand(len(self.experts), -1, -1)

        # 批量计算所有专家的输出
        gate = torch.bmm(
            hidden_states_expanded, self.gate_proj.permute(0, 2, 1)
        )  # [n_experts, ffn_dim, hidden_size_dim]
        up = torch.bmm(hidden_states_expanded, self.up_proj.permute(0, 2, 1))  # [n_experts, ffn_dim, hidden_size_dim]
        glu = self.act_fn(gate) * up  # [n_experts, ffn_dim, hidden_size_dim]
        expert_outputs = torch.bmm(glu, self.down_proj.permute(0, 2, 1))  # [n_experts, hidden_size_dim, hidden_size]

        # 处理零专家的输出
        if self.zero_expert_num is not None:
            if self.zero_expert_type == "drop":
                zero_expert_output = torch.zeros_like(hidden_states)
            elif self.zero_expert_type == "identity":
                zero_expert_output = hidden_states
            else:
                raise ValueError("Unknown zero_expert_type")
            zero_expert_output = zero_expert_output.unsqueeze(0).expand(self.zero_expert_num, -1, -1)
            expert_outputs = torch.cat(
                [expert_outputs, zero_expert_output], dim=0
            )  # [total_experts, batch*seq_len, hidden_size]

        # 计算每个token的专家加权输出
        # 扩展权重维度以匹配专家维度 [total_experts, batch*seq_len, top_k]
        expanded_weights = topk_weights.unsqueeze(0).expand(total_experts, -1, -1)

        # 将权重应用到专家掩码上 [total_experts, batch*seq_len, top_k]
        weighted_mask = expert_mask * expanded_weights

        # 对top_k维度求和，得到每个token对应每个专家的总权重 [total_experts, batch*seq_len]
        token_expert_weights = weighted_mask.sum(dim=-1)

        # 扩展权重维度以匹配hidden_size [total_experts, batch*seq_len, 1]
        token_expert_weights = token_expert_weights.unsqueeze(-1)

        # 应用权重到专家输出 [total_experts, batch*seq_len, hidden_size]
        weighted_outputs = expert_outputs * token_expert_weights

        # 对专家维度求和，得到最终输出 [batch*seq_len, hidden_size]
        final_output = weighted_outputs.sum(dim=0)

        del (
            hidden_states_expanded,
            gate,
            up,
            glu,
            expert_outputs,
            expert_mask,
            expanded_weights,
            weighted_mask,
            token_expert_weights,
            weighted_outputs,
        )

        return final_output.type(hidden_states.dtype)

    def moe_opt_v2(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        total_experts = len(self.experts) if self.zero_expert_num is None else len(self.experts) + self.zero_expert_num

        # 从token维度分批次，避免一次性处理所有token
        batch_size = hidden_states.shape[0]
        token_batch_size = min(1024, batch_size)  # 可调整的token批次大小

        final_output = torch.zeros_like(hidden_states, dtype=hidden_states.dtype)

        # 按token批次处理
        for start_idx in range(0, batch_size, token_batch_size):
            end_idx = min(start_idx + token_batch_size, batch_size)

            # 当前批次的token
            batch_hidden_states = hidden_states[start_idx:end_idx]  # [token_batch_size, hidden_size]
            batch_topk_indices = topk_indices[start_idx:end_idx]  # [token_batch_size, top_k]
            batch_topk_weights = topk_weights[start_idx:end_idx]  # [token_batch_size, top_k]

            # 创建专家掩码 [total_experts, token_batch_size, top_k]
            expert_mask = torch.nn.functional.one_hot(batch_topk_indices, num_classes=total_experts)
            expert_mask = expert_mask.permute(2, 0, 1)  # [total_experts, token_batch_size, top_k]

            # 计算实际专家的输出
            hidden_states_expanded = batch_hidden_states.unsqueeze(0).expand(len(self.experts), -1, -1)

            # 批量计算所有专家的输出
            gate = torch.bmm(hidden_states_expanded, self.gate_proj.permute(0, 2, 1))
            up = torch.bmm(hidden_states_expanded, self.up_proj.permute(0, 2, 1))
            glu = self.act_fn(gate) * up
            expert_outputs = torch.bmm(glu, self.down_proj.permute(0, 2, 1))

            # 处理零专家
            if self.zero_expert_num is not None:
                if self.zero_expert_type == "drop":
                    zero_expert_output = torch.zeros_like(batch_hidden_states)
                elif self.zero_expert_type == "identity":
                    zero_expert_output = batch_hidden_states
                else:
                    raise ValueError("Unknown zero_expert_type")

                zero_expert_output = zero_expert_output.unsqueeze(0).expand(self.zero_expert_num, -1, -1)
                expert_outputs = torch.cat([expert_outputs, zero_expert_output], dim=0)

            # 计算权重
            expanded_weights = batch_topk_weights.unsqueeze(0).expand(total_experts, -1, -1)
            weighted_mask = expert_mask * expanded_weights
            token_expert_weights = weighted_mask.sum(dim=-1).unsqueeze(-1)

            # 应用权重并求和
            weighted_outputs = expert_outputs * token_expert_weights
            batch_final_output = weighted_outputs.sum(dim=0)

            # 将结果放回最终输出
            final_output[start_idx:end_idx] = batch_final_output

            # 清理当前批次的临时张量
            del batch_hidden_states, batch_topk_indices, batch_topk_weights, expert_mask
            del hidden_states_expanded, gate, up, glu, expert_outputs
            del expanded_weights, weighted_mask, token_expert_weights, weighted_outputs, batch_final_output

        return final_output

    def forward(self, hidden_states):
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.router(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        if self.moe_impl == "naive":
            hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        elif self.moe_impl == "bmm_v1":
            hidden_states = self.moe_opt_v1(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        elif self.moe_impl == "bmm_v2":
            hidden_states = self.moe_opt_v2(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        elif self.moe_impl == "mix":
            token_num = hidden_states.shape[0]
            if token_num > self.moe_switch_token_num:
                hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
            else:
                hidden_states = self.moe_opt_v2(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        else:
            raise ValueError(f"Unknown moe_impl: {self.moe_impl}")

        return hidden_states


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[str],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1, use_mla=False):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    if use_mla:
        b, h, s, d = q.shape
        q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

        b, h, s, d = k.shape
        k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LongcatGQA(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LongcatConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LongcatMLA(nn.Module):
    """Modified from Deepseek MLA"""

    def __init__(self, config: LongcatConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_attention_heads
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_head_dim

        self.is_causal = True
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = LongcatRMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = LongcatRMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        if config.mla_scale_q_lora:
            self.mla_scale_q_lora = (config.hidden_size / self.q_lora_rank) ** 0.5
        if config.mla_scale_kv_lora:
            self.mla_scale_kv_lora = (config.hidden_size / self.kv_lora_rank) ** 0.5
        self.scaling = self.qk_head_dim ** (-0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states))).view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # apply q_lora scaling
        if self.mla_scale_q_lora is not None:
            q_pass = q_pass * self.mla_scale_q_lora
            q_rot = q_rot * self.mla_scale_q_lora

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pass = self.kv_a_layernorm(k_pass)

        # apply kv_lora scaling
        if self.mla_scale_kv_lora is not None:
            k_pass = k_pass * self.mla_scale_kv_lora

        k_pass = self.kv_b_proj(k_pass).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin, use_mla=True)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def create_attention_block(class_name, *args, **kwargs):
    attention_mapping = {"MLA": LongcatMLA, "GQA": LongcatGQA}

    chosen_class = attention_mapping.get(class_name)
    if not chosen_class:
        raise ValueError(f"No class found for name: {class_name}")

    return chosen_class(*args, **kwargs)


# class LongcatDecoderLayer(GradientCheckpointingLayer):
#     def __init__(self, config: LongcatConfig, layer_idx: int):
#         super().__init__()
#         self.layer_idx = layer_idx
#         self.hidden_size = config.hidden_size
#         self.mlp = LongcatMoE(config)

#         self_attn = []
#         mlps = []
#         input_layernorm = []
#         post_attention_layernorm = []
#         for i in range(2):
#             self_attn.append(
#                 create_attention_block(config.attention_method, config=config, layer_idx=layer_idx * 2 + i)
#             )
#             mlps.append(LongcatMLP(config))
#             input_layernorm.append(LongcatRMSNorm(config.hidden_size, eps=config.rms_norm_eps))
#             post_attention_layernorm.append(LongcatRMSNorm(config.hidden_size, eps=config.rms_norm_eps))

#         self.self_attn = nn.ModuleList(self_attn)
#         self.mlps = nn.ModuleList(mlps)
#         self.input_layernorm = nn.ModuleList(input_layernorm)
#         self.post_attention_layernorm = nn.ModuleList(post_attention_layernorm)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Cache] = None,
#         use_cache: Optional[bool] = False,
#         cache_position: Optional[torch.LongTensor] = None,
#         position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
#         **kwargs: Unpack[FlashAttentionKwargs],
#     ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
#         for i in range(2):
#             residual = hidden_states

#             hidden_states = self.input_layernorm[i](hidden_states)

#             # Self Attention
#             hidden_states, self_attn_weights = self.self_attn[i](
#                 hidden_states=hidden_states,
#                 attention_mask=attention_mask,
#                 position_ids=position_ids,
#                 past_key_value=past_key_value,
#                 use_cache=use_cache,
#                 cache_position=cache_position,
#                 position_embeddings=position_embeddings,
#                 **kwargs,
#             )
#             hidden_states = residual + hidden_states

#             # Fully Connected
#             residual = hidden_states
#             hidden_states = self.post_attention_layernorm[i](hidden_states)

#             if i == 0:
#                 shortcut_mlp_output = self.mlp(hidden_states)  # shortcut output (MoE output)

#             hidden_states = self.mlps[i](hidden_states)
#             hidden_states = residual + hidden_states
#             if i == 1:
#                 hidden_states = hidden_states + shortcut_mlp_output

#         return hidden_states


# @auto_docstring
class LongcatPreTrainedModel(PreTrainedModel):
    config: LongcatConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LongcatDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        # "hidden_states": LongcatDecoderLayer,
        "attentions": LongcatMLA,
    }


def padding_vocab(vocab_size: int, parallel: int) -> int:
    if vocab_size % parallel == 0:
        return vocab_size
    else:
        return ((vocab_size // parallel) + 1) * parallel


class NgramEmbedding(nn.Module):
    def __init__(self, config, base_embeddings):
        super().__init__()
        self.config = config
        self.m = config.oe_vocab_size_ratio * config.vocab_size_text
        self.k = config.oe_split_num
        self.n = config.oe_neighbor_num
        self.word_embeddings = base_embeddings
        self.sp_tokens_id = config.multimodal_special_token_list
        if 'text_special_tokens_list' in config:
            self.sp_tokens_id.extend(config.text_special_tokens_list)
        oe_embedders = []
        emb_hidden_size = config.hidden_size
        emb_dim = emb_hidden_size // (self.k * (self.n - 1))
        for i in range(self.k * (self.n - 1)):
            emb_vocab_dim = int(self.m + i * 2 + 1)
            print(f"{emb_vocab_dim=}")
            emb = nn.Embedding(
                padding_vocab(emb_vocab_dim, config.embP), emb_dim, config.pad_token_id
            )  # 需要做padding，同embP对齐
            oe_embedders.append(emb)

        self.oe_embedders = torch.nn.ModuleList(oe_embedders)
        self.oe_projs = torch.nn.ModuleList(
            [torch.nn.Linear(emb_dim, emb_hidden_size, bias=False) for _ in range(self.k * (self.n - 1))]
        )

    def shift_right_ignore_eos(self, tensor, n, eos_token_id=2):
        if n == 0:
        # 如果不移动,直接返回副本
            return tensor.clone()
        
        p, q = tensor.shape
        result = torch.zeros_like(tensor)  # 初始化结果tensor
        if n >= q:
        # 如果移动距离>=序列长度,全部变0
            return result
        
        # 找到所有special_token/modal/EOS位置
        special_mask = (tensor == 0) # special_token / modal 都已经置零
        total_mask = (tensor == eos_token_id | special_mask)
        # mask = tensor == eos_token_id

        # 计算每个位置所属的段ID
        eos_cumsum = total_mask.long().cumsum(dim=1)
        # 右移1位,使得第1个EOS位置仍属于段0,第2个EOS位置属于段1
        segment_ids = torch.cat([
            torch.zeros(p, 1, dtype=torch.long, device=tensor.device),
            eos_cumsum[:, :-1]
        ], dim=1)

        col_indices = torch.arange(q, device=tensor.device).unsqueeze(0).expand(p, q)
        # 段的数量
        max_segments = segment_ids.max().item() + 1
        segment_starts = torch.full((p, max_segments), q, dtype=torch.long, device=tensor.device)
        # 计算每个段的起始位置
        segment_starts.scatter_reduce_(1, segment_ids, col_indices, reduce='amin', include_self=False)

        # 获取每个位置所属段的起始位置
        segment_start_per_pos = torch.gather(segment_starts, 1, segment_ids)

        # 计算每个位置在段内的偏移量
        offset_in_segment = col_indices - segment_start_per_pos

        # 每个位置要从段内偏移-n的位置取数据
        source_offset = offset_in_segment - n
        valid_mask = source_offset >= 0

        # 计算实际的源索引
        source_indices = segment_start_per_pos + torch.clamp(source_offset, min=0)

        # 按source_indices收集数据收集数据
        result = torch.gather(tensor, 1, source_indices)

        # 将无效位置置零
        result = result * valid_mask * (~special_mask)

        return result

    def precompute_vocab_mods(self):
        vocab_mods = {}
        for i in range(2, self.n + 1):
            for j in range(self.k):
                vocab_mods[(i, j)] = []
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)
                # 初始化 power_value_mod
                power_value_mod = 1
                for _ in range(i - 1):
                    # 逐步取模以防止溢出
                    power_value_mod = (power_value_mod * self.config.vocab_size_text) % emb_vocab_dim
                    vocab_mods[(i, j)].append(power_value_mod)
        return vocab_mods

    def get_ngram_ids(self, input_ids, shift_value_dict, vocab_moded_power, ngram):
        # 每个embedding对应的模数不同，为了防止溢出，对每一个矩阵都从头计算ngram ids
        ngram_ids = input_ids.clone()
        for k in range(2, ngram + 1):
            shift_value = shift_value_dict[k]
            ngram_ids += shift_value * vocab_moded_power[k - 2]
        return ngram_ids

    def forward(self, input_ids):
        assert input_ids.dtype == torch.int64, (
            "input_ids必须为int64类型，否则在计算2-gram index时就会数值溢出，无法避免"
        )
        # print(f'input_ids shape:{input_ids.shape}')
        input_seq_len = input_ids.size(-1)
        # print(f"{input_ids=}")
        # 完整ids进base_emb层
        x = self.word_embeddings(input_ids.to(self.word_embeddings.weight.device)).clone()
        x_orig = x.clone()
        # special token置零（包括多模pad）再进oe
        input_ids_no_sp_token = input_ids.clone()
        for sp_token_id in self.sp_tokens_id:
            input_ids_no_sp_token[input_ids == sp_token_id] = 0
        spt_mask = (input_ids_no_sp_token == 0).unsqueeze(-1)  # 形状为 [1, len, 1]
        
        if input_seq_len > 1:
            vocab_mods = self.precompute_vocab_mods()
            # 预先计算shifted ids
            shifted_ids = {}
            for i in range(2, self.n + 1):
                shifted_ids[i] = self.shift_right_ignore_eos(input_ids_no_sp_token, i - 1)
            for i in range(2, self.n + 1):
                for j in range(self.k):
                    index = (i - 2) * self.k + j
                    emb_vocab_dim = int(self.m + index * 2 + 1)
                    n_gram_ids = self.get_ngram_ids(input_ids_no_sp_token, shifted_ids, vocab_mods[(i, j)], ngram=i)
                    text_mask = (n_gram_ids > 0).unsqueeze(-1)
                    new_ids = n_gram_ids % emb_vocab_dim
                    x_oe = self.oe_embedders[index](new_ids.to(self.oe_embedders[index].weight.device))
                    x_oe = x_oe * text_mask  # 广播并置零
                    x += self.oe_projs[index](x_oe.to(self.oe_projs[index].weight.device)).to(x.device)
            x /= 1 + self.k * (self.n - 1)
            self.pv_input_ids = input_ids_no_sp_token[..., -self.n :]
        elif input_seq_len == 1:
            new_input_ids = (
                torch.cat([self.pv_input_ids[..., 1:], input_ids_no_sp_token], dim=-1)
                if hasattr(self, "pv_input_ids")
                else input_ids_no_sp_token
            )
            vocab_mods = self.precompute_vocab_mods()
            # 预先计算shifted ids
            shifted_ids = {}
            for i in range(2, self.n + 1):
                shifted_ids[i] = self.shift_right_ignore_eos(new_input_ids, i - 1)

            for i in range(2, self.n + 1):
                for j in range(self.k):
                    index = (i - 2) * self.k + j
                    emb_vocab_dim = int(self.m + index * 2 + 1)
                    n_gram_ids = self.get_ngram_ids(new_input_ids, shifted_ids, vocab_mods[(i, j)], ngram=i)
                    text_mask = (n_gram_ids > 0).unsqueeze(-1)
                    new_ids = n_gram_ids % emb_vocab_dim
                    new_ids = new_ids[..., -1].unsqueeze(-1)
                    x_oe = self.oe_embedders[index](new_ids.to(self.oe_embedders[index].weight.device))
                    x_oe = x_oe * text_mask  # 广播并置零
                    x += self.oe_projs[index](x_oe.to(self.oe_projs[index].weight.device)).to(x.device)
            x /= 1 + self.k * (self.n - 1)
            self.pv_input_ids = new_input_ids
        else:
            raise ValueError(f"Unknown seq len:{input_seq_len}")
        # 恢复special token位置的emb：spt_mask为1的位置恢复原值，0的位置保持不变，确保spt对应的emb没有被除以13
        x = torch.where(spt_mask, x_orig, x)
        return x


# @auto_docstring
class LongcatModel(LongcatPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"model\.mtp.*"]

    def __init__(self, config: LongcatConfig, oe_delegate_fn=None):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        if config.visual_config.enable:
            self.visual_model        =  DiscreteQwenVitEncoder(config)
            self.visual_bridge_model =  VisualEmbeddingBridge(config)
        if config.audio_config.enable:
            self.audio_embed_layers = nn.ModuleList([
                nn.Embedding(codedim, config.hidden_size)
                    for i, codedim in enumerate(config.audio_config.vq_config.codebook_sizes)
            ])
            
        # emb scaling相关初始化
        self.embed_tokens = nn.Embedding(config.vocab_size_special, config.hidden_size, self.padding_idx)
        # llm 部分用不到
        # self.layers = nn.ModuleList(
        #     [LongcatDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        # )
        self.norm = LongcatRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.rotary_emb = LongcatRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.oe_delegate_fn = oe_delegate_fn
        if oe_delegate_fn:
            self.oe_embeddings = None
        else:
            self.oe_embeddings = NgramEmbedding(config, self.embed_tokens)
        # Initialize weights and apply final processing
        self.post_init()

    def load_weights_from_safetensors(self, safetensors_path: str, image_model_path, device="cuda:0"):
        # 加载通用权重
        key_words_list = ["model.embed_tokens.", "model.norm.", "model.oe_embed_"]
        state_dicts = load_weights_from_safetensors_helper(safetensors_path, key_words_list, device)
        self.norm.load_state_dict(state_dicts[1], strict=True)
        
        if self.config.visual_config.enable:
            # visual_model部分权重从image_model_path加载
            key_words_list = ["model.visual_tokenizer.visual_model.","model.visual_tokenizer.visual_bridge_model."]
            visual_model_state_dicts = load_weights_from_safetensors_helper(image_model_path, key_words_list, device)
            self.visual_model.visual_model.load_state_dict(visual_model_state_dicts[0], strict=True)
            self.visual_model.visual_bridge_model.load_state_dict(visual_model_state_dicts[1], strict=True)

            # visual_bridge_model的embedding_layers需要从通用权重的"model.embed_tokens."中切分
            emb_state_dicts = {}
            offset = self.config.visual_offset
            for i, codedim in enumerate(self.config.visual_quantizer_config.codebook_sizes):
                # 16384 * 8层
                emb_state_dicts[f"{i}.weight"] = state_dicts[0]["weight"][offset:offset+codedim, :]
                offset += codedim
            self.visual_bridge_model.embedding_layers.load_state_dict(emb_state_dicts, strict=True)
            # model.visual_bridge_model.transformer_block.的权重单独读取
            key_words_list = ["model.visual_tokenizer.visual_embedding_layer.pre_buffer."]
            visual_bridge_state_dicts = load_weights_from_safetensors_helper(safetensors_path, key_words_list, device)
            self.visual_bridge_model.transformer_block.load_state_dict(visual_bridge_state_dicts[0], strict=True)
        if self.config.audio_config.enable:
            emb_state_dicts = {}
            offset = self.config.audio_offset
            for i, codedim in enumerate(self.config.audio_config.vq_config.codebook_sizes):
                # 8192,4096,2048, 1024*5, 共8层
                emb_state_dicts[f"{i}.weight"] = state_dicts[0]["weight"][offset:offset+codedim, :]
                offset += codedim
            self.audio_embed_layers.load_state_dict(emb_state_dicts, strict=True)

        if self.oe_delegate_fn is None:
            print("start load oe", flush=True)
            # oe_embedding的word_embeddings需要从"model.embed_tokens."中切分
            vocab_size_special = self.config.vocab_size_special
            text_state_dicts = {"weight":state_dicts[0]["weight"][:vocab_size_special,:]}
            self.oe_embeddings.word_embeddings.load_state_dict(text_state_dicts, strict=True)
            # oe其他权重从通用权重里加载
            oe_embedders_dict = {}
            oe_projs_dict = {}
            oe_embedder_num = (self.oe_embeddings.n - 1) * self.oe_embeddings.k
            for i in range(oe_embedder_num):
                oe_embedders_dict[f"{i}.weight"]= state_dicts[2][f"tokens{i}.weight"]
                oe_projs_dict[f"{i}.weight"] = state_dicts[2][f"proj{i}.weight"]
            self.oe_embeddings.oe_embedders.load_state_dict(oe_embedders_dict, strict=True)
            self.oe_embeddings.oe_projs.load_state_dict(oe_projs_dict, strict=True)
            print("load weights done", flush=True)

    @torch.no_grad()
    def get_multimodal_mask(self, input_ids, pad_token_id, special_token_list):
        '''
        获取任意模态的特殊mask，包含以下
        1. pad mask 表示文本中图像/语音/视频模态提前留出的token位置
        2. special token mask 特殊token 例如对理解模型<start> <end> 不需要next token prediction
        3. embedding mask / lm_head mask 标记出特殊token在embedding中的mask
        '''
        pad_mask = torch.eq(input_ids, pad_token_id)
        sp_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        lm_head_mask = torch.zeros([self.config.vocab_size, 1], dtype=torch.bool)
        for sp_id in special_token_list:
            sp_mask = torch.logical_or(sp_mask, torch.eq(input_ids, sp_id))
            lm_head_mask[sp_id, 0] = True
        return pad_mask, sp_mask, lm_head_mask

    def get_multimodal_embed(
            self, 
            input_ids,
            text_embedding,  # 1. self.embed_tokens(input_ids) 2. 其他模态结果
            multimodal_embed,
            pad_token_id,
            fake_input,
            group_index=None,  # 某种模态的编号 for MoE
        ):
        pad_mask, sp_mask, _ = self.get_multimodal_mask(input_ids, pad_token_id, self.config.multimodal_special_token_list)
        if not self.training:  # 推理支持auto map 把多模态模块输出和input_ids 统一到一个device
            multimodal_embed = multimodal_embed.to(input_ids.device)
        if not fake_input:  # 检查多模态token 和 pad mask数量一致 （不正确的截断会导致该问题）
            assert pad_mask.sum() == multimodal_embed.shape[0], f"{pad_mask.sum()} vs {multimodal_embed.shape[0]}"
        else:
            assert pad_mask.sum() <= 0  # 0 vs 1

        # 合并 当前模态embeddings 和text embeddings
        input_ids = torch.where(pad_mask, torch.cumsum(pad_mask.view(-1).to(input_ids), dim=0).view(input_ids.shape)-1, input_ids)
        if self.config.train_multimodal_special_tokens_only and self.training:  
            # 仅special token传梯度到embedding weight, 保证LLM部分不变
            # 注意: 多种模态之间special token list应该共享，否则会有部分被stop gradient
            sp_mask = sp_mask.unsqueeze(-1).to(text_embedding)
            text_embedding = (1 - sp_mask) * text_embedding.detach() + sp_mask * text_embedding 
        text_embedding = (1 - pad_mask.to(text_embedding)).unsqueeze(-1) * text_embedding  # pad token位置填0 (不传梯度)
        multimodal_embedding = torch.embedding(multimodal_embed, input_ids * pad_mask)  # 非 pad token 位置填idx=0位置结果
        multimodal_embedding = pad_mask.to(multimodal_embedding).unsqueeze(-1) * multimodal_embedding  # 非pad token 位置填0
        
        final_embedding = multimodal_embedding.to(text_embedding) + text_embedding

        if group_index is None:
            group_index = pad_mask.to(torch.int32)
        else:
            current_index = torch.max(group_index) + 1
            group_index += pad_mask.to(torch.int32) * current_index  # 假设模态无重叠

        return final_embedding, group_index  # group_index 不传None 防止MoE部分参数无梯度

    def get_visual_embed(
            self, 
            input_ids,
            text_embedding,  # 1. self.embed_tokens(input_ids) 2. 其他模态结果
            images = None,
            patch_nums = None, 
            images_grid = None,
            videos = None,
            videos_patch_nums = None, 
            videos_grid = None,
            group_index = None,  # 某种模态的编号
        ): 
        if images is None or len(images) <= 0:
            images, images_grid, patch_nums = self.visual_model.fake_input(input_ids.device)
            image_fake_input = True
        else:
            image_fake_input = False
            
        if videos is None or len(videos) <= 0 :
            videos, videos_grid, videos_patch_nums = self.visual_model.fake_input(input_ids.device)
            video_fake_input = True
        else:
            video_fake_input = False
        
        visual_input = images + videos
        visual_grid = images_grid + videos_grid
        
        visual_input = torch.cat(visual_input, dim=0)
        visual_grid = torch.tensor(np.array(visual_grid))

        visual_indices, cmt_loss, codebook_usage = self.visual_model(visual_input, grid_thw=visual_grid)
        visual_embed = self.visual_bridge_model(visual_indices)


        assert sum(patch_nums) + sum(videos_patch_nums) == visual_embed.shape[0]
        images_embed = visual_embed[:sum(patch_nums)]
        videos_embed = visual_embed[sum(patch_nums):]

        final_embedding, group_index = self.get_multimodal_embed(input_ids, text_embedding, images_embed, self.config.visual_config.image_pad_token_id, image_fake_input, group_index=group_index)

        ret_visual_indices = []
        visual_indices = visual_indices.view(-1, visual_indices.shape[-1]) # BF,L,Lv -> -1,Lv
        if not image_fake_input:
            ret_visual_indices.append(visual_indices[:sum(patch_nums)])
        if not video_fake_input:
            ret_visual_indices.append(visual_indices[sum(patch_nums):])
        if len(ret_visual_indices) == 0:
            ret_visual_indices = None
        else:
            ret_visual_indices = torch.cat(ret_visual_indices, dim=0)
        
        return final_embedding, group_index, cmt_loss, codebook_usage, ret_visual_indices
    
    def get_visual_embed_given_tokens(
            self, 
            input_ids,
            text_embedding,  # 1. self.embed_tokens(input_ids) 2. 其他模态结果
            vision_tokens = None, 
            group_index = None,  # 某种模态的编号
        ): 

        vision_embed = self.visual_bridge_model(vision_tokens)
        # 目前还不支持video
        final_embedding, group_index = self.get_multimodal_embed(input_ids, text_embedding, vision_embed, self.config.visual_config.image_pad_token_id, False, group_index=group_index)
        return final_embedding, group_index, vision_tokens
    
    def get_audio_embed(self, inputs_embeds, input_ids, audios_tokens, audiotext_ids, group_index):
        if audios_tokens is None or len(audios_tokens) <= 0 :
            audios_tokens = torch.zeros(5, len(self.config.audio_config.vq_config.codebook_sizes), dtype=torch.int32, device=input_ids.device)  # a fake input
            fake_input = True
        else:
            fake_input = False
        for i, audio_emb_layer in enumerate(self.audio_embed_layers):
            if i==0:
                audio_embs = audio_emb_layer(audios_tokens[..., i]) 
            else:
                audio_embs += audio_emb_layer(audios_tokens[..., i]) 
        inputs_embeds, group_index = self.get_multimodal_embed(input_ids, inputs_embeds, audio_embs, self.config.audio_config.audio_pad_token_id, fake_input, group_index=group_index)
        if audiotext_ids is not None:
            # audiotext_embs = self.embed_tokens(audiotext_ids)
            audiotext_embs = self.get_text_pad_embeddings(audiotext_ids.unsqueeze(0)).squeeze()
            audiotext_embs = audiotext_embs.to('cuda')
        else:
            audiotext_ids = torch.zeros(5, dtype=input_ids.dtype, device=input_ids.device)
            audiotext_embs = self.embed_tokens(audiotext_ids) # a fake input for zero3
        # audiotext_ids中操作audiotext_embedding, pad_token位置置0
        audiotext_embs = audiotext_embs * (audiotext_ids != self.config.audio_config.audiotext_pad_token_id).unsqueeze(-1).to(audiotext_embs.dtype)
        # audiotext在完整input_ids中的位置，包括了audiotext_pad_token, audiotext_start_token, audio_pad_token
        audiotext_pad_mask = torch.logical_or(torch.logical_or(input_ids == self.config.audio_config.audiotext_pad_token_id, input_ids == self.config.audio_config.audiotext_start_token_id), input_ids == self.config.audio_config.audio_pad_token_id)
        # audiotext在完整input_ids中的delay token位置，即audiotext_pad_token
        audiotext_delay_mask = input_ids == self.config.audio_config.audiotext_pad_token_id
        # 去掉input_embeds中delay token位置
        inputs_embeds = (1 - audiotext_delay_mask.to(inputs_embeds)).unsqueeze(-1) * inputs_embeds
        # 把audiotext_embedding填充到input_ids长度，并加到input_embeds中
        audiotext_input_ids = torch.where(audiotext_pad_mask, torch.cumsum(audiotext_pad_mask.view(-1).to(input_ids), dim=0).view(input_ids.shape)-1, input_ids)
        audiotext_input_ids = audiotext_input_ids.cuda()
        audiotext_pad_mask = audiotext_pad_mask.cuda()
        audiotext_embeds = torch.embedding(audiotext_embs, audiotext_input_ids * audiotext_pad_mask)
        audiotext_embeds = audiotext_pad_mask.to(audiotext_embeds).unsqueeze(-1) * audiotext_embeds
        inputs_embeds = inputs_embeds + audiotext_embeds.to(inputs_embeds)
        return inputs_embeds

    def get_text_pad_embeddings(self, input_ids: torch.LongTensor, enbale = False):
        # 这里要mask掉image pad token给oe
        input_ids_mask_pad = input_ids.clone().to("cuda")
        input_ids_mask_pad[input_ids_mask_pad==self.config.visual_config.image_pad_token_id] = 0
        input_ids_mask_pad[input_ids_mask_pad==self.config.audio_config.audio_pad_token_id] = 0
        # if enbale:
        if self.oe_delegate_fn:
            if isinstance(self.oe_delegate_fn, str):
                embeddings = torch.randn(input_ids_mask_pad.shape[1], 3072, device='cuda')
                print("random emb")
            else:   
                assert input_ids_mask_pad.dim() == 2, input_ids_mask_pad.dim()
                assert input_ids_mask_pad.shape[0] == 1, f"仅支持 bs=1"
                # embeddings = torch.randn(input_ids_mask_pad.shape[1], 3072, device='cuda')
                tmp_embeddings, cb = self.oe_delegate_fn(input_ids_mask_pad.tolist())
                embeddings = tmp_embeddings[0].clone()
                cb()

            embeddings = embeddings.unsqueeze(0)
        else:
            embeddings = self.oe_embeddings(input_ids_mask_pad)

        return embeddings

    # @check_model_inputs
    # @auto_docstring
    @torch.no_grad() 
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        audios_tokens: Optional[Union[List, torch.Tensor]] = None, # 音频token bs*seqlen*vq_num
        audiotext_ids: Optional[Union[List, torch.LongTensor]] = None,
        vision_tokens: Optional[Union[List, torch.Tensor]] = None, # 视觉token bs*seqlen*vq_num
        images: Optional[Union[List, torch.Tensor]] = None,
        patch_nums: Optional[torch.Tensor] = None, 
        images_grid: Optional[Union[List, torch.Tensor]] = None,
        videos: Optional[Union[List, torch.Tensor]] = None,
        videos_patch_nums: Optional[torch.Tensor] = None, 
        videos_grid: Optional[Union[List, torch.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[str],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = True if (return_dict is not None or self.training) else self.config.use_return_dict
        
        # if any(key in self.config.moe_impl for key in ["bmm", "mix"]) and not self.layers[0].mlp._parameters_organized:
        #     for layer in self.layers:
        #         layer.mlp.reorganize_parameters()
        #     torch.cuda.empty_cache()

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        
        group_index = None
        if inputs_embeds is None:
            inputs_embeds = self.get_text_pad_embeddings(input_ids, True)

            if self.config.audio_config.enable:
                import traceback
                try:
                    inputs_embeds = self.get_audio_embed(inputs_embeds, input_ids, audios_tokens, audiotext_ids, group_index)
                except Exception as e:
                    print("error!", repr(e))
                    traceback.print_exc()  # 打印异常的完整回溯
            if (self.config.visual_config.enable or self.config.video_config.enable):
                if vision_tokens is None:
                    inputs_embeds, group_index, cmt_loss, codebook_usage, visual_indices = \
                        self.get_visual_embed(
                            input_ids, 
                            inputs_embeds, 
                            images, 
                            patch_nums, 
                            images_grid, 
                            videos, 
                            videos_patch_nums, 
                            videos_grid, 
                            group_index=group_index)  # 注意更新group index
                else: # 这种情况用于视觉生成
                    inputs_embeds, group_index, visual_indices = \
                        self.get_visual_embed_given_tokens(
                            input_ids, 
                            inputs_embeds, 
                            vision_tokens,
                            group_index=group_index)  # 注意更新group index
            else:
                visual_indices = None
        
        return inputs_embeds
    
        # if use_cache and past_key_values is None:
        #     past_key_values = DynamicCache()

        # if cache_position is None:
        #     past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        #     cache_position: torch.Tensor = torch.arange(
        #         past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        #     )

        # if position_ids is None:
        #     position_ids = cache_position.unsqueeze(0)

        # causal_mask = create_causal_mask(
        #     config=self.config,
        #     input_embeds=inputs_embeds,
        #     attention_mask=attention_mask,
        #     cache_position=cache_position,
        #     past_key_values=past_key_values,
        #     position_ids=position_ids,
        # )

        # hidden_states = inputs_embeds
        # position_ids = position_ids.to(hidden_states.device)
        # position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        #     hidden_states = decoder_layer(
        #         hidden_states,
        #         attention_mask=causal_mask,
        #         position_ids=position_ids,
        #         past_key_value=past_key_values,
        #         cache_position=cache_position,
        #         position_embeddings=position_embeddings,
        #         **kwargs,
        #     )

        # hidden_states = self.norm(hidden_states)
        # return BaseModelOutputWithPast(
        #     last_hidden_state=hidden_states,
        #     past_key_values=past_key_values,
        # )


# @auto_docstring
class LongcatForCausalLM(LongcatPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _keys_to_ignore_on_load_unexpected = [r"model\.mtp.*"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LongcatModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # @can_return_tuple
    # @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[str],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LongcatForCausalLM

        >>> model = LongcatForCausalLM.from_pretrained("meta-longcat/Longcat-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-longcat/Longcat-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class LongcatAudioTokenizer(LongcatPreTrainedModel):
    """
    Construct an audio tokenizer and decoder.
    """
    def __init__(self, config: LongcatConfig):
        super().__init__(config)
        self.padding_idx = None
        self.vocab_size = config.vocab_size
        self.training = False
        self.eval()
        self.audio_model = OmniAudioEncoder(config.audio_config)
        self.audio_bridge_model = OmniAudioVQBridgeTokenizer(config)
        if config.vocoder_config.enable:
            self.audio_decoder = OmniAudioDecoder(config)
            if config.flow_matching_config.enable:
                self.audio_flow_matching_decoder = OmniAudioFlowMatchingDecoder(config)
        
        for param in self.parameters():  # 修复SFT阶段可能错误加载导致OOM的问题
            param.requires_grad = False

    def encode(self, x, encoder_length: Optional[torch.Tensor] = None, bridge_length: Optional[torch.Tensor] = None):
        audio_emb = self.audio_model(x, encoder_length)
        # torch.save(audio_emb,"audio_emb.pt")
        audios_tokens = self.audio_bridge_model(audio_emb, bridge_length)
        return audios_tokens
    
    def decode(self, audio_code_ids, bridge_length: Optional[torch.Tensor] = None):
        assert self.config.vocoder_config.enable, "Vocoder is not enabled in config."
        audio_emb = self.audio_bridge_model.decode(audio_code_ids)
        audio_dec = self.audio_decoder(
            audio_emb.to(next(self.audio_decoder.parameters())), bridge_length
        )
        if self.config.flow_matching_config.enable:
            if self.config.flow_matching_config.use_hidden_states_before_dconv2:
                hidden_states, hidden_states_length = (
                    self.audio_flow_matching_decoder.unpack_hidden_states(
                        audio_dec.hidden_states_before_dconv2,
                        audio_dec.output_length_before_dconv2,
                    )
                )
                audio_flow_matching_decoder_ret = self.audio_flow_matching_decoder(
                    hidden_states, hidden_states_length
                )

            else:
                audio_flow_matching_decoder_ret = self.audio_flow_matching_decoder(
                    audio_dec.refined_mel, audio_dec.mel_length
                )
            return audio_flow_matching_decoder_ret
        else:
            return audio_dec
    
    @torch.no_grad()    
    def forward(self, audios, encoder_length: Optional[torch.Tensor] = None, bridge_length: Optional[torch.Tensor] = None):
        self.eval()
        audios_tokens = self.encode(audios, encoder_length, bridge_length)
        return audios_tokens

__all__ = ["LongcatPreTrainedModel", "LongcatModel", "LongcatForCausalLM", "LongcatAudioTokenizer"]

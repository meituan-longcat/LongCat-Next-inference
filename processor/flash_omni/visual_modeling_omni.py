
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from flash_attn import flash_attn_varlen_func
from transformers.activations import ACT2FN
from torch.nn import functional as F
import time
from transformers.activations import ACT2FN
from transformers import PreTrainedModel
from .configuration_omni import OmniConfig
from .audio_modeling_omni import RMSNorm
from .navit_vq_model  import OmniVisualEncoder, OmniVisualBridge, VisualQuantizer

class DiscreteQwenVitEncoder(PreTrainedModel):
    _supports_flash_attn_2 = True
    _attn_implementation = "flash_attention_2"
    def __init__(self, config: OmniConfig):
        config._attn_implementation = "flash_attention_2"
        super().__init__(config)
        self.padding_idx = config.pad_token_id

        self.config = config.visual_config
        self.patch_size = self.config.patch_size 
        self.merge_size = self.config.merge_size
        self.temporal_patch_size = self.config.temporal_patch_size
        self.merge_size = self.config.merge_size
        self.temporal_patch_size = self.config.temporal_patch_size
        self.image_channel = 3
        
        start = time.time()        
        self.visual_model = OmniVisualEncoder(config.visual_config)
        # 创建 visual_bridge_model 容器模块
        self.visual_bridge_model = nn.Module()
        self.visual_bridge_model.bridge = OmniVisualBridge(config.visual_config)
        self.visual_bridge_model.quantizer = VisualQuantizer(config.visual_quantizer_config)

        # 强制开启
        self.gradient_checkpointing = True  
        self.visual_model.gradient_checkpointing = True 
        self.visual_model._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint 
        self.codebook_size =config.visual_quantizer_config.codebook_size# 多级量化的情况下，要考虑是否共享码本
        self.num_levels = config.visual_quantizer_config.depth
        self.ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}["bf16"]
        self.post_init()

        stop = time.time()
        print(f'Done. cost: {stop-start}', flush=True)
        self.freeze = self.config.freeze
        if self.freeze:
            for param in self.parameters():  # 修复SFT阶段可能错误加载导致OOM的问题
                param.requires_grad = False
            self.visual_model = self.visual_model.eval()
            self.visual_bridge_model.bridge = self.visual_bridge_model.bridge.eval()
            self.visual_bridge_model.quantizer = self.visual_bridge_model.quantizer.eval()

    def get_quantizer_ret(self, input_embed, input_grid, fake_input=False):
        quantizer_ret = self.visual_bridge_model.quantizer(input_embed, input_grid, fake_input)
        projected_embeds = quantizer_ret.projected_embeds
        input_embed = projected_embeds if projected_embeds is not None else input_embed
        return input_embed, quantizer_ret

    def encode(
        self, 
        pixel_values = None,
        grid_thw = None,
        ): 
        image_fake_input = False
        visual_embed, window_index = self.visual_model(pixel_values, grid_thw=grid_thw, require_window_index=True)
        visual_embed = self.visual_bridge_model.bridge(visual_embed, window_index=window_index)
        merged_grid_thw = grid_thw.clone()
        merged_grid_thw[:,1] = merged_grid_thw[:,1] // self.visual_bridge_model.bridge.merge_size
        merged_grid_thw[:,2] = merged_grid_thw[:,2] // self.visual_bridge_model.bridge.merge_size
        visual_embed, visual_quantizer_ret = self.get_quantizer_ret(visual_embed,merged_grid_thw,image_fake_input)
        indices = visual_quantizer_ret.indices
        quants = visual_quantizer_ret.quants
        vq_loss = visual_quantizer_ret.vq_loss
        cmt_loss = visual_quantizer_ret.cmt_loss
        entropy_loss = visual_quantizer_ret.entropy_loss
        codebook_usage = visual_quantizer_ret.codebook_usage

        return indices, quants, vq_loss, cmt_loss, entropy_loss, codebook_usage

    def _to_id(self, x, num_patches_list=None):
        with torch.cuda.amp.autocast(dtype=self.ptdtype):
            with torch.no_grad():
                indices, quants, vq_loss, cmt_loss, entropy_loss, codebook_usage = self.encode(x, num_patches_list) # B*Num-of-Tok, Lv
        return indices.long(), cmt_loss, codebook_usage# B*Num-of-Tok, Lv -> B, Num-of-Tok, Lv
    
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
    
    def forward(
        self,
        pixel_values: torch.Tensor, 
        grid_thw: torch.Tensor,
    ):
        '''
        grid_thw.shape = [NumOfSamples,3]. [1, 32, 76]
        pixel_values.shape = [N,C,H,W]
        '''
        # 这里有个强假设，默认所有grid_thw的值都是一样的
        if not self.freeze:
            indices, cmt_loss, codebook_usage = self._to_id(pixel_values,grid_thw)
        else:
            with torch.no_grad():
                indices, cmt_loss, codebook_usage = self._to_id(pixel_values,grid_thw)
        return indices, cmt_loss, codebook_usage

class Qwen2_5_VLMLP(nn.Module):
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.hidden_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        # self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        # 默认激活函数为silu，如果config.hidden_act不存在则用silu
        # self.act_fn = ACT2FN[getattr(config, "mlp_hidden_act", "silu")]
        self.act_fn = nn.GELU()
        self.prenorm = nn.Identity()

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(self.prenorm(hidden_state))))

class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

# Copied from transformers.models.llama.modeling_llama.repeat_kv
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

class DecoderLayer(nn.Module):
    def __init__(self, config: OmniConfig, is_sparse=False):
        super().__init__()
        self.hidden_size = config.hidden_size
        # self.self_attn = Attention(config=config, is_sparse=is_sparse)
        self.mlp = MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        # self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.post_attention_layernorm = RMSNorm_no_weight(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seqlens: Optional[torch.IntTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        group_index=None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        # residual = hidden_states

        # hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        # hidden_states, self_attn_weights, present_key_value = self.self_attn(
        #     hidden_states=hidden_states,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     seqlens=seqlens,
        #     past_key_value=past_key_value,
        #     output_attentions=output_attentions,
        #     use_cache=use_cache,
        # )
        # hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.pre_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class VisualEmbeddingBridge(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(codedim, config.hidden_size)
            for _, codedim in enumerate(config.visual_quantizer_config.codebook_sizes)
        ])
        self.config = config
        self.hidden_size = config.hidden_size
        self.codebook_num = len(config.visual_quantizer_config.codebook_sizes)
        # 添加transformer block
        self.transformer_block = DecoderLayer(config)

        # mlp modified from Qwen2_5_VLMLP
        # self.mlp = Qwen2_5_VLMLP(config)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        # indices.shape: [B, num-of-token, lv]
        for i, embedding_layer in enumerate(self.embedding_layers):
            if i == 0:
                embeding = embedding_layer(indices[..., i])
            else:
                embeding += embedding_layer(indices[..., i])

        embeding = self.transformer_block(embeding)[0]
        
        return embeding.view(-1, embeding.shape[-1])


class FlashVarLenAttention(nn.Module):
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

        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, self.num_heads, self.head_dim).contiguous()
        key_states = self.k_proj(hidden_states)
        key_states = key_states.view(bsz, self.num_heads, self.head_dim).contiguous()
        value_states = self.v_proj(hidden_states)
        value_states = value_states.view(bsz, self.num_heads, self.head_dim).contiguous()

        cu_len = F.pad(torch.cumsum(seq_len, dim=0), (1, 0), "constant", 0).to(torch.int32)
        max_seqlen = torch.max(seq_len).to(torch.int32).detach()
        attn_output = flash_attn_varlen_func(query_states, key_states, value_states, cu_len, cu_len, max_seqlen,
                                             max_seqlen, causal=self.causal, window_size=self.window_size)  # (bsz * qlen, nheads, headdim)
        attn_output = attn_output.reshape(bsz, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output

class CasualDepthTransformerLayer(nn.Module):
    def __init__(self, config, depth):
        super().__init__()
        self.config = config.visual_config.image_head_config
        self.depth = depth
        self.llm_hidden_size = config.hidden_size
        self.num_heads = self.config.image_head_transformer_dims // 128
        self.transformer_ffn_scale = self.config.image_head_transformer_ffn_scale
        self.transformer_dims = self.config.image_head_transformer_dims
        
        assert self.transformer_dims % 128 == 0
        assert self.transformer_dims % depth == 0

        # self.self_attention = nn.MultiheadAttention(embed_dim=self.transformer_dims, num_heads=self.num_heads,batch_first=True)
        self.self_attention = FlashVarLenAttention(self.transformer_dims, self.num_heads, causal=True)
        self.layernorm1 = RMSNorm(self.transformer_dims)
        self.layernorm2 = RMSNorm(self.transformer_dims)
        if self.transformer_ffn_scale <= 0:  # 兼容baichuan-omni-1d5老逻辑
            self.linear1 = nn.Linear(self.transformer_dims * self.depth, 2 * self.transformer_dims)
            self.linear2 = nn.Linear(2 * self.transformer_dims * self.depth, self.transformer_dims)
        else:
            self.linear1 = nn.Linear(
                self.transformer_dims, 
                self.transformer_ffn_scale * self.transformer_dims
            )
            self.linear2 = nn.Linear(
                self.transformer_ffn_scale * self.transformer_dims,
                self.transformer_dims
            )

    def forward(self, x):
        bsz = x.shape[0]
        res = x
        x = self.layernorm1(x)

        seqlens = self.depth * torch.ones((bsz,), dtype=torch.int32, device=x.device)
        _x = self.self_attention(x.view(-1, self.transformer_dims), seqlens)
        _x = _x.view(bsz, self.depth, self.transformer_dims).contiguous()

        # seq_len = x.size(1)
        # src_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        # # _x, _ = self.self_attention(x, x, x,  is_causal=True, attn_mask=src_mask)
        # _x, _ = self.self_attention(x, x, x, attn_mask=src_mask)

        
        # if _x.isnan().any():
        #     raise NotImplementedError(f"{seqlens}, 0.2")
        
        _res = _x + res  # (bs, sl, d)
        res = self.layernorm2(_res)

        if self.transformer_ffn_scale <= 0:  # 兼容baichuan-omni-1d5老逻辑
            x = torch.einsum('bld,tld->blt', res, torch.reshape(self.linear1.weight, (2 * self.llm_hidden_size, -1, self.llm_hidden_size)))
            x = torch.nn.functional.gelu(x)
            x = torch.einsum('blt,dlt->bld', x, torch.reshape(self.linear2.weight, (self.llm_hidden_size, -1, 2 * self.llm_hidden_size)))
        else:
            if self.depth > 1:
                x = torch.einsum('bld,tld->blt', 
                    res, 
                    torch.reshape(self.linear1.weight, 
                        (self.transformer_ffn_scale * self.transformer_dims // self.depth,
                        self.depth,
                        self.transformer_dims)
                    ))
            else:
                x = self.linear1(res)
            x = torch.nn.functional.gelu(x)
            if self.depth > 1:
                x = torch.einsum('blt,dlt->bld',
                    x, 
                    torch.reshape(self.linear2.weight, 
                        (self.transformer_dims,
                        self.depth,
                        self.transformer_ffn_scale * self.transformer_dims // self.depth)
                    ))
            else:
                x = self.linear2(x)
        return _res + x

class OmniImageHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.llm_hidden_size = config.hidden_size
        self.codebook_sizes = config.visual_quantizer_config.codebook_sizes # list
        self.config = config.visual_config.image_head_config
        self.transformer_ffn_scale = self.config.image_head_transformer_ffn_scale
        self.transformer_dims = self.config.image_head_transformer_dims
        self.transformer_layers = self.config.image_head_transformer_layers

        if self.transformer_ffn_scale > 0:
            self.hidden_norm = RMSNorm(self.llm_hidden_size)
            self.hidden_proj = nn.Linear(self.llm_hidden_size, self.transformer_dims, bias=False)
        self.transformer_layers = nn.ModuleList([
            CasualDepthTransformerLayer(config, len(self.codebook_sizes)) for _ in range(self.transformer_layers)])
        # self.head_proj = nn.Linear(self.transformer_dims, self.llm_hidden_size, bias=False)
        self.headnorm = RMSNorm( self.transformer_dims) 
        self.heads = nn.ModuleList([ nn.Linear(self.transformer_dims, vq_size+1) for vq_size in self.codebook_sizes])
        self.gradient_checkpointing = True
        if not self.config.enable:
            for param in self.parameters():  # 修复SFT阶段可能错误加载导致OOM的问题
                param.requires_grad = False

    
    def forward(self, x, visual_tokens, visual_emb_layers):
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
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs)

                    return custom_forward

                hidden_states  = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(tlayer), hidden_states,
                )
            else:
                hidden_states  = tlayer(
                    hidden_states,
                )
        # hidden_states = self.head_proj(hidden_states)
        hidden_states = self.headnorm(hidden_states)
        logits = [head(hidden_states[:,i]) for i,head in enumerate(self.heads)]
        # # 使用NormHead 对tie embedding进行归一化后计算logits
        # logits = [
        #     nn.functional.linear(
        #         hidden_states[:, i], 
        #         nn.functional.normalize(visual_emb_layers[i].weight, eps=1e-6)
        #         )
        #     for i, _ in enumerate(self.codebook_sizes)
        # ]
        return logits
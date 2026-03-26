import torch
from torch import nn
from typing import List, Optional, Tuple, Union
from transformers.activations import ACT2FN

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

class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act, rms_norm_eps, is_sparse=False):
        super().__init__()
        self.hidden_size = hidden_size
        # self.self_attn = Attention(config=config, is_sparse=is_sparse)
        self.mlp = MLP(
            hidden_size=self.hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
        )
        # self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.post_attention_layernorm = RMSNorm_no_weight(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_layernorm = nn.LayerNorm(hidden_size, eps=rms_norm_eps)

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
    def __init__(self, codebook_sizes, hidden_size, intermediate_size, hidden_act, rms_norm_eps):
        super().__init__()
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(codedim+1, hidden_size)
            for _, codedim in enumerate(codebook_sizes)
        ])
        self.hidden_size = hidden_size
        self.codebook_num = len(codebook_sizes)
        # 添加transformer block
        self.transformer_block = DecoderLayer(hidden_size, intermediate_size, hidden_act, rms_norm_eps)

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

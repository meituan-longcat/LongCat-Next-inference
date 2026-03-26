# Copyright (c) 2022-present, Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Iterable

import numpy as np
import torch
# import torch.distributed as dist
import deepspeed.comm as dist
from torch import nn
import torch.distributed
from torch.nn import functional as F
from torch.amp import autocast # 导入 AMP 相关的工具

def uniform_init(*shape):
    t = torch.zeros(shape)
    nn.init.kaiming_uniform_(t)
    return t

class VQEmbedding(nn.Module):
    """VQ embedding module with ema update."""

    def __init__(self, n_embed, embed_dim, ema=True, decay=0.99, restart_unused_codes=True, eps=1e-5, init_std=0.02):
        super().__init__()

        self.ema = ema
        self.decay = decay
        self.eps = eps
        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed
        self.init_std = init_std

        assert self.ema, "RVQ目前仅支持ema更新码本"
        embed = uniform_init(n_embed + 1, embed_dim).to(torch.float32)
        self.embed = nn.Parameter(embed)
        self.embed_ema = nn.Parameter(embed[:-1, :].clone())
        self.cluster_size_ema = nn.Parameter(torch.ones(n_embed))
        del embed
        _ = [p.requires_grad_(False) for p in self.parameters()]

    @torch.no_grad()
    def compute_distances(self, inputs):
        codebook_t = self.embed[:-1, :].t()

        (embed_dim, _) = codebook_t.shape
        inputs_shape = inputs.shape
        assert inputs_shape[-1] == embed_dim

        inputs_flat = inputs.reshape(-1, embed_dim)

        inputs_norm_sq = inputs_flat.pow(2.).sum(dim=1, keepdim=True)
        codebook_t_norm_sq = codebook_t.pow(2.).sum(dim=0, keepdim=True)
        distances = torch.addmm(
            inputs_norm_sq + codebook_t_norm_sq,
            inputs_flat,
            codebook_t,
            alpha=-2.0,
        )
        distances = distances.reshape(*inputs_shape[:-1], -1)  # [B, h, w, n_embed or n_embed+1]
        return distances

    @torch.no_grad()
    def find_nearest_embedding(self, inputs):
        distances = self.compute_distances(inputs)  # [B, h, w, n_embed or n_embed+1]
        embed_idxs = distances.argmin(dim=-1)  # use padding index or not

        return embed_idxs
    
    @autocast('cuda', enabled=True, dtype=torch.float32)
    @torch.no_grad()
    def forward(self, inputs, input_is_fake=False):
        if inputs.dtype != torch.float32:
            inputs = inputs.to(torch.float32)
        embed_idxs = self.find_nearest_embedding(inputs)
        embeds = self.embed[embed_idxs]
        return embeds, embed_idxs

class RQBottleneck(nn.Module):
    """
    Quantization bottleneck via Residual Quantization.

    Arguments:
        latent_shape (Tuple[int, int, int]): the shape of latents, denoted (H, W, D)
        code_shape (Tuple[int, int, int]): the shape of codes, denoted (h, w, d)
        n_embed (int, List, or Tuple): the number of embeddings (i.e., the size of codebook)
            If isinstance(n_embed, int), the sizes of all codebooks are same.
        shared_codebook (bool): If True, codebooks are shared in all location. If False,
            uses separate codebooks along the ``depth'' dimension. (default: False)
        restart_unused_codes (bool): If True, it randomly assigns a feature vector in the curruent batch
            as the new embedding of unused codes in training. (default: True)
    """

    def __init__(self,
                 latent_shape,
                 code_shape,
                 n_embed,
                 decay=0.99,
                 shared_codebook=False,
                 restart_unused_codes=True,
                 commitment_loss='cumsum'
                 ):
        super().__init__()

        if not len(code_shape) == len(latent_shape) == 3:
            raise ValueError("incompatible code shape or latent shape")
        if any([y % x != 0 for x, y in zip(code_shape[:2], latent_shape[:2])]):
            raise ValueError("incompatible code shape or latent shape")

        #residual quantization does not divide feature dims for quantization.
        embed_dim = np.prod(latent_shape[:2]) // np.prod(code_shape[:2]) * latent_shape[2]
        print(f"shared_codebook: {shared_codebook}. embed_dim: {embed_dim}")

        self.latent_shape = torch.Size(latent_shape)
        self.code_shape = torch.Size(code_shape)
        self.shape_divisor = torch.Size([latent_shape[i] // code_shape[i] for i in range(len(latent_shape))])
        
        self.shared_codebook = shared_codebook
        if self.shared_codebook:
            if isinstance(n_embed, Iterable) or isinstance(decay, Iterable):
                raise ValueError("Shared codebooks are incompatible \
                                    with list types of momentums or sizes: Change it into int")

        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed if isinstance(n_embed, Iterable) else [n_embed for _ in range(self.code_shape[-1])]
        self.decay = decay if isinstance(decay, Iterable) else [decay for _ in range(self.code_shape[-1])]
        assert len(self.n_embed) == self.code_shape[-1]
        assert len(self.decay) == self.code_shape[-1]

        if self.shared_codebook:
            codebook0 = VQEmbedding(self.n_embed[0], 
                                    embed_dim, 
                                    decay=self.decay[0], 
                                    restart_unused_codes=restart_unused_codes,
                                    ).to(torch.float32)
            self.codebooks = nn.ModuleList([codebook0 for _ in range(self.code_shape[-1])])
            # self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536)))
        else:
            codebooks = [VQEmbedding(self.n_embed[idx], 
                                     embed_dim, 
                                     decay=self.decay[idx], 
                                     restart_unused_codes=restart_unused_codes,
                                     ).to(torch.float32) for idx in range(self.code_shape[-1])]
            self.codebooks = nn.ModuleList(codebooks)
            # self.register_buffer("codebook_used", nn.Parameter(torch.zeros(self.code_shape[-1], 65536)))

        self.commitment_loss = commitment_loss

    def to_code_shape(self, x):
        (B, H, W, D) = x.shape
        (rH, rW, _) = self.shape_divisor

        x = x.reshape(B, H//rH, rH, W//rW, rW, D)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, H//rH, W//rW, -1)

        return x

    def to_latent_shape(self, x):
        (B, h, w, _) = x.shape
        (_, _, D) = self.latent_shape
        (rH, rW, _) = self.shape_divisor

        x = x.reshape(B, h, w, rH, rW, D)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, h*rH, w*rW, D)

        return x

    def quantize(self, x, input_is_fake):
        r"""
        Return list of quantized features and the selected codewords by the residual quantization.
        The code is selected by the residuals between x and quantized features by the previous codebooks.

        Arguments:
            x (Tensor): bottleneck feature maps to quantize.

        Returns:
            quant_list (list): list of sequentially aggregated and quantized feature maps by codebooks.
            codes (LongTensor): codewords index, corresponding to quants.

        Shape:
            - x: (B, h, w, embed_dim)
            - quant_list[i]: (B, h, w, embed_dim)
            - codes: (B, h, w, d)
        """
        B, h, w, embed_dim = x.shape
        ori_dtype = x.dtype
        x = x.to(torch.float32)
        self.codebooks = self.codebooks.to(torch.float32)

        residual_feature = x.detach().clone()

        quant_list = []
        code_list = []
        aggregated_quants = torch.zeros_like(x)
        for i in range(self.code_shape[-1]):
            quant, code = self.codebooks[i](residual_feature, input_is_fake)
            residual_feature.sub_(quant)
            aggregated_quants.add_(quant)
            quant_list.append(aggregated_quants.clone().to(dtype=ori_dtype))
            code_list.append(code.unsqueeze(-1))
        
        codes = torch.cat(code_list, dim=-1)
        return quant_list, codes

    def forward(self, x, input_is_fake):
        x_reshaped = self.to_code_shape(x)
         # 强制使用float32精度来执行
        quant_list, codes = self.quantize(x_reshaped, input_is_fake)
        # quant_list, codes = self.quantize(x_reshaped)

        commitment_loss = self.compute_commitment_loss(x_reshaped, quant_list)
        quants_trunc = self.to_latent_shape(quant_list[-1])
        quants_trunc = x + (quants_trunc - x).detach()
        
        '''
        if self.shared_codebook:
            cur_len = codes.view(-1).shape[0]
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = codes.view(-1)
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_embed[0]
        else:
            # info|code: torch.Size([10, 16, 16, 4])
            codebook_usage = 0
            for idx in range(self.code_shape[-1]):
                cur_len = codes[..., idx].view(-1).shape[0]
                self.codebook_used[idx, :-cur_len] = self.codebook_used[idx, cur_len:].clone()
                self.codebook_used[idx, -cur_len:] = codes[..., idx].view(-1)
                codebook_usage += len(torch.unique(self.codebook_used[idx]))
            codebook_usage /= (self.n_embed[0] * self.code_shape[-1])
        '''
        codebook_usage = 0
        # (vq_loss, commit_loss, entropy_loss, codebook_usage) # 格式对齐
        codebook_loss = [0, commitment_loss, 0, codebook_usage]

        return quants_trunc, codebook_loss, codes
    
    def compute_commitment_loss(self, x, quant_list):
        r"""
        Compute the commitment loss for the residual quantization.
        The loss is iteratively computed by aggregating quantized features.
        """
        loss_list = []
        
        for idx, quant in enumerate(quant_list):
            partial_loss = (x-quant.detach()).pow(2.0).mean()
            loss_list.append(partial_loss)
        
        commitment_loss = torch.mean(torch.stack(loss_list))
        return commitment_loss
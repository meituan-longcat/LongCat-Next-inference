from sglang.global_config import global_config
from torch import Tensor
from sglang.srt.models.longcat_flash import FLASHModel
from .context import LongcatOOverEmbContext
from .state_machine import StateEnum, StateMachine, StateMachineInput
from sglang.srt.managers.schedule_batch import Req
from typing import Dict, List
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.layers.dp_attention import get_attention_tp_rank
from .mllm_over_embedding import  MllmOverEmbedding
from .special_token import init_spt, get_spt
from utils.logger import logger
import torch

class LongcatOOverEmbInputProcessor():

    def __init__(self, base_lm, context:LongcatOOverEmbContext, config_dict):
        self.base_lm = base_lm
        self.ctx = context
        
        if self.ctx.use_oe: # 只有flash模型用到了这些
            self.flash_model: FLASHModel = self.base_lm.model
            self.oe = MllmOverEmbedding(self.flash_model.over_embedding, config_dict)
        self.special_tokens_set = set(self.ctx.config.multimodal_special_token_list)
        self.sm_dict = self.ctx.sm_dict
        if self.ctx.visual_enable:
            self.visual_bridge_model = self.ctx.visual_bridge_model
        self.hidden_size = self.ctx.hidden_size
        print(f"\033[32m[{self.special_tokens_set=}]\033[0m")
        
        self.enable_cuda_graph = config_dict.get("enable_cuda_graph", False)
        self.replay_cuda_graph = False
        if self.enable_cuda_graph:
            self._inited_for_graphs = False
            self.init_for_graphs()

    def init_for_graphs(self):
        raise NotImplementedError ("cuda图还没实现TODO")
    
    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor=None,
        input_multi_ids: Tensor=None,
        forward_batch: ForwardBatch=None,
        input_embeds: Tensor = None,
    ) -> Tensor:
    
        assert input_embeds is None
        # assert forward_batch.forward_mode == ForwardMode.DECODE
        # batch 中全是生文任务，不需要执行图像多级 token id 转 embedding，直接返回 text embedding 结果即可。
        if self.ctx.check_text_generation_only_in_batch(forward_batch):
            if not self.ctx.use_oe:
                return self.base_lm.model.embed_tokens(input_ids)
            else:
                return self.decode_oe_with_sp_new(input_ids, forward_batch)
        
        tp_num_tokens = input_ids.shape[0]
        if input_multi_ids is None:
            input_multi_ids = torch.as_tensor(forward_batch.input_multi_ids, dtype=torch.int64, device=input_ids.device)
            input_multi_ids = input_multi_ids.reshape(tp_num_tokens, -1)
        assert tp_num_tokens == forward_batch.batch_size
        return self.forward_decode(input_ids, input_multi_ids, forward_batch)
        

    def get_emb(self, input_ids_list):
        return self.forward_2d_ids_with_sp(input_ids_list)
    
    def forward_extend(
        self,
        input_ids: Tensor=None,
        positions: Tensor=None,
        forward_batch: ForwardBatch=None,
        input_embeds: Tensor = None,
    ):
        for req_idx in range(len(forward_batch.reqs)):
            req: Req = forward_batch.reqs[req_idx]
            sm = StateMachine(max_gen=1000) # TODO:改为配置
            self.ctx.sm_dict[req.rid] = sm
            gen_image=req.input_extra_infos[0].get("gen_image", False)
            gen_audio=req.input_extra_infos[0].get("gen_audio", False)
            sm_input = StateMachineInput(gen_image=gen_image, gen_audio=gen_audio)
            trans = sm.process(sm_input)
            if trans and get_attention_tp_rank() == 0:
                logger.trace(f"\033[34m[[{req_idx=}, Rid {req.rid}] {sm.to_string()}]\033[0m")
        return input_embeds
    
    def forward_decode(
        self,
        input_ids: Tensor,
        input_multi_ids: Tensor,
        forward_batch: ForwardBatch,
    ):  
        if not self.ctx.use_oe:
            inputs_embeds =self.base_lm.model.embed_tokens(input_ids)
        else:
            inputs_embeds = self.decode_oe_with_sp_new(input_ids, forward_batch)
        input_ids = input_ids.reshape(-1)
        assert input_ids.dim() == 1, f"{input_ids.shape=}"
        text_indices, image_indices, audio_indices, _, _ = self.ctx.collect_gen_type_indices(forward_batch)
        # TODO: 都enable时需要考虑冗余计算，最后取mask，以及考虑视觉的id 超过音频emb词表大小怎么处理
        if self.ctx.audio_enable and len(audio_indices) > 0:
            ext_ids = []
            for req in forward_batch.reqs:
                ext_ids.append(req.input_extra_infos[0].get("ext_ids", get_spt().AUDIOTEXT_PAD_TOKEN_ID))
            ext_ids_tensor = torch.tensor(ext_ids, dtype=torch.long, device="cuda")  # [B]
            all_embeds = self.get_audio_embeddings(input_ids[audio_indices], input_multi_ids[audio_indices], inputs_embeds[audio_indices], ext_ids_tensor[audio_indices])
            all_embeds = all_embeds.reshape(len(audio_indices), -1)
            inputs_embeds[audio_indices] = all_embeds
        if self.ctx.visual_enable and len(image_indices) > 0:
            all_embeds = self.get_visual_embed_given_tokens(input_ids[image_indices], inputs_embeds[image_indices], input_multi_ids[image_indices])
            inputs_embeds[image_indices] = all_embeds
        return inputs_embeds
    
    def get_audio_embeddings(
        self,
        input_ids: Tensor,
        input_multi_ids: Tensor,
        inputs_embeds: Tensor,
        ext_ids_tensor: Tensor
    ) -> Tensor:
        """
        Compute audio-based embeddings and add them to the input embeddings.
        """
        input_ids_mask = (input_ids != get_spt().AUDIOTEXT_PAD_TOKEN_ID).unsqueeze(-1)
        ext_text_mask = (ext_ids_tensor != get_spt().AUDIOTEXT_PAD_TOKEN_ID).unsqueeze(-1)
        valid_audio_tokens = torch.clamp(input_multi_ids, min=0)
        multi_ids_row_mask = (
            (valid_audio_tokens[:, 0] != 0) & 
            (valid_audio_tokens[:, 0] != self.ctx.audio_codebook_sizes[0])
        ).unsqueeze(-1)   # Mask rows where the first-level ID is 0 or 8192

        inputs_embeds = inputs_embeds * input_ids_mask
        if not self.ctx.use_oe:
            ext_ids_emb =self.base_lm.model.embed_tokens(ext_ids_tensor)
        else: # ext_id一定是spt，所以只过oe的基础emb
            ext_ids_emb = self.oe.word_embeder(ext_ids_tensor)
        ext_ids_emb  = ext_ids_emb*ext_text_mask
        for i, audio_emb_layer in enumerate(self.ctx.audio_embed_layers):
            if i == 0:
                audio_embs = audio_emb_layer(valid_audio_tokens[..., i])
            else:
                audio_embs += audio_emb_layer(valid_audio_tokens[..., i])
        audio_embs *= multi_ids_row_mask  # Mask out invalid rows
        inputs_embeds = ext_ids_emb + inputs_embeds + audio_embs

        return inputs_embeds
    
    def get_visual_embed_given_tokens(
        self, 
        input_ids,
        text_embedding,  # 1. self.embed_tokens(input_ids) 2. 其他模态结果
        vision_tokens = None,
    ): 
        # 如果用 pad_mask 直接取对应 index 的结果，在 capture cuda graph 时会报错，因为shape不确定
        # valid_vision_tokens = vision_tokens[pad_mask]
        # 只在 <img_pad> 位置保留真实的多级 id；其他位置（纯文本或 <img_newline>）强制置 0，让它们走文本 embedding。
        pad_mask_for_img = input_ids.eq(get_spt().IMAGE_PAD_TOKEN_ID).unsqueeze(-1)
        safe_vision_tokens = torch.where(pad_mask_for_img, vision_tokens, torch.zeros_like(vision_tokens))
        # 多级id中的负值无法做lookup，转成0做完lookup后，在get_multimodal_embed中会被mask。
        valid_vision_tokens = torch.clamp(safe_vision_tokens, min=0)
        vision_embed = self.visual_bridge_model(valid_vision_tokens)
        # 目前还不支持video
        final_embedding = self.get_multimodal_embed(input_ids, text_embedding, vision_embed, get_spt().IMAGE_PAD_TOKEN_ID)
        return final_embedding
    
    def get_multimodal_embed(
            self, 
            input_ids,
            text_embedding,  # 1. self.embed_tokens(input_ids) 2. 其他模态结果
            multimodal_embed,
            pad_token_id,
        ):
        pad_mask = torch.eq(input_ids, pad_token_id)
        assert multimodal_embed.device == input_ids.device

        # 合并 当前模态embeddings 和text embeddings
        text_embedding = (1 - pad_mask.to(text_embedding)).unsqueeze(-1) * text_embedding  # pad token位置填0 (不传梯度)
        pad_mask = pad_mask.to(multimodal_embed).view(-1, 1)
        multimodal_embedding = pad_mask * multimodal_embed  # 非pad token 位置填0
        final_embedding = multimodal_embedding.to(text_embedding) + text_embedding
        # 每次 decode 生成一个 token，在 LLM 部分使用时没有这个维度
        final_embedding = final_embedding.squeeze(dim=1)

        return final_embedding
    
    def forward_ids_list(self, input_ids_list):
        return self.oe.forward_ids_list(input_ids_list)

    def get_last_tokens_split(self, fill_ids: List[int], special_tokens_set: set) -> List[int]:
        last_spt_idx = None
        for i in range(len(fill_ids) - 1, -1, -1):
            if fill_ids[i] in special_tokens_set:
                last_spt_idx = i
                break

        if last_spt_idx is None:
            return fill_ids
        res = fill_ids[last_spt_idx + 1 :]
        print(f"\033[31m[Cut: {fill_ids[last_spt_idx]=}. {res=}]\033[0m")
        return res

    def decode_oe_with_sp_new(self, input_ids: Tensor, forward_batch: ForwardBatch):
        res = torch.empty(
            (forward_batch.batch_size, self.hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
        )
        masks = torch.zeros(forward_batch.batch_size, dtype=torch.bool)
        tokens_without_spt_list = []
        for req_idx, req in enumerate(forward_batch.reqs):
            # 这时候req.output_ids还没有拼接当前步的输出id
            if req.output_ids and len(req.output_ids) > 0:
                output_ids = req.output_ids + [input_ids[req_idx].item()]
            else:
                output_ids = [input_ids[req_idx].item()]
            if int(output_ids[-1]) in self.special_tokens_set:
                # print(f"\033[31m[直接过 Base Embedding]{req.output_ids[-1]=}\033[0m")
                output_id = torch.tensor(output_ids[-1], dtype=torch.long, device="cuda")
                emb = self.oe.word_embeder(output_id)
                res[req_idx] = emb

                masks[req_idx] = True  # 标记为已处理
                tokens_without_spt = [0] # 设置一个无效token,方便batch过oe,当前请求的oe结果会被mask掉
            else:
                fill_ids = req.origin_input_ids + output_ids
                # decode的话，取最后n-1个token，不足则前面补0
                if len(fill_ids) >= self.oe.over_embedding_n:
                    look_ahead_tokens = fill_ids[-self.oe.over_embedding_n :]
                else:
                    pad_len = self.oe.over_embedding_n - len(fill_ids)
                    look_ahead_tokens = [0] * pad_len + fill_ids
                tokens_without_spt = self.get_last_tokens_split(look_ahead_tokens, self.special_tokens_set)
            tokens_without_spt_list.append(tokens_without_spt)

        oe_res = self.oe.forward_ids_list_decode(tokens_without_spt_list)

        res[~masks] = oe_res[~masks]
        return res

    def forward_2d_ids_with_sp(self, ids_2d):
        """
        将 batch 中的 normal tokens 按 chunk 分组 forward，
        special tokens 单独 forward，最后重构为 [B, L_max, D] 输出。
        """
        assert isinstance(ids_2d[0], list), "输入不是列表类型"
        B = len(ids_2d)
        D = self.hidden_size
        if B == 0:
            return torch.empty(0, 0, D, device="cuda")

        L_max = max(len(seq) for seq in ids_2d)
        device = "cuda"

        all_normal_chunks = []
        all_special_ids = []

        normal_positions_list = []
        special_positions_list = []
        normal_chunk_indices_list = []
        special_token_indices_list = []

        for seq in ids_2d:
            if len(seq) == 0:
                normal_positions_list.append([])
                special_positions_list.append([])
                normal_chunk_indices_list.append([])
                special_token_indices_list.append([])
                continue
            # 创建 mask
            is_special = torch.tensor([x in self.special_tokens_set for x in seq], dtype=torch.bool, device=device)
            is_normal = ~is_special

            normal_pos = torch.where(is_normal)[0].cpu().tolist()
            special_pos = torch.where(is_special)[0].cpu().tolist()

            normal_positions_list.append(normal_pos)
            special_positions_list.append(special_pos)

            # 处理 normal tokens: 分 chunk（连续 token 视为一个 chunk）
            normal_ids = [seq[i] for i in normal_pos]
            if len(normal_ids) > 0:
                # 判断是否连续：diff == 1 表示连续
                diff = [normal_pos[i] - normal_pos[i - 1] for i in range(1, len(normal_pos))]
                is_new_chunk = [d != 1 for d in diff]
                is_new_chunk.insert(0, True)  # 第一个元素总是新 chunk 的开始

                chunk_ids = []
                current_chunk = 0
                for is_new in is_new_chunk:
                    if is_new:
                        current_chunk += 1
                    chunk_ids.append(current_chunk)

                # 按 chunk 分组
                chunks = []
                for cid in range(1, max(chunk_ids) + 1):
                    chunk = [normal_ids[i] for i in range(len(normal_ids)) if chunk_ids[i] == cid]
                    chunks.append(chunk)

                # 记录每个 normal token 在 all_normal_chunks 中的全局索引
                normal_global_indices = []
                lens = 0
                for chunk in chunks:
                    # start = len(all_normal_chunks)
                    start = lens
                    lens += len(chunk)
                    all_normal_chunks.append(chunk)
                    normal_global_indices.extend(range(start, start + len(chunk)))
                normal_chunk_indices_list.append(normal_global_indices)
            else:
                normal_chunk_indices_list.append([])

            # 处理 special tokens
            special_ids = [seq[i] for i in special_pos]
            if len(special_ids) > 0:
                start = len(all_special_ids)
                all_special_ids.extend(special_ids)
                special_token_indices_list.append(list(range(start, start + len(special_ids))))
            else:
                special_token_indices_list.append([])
        # === 批量 forward（确定性，可复现）===
        if all_normal_chunks:
            # print("all_normal_chunks", all_normal_chunks)
            normal_result = self.forward_ids_list(all_normal_chunks)  # [L_normal, D]
        else:
            normal_result = torch.empty(0, D, device=device)

        if all_special_ids:
            special_result = self.oe.word_embeder(torch.as_tensor(all_special_ids, dtype=torch.int32, device=device))  # [L_special, D]
        else:
            special_result = torch.empty(0, D, device=device)

        # === 重构每个样本 ===
        reconstructed_batch = torch.zeros(B, L_max, D, device=device, dtype=torch.bfloat16)

        for i in range(B):
            seq_len = len(ids_2d[i])
            if seq_len == 0:
                continue

            # 填充 normal
            if normal_positions_list[i]:
                global_idx = normal_chunk_indices_list[i]
                local_pos = normal_positions_list[i]
                reconstructed_batch[i, local_pos] = normal_result[global_idx]

            # 填充 special
            if special_positions_list[i]:
                global_idx = special_token_indices_list[i]
                local_pos = special_positions_list[i]
                reconstructed_batch[i, local_pos] = special_result[global_idx]
        return reconstructed_batch

import torch
from typing import Iterable, Optional, Tuple, List
from sglang.srt.utils import get_colorful_logger
logger = get_colorful_logger(__name__)

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.layers.over_embedding import FusedOverEmbedding
import traceback
# try:
#     from sgl_kernel.embedding_scaling import compute_n_gram_ids
# except:
#     print(f"\033[31m[============ compute_n_gram_ids 不可用！{traceback.format_exc()} ============]\033[0m")


class MllmOverEmbedding:
    def __init__(self, over_embedding, config_dict):
        self.oe: FusedOverEmbedding = over_embedding
        self.oe_mods = self.oe.oe_mods
        self.oe_weights = self.oe.oe_weights
        self.over_embedding_n = self.oe.over_embedding_n
        self.over_embedding_k = self.oe.over_embedding_k
        self.n_grams = self.oe.n_grams
        self.exclusive_oe_embeder_size_sums = self.oe.exclusive_oe_embeder_size_sums
        self.word_embeder = self.oe.word_embeder
        # self.special_tokens_set = set(config_dict.special_tokens_list)
        # self.hidden_size = config_dict.hidden_size
        # self.dtype = config_dict.dtype
        # self.config_dict = config_dict
        # self.oe_ignore_tokens = torch.tensor(self.config_dict.oe_ignore_tokens, device='cuda')
        # logger.info(f"\033[32m[{self.special_tokens_set=}]\033[0m")


    def forward_oe_lookup(self, 
                          input_ids,
                          over_embedding_ids,
                          oe_exclusive_req_len_sums,
                          oe_exclusive_oe_info_len_sums, 
                          ):
        device = input_ids.device
        over_embedding_input_ids = torch.tensor(sum(over_embedding_ids, [])).view(-1).to(device, dtype=torch.int32)
        oe_exclusive_req_len_sums = torch.tensor(oe_exclusive_req_len_sums, device=device, dtype=torch.int32)
        oe_exclusive_oe_info_len_sums = torch.tensor(oe_exclusive_oe_info_len_sums, device=device, dtype=torch.int32)
        oe_n_gram_ids = torch.zeros([oe_exclusive_req_len_sums[-1] - oe_exclusive_oe_info_len_sums[-1],
                                            (self.oe.over_embedding_n - 1) * self.oe.over_embedding_k],
                                    device = device,
                                    dtype = torch.int32)
        # todo
        oe_ignore_oe_input_ids_flags = torch.isin(over_embedding_input_ids, self.oe_ignore_tokens)
        oe_ignore_input_ids_flags = torch.isin(input_ids, self.oe_ignore_tokens).unsqueeze(1)
        # logger.info(f"forward_oe_lookup oe_n_gram_ids:{oe_n_gram_ids.shape} {over_embedding_input_ids.shape}")
        compute_n_gram_ids(
            self.oe.over_embedding_n,
            self.oe.over_embedding_k,
            self.oe.oe_weights,
            self.oe.oe_mods,
            over_embedding_input_ids,
            oe_ignore_oe_input_ids_flags,
            oe_exclusive_req_len_sums,
            oe_exclusive_oe_info_len_sums,
            self.oe.exclusive_oe_embeder_size_sums,
            oe_n_gram_ids,
        )
        
        #logger.info(f"forward_oe_lookup 2 oe_n_gram_ids:{oe_n_gram_ids.shape} {over_embedding_input_ids.shape}")
        # [13, seq_len, hidden_dim]
        all_hidden_states = torch.empty([(self.oe.over_embedding_n - 1) * self.oe.over_embedding_k + 1, len(input_ids), self.oe.embedding_dim],
                                        dtype =  self.oe.oe_projection.dtype, device = 'cuda')
        #logger.info(f"forward_oe_lookup 3 oe_n_gram_ids:{all_hidden_states.shape} {input_ids.shape} {input_ids.tolist()=}")
        input_ids_clamped_tensor = torch.clamp(input_ids, min=0)
        all_hidden_states[0] = self.oe.word_embeder(input_ids_clamped_tensor)
        #logger.info(f"forward_oe_lookup 4 oe_n_gram_ids:{oe_n_gram_ids.shape} ")
        # oe_hidden_states: [12, seq_len, hidden_dim / 12]
        oe_hidden_states = self.oe.oe_embeder(oe_n_gram_ids.permute(1, 0).contiguous())
        #logger.info(f"forward_oe_lookup 5 oe_n_gram_ids:{oe_hidden_states.shape} {all_hidden_states.shape}")
        torch.bmm(oe_hidden_states, self.oe.oe_projection, out=all_hidden_states[1:])
        #logger.info(f"forward_oe_lookup 6 oe_n_gram_ids:")
        # 增加一路word embedding
        mean_hidden_states = all_hidden_states.mean(dim=0)
        #logger.info(f"forward_oe_lookup 7 oe_n_gram_ids:{mean_hidden_states.shape} {all_hidden_states.shape} {oe_ignore_input_ids_flags.shape}")
        results = torch.where(oe_ignore_input_ids_flags, all_hidden_states[0], mean_hidden_states)
        #logger.info(f"forward_oe_lookup 8 oe_n_gram_ids:{results.shape}")
        return results

    def compute_n_gram_ids(self, input_ids, n, k):
        mod = self.oe_mods[n - 2][k]
        # 检查输入是否为一维张量，如果是则扩展为二维
        is_1d = len(input_ids.shape) == 1
        if is_1d:
            input_ids = input_ids.unsqueeze(0)  # [seq_len] -> [1, seq_len]
        input_ids = input_ids.to(torch.int64)
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        # 创建结果张量
        result = torch.zeros((batch_size, seq_len), device=device, dtype=torch.int64)

        # 计算n-gram IDs，使用num_embeddings作为基数
        for i in range(seq_len):
            for j in range(max(0, i - n + 1), i + 1):
                # 计算当前位置的权重：num_embeddings^(i-j)
                weight = self.oe_weights[n - 2][k][i - j]
                # 计算当前项并取模
                term = (input_ids[:, j] * weight) % mod
                # 累加并取模
                result[:, i] = (result[:, i] + term) % mod

        # 如果输入是一维的，则将结果也转回一维
        if is_1d:
            result = result.squeeze(0)
        return result.to(torch.int32)
    
    def forward_ids_list(self, input_ids_list: List[List[int]]):
        '''
            根据id list计算oe emb
            输入: input_ids_list: List[List[int]] 每个list表示一个segment
            输出: 拼接后的emb，需要在外部根据长度进行切分
        '''
        device = "cuda"
        # Convert each segment to tensor
        input_ids_tensors = []
        for input_ids in input_ids_list:
            input_ids_tensors.append(torch.tensor(list(input_ids), device=device, dtype=torch.int32))
        
        # Compute total length
        total_len = sum(len(ids) for ids in input_ids_tensors)
        
        # Create output tensors
        over_embedding_input_ids = torch.cat(input_ids_tensors, dim=0)
        oe_n_gram_ids = torch.zeros(
            [total_len, self.over_embedding_n - 1, self.over_embedding_k],
            device=device,
            dtype=torch.int32,
        )
        
        # Compute n_gram_ids for each segment separately to avoid cross-segment contamination
        offset = 0
        for input_ids_tensor in input_ids_tensors:
            seg_len = len(input_ids_tensor)
            for i in range(2, self.over_embedding_n + 1):
                for j in range(self.over_embedding_k):
                    index = (i - 2) * self.over_embedding_k + j
                    oe_n_gram_ids[offset:offset + seg_len, i - 2, j] = self.compute_n_gram_ids(input_ids_tensor, i, j)
                    oe_n_gram_ids[offset:offset + seg_len, i - 2, j] += self.exclusive_oe_embeder_size_sums[index]
            offset += seg_len
        
        # Reshape from [total_len, n-1, k] to [total_len, n_grams] for oe_embeder
        oe_n_gram_ids = oe_n_gram_ids.view(total_len, self.n_grams)
        return self.oe.compute_hidden_states(over_embedding_input_ids, oe_n_gram_ids, total_len)
    
    def forward_ids_list_decode(self, input_ids_list: List[List[int]]):
        '''
            根据id list计算oe emb decode版，每个输入只保留最后一个id的emb
            输入: input_ids_list: List[List[int]] 每个list表示一个segment，包括当前token和lookuptoken
            输出: bs*hidden_dim的emb
        '''
        device = "cuda"
        num_segments = len(input_ids_list)
        
        # Compute n_gram_ids for each segment separately and keep only the last token
        last_input_ids = []  # last token id of each segment
        last_n_gram_ids = []  # last n_gram_ids of each segment
        
        for input_ids in input_ids_list:
            input_ids_tensor = torch.tensor(list(input_ids), device=device, dtype=torch.int32)
            seg_len = len(input_ids_tensor)
            
            # Compute n_gram_ids for this segment
            seg_n_gram_ids = torch.zeros(
                [seg_len, self.over_embedding_n - 1, self.over_embedding_k],
                device=device,
                dtype=torch.int32,
            )
            for i in range(2, self.over_embedding_n + 1):
                for j in range(self.over_embedding_k):
                    index = (i - 2) * self.over_embedding_k + j
                    seg_n_gram_ids[:, i - 2, j] = self.compute_n_gram_ids(input_ids_tensor, i, j)
                    seg_n_gram_ids[:, i - 2, j] += self.exclusive_oe_embeder_size_sums[index]
            
            # Keep only the last token's n_gram_ids
            last_input_ids.append(input_ids_tensor[-1:])
            last_n_gram_ids.append(seg_n_gram_ids[-1:])  # [1, n-1, k]
        
        # Concatenate last tokens from all segments
        over_embedding_input_ids = torch.cat(last_input_ids, dim=0)  # [num_segments]
        oe_n_gram_ids = torch.cat(last_n_gram_ids, dim=0)  # [num_segments, n-1, k]
        
        # Reshape from [num_segments, n-1, k] to [num_segments, n_grams] for oe_embeder
        oe_n_gram_ids = oe_n_gram_ids.view(num_segments, self.n_grams)
        hidden_states = self.oe.compute_hidden_states(over_embedding_input_ids, oe_n_gram_ids, num_segments)
        return hidden_states


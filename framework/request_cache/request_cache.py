import json
import math
from typing import Callable, Dict, List, Optional, Tuple
import time
import concurrent
import torch
import threading
import os

# from sglang.srt.utils import broadcast_wrapper
import torch.distributed as dist
from sglang.srt.server_args import ServerArgs
from sglang.srt.managers.req import FINISH_ABORT, Req
from sglang.srt.distributed import (
    get_tp_group,
)

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode, CaptureHiddenMode

from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    get_attention_tp_size,
    get_attention_tp_rank,
)

from sglang.srt.utils import (
    get_local_ip_by_remote,
    get_colorful_logger,
)

from framework.request_cache.utils import hash_to_int, parse_tmp_cache_schema_string, build_rid, parse_rid, dtype_map, parse_cache_name_dtype_dim
from framework.request_cache.request_cache_info import RequestCacheInfo
from framework.request_cache.shared_memory import RequestCacheInitInfo, SharedMemoryHandler

logger = get_colorful_logger(__name__)

EMBEDDING_CACHE_KEY="llm_encoder"
EMBEDDING_SUB_KEY="input_embedding"

ENABLE_GPU_OUTPUT_CACHE=False
# 定义global_state_tensor每一位的含义
_GLOBAL_STATE_BITS = {
    "SESSION_ID": 0,
    "ROUND_ID": 1,
    "TOTAL_STEP": 2,
    "SESSION_STATE_1": 3,
}

def is_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()

_is_npu = is_npu()

def broadcast_wrapper(tensor_size, group_src, dist_group):
    if not _is_npu:
        return dist.broadcast(tensor_size, group_src=group_src, group=dist_group)
    else:
        return dist.broadcast(tensor_size, src=group_src, group=dist_group)


class RequestCache:
    # 类变量：存储全局单例实例
    _instance = None
    _instance_lock = threading.RLock()

    @classmethod
    def get_instance(cls,
            server_args: ServerArgs=None, gpu_id: int=None, run_device = None, create: bool=None):
        with cls._instance_lock:
            if cls._instance is None:
                # 第一次创建实例
                if all(param is None for param in [server_args, gpu_id, create, run_device]):
                    raise ValueError("首次创建 RequestCache 实例时，必须提供所有配置参数")

                cls._instance = cls(server_args, gpu_id=gpu_id, run_device=run_device, create=create)
                return cls._instance
            else:
                if not all(param is None for param in [server_args, gpu_id, run_device, create]):
                    raise ValueError("已经创建，无需传递参数")
                return cls._instance

    @classmethod
    def has_instance(cls):
        """检查是否已经创建了实例"""
        return cls._instance is not None

    @classmethod
    def reset_instance(cls):
        """重置单例实例（主要用于测试）"""
        with cls._instance_lock:
            cls._instance = None

    def __init__(self,
            server_args: ServerArgs, gpu_id: int, run_device, create: bool):
        self.server_args = server_args
        self.page_size = server_args.page_size
        self.max_cache_size = server_args.request_cache_size
        assert self.max_cache_size >= server_args.max_running_requests
        self.output_max_cache_size =  self.max_cache_size# * 2
        self.max_input_len = server_args.request_max_input_len
        self.max_input_len = math.ceil(self.max_input_len/self.page_size) * self.page_size
        self.max_output_len = server_args.request_max_output_len
        self.max_output_len = math.ceil(self.max_output_len/self.page_size) * self.page_size
        self.max_seq_len = self.max_input_len + self.max_output_len
        assert self.max_input_len > 0
        assert self.max_seq_len > 0
        assert self.max_seq_len > 0
        self.buffer_max_seq_len = server_args.chunked_prefill_size * server_args.max_running_requests
        self.cache_device = "cpu"
        self.run_device = run_device

        self.global_state_dict = _GLOBAL_STATE_BITS
        self.global_state_tensor_dtype = torch.int64

        self.enable_pd = self.server_args.disaggregation_mode != "null"
        
        # dtype_mapping = {
        #     "float16": torch.float16,
        #     "bfloat16": torch.bfloat16,
        #     "int16": torch.int16,
        #     "int32": torch.int32,
        #     "int64": torch.int64,
        #     # "bool": torch.bool,
        # }
        # ids_dtype = dtype_mapping.get(ids_dtype)
        # embs_dtype = dtype_mapping.get(embs_dtype)
        request_cache_config = json.loads(server_args.request_cache_config)
        all_cache_info = parse_cache_name_dtype_dim(request_cache_config["request_cache_name_dtype_dim"])
        self.input_keys = request_cache_config["request_cache_input_names"]
        self.output_keys = request_cache_config["request_cache_output_names"]
        self.server_output_keys = request_cache_config["request_cache_server_output_names"]
        for key in self.input_keys:
            if key not in self.output_keys:
                assert False, f"output_keys must have all input_keys: {self.output_keys} {self.input_keys}"
        for key in self.server_output_keys:
            if key not in self.output_keys:
                assert False, f"output_keys must have all server_output_keys: {self.output_keys} {self.server_output_keys}"
        input_cache_init_info_list = []
        output_cache_init_info_list = []
        self.input_cache_init_info_map = {}
        for (name, dtype, dim) in all_cache_info:
            if name in self.input_keys:
                input_cache_init_info_list.append(RequestCacheInitInfo(name=name, dtype=dtype, lenth=self.max_seq_len, dim=dim))
                self.input_cache_init_info_map[name] = RequestCacheInitInfo(name=name, dtype=dtype, lenth=self.max_seq_len, dim=dim)
            if name in self.output_keys:
                output_cache_init_info_list.append(RequestCacheInitInfo(name=name, dtype=dtype, lenth=self.max_output_len, dim=dim))

        self.round_info_dtype = torch.int64
        input_cache_init_info_list.append(RequestCacheInitInfo(name="__system_input_ids", dtype=torch.int32, dim=1, lenth=self.max_seq_len))
        input_cache_init_info_list.append(RequestCacheInitInfo(name="__system_paged_hash_ids", dtype=torch.int64, dim=1, lenth=math.ceil(self.max_seq_len / server_args.page_size) * server_args.page_size))
        input_cache_init_info_list.append(RequestCacheInitInfo(name="__system_global_state", dtype=self.global_state_tensor_dtype, dim=1, lenth=len(self.global_state_dict)))
        input_cache_init_info_list.append(RequestCacheInitInfo(name="__system_round_info", dtype=self.round_info_dtype, dim=5, lenth=self.max_seq_len))

        output_cache_init_info_list.append(RequestCacheInitInfo(name="__system_input_ids", dtype=torch.int32, dim=1, lenth=self.max_output_len))
        # output_cache_init_info_list.append(RequestCacheInitInfo(name="__system_global_state", dtype=self.global_state_tensor_dtype, dim=1, lenth=len(self.global_state_dict)))

        logger.info(f"RequestCache init max_cache_size: {self.max_cache_size} max_seq_len: {self.max_seq_len} max_input_len:{self.max_input_len}, max_output_len:{self.max_output_len} input_cache_init_info_list:{input_cache_init_info_list} output_cache_init_info_list {output_cache_init_info_list} buffer_max_seq_len:{self.buffer_max_seq_len} input_keys:{self.input_keys} output_keys:{self.output_keys} server_output_keys:{self.server_output_keys}")


        self.input_cache_init_info_list = input_cache_init_info_list
        self.output_cache_init_info_list = output_cache_init_info_list

        if self.enable_pd:
            from sglang.srt.disaggregation.mooncake.transfer_engine import MooncakeTransferEngine
            self.transfer_engine = MooncakeTransferEngine(
                hostname=get_local_ip_by_remote(),
                gpu_id=gpu_id,
                ib_device=self.server_args.disaggregation_ib_device,
            )
        self.input_cache = SharedMemoryHandler(
            self.max_cache_size,
            self.input_cache_init_info_list,
            self.cache_device,
        )
        self.output_cache = SharedMemoryHandler(
            self.output_max_cache_size,
            self.output_cache_init_info_list,
            self.cache_device if gpu_id < 0 or not ENABLE_GPU_OUTPUT_CACHE else self.run_device,
        )
        self.suffix = os.environ.get("REQUEST_CACHE_SUFFIX")
        print(f"\033[32m[============ {self.suffix=} ============]\033[0m")
        self.input_cache.create_tensor_by_share_memory(f"___RequestCache_INPUT___{self.suffix}", create=create)
        self.output_cache.create_tensor_by_share_memory(f"___RequestCache_OUTPUT___{self.suffix}", create=create)


        def transfer_engine_register(ptrs, data_lens, row_data_lens):
            if not self.enable_pd:
                return
            for ptr, data_len, row_data_len in zip(ptrs, data_lens, row_data_lens):
                assert data_len % row_data_len == 0, f"transfer_engine_register error {data_len}/{row_data_len}"
                chunk_size = 4 * 1024 * 1024 * 1024
                if data_len <= chunk_size:
                    self.transfer_engine.register(ptr, data_len)
                else:
                    if row_data_len > chunk_size:
                        logger.warning(f"transfer_engine_register row_data_len too big may block transfer_engine.register")
                    # offset = 0
                    # while offset < data_len:
                    #     current_chunk_size = min(chunk_size, data_len - offset)
                    #     self.transfer_engine.register(ptr + offset, current_chunk_size)
                    #     offset += current_chunk_size
                    # for row_idx in range(0, row_data_len):
                    #     self.transfer_engine.register(ptr + row_data_len * row_idx, row_data_len)
                    
                    # Register in chunks of rows to avoid too many register calls and respect chunk_size
                    num_rows = data_len // row_data_len
                    rows_per_chunk = chunk_size // row_data_len
                    if rows_per_chunk == 0: 
                        rows_per_chunk = 1
                    
                    for i in range(0, num_rows, rows_per_chunk):
                        current_rows = min(rows_per_chunk, num_rows - i)
                        current_size = current_rows * row_data_len
                        self.transfer_engine.register(ptr + i * row_data_len, current_size)

        if gpu_id < 0:
            names, ptrs, data_lens, row_data_lens, item_lens = self.input_cache.get_tensor_buf_infos(exclude_keys=["__system_global_state", "__system_round_info"])
            if self.enable_pd:
                transfer_engine_register(ptrs, data_lens, row_data_lens)
            self.input_cache_ptrs = ptrs
            self.input_cache_names = names
            self.input_cache_data_lens = data_lens
            self.input_cache_item_lens = item_lens
        if ENABLE_GPU_OUTPUT_CACHE:
            names, ptrs, data_lens, row_data_lens, item_lens = self.output_cache.get_tensor_buf_infos()
            transfer_engine_register(ptrs, data_lens, row_data_lens)
            self.output_cache_ptrs = ptrs
            self.output_cache_names = names
            self.output_cache_data_lens = data_lens
            self.output_cache_item_lens = item_lens
        
        self.tmp_cache = {}
        self.tmp_cache_ptr_infos = {}
        if request_cache_config.get("request_cache_tmp_cache_name_len_dtypes_dims", None) is not None:
            for encoder_name_len_dtypes_dims in request_cache_config["request_cache_tmp_cache_name_len_dtypes_dims"]:
                (name, cache_size, length, parsed_tuples) = parse_tmp_cache_schema_string(encoder_name_len_dtypes_dims)
                logger.info(f"request_cache_tmp_cache: {(name, cache_size, length, parsed_tuples)}")
                tmp_cache_init_info_list = []
                for (sub_name, dtype, dim) in parsed_tuples:
                    tmp_cache_init_info_list.append(RequestCacheInitInfo(name=sub_name, dtype=dtype_map(dtype), lenth=length, dim=dim))
                cache = SharedMemoryHandler(
                    cache_size,
                    tmp_cache_init_info_list,
                    self.cache_device,
                )
                cache.create_tensor_by_share_memory(f"___RequestCache_ENCODER__{name}__{self.suffix}", create=create)
                names, ptrs, data_lens, row_data_lens, item_lens = cache.get_tensor_buf_infos()
                if gpu_id < 0:
                    transfer_engine_register(ptrs, data_lens, row_data_lens)
                self.tmp_cache[name] = cache
                self.tmp_cache_ptr_infos[name] = (cache_size, length, names, ptrs, data_lens, item_lens)
        
        self.cache_copy_exector = concurrent.futures.ThreadPoolExecutor(max_workers=server_args.max_running_requests * len(self.input_cache_init_info_list))

        self.SESSION_ID_offset = self.global_state_dict.get("SESSION_ID", None)
        assert self.SESSION_ID_offset is not None
        self.ROUND_ID_offset = self.global_state_dict.get("ROUND_ID", None)
        assert self.ROUND_ID_offset is not None

        self.embedding_lookup_stream = None
    
    '''
    fluent llm inner use
    '''
    def init_buffer(self, model_config, model, model_runner):
        self.input_buffer_tensor_map = {}
        for cache_init_info in self.input_cache_init_info_list:
            if cache_init_info.name.startswith("__system_"):
                continue
            tensor = torch.empty((self.buffer_max_seq_len, cache_init_info.dim), dtype=cache_init_info.dtype, device=self.run_device)
            self.input_buffer_tensor_map[cache_init_info.name] = tensor
        self.model_config = model_config
        self.model = model
        self.model_runner = model_runner

        self.tp_group = get_tp_group()
        self.tp_rank = torch.distributed.get_rank(group=self.tp_group.device_group)
        self.tp_size = torch.distributed.get_world_size(group=self.tp_group.device_group)

        self.attn_tp_group = get_attention_tp_group()
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()

        self.embedding_lookup_stream = torch.get_device_module(self.run_device).Stream()
        # logger.info(f"read_from_request_cache attn_tp_rank:{attn_tp_rank} {attn_tp_size} {tp_rank} {tp_size}")
        return self

    def get_input_ids_and_paged_hash_ids(self, input_cache_idx, max_new_tokens, prefill_len):
        # assert self.server_args.nnodes == 1, f"not support nnodes > 1, need to broadcast"
        # if prefill_len is None:
        #     prefill_len = self.get_global_state("input", input_cache_idx, "TOTAL_STEP")
        end_page = (prefill_len + max_new_tokens + self.page_size - 1) // self.page_size + 1
        all_input_ids = self.input_cache.read_use_index_range(["__system_input_ids"], input_cache_idx, 0, prefill_len)["__system_input_ids"].view(-1)
        # logger.info(f"append_input: {step} {extend_step} {all_input_ids.shape} {all_input_ids}")
        paged_hash_ids = self.input_cache.read_use_index_range(["__system_paged_hash_ids"], input_cache_idx, 0, end_page)["__system_paged_hash_ids"].view(-1)

            
        return all_input_ids.tolist(), paged_hash_ids.tolist()
    
    def patch_req_info(self, recv_req):
        parse_res = parse_rid(recv_req.rid)
        if parse_res is None:
            logger.info(f"[WARNING][{recv_req.rid}] read_from_request_cache rid error")
            return
        session_id, cache_idx, output_cache_idx, round_id, prefill_len, req_id = parse_res
        # return output_cache_idx * self.output_max_cache_size
        input_ids, paged_hash_ids = self.get_input_ids_and_paged_hash_ids(cache_idx, recv_req.sampling_params.max_new_tokens, prefill_len=prefill_len)
        # logger.info(f"patch_req_info input_ids:{input_ids} paged_hash_ids:{paged_hash_ids}")
        recv_req.input_ids = input_ids
        recv_req.input_extra_infos[0]["paged_hash_ids"] = paged_hash_ids

    def embedding_lookup_build_rid(self, round_id, cache_idx, prefill_len):
        return build_rid(session_id=0, round_id=round_id, input_cache_idx=0, output_cache_idx=cache_idx, prefill_len=prefill_len, request_id="mock")

    def embedding_lookup(self, rid, input_ids_list, kwargs={}):
        if not hasattr(self.model, "embedding_lookup"):
            logger.info(f"[WARNING][{rid}] No embedding_lookup")
            return None
        assert EMBEDDING_CACHE_KEY in self.tmp_cache
        assert EMBEDDING_SUB_KEY in self.tmp_cache[EMBEDDING_CACHE_KEY].cache_init_info_map
        parse_res = parse_rid(rid)
        if parse_res is None:
            logger.info(f"[WARNING][{rid}] embedding_lookup rid error")
            return None
        session_id, cache_idx, output_cache_idx, round_id, prefill_len, req_id = parse_res
        # logger.info(f"embedding_lookup input_ids:{input_ids} kwargs:{kwargs}")
        with torch.get_device_module(self.run_device).stream(self.embedding_lookup_stream):
            try:
                emb = self.model.embedding_lookup(input_ids_list, kwargs=kwargs, device=self.run_device)
                tensor_list = []
                tensor_len = []
                if isinstance(emb, torch.Tensor):
                    assert emb.shape[0] == len(input_ids_list), f"emb.shape[0] {emb.shape[0]} != len(input_ids_list) {len(input_ids_list)}"
                    for idx, input_ids in enumerate(input_ids_list):
                        assert emb[idx].shape[0] >= len(input_ids), f"emb[idx].shape[0] {emb[idx].shape[0]} < len(input_ids) {len(input_ids)}"
                        tensor_list.append(emb[idx, :len(input_ids)])
                        tensor_len.append(len(input_ids))
                else:
                    assert isinstance(emb, list)
                    for idx, input_ids in enumerate(input_ids_list):
                        assert emb[idx].shape[0] == len(input_ids), f"emb[idx].shape[0] {emb[idx].shape[0]} == len(input_ids) {len(input_ids)}"
                        tensor_list.append(emb[idx][:len(input_ids)])
                        tensor_len.append(len(input_ids))
                assert sum(tensor_len) <= self.tmp_cache_ptr_infos[EMBEDDING_CACHE_KEY][1], f"sum(tensor_len) {sum(tensor_len)} > self.tmp_cache_ptr_infos[{EMBEDDING_CACHE_KEY}][1] {self.tmp_cache_ptr_infos[EMBEDDING_CACHE_KEY][1]}"
                if self.attn_tp_rank == 0:
                    cache_start = output_cache_idx * self.tmp_cache_ptr_infos[EMBEDDING_CACHE_KEY][1]
                    for tensor, input_ids_len in zip(tensor_list, tensor_len):
                        self.tmp_cache[EMBEDDING_CACHE_KEY].write([(cache_start, cache_start + input_ids_len)], [(0, input_ids_len)], {EMBEDDING_SUB_KEY: tensor})
                        cache_start += input_ids_len
                    # self.tmp_cache[EMBEDDING_CACHE_KEY].write([(cache_start, cache_start + prefill_len)], [(0, prefill_len)], {EMBEDDING_SUB_KEY: emb})
                sync_tensor = torch.zeros((1), device=self.run_device)
                broadcast_wrapper(sync_tensor, 0, self.attn_tp_group.device_group)
                return {
                    "tensor_len": tensor_len,
                }
            except Exception as e:
                logger.error(f"embedding_lookup raise error : {e}", exc_info=True)
                return None
        return None

    def read_from_request_cache_wrapper(self, forward_batch, new_output_dict:bool=False):
        if forward_batch.forward_mode.is_decode():
            self.read_from_request_cache(
                forward_batch,
                forward_batch.batch_size,
                [seq_len - 1 for seq_len in forward_batch.seq_lens_cpu.tolist()],
                [1] * forward_batch.batch_size, 
                forward_batch.reqs,
                new_output_dict,
            )
        elif forward_batch.forward_mode.is_extend():
            self.read_from_request_cache(
                forward_batch,
                forward_batch.batch_size, 
                forward_batch.extend_prefix_lens_cpu,
                forward_batch.extend_seq_lens_cpu,
                forward_batch.reqs,
                new_output_dict,
            )
        elif forward_batch.forward_mode.is_idle():
            # total_len = len(forward_batch.positions)
            # assert forward_batch.batch_size == total_len
            self.read_from_request_cache(
                forward_batch,
                forward_batch.batch_size, 
                [0] * forward_batch.batch_size,
                [1] * forward_batch.batch_size, 
                [None] * forward_batch.batch_size,
                new_output_dict,
            )

    def init_request_cache_capture(self, forward_batch, extend_lens):
        extend_lens_sum = sum(extend_lens)
        ret_tensor_dict = {}
        for key, value in self.input_buffer_tensor_map.items():
            part_value = value[:extend_lens_sum]
            ret_tensor_dict[key] = part_value
        # logger.info(f"read_from_request_cache: ret_tensor_dict:{ret_tensor_dict}")
        setattr(forward_batch, f"request_cache_input", ret_tensor_dict)
        return ret_tensor_dict

    def init_request_cache_replay(self, forward_batch, bs, prefix_lens, extend_lens, reqs):
        return self.read_from_request_cache(forward_batch, bs, prefix_lens, extend_lens, reqs)

    def read_from_request_cache(self, forward_batch, bs, prefix_lens, extend_lens, reqs, new_output_dict:bool=False):
        rids = [req.rid if req is not None else "" for req in reqs]
        # logger.info(f"read_from_request_cache bs:{bs}, prefix_lens:{prefix_lens}, extend_lens:{extend_lens}, rids:{rids}")
        output_prefix = 0
        cache_offset = []
        output_offset = []
        cache_offset2 = []
        output_offset2 = []
        # global_state = [None for _ in range(bs)]
        assert bs == len(prefix_lens), f"ERROR {bs} != {len(prefix_lens)}"
        assert bs == len(extend_lens), f"ERROR {bs} != {len(extend_lens)}"
        assert bs == len(rids), f"ERROR {bs} != {len(rids)}"
        extend_lens_sum = sum(extend_lens)
        for idx, (req, prefix_len, extend_len) in enumerate(zip(reqs, prefix_lens, extend_lens)):
            output_prefix += extend_len
            if req is None:
                continue
            rid = req.rid
            parse_res = parse_rid(rid)
            if parse_res is None:
                if rid != "":
                    logger.info(f"[WARNING][{rid}] read_from_request_cache rid error")
                continue
            session_id, input_cache_idx, output_cache_idx, round_id, prefill_len, req_id = parse_res
            if ENABLE_GPU_OUTPUT_CACHE:
                if "output_cache_info" not in req.output_extra_info:
                    output_session_id = self.get_int_session_id(f"{req_id}_{round_id}")
                    req.output_extra_info["output_cache_info"] = {
                        "mooncake_session_id": self.transfer_engine.get_session_id(),
                        "dst_ptrs": self.output_cache_ptrs,
                        "dst_index": self.output_cache.get(output_session_id)._cache_idx,
                    }
                output_cache_idx = req.output_extra_info["output_cache_info"]["dst_index"]
            
            input_extend_len = 0
            output_extend_len = 0
            input_prefix_len = 0
            output_prefix_len = 0
            if prefix_len < prefill_len:
                input_prefix_len = prefix_len
                if prefix_len + extend_len > prefill_len:
                    input_extend_len = prefill_len - prefix_len
                    output_prefix_len = prefill_len
                    output_extend_len = prefix_len + extend_len - prefill_len
                else:
                    input_extend_len = extend_len
                    output_prefix_len = 0
                    output_extend_len = 0
            else:
                output_extend_len = extend_len
                output_prefix_len = prefix_len

            if input_extend_len > 0:
                # assert prefix_len + extend_len <= prefill_len, f"[ERROR][{rid}] prefix_len:{prefix_len} + extend_len:{extend_len} > prefill_len:{prefill_len}"
                # cache_start = self.input_cache.get_cache_offset("__system_input_ids", input_cache_idx, prefix_len)
                cache_start = input_cache_idx * self.max_seq_len + input_prefix_len
                # logger.info(f"read_from_request_cache prefill [{rid}] prefix_len:{prefix_len}, prefill_len:{prefill_len}, extend_len:{extend_len}, cache_start:{cache_start}")
                cache_offset.append((cache_start, cache_start + input_extend_len))
                # Fix: 扣除 output_extend_len，防止在跨越 input/output 边界时位置重叠
                output_offset.append((output_prefix - input_extend_len - output_extend_len, output_prefix - output_extend_len))
            if output_extend_len > 0:
                cache_start = output_cache_idx * self.max_output_len + output_prefix_len-prefill_len
                cache_offset2.extend(list(range(cache_start, cache_start + output_extend_len)))
                output_offset2.extend(list(range(output_prefix - output_extend_len, output_prefix)))

            # logger.info(f"read_from_request_cache cache_offset:{len(cache_offset)} output_offset:{len(output_offset)} cache_offset2:{cache_offset2} output_offset2:{output_offset2} ")
            # {self.output_ids_tmp_tensor.shape if self.output_ids_tmp_tensor is not None else None} {self.output_embs_tmp_tensor.shape if self.output_embs_tmp_tensor is not None else None} {self.buffer_tmp_tensor_index_map}

        # tensor_dict = {}

        # cuda graph decode input要和prefill read cache分开，否则会有乱码
        if new_output_dict:
            input_new_output_dict = {}
            for cache_init_info in self.input_cache_init_info_list:
                if cache_init_info.name.startswith("__system_"):
                    continue
                tensor = torch.empty((self.buffer_max_seq_len, cache_init_info.dim), dtype=cache_init_info.dtype, device=self.run_device)
                input_new_output_dict[cache_init_info.name] = tensor
        else:
            input_new_output_dict = self.input_buffer_tensor_map
        if self.attn_tp_rank == 0 or new_output_dict:
            self.input_cache.read(self.input_keys, cache_offset, self.run_device, out_dict=input_new_output_dict, output_offset=output_offset)
        self.output_cache.read(self.input_keys, cache_offset2, self.run_device, out_dict=input_new_output_dict, output_offset=output_offset2)
        # for key, value in self.input_buffer_tensor_map.items():
        #     if key in tensor_dict:
        #         value[output_offset] = tensor_dict[key]
        #     if key in tensor_dict2:
        #         value[output_offset2] = tensor_dict2[key]
        
        ret_tensor_dict = {}
        for key, value in input_new_output_dict.items():
            part_value = value[:extend_lens_sum]
            if self.attn_tp_size > 1 and len(cache_offset) > 0 and not new_output_dict:
                broadcast_wrapper(part_value, 0, self.attn_tp_group.device_group)
            ret_tensor_dict[key] = part_value
        # logger.info(f"{prefix_lens=} {extend_lens=} ret_tensor_dict:{ret_tensor_dict}")
        setattr(forward_batch, f"request_cache_input", ret_tensor_dict)
        return ret_tensor_dict

    def resolve_future_input_cache(self, model_worker_batch):
        reqs = model_worker_batch.reqs
        ret_tensor_dict = {}
        bs = len(model_worker_batch.input_ids)
        for name, _ in self.input_cache_init_info_map.items():
            cache0 = reqs[0].output_cache_dict[name]
            value = torch.zeros((bs, cache0.shape[1]), dtype=cache0.dtype, device=cache0.device)
            start_idx = 0
            for req in reqs:
                value[start_idx:start_idx+1].copy_(req.output_cache_dict[name])
                start_idx += 1
            ret_tensor_dict[name] = value
        setattr(model_worker_batch, f"request_cache_input", ret_tensor_dict)
        return ret_tensor_dict

    def save_output_cache(self, reqs, output_tensor_dict):
        assert output_tensor_dict is not None, f"output_tensor_dict is None"

        for key, value in output_tensor_dict.items():
            start_idx = 0
            for req in reqs:
                req.output_cache_dict[key] = value[start_idx:start_idx+1]
                start_idx += 1

    def write_to_request_cache_overlap(self, forward_batch, ids, output_tensor_dict):
        batch_size = forward_batch.batch_size
        self.write_to_request_cache(
                batch_size,
                forward_batch.seq_lens_cpu.view(batch_size).tolist(),
                [1] * batch_size, 
                forward_batch.reqs,
                ids,
                output_tensor_dict,
                False,
            )
    def write_to_request_cache_sample(self, forward_batch, batch_size, next_token_ids, output_tensor_dict, enable_overlap=False):
        if not enable_overlap:
            next_token_ids, output_tensor_dict = self.write_to_request_cache(
                    batch_size,
                    forward_batch.seq_lens_cpu.view(batch_size).tolist(),
                    [1] * batch_size, 
                    forward_batch.reqs,
                    next_token_ids,
                    output_tensor_dict,
                )
        return next_token_ids, output_tensor_dict

    def read_from_request_cache_overlap(self, model_worker_batch):
        tmp_forward_batch = self.init_new_for_preprocess(model_worker_batch)
        self.read_from_request_cache_wrapper(tmp_forward_batch, True)
        extend_input_dict = tmp_forward_batch.request_cache_input
        setattr(model_worker_batch, f"request_cache_input", extend_input_dict)

    def write_to_request_cache(self, bs, prefix_lens, extend_lens, reqs, output_ids, output_tensor_dict, abort_req:bool=True):

        # logger.info(f"{self.attn_tp_rank=} before finish_req_flag")
        if abort_req:
            if self.attn_tp_size > 1:
                finish_req_flag = torch.zeros(bs, dtype=torch.uint8, device=self.run_device)
            else:
                finish_req_flag = torch.zeros(bs, dtype=torch.uint8, device='cpu')
        # logger.info(f"{self.attn_tp_rank=}  after finish_req_flag")
        
        output_prefix = 0
        offset_map = {}
        # cache_offset = []
        # output_offset = []
        # finish_update_fn_list = []
        # buffer_tmp_tensor_index_map = {}
        
        assert bs == len(prefix_lens), f"ERROR {bs} != {len(prefix_lens)}"
        assert bs == len(extend_lens), f"ERROR {bs} != {len(extend_lens)}"
        assert bs == len(reqs), f"ERROR {bs} != {len(reqs)}"
        # prefix_lens = copy.deepcopy(prefix_lens)
        # extend_lens = copy.deepcopy(extend_lens)
        # logger.info(f"{self.attn_tp_rank=}  before process req")
        for idx, (req, prefix_len, extend_len) in enumerate(zip(reqs, prefix_lens, extend_lens)):
            output_prefix += extend_len
            if req.is_chunked > 0:
                continue
            rid = req.rid
            parse_res = parse_rid(rid)
            if parse_res is None:
                if rid != "":
                    logger.info(f"[WARNING][{rid}] read_from_request_cache rid error")
                continue
            session_id, input_cache_idx, output_cache_idx, round_id, prefill_len, req_id = parse_res
            if ENABLE_GPU_OUTPUT_CACHE:
                output_cache_idx = req.output_extra_info["output_cache_info"]["dst_index"]
            if self.attn_tp_rank == 0 and abort_req:
                cur_session_id = self.get_global_state("input", input_cache_idx, "SESSION_ID")
                cur_round_id = self.get_global_state("input", input_cache_idx, "ROUND_ID")
                if session_id != cur_session_id:
                    err_msg = f"[WARNING][{rid}] write_to_request_cache cache_session_id error: {cur_session_id} {session_id}"
                    logger.info(err_msg)
                    finish_req_flag[idx] = 1
                    # continue
                elif round_id != cur_round_id:
                    err_msg = f"[WARNING][{rid}] write_to_request_cache round_id error: {cur_round_id} {round_id}"
                    logger.info(err_msg)
                    finish_req_flag[idx] = 2
                    # continue
            # decode_step = self.get_global_state("output", output_cache_idx, "TOTAL_STEP")
            # if decode_step != prefix_len - prefill_len:
            #     logger.info(f"[ERROR][{rid}] write_to_request_cache decode_step error: {decode_step} {prefix_len} {extend_len} {prefill_len}")
            #     continue
            # finish_update_fn_list.append((output_cache_idx, extend_len, prefix_len, prefill_len))
            # update_fn()
            # new_total_step = cache_info.get_global_state("TOTAL_STEP")
            cache_start = output_cache_idx * self.max_output_len + prefix_len-prefill_len
            # cache_start = self.output_cache.get_cache_offset("__system_input_ids", output_cache_idx, prefix_len-prefill_len)
            # logger.info(f"write_to_request_cache [{rid}] prefill_len:{prefill_len}, prefix_len:{prefix_len}, extend_len:{extend_len}, decode_step:{decode_step}, cache_start:{cache_start}")
            # cache_offset.extend(list(range(cache_start, cache_start + extend_len)))
            # output_offset.extend(list(range(output_prefix - extend_len, output_prefix)))
            # cache_offset.append((cache_start, cache_start + extend_len))
            # output_offset.append((output_prefix - extend_len, output_prefix))
            if session_id not in offset_map or offset_map[session_id][0] <= round_id:
                if session_id in offset_map:
                    if self.attn_tp_rank == 0 and abort_req:
                        finish_req_flag[offset_map[session_id][1]] = 2
                    logger.warning(f"same session id {rid} run in one batch {offset_map[session_id][0]} -> {round_id}")
                offset_map[session_id] = (round_id, idx, (cache_start, cache_start + extend_len), (output_prefix - extend_len, output_prefix))

        cache_offset = []
        output_offset = []
        for k, v in offset_map.items():
            cache_offset.append(v[2])
            output_offset.append(v[3])
        output_tensor_dict["__system_input_ids"] = output_ids
        self.output_cache.write(cache_offset, output_offset, output_tensor_dict)
            # for (output_cache_idx, extend_len, prefix_len, prefill_len) in finish_update_fn_list:
            #     self.set_global_state("output", output_cache_idx, "TOTAL_STEP", prefix_len + extend_len - prefill_len)
        
        # logger.info(f"{self.attn_tp_rank=}  before broadcast_wrapper")
        if self.attn_tp_size > 1 and abort_req:
            finish_req_flag = finish_req_flag.cuda()
            broadcast_wrapper(finish_req_flag, 0, self.attn_tp_group.device_group)
            finish_req_flag = finish_req_flag.tolist()
        # logger.info(f"{self.attn_tp_rank=}  before broadcast_wrapper")
        # need special prefix [REQUEST_CACHE_ERROR]

        if abort_req:
            for idx, req in enumerate(reqs):
                if finish_req_flag[idx] == 1:
                    err_msg = f"[REQUEST_CACHE_ERROR] session changed"
                    # req.finished_reason = FINISH_ABORT(message=f"{err_msg}")
                    req.to_abort = True
                    req.to_abort_message = err_msg
                if finish_req_flag[idx] == 2:
                    err_msg = f"[REQUEST_CACHE_ERROR] round changed"
                    # req.finished_reason = FINISH_ABORT(message=f"{err_msg}")
                    req.to_abort = True
                    req.to_abort_message = err_msg
        return output_ids, output_tensor_dict
    
    def forward_postprocess_for_pd_decode(self, req: Req, output_id, hidden_states):
        '''
        1,
        [len(decode_req.req.origin_input_ids)],
        [1], 
        [decode_req.req],
        logits_and_hidden_states,
        '''
        schedule_batch = ScheduleBatch(
            reqs=[req],
            device=self.run_device,
        )
        seq_len = len(req.origin_input_ids)
        input_ids = torch.tensor([req.origin_input_ids[-1]], device=self.run_device)
        seq_lens = torch.tensor([seq_len], device=self.run_device)
        seq_lens_cpu = seq_lens.cpu()
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=1,
            input_ids=input_ids,
            req_pool_indices=None,
            sampling_info=SamplingBatchInfo.from_schedule_batch(
                schedule_batch,
                self.model_config.vocab_size,
            ),
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            extend_seq_lens=seq_lens,
            extend_seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=None,
            reqs=[req],
            return_logprob=False,
            capture_hidden_mode = CaptureHiddenMode.LAST,
        )
        # hidden_states = logits_and_hidden_states[self.model_config.hidden_size:]
        # TODO @xiaobin support last n hidden states
        hidden_states = hidden_states.to(self.run_device).view(1, -1)
        logits_output = self.model.get_logits_output(
            input_ids,
            hidden_states,
            forward_batch)
        # logits_output = LogitsProcessorOutput(
        #     next_token_logits=next_token_logits,
        #     hidden_states=hidden_states,
        # )
        return self.model_runner.sample(logits_output, forward_batch)

    '''
    mllm infer inner use
    '''
    def get_int_session_id(self, raw_session_id):
        if isinstance(raw_session_id, str):
            assert self.global_state_tensor_dtype == torch.int64
            session_id = hash_to_int(raw_session_id, "int64")
        else:
            session_id = raw_session_id
        return session_id

    def get(self, raw_session_id, get_without_create = False):
        session_id = self.get_int_session_id(raw_session_id)
        def input_cache_init_fn(input_cache, cache_idx):
            input_cache.write_element("__system_global_state", cache_idx, self.SESSION_ID_offset, session_id)
        input_cache_info = self.input_cache.get(session_id, input_cache_init_fn, get_without_create=get_without_create)
        if input_cache_info is None:
            return session_id, None
        if get_without_create:
            return session_id, input_cache_info
        with input_cache_info._lock:
            # round_id = self.get_global_state("input", input_cache_info._cache_idx, "ROUND_ID")
            # output_session_id = self.get_int_session_id(f"{raw_session_id}_{round_id}")
            # def output_cache_init_fn(input_cache, cache_idx):
            #     input_cache.write_element("__system_global_state", cache_idx, self.SESSION_ID_offset, session_id)
            #     input_cache.write_element("__system_global_state", cache_idx, self.ROUND_ID_offset, round_id)
            output_cache_info = self.output_cache.get(raw_session_id, None)
            input_cache_info.req_cache = self
            input_cache_info.input_cache_idx = input_cache_info._cache_idx
            input_cache_info.output_cache_idx = output_cache_info._cache_idx
            #input_cache_info.output_cache_idx_map[round_id] = (output_cache_info._cache_idx, output_session_id)
        return session_id, input_cache_info
    
    # def delete(self, raw_session_id, cache_info, free_input=False, free_output=False, cur_round_id=None):
    #     session_id = self.get_int_session_id(raw_session_id)
    #     # round_id = self.get_global_state("input", cache_info.input_cache_idx, "ROUND_ID")
    #     if free_input:
    #         self.input_cache.delete(session_id)
    #     running_rids = []
    #     if free_output:
    #         left_running_rids_dict = {}
    #         for round_id_i, rids in cache_info.running_rids_dict.items():
    #             if cur_round_id is None or round_id_i <= cur_round_id:
    #                 running_rids.extend(rids)
    #             else:
    #                 left_running_rids_dict[round_id_i] = rids
    #         cache_info.running_rids_dict = left_running_rids_dict
    #         left_output_cache_idx_map = {}
    #         for round_id_i, (output_cache_idx, output_session_id) in cache_info.output_cache_idx_map.items():
    #             if cur_round_id is None or round_id_i <= cur_round_id:
    #                 self.output_cache.delete(output_session_id)
    #             else:
    #                 left_output_cache_idx_map[round_id_i] = (output_cache_idx, output_session_id)
    #         cache_info.output_cache_idx_map = left_output_cache_idx_map
    #     return running_rids
    
    def get_input_buf_idx(self, rid):
        parse_res = parse_rid(rid)
        if parse_res is None:
            logger.info(f"[WARNING][{rid}] read_from_request_cache rid error")
            return 0
        session_id, cache_idx, output_cache_idx, round_id, prefill_len, req_id = parse_res
        # return output_cache_idx * self.output_max_cache_size
        return cache_idx * self.max_seq_len
    
    def get_input_ids(self, input_cache_idx, start=0, end=-1):
        all_input_ids = self.input_cache.read_use_index_range(["__system_input_ids"], input_cache_idx, start, end)["__system_input_ids"].view(-1)
        return all_input_ids.tolist()

    def copy_output_to_input(self, resp_info, output_cache_idx, input_cache_idx, output_start_idx, input_start_idx, extend_len, output_ids, clone=True, timeout=0.5):
        if ENABLE_GPU_OUTPUT_CACHE:
            self.output_cache_copy_from_gpu(resp_info, output_cache_idx, output_start_idx, output_start_idx+extend_len, timeout=timeout)
        input_keys = self.input_keys + ["__system_input_ids"]
        server_output_keys = self.server_output_keys + ["__system_input_ids"]
        output_tensor_dict = self.output_cache.read_use_index_range(list(set(input_keys + server_output_keys)), output_cache_idx, output_start_idx, output_start_idx+extend_len)
        # output_tensor_dict["__system_input_ids"] = torch.tensor(output_ids).view(extend_len, 1)
        input_tensor_dict = {}
        ret_output_tensor_dict = {}
        # logger.info(f"copy_output_to_input {output_tensor_dict.keys()} {server_output_keys}")
        for key, value in output_tensor_dict.items():
            if key in input_keys:
                input_tensor_dict[key] = value
            if key in server_output_keys:
                ret_output_tensor_dict[key] = value.clone() if clone else value
        self.input_cache.write_use_index_range(input_cache_idx, input_start_idx, input_start_idx+extend_len, input_tensor_dict)
        self.update_global_state("input", input_cache_idx, "TOTAL_STEP", lambda x: x + extend_len)

        return ret_output_tensor_dict

    def gen_hash_input_ids(self, input_cache_idx, input_ids, vocab_size=-1, hash_base_str="", dtype="uint32"):
        # cur_session_id = self.get_global_state("input", input_cache_idx, "SESSION_ID")
        # cur_round_id = self.get_global_state("input", input_cache_idx, "ROUND_ID")
        ret_input_ids = []
        for id in input_ids:
            hash_id = hash_to_int(f"{hash_base_str}_{id}", dtype) #_{cur_session_id}_{cur_round_id}
            if vocab_size > 0:
                hash_id = hash_id % vocab_size
            ret_input_ids.append(hash_id)
        return ret_input_ids

    def append_input(self, input_cache_idx, input_token_ids, max_new_tokens, input_tensor_dict, kv_paged_base_str=None, step_check=None):
        step = self.get_global_state("input", input_cache_idx, "TOTAL_STEP")
        if step_check is not None:
            if step_check < 0:
                # logger.info(f"append_input skip step_check({step_check}) < 0, skip, step:{step}")
                pass
            elif step_check > step:
                logger.warning(f"append_input error step_check > step:{step_check} {step}")
                step_check = step
                # return None
            elif step_check < step:
                logger.info(f"step_check < step:{step_check} {step}")
                step = step_check
            # elif step_check != step:
            #     logger.info(f"append_input error step_check != step:{step_check} {step}")
            #     step = step_check
        # logger.info(f"append_input input_token_ids:{input_token_ids}, input_ids_list:{input_ids_list}, input_embs_list:{input_embs_list}")
        if isinstance(input_token_ids, list):
            extend_step = len(input_token_ids)
        # elif isinstance(input_token_ids, torch.Tensor):
        #     extend_step = input_token_ids.shape[0]
        else:
            # assert False, f"not support input_token_ids:{input_token_ids}"
            extend_step = input_token_ids.shape[0]
        for input_ids in input_tensor_dict.values():
            assert extend_step == input_ids.shape[0], f"shape not equal {extend_step} {input_ids.shape}"
        
        input_token_ids = torch.tensor(input_token_ids, dtype=torch.int32, device=self.cache_device).view(extend_step, 1)
        input_tensor_dict["__system_input_ids"] = input_token_ids
        total_step = step+extend_step
        self.input_cache.write_use_index_range(input_cache_idx, step, total_step, input_tensor_dict)
        start_page = step // self.page_size
        end_page = (total_step + max_new_tokens + self.page_size - 1) // self.page_size + 1
        kv_paged_hash_id = self.gen_hash_input_ids(input_cache_idx, list(range(start_page, end_page)), hash_base_str=kv_paged_base_str, dtype="int64")
        assert len(kv_paged_hash_id) == end_page - start_page, f"kv_paged_hash_id len({len(kv_paged_hash_id)}):{kv_paged_hash_id}, end_page:{end_page}, start_page:{start_page}"
        # logger.info(f"append_input start_page:{start_page}, end_page:{end_page}, kv_paged_hash_id:{kv_paged_hash_id}")
        kv_paged_hash_id = torch.tensor(kv_paged_hash_id, dtype=torch.int64).view(-1, 1)
        self.input_cache.write_use_index_range(input_cache_idx, start_page, end_page, {"__system_paged_hash_ids": kv_paged_hash_id})
        
        self.set_global_state("input", input_cache_idx, "TOTAL_STEP", total_step)
        return total_step, step_check
    
    def update_round_info(self, cache_idx, round_id, hash_id=None, input_tokens=None, output_tokens=None, back_tokens=None, step_check=None, step_check_round=None):
        round_info_tensor = self.input_cache.get_tensor("__system_round_info")
        assert round_info_tensor is not None
        round_info_tensor[cache_idx, round_id, 0] = round_id
        if hash_id is not None:
            # 写入新 round 时，清空当前 round 的旧数据，并清空所有后续 round，防止匹配到失效的旧缓存
            round_info_tensor[cache_idx, round_id, 1:] = 0
            round_info_tensor[cache_idx, round_id, 1] = hash_id
            if round_id + 1 < round_info_tensor.shape[1]:
                round_info_tensor[cache_idx, round_id + 1:, 1:] = 0
        if input_tokens is not None:
            round_info_tensor[cache_idx, round_id, 2] = input_tokens
        if output_tokens is not None:
            round_info_tensor[cache_idx, round_id, 3] = output_tokens
        if back_tokens is not None:
            round_info_tensor[cache_idx, round_id, 4] = back_tokens
        if step_check is not None and step_check >= 0:
            if step_check_round is None:
                step_check_round = round_id
            cur_sum = 0
            for input_output_size in round_info_tensor[cache_idx, : step_check_round + 1]:
                sum_input_output_size = input_output_size[2] + input_output_size[3] + input_output_size[4]
                if sum_input_output_size == 0:
                    continue
                cur_sum += sum_input_output_size
                if step_check < cur_sum:
                    reduce_size = cur_sum - step_check
                    cur_sum -=reduce_size
                    input_output_size[4] = -reduce_size

    def get_round_info(self, cache_idx, end_round_id, start_round_id=0):
        round_info_tensor = self.input_cache.get_tensor("__system_round_info")
        assert round_info_tensor is not None
        assert round_info_tensor is not None
        round_info = round_info_tensor[cache_idx, start_round_id:end_round_id]#.view(-1)
        # return [(input_output_size[0], hash_id, input_output_size[1], input_output_size[2], input_output_size[3]) for hash_id, input_output_size in zip(round_info_hash_tensor.tolist(), round_info_input_output_size_tensor.tolist())]
        return round_info.tolist()

    def set_global_state(self, cache_type:str, cache_idx:int, state_name: str, value: int) -> int:
        global_state_tensor = None
        if cache_type == "input":
            global_state_tensor = self.input_cache.get_tensor("__system_global_state")
        elif cache_type == "output":
            global_state_tensor = self.output_cache.get_tensor("__system_global_state")
        if global_state_tensor is None:
            raise RuntimeError(f"global_state_tensor get error {cache_type}")
        if state_name not in self.global_state_dict:
            raise ValueError(f"未知的位名称: {state_name}")
        bit_position = self.global_state_dict[state_name]
        pre_value = global_state_tensor[cache_idx, bit_position]
        global_state_tensor[cache_idx, bit_position] = value
        return pre_value.item()

    def get_global_state(self, cache_type:str, cache_idx, state_name: str, check=True) -> int:
        global_state_tensor = None
        if cache_type == "input":
            global_state_tensor = self.input_cache.get_tensor("__system_global_state")
        elif cache_type == "output":
            global_state_tensor = self.output_cache.get_tensor("__system_global_state")
        if global_state_tensor is None:
            raise RuntimeError(f"global_state_tensor get error {cache_type}")
        if state_name not in self.global_state_dict:
            raise ValueError(f"未知的位名称: {state_name}")
        bit_position = self.global_state_dict[state_name]
        pre_value = global_state_tensor[cache_idx, bit_position]
        return pre_value.item()
    
    def update_global_state(self, cache_type:str, cache_idx, state_name: str, update_fn: Callable) -> int:
        global_state_tensor = None
        if cache_type == "input":
            global_state_tensor = self.input_cache.get_tensor("__system_global_state")
        elif cache_type == "output":
            global_state_tensor = self.output_cache.get_tensor("__system_global_state")
        if global_state_tensor is None:
            raise RuntimeError(f"global_state_tensor get error {cache_type}")
        if state_name not in self.global_state_dict:
            raise ValueError(f"未知的位名称: {state_name}")
        bit_position = self.global_state_dict[state_name]
        pre_value = global_state_tensor[cache_idx, bit_position]
        new_value = update_fn(pre_value)
        # logger.info(f"update_global_state:{state_name} {pre_value} {new_value}")
        global_state_tensor[cache_idx, bit_position] = new_value
        return pre_value.item(), new_value.item()
    
    def get_tmp_cache(self, name, session_id, no_capacity_return_none = True):
        assert name in self.tmp_cache
        input_cache_info = self.tmp_cache[name].get(session_id, no_capacity_return_none=no_capacity_return_none)
        if input_cache_info is None:
            logger.error(f"get_tmp_cache for {name} capacity full")
            return None, None
        tmp_cache_tensor = self.tmp_cache[name].get_tensor_with_row(input_cache_info._cache_idx)
        assert len(tmp_cache_tensor) > 0
        tmp_cache_max_len = self.tmp_cache_ptr_infos[name][1]
        return tmp_cache_tensor, input_cache_info._cache_idx, tmp_cache_max_len
    
    def delete_tmp_cache(self, name, session_id):
        assert name in self.tmp_cache
        self.tmp_cache[name].delete(session_id)


    def pd_input_copy(self, 
                    mooncake_session_id,
                    dst_ptrs, dst_index, 
                    prefill_len, prefill_index, timeout,
                    prefill_start_idx = 0):

        self.cache_copy(mooncake_session_id, 
            self.input_cache_names, self.input_cache_item_lens, 
            self.input_cache_ptrs, prefill_index,
            dst_ptrs, dst_index, 
            prefill_len, timeout,
            seq_start_idx = prefill_start_idx
        )

    def output_cache_copy_from_gpu(self, 
                    resp_info,
                    host_index,
                    seq_start_idx,
                    seq_end_idx,
                    timeout,):
        output_cache_info = resp_info["output_extra_info"]["output_cache_info"]
        mooncake_session_id = output_cache_info["mooncake_session_id"]
        dst_ptrs = output_cache_info["dst_ptrs"]
        dst_index = output_cache_info["dst_index"] * self.max_output_len
        self.cache_copy(mooncake_session_id, 
            self.output_cache_names, self.output_cache_item_lens, 
            self.output_cache_ptrs, host_index * self.max_output_len,
            dst_ptrs, dst_index, 
            seq_end_idx, timeout,
            seq_start_idx = seq_start_idx
        )

    def gen_item_len_by_name(self, key, input_len, extend_len):
        if key == "__system_paged_hash_ids":
            return input_len // self.page_size, (input_len + extend_len + self.page_size - 1) // self.page_size + 1
        else:
            return input_len, input_len+extend_len

    def _transfer_task_wrapper(self, mooncake_session_id: str, buffer: int, peer_buffer_address: int, length: int):
        start_time = time.time()
        try:
            # before_at = time.time()
            # logger.info(f"log_transfer_sync_read before {mooncake_session_id=} {length=} {before_at=}")
            result = self.transfer_engine.transfer_sync_read(mooncake_session_id, buffer, peer_buffer_address, length)
            # after_at = time.time()
            # logger.info(f"log_transfer_sync_read after {mooncake_session_id=} {length=} {after_at=} {after_at-before_at}")
            error = None
        except Exception as e:
            result = None
            error = e
        end_time = time.time()
        return result, error, start_time, end_time

    def cache_copy(self, 
                    mooncake_session_id,
                    names, item_lens, 
                    source_ptrs, source_indx,
                    dst_ptrs, dst_index, 
                    seq_end_idx, timeout,
                    seq_start_idx = 0):
        futures = []
        enqueue_time = time.time()
        for idx, (request_cache_name, request_cache_data_ptr, request_cache_item_len) in enumerate(zip(
            names, source_ptrs, item_lens
        )):
            start_idx, end_idx = self.gen_item_len_by_name(request_cache_name, seq_start_idx, seq_end_idx - seq_start_idx)
            item_len = (end_idx - start_idx) * request_cache_item_len
            prefill_addr = request_cache_data_ptr + (source_indx + start_idx) * request_cache_item_len
            decode_addr = dst_ptrs[idx] + (dst_index + start_idx) * request_cache_item_len
            logger.info(f"cache_copy: {request_cache_name} {mooncake_session_id} {request_cache_item_len}*{end_idx - start_idx}={item_len}  {decode_addr}-{dst_ptrs[idx]}:{dst_index}*{request_cache_item_len} ====> {prefill_addr}-{request_cache_data_ptr}:{source_indx}*{request_cache_item_len}")
            future = self.cache_copy_exector.submit(
                self._transfer_task_wrapper,
                mooncake_session_id, prefill_addr, decode_addr, item_len
            )
            futures.append(future)
        
        # ✓ 优化：使用wait()替代忙轮询，避免浪费CPU资源
        remaining_timeout = timeout - (time.time() - enqueue_time)
        if remaining_timeout <= 0:
            logger.warning(f"[Prefill] Timeout before waiting for futures")
            raise RuntimeError("wait input cache copy timeout")
        
        try:
            done, not_done = concurrent.futures.wait(
                futures,
                timeout=remaining_timeout,
                return_when=concurrent.futures.ALL_COMPLETED
            )
            
            if not_done:
                logger.warning(f"[Prefill] Input cache copy timeout: {len(not_done)}/{len(futures)} tasks not completed")
                raise RuntimeError("wait input cache copy timeout")
            
            total_queue_time = 0
            total_exec_time = 0
            max_queue_time = 0
            max_exec_time = 0

            # 检查所有任务的结果
            for future in done:
                try:
                    result, error, start_time, end_time = future.result()
                    if error:
                        raise error
                    
                    q_time = start_time - enqueue_time
                    e_time = end_time - start_time
                    total_queue_time += q_time
                    total_exec_time += e_time
                    max_queue_time = max(max_queue_time, q_time)
                    max_exec_time = max(max_exec_time, e_time)

                    if result != 0:
                        err_msg = f"get_request_input error {result}"
                        logger.error(err_msg)
                        raise RuntimeError(err_msg)
                except Exception as e:
                    logger.error(f"[Prefill] Exception getting result: {e}", exc_info=True)
                    raise RuntimeError(f"Error: {str(e)}")
            
            if len(done) > 0:
                logger.info(f"Cache copy stats: tasks={len(done)}, "
                            f"avg_queue={total_queue_time/len(done)*1000:.2f}ms, "
                            f"max_queue={max_queue_time*1000:.2f}ms, "
                            f"avg_exec={total_exec_time/len(done)*1000:.2f}ms, "
                            f"max_exec={max_exec_time*1000:.2f}ms")

        except Exception as e:
            logger.error(f"[Prefill] Error waiting for futures: {e}", exc_info=True)
            raise RuntimeError(f"Error: {str(e)}")

    def init_new_for_preprocess(self, batch):

        ret = ForwardBatch(
            forward_mode=batch.forward_mode,
            batch_size=len(batch.seq_lens),
            seq_lens_cpu=batch.seq_lens_cpu,
            reqs=batch.reqs,
            input_ids=None,
            req_pool_indices=None,
            seq_lens=None,
            seq_lens_sum=None,
        )

        # Init position information
        if not ret.forward_mode.is_decode():
            ret.extend_prefix_lens_cpu = batch.extend_prefix_lens
            ret.extend_seq_lens_cpu = batch.extend_seq_lens

        return ret
    

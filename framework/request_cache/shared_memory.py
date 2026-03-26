import copy
import math
from typing import List, Optional
import torch
import dataclasses
from collections import OrderedDict
import threading
from multiprocessing import shared_memory

from sglang.srt.utils import (
    get_colorful_logger,
)
from framework.request_cache.request_cache_info import RequestCacheInfo
from utils.timeout_lock import TimeoutRLock
logger = get_colorful_logger(__name__)

@dataclasses.dataclass
class RequestCacheInitInfo:
    name: str
    dtype: torch.dtype
    lenth: int
    dim: int
    tensor: Optional[torch.Tensor] = None

class SharedMemoryHandler:
    def __init__(self,
            max_size: int,
            cache_init_info_list: List[RequestCacheInitInfo],
            device: str,
        ):
        self.max_cache_size = max_size
        self.device = device
        self._shm = None
        self._shm_create = False

        self.cache_init_info_list = cache_init_info_list
        self.cache_init_info_key_list = []
        self.cache_init_info_map = {}
        for cache_init_info in self.cache_init_info_list:
            assert cache_init_info.dim > 0 and cache_init_info.lenth > 0
            self.cache_init_info_key_list.append(cache_init_info.name)
            self.cache_init_info_map[cache_init_info.name] = copy.deepcopy(cache_init_info)

        # 使用 OrderedDict 同时管理缓存映射和 LRU 顺序
        self.cache = OrderedDict()  # 存储 key -> cache_idx 的映射，按访问顺序排序
        self.available_indices = list(range(self.max_cache_size))  # 可用的缓存索引
        self._lock = TimeoutRLock("SharedMemoryHandler")
    
    def __del__(self):
        if self._shm is not None:
            self._shm.close()
            if self._shm_create:
                self._shm.unlink()
    
    def create_tensor_by_share_memory(self, shared_memory_name, create=False):
        total_buffer_size = 0
        tensor_size_list = []
        for key in self.cache_init_info_key_list:
            cache_init_info = self.cache_init_info_map[key]
            shape_and_byte_size = (self.max_cache_size, cache_init_info.lenth, cache_init_info.dim) 
            byte_size = torch.tensor([1], dtype=cache_init_info.dtype, device=self.device).element_size()
            size = math.prod(shape_and_byte_size) * byte_size
            tensor_size_list.append((total_buffer_size, size, shape_and_byte_size, byte_size))
            logger.info(f"SharedMemory {key} tensor_size: {size/1024/1024} M")
            total_buffer_size += size

        shm = None
        if self.device == "cpu":
            shm = shared_memory.SharedMemory(
                name=shared_memory_name,
                create=create,
                size=total_buffer_size
            )
            for key, tensor_size in zip(self.cache_init_info_key_list, tensor_size_list):
                cache_init_info = self.cache_init_info_map[key]
                (offset, size, shape_and_byte_size, byte_size) = tensor_size
                cache_init_info.tensor = torch.frombuffer(
                    shm.buf[offset:offset + size],
                    dtype=cache_init_info.dtype,
                ).view(shape_and_byte_size)
        else:
            for key, tensor_size in zip(self.cache_init_info_key_list, tensor_size_list):
                cache_init_info = self.cache_init_info_map[key]
                (offset, size, shape_and_byte_size, byte_size) = tensor_size
                cache_init_info.tensor = torch.zeros(shape_and_byte_size, dtype=cache_init_info.dtype, device=self.device)
                # logger.info(f"create_tensor_by_share_memory gpu tensor {key} {self.device}: {cache_init_info.tensor}")

        self._shm = shm
        self._shm_create = create

        return self

    def get(self, session_id, init_fn=None, get_without_create=False, no_capacity_return_none=False):
        with self._lock:
            if session_id in self.cache:
                cache_init_info = self.cache.pop(session_id)
                self.cache[session_id] = cache_init_info
                # logger.info(f"request_cache get from cache session_id:{session_id}, cache_idx:{cache_init_info._cache_idx}")
                return cache_init_info
            elif get_without_create:
                return None

            if len(self.available_indices) == 0:
                if no_capacity_return_none:
                    return None
                oldest_session_id, cache_init_info_to_delete = self.cache.popitem(last=False)
                with cache_init_info_to_delete._lock:
                    delete_cache_idx = cache_init_info_to_delete._cache_idx
                    cache_init_info_to_delete.clear()
                self.clear(delete_cache_idx, oldest_session_id)

            cache_idx = self.available_indices.pop(0)
            cache_init_info = RequestCacheInfo(cache_idx)
            self.cache[session_id] = cache_init_info
            if init_fn is not None:
                init_fn(self, cache_idx)
            # logger.info(f"request_cache get session_id:{session_id}, cache_idx:{cache_idx}")
            return cache_init_info

    def clear(self, cache_idx, session_id):
        # logger.info(f"request_cache clear session_id:{session_id}, cache_idx:{cache_idx}")
        for key in self.cache_init_info_key_list:
            cache_init_info = self.cache_init_info_map[key]
            cache_init_info.tensor[cache_idx].zero_()
        self.available_indices.append(cache_idx)

    def delete(self, session_id):
        with self._lock:
            if session_id in self.cache:
                cache_init_info = self.cache.pop(session_id)
                self.clear(cache_init_info._cache_idx, session_id)
            else:
                logger.warning(f"{self.max_cache_size=} {session_id=}")

    def get_tensor_with_row(self, cache_idx):
        ret_tensor_dict = {}
        for key, cache_init_info in self.cache_init_info_map.items():
            ret_tensor_dict[key] = cache_init_info.tensor[cache_idx]
        return ret_tensor_dict

    def get_tensor_buf_infos(self, include_keys=None, exclude_keys=None, exclude_keys_prefix=None):
        names = []
        ptrs = []
        data_lens = []
        row_data_lens = []
        item_lens = []
        for key in self.cache_init_info_key_list:
            if include_keys is not None:
                if key not in include_keys:
                    continue
            if exclude_keys is not None:
                if key in exclude_keys:
                    continue
            if exclude_keys_prefix is not None:
                for prefix in exclude_keys_prefix:
                    if key.startswith(prefix):
                        continue
            cache_init_info = self.cache_init_info_map[key]
            extend_tensor = cache_init_info.tensor
            # logger.info(f"get_tensor_buf_infos: {extend_tensor.data_ptr()}, {extend_tensor.nbytes} {extend_tensor[0][0].nbytes} {extend_tensor[-1]}")
            names.append(key)
            ptrs.append(extend_tensor.data_ptr())
            data_lens.append(extend_tensor.nbytes)
            row_data_lens.append(extend_tensor[0].nbytes)
            item_lens.append(extend_tensor[0][0].nbytes)
        return names, ptrs, data_lens, row_data_lens, item_lens
    
    def get_tensor(self, key):
        assert key in self.cache_init_info_key_list
        cache_init_info = self.cache_init_info_map[key]
        return cache_init_info.tensor
    
    def write_element(self, key, index, offset, value):
        assert key in self.cache_init_info_key_list
        cache_init_info = self.cache_init_info_map[key]
        cache_init_info.tensor[index, offset] = value


    def write(self, cache_offset, input_offset, tensor_dict):
        if len(cache_offset) == 0:
            return
        
        for key, input_tensor in tensor_dict.items():
            assert key in self.cache_init_info_key_list
            cache_init_info = self.cache_init_info_map[key]
            tensor = cache_init_info.tensor.view(self.max_cache_size * cache_init_info.lenth, cache_init_info.dim)
            # logger.info(f"xxxxx write {key} {self.device}: {tensor} {input_tensor[input_offset]}")
            # if tensors_slice is not None and key in tensors_slice:
            #     cur_tensor_slice = tensors_slice[key]
            #     tensor[cache_offset, cur_tensor_slice[0]:cur_tensor_slice[1]] = input_tensor[input_offset].to(self.device)
            # else:
            if isinstance(cache_offset[0], int):
                tensor[cache_offset] = input_tensor[input_offset].to(self.device)
            else:
                for cache_offset_part, input_offset_part in zip(cache_offset, input_offset):
                    tensor[cache_offset_part[0]:cache_offset_part[1]] = input_tensor[input_offset_part[0]:input_offset_part[1]].to(self.device)
        
    def read(self, keys, cache_offset, to_device, out_dict=None, output_offset=None):
        ret_tensor_dict = {}
        if len(cache_offset) == 0:
            return ret_tensor_dict
        
        for key in keys:
            assert key in self.cache_init_info_key_list
            cache_init_info = self.cache_init_info_map[key]
            tensor = cache_init_info.tensor.view(self.max_cache_size * cache_init_info.lenth, cache_init_info.dim)

            if isinstance(cache_offset[0], int):
                if out_dict is None or output_offset is None:
                    ret_tensor_dict[key] = tensor[cache_offset].to(to_device)
                else:
                    out_dict[key][output_offset] = tensor[cache_offset].to(to_device)
            else:
                assert out_dict is not None and output_offset is not None
                for cache_offset_part, output_offset_part in zip(cache_offset, output_offset):
                    out_dict[key][output_offset_part[0]:output_offset_part[1]] = tensor[cache_offset_part[0]:cache_offset_part[1]].to(to_device)

            # assert cache_init_info.dim == tensor.shape[1], f"dim error {cache_init_info.name} {cache_init_info.dim} {tensor.shape[1]}"

        return ret_tensor_dict

    def read_use_index_range(self, keys, cache_idx, range_start, range_end):
        ret_tensor_dict = {}
        
        for key in keys:
            assert key in self.cache_init_info_key_list
            cache_init_info = self.cache_init_info_map[key]
            assert range_start < range_end, f"range_start:{range_start} >= range_end:{range_end}"
            assert range_end <= cache_init_info.lenth, f"range_end:{range_end} >= cache_init_info.lenth:{cache_init_info.lenth}"
            tensor = cache_init_info.tensor[cache_idx, range_start:range_end]
            # if tensors_slice is not None and key in tensors_slice:
            #     cur_tensor_slice = tensors_slice[key]
            #     tensor = tensor[:, cur_tensor_slice[0]:cur_tensor_slice[1]]
            # tensor = tensor.to(to_device)
            ret_tensor_dict[key] = tensor

        return ret_tensor_dict

    def write_use_index_range(self, cache_idx, range_start, range_end, tensor_dict):
        for key in tensor_dict.keys():
            assert key in self.cache_init_info_key_list
            cache_init_info = self.cache_init_info_map[key]
            assert range_start < range_end, f"range_start:{range_start} >= range_end:{range_end}"
            assert range_end <= cache_init_info.lenth, f"range_end:{range_end} >= cache_init_info.lenth:{cache_init_info.lenth}"
            assert cache_init_info.dim == tensor_dict[key].shape[1], f"dim error {cache_init_info.name} {cache_init_info.dim} {tensor_dict[key].shape[1]}"
            cache_init_info.tensor[cache_idx, range_start:range_end] = tensor_dict[key]
            # logger.info(f"write_use_index_range: {key} {range_start}:{range_end} {tensor_dict[key]}")
    
    # def get_cache_offset(self, key, cache_idx, offset):
    #     assert key in self.cache_init_info_key_list
    #     cache_init_info = self.cache_init_info_map[key]
    #     assert offset <= cache_init_info.lenth
    #     return cache_idx * cache_init_info.lenth + offset


from typing import Callable, Dict, List, Optional, Tuple
import re
import torch
import ctypes
import hashlib

from sglang.srt.utils import (
    get_colorful_logger,
)

logger = get_colorful_logger(__name__)

def hash_to_int(s, dtype):
    """
    使用 MD5 哈希函数生成确定性的 uint32 值
    确保在不同进程/机器上返回相同的结果
    
    Args:
        s: 输入字符串或对象
        
    Returns:
        uint32: 确定性哈希值
    """
    if not isinstance(s, str):
        s = str(s)
    # 使用 MD5 生成确定性哈希
    hash_bytes = hashlib.md5(s.encode()).digest()
    
    if dtype == "int32":
        hash_value = int.from_bytes(hash_bytes[:4], byteorder='big', signed=False)
        return ctypes.c_int32(hash_value).value
    elif dtype == "uint32":
        hash_value = int.from_bytes(hash_bytes[:4], byteorder='big', signed=False)
        return ctypes.c_uint32(hash_value).value
    elif dtype == "int64":
        hash_value = int.from_bytes(hash_bytes[:8], byteorder='big', signed=False)
        return ctypes.c_int64(hash_value).value
    elif dtype == "uint64":
        hash_value = int.from_bytes(hash_bytes[:8], byteorder='big', signed=False)
        return ctypes.c_uint64(hash_value).value
    else:
        raise NotImplementedError()


RID_PATTERN = re.compile(r'^SESSION::(-?\d+)::IN_CACHE_IDX::(\d+)::OUT_CACHE_IDX::(\d+)::ROUND::(\d+)::PRFILL::(\d+)::RID::([^:]+)$')
def parse_rid(rid: str) -> Optional[Tuple[int, int, int, int, int, str]]:
    match = RID_PATTERN.match(rid)
    if match:
        session_id, input_cache_idx, output_cache_idx, round_id, prefill_len, request_id = match.groups()
        return int(session_id), int(input_cache_idx), int(output_cache_idx), int(round_id), int(prefill_len), request_id
    return None

def build_rid(session_id: int, round_id: int, input_cache_idx: int, output_cache_idx: int, prefill_len:int, request_id: str) -> str:
    return f"SESSION::{session_id}::IN_CACHE_IDX::{input_cache_idx}::OUT_CACHE_IDX::{output_cache_idx}::ROUND::{round_id}::PRFILL::{prefill_len}::RID::{request_id}"


def dtype_map(dtype_str):
    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint16": torch.uint16,
        "uint32": torch.uint32,
        "uint64": torch.uint64,
        # "bool": torch.bool,
    }
    if dtype_str in dtype_mapping:
        return dtype_mapping[dtype_str]
    else:
        raise NotImplemented(f"not support {dtype_str}")

def parse_cache_name_dtype_dim(request_cache_name_dtype_dim):
    request_cache_name_dtype_dim_new = []
    for name_dtype_dim_str in request_cache_name_dtype_dim:
        name_dtype_dim_list = name_dtype_dim_str.split(",")
        assert len(name_dtype_dim_list) == 3
        name = name_dtype_dim_list[0]
        dtype = dtype_map(name_dtype_dim_list[1])
        dim = int(name_dtype_dim_list[2])
        cache_name_dtype_dim = (name, dtype, dim)
        request_cache_name_dtype_dim_new.append(cache_name_dtype_dim)
    return request_cache_name_dtype_dim_new

def parse_tmp_cache_schema_string(schema_str):
    """
    更好的解析方法，使用正则表达式处理未加引号的字符串
    
    参数:
        schema_str: 要解析的字符串
    
    返回:
        元组 (name, cache_size, length, [(sub_name,dtype,dim)])
    """
    try:
        print(f"schema_str:{schema_str}")
        
        # 方法1: 使用正则表达式解析
        # 匹配模式: name,cache_size,length,[(sub_name1,dtype1,dim1),(sub_name2,dtype2,dim2),...]
        pattern = r'^([^,]+),\s*([^,]+),\s*([^,]+),\s*\[(.*)\]$'
        match = re.match(pattern, schema_str.strip())
        
        if not match:
            raise ValueError("字符串格式不正确")
        
        name = match.group(1).strip()
        cache_size = int(match.group(2).strip())
        length = int(match.group(3).strip())
        list_content = match.group(4).strip()
        
        print(f"解析出的部分: name={name}, cache_size={cache_size}, length={length}, list_content={list_content}")
        
        # 解析列表部分 - 处理元组
        parsed_tuples = []
        if list_content:
            # 使用正则表达式匹配元组: (item1,item2,item3)
            tuple_pattern = r'\(([^,]+),\s*([^,]+),\s*([^,]+)\)'
            tuple_matches = re.findall(tuple_pattern, list_content)
            
            for tuple_match in tuple_matches:
                sub_name = tuple_match[0].strip()
                dtype = tuple_match[1].strip()
                dim = int(tuple_match[2].strip())
                parsed_tuples.append((sub_name, dtype, dim))
        
        return (name, cache_size, length, parsed_tuples)
        
    except Exception as e:
        raise ValueError(f"解析错误: {e}")



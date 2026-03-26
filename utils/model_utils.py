import json
from collections import defaultdict
from safetensors import safe_open
import inspect
import random


import torch
import os
import glob
from safetensors.torch import load_file
from typing import Optional, Type, Union, Dict, Any
from transformers import PretrainedConfig, PreTrainedModel

def load_model_from_pretrained(
    model_path: str,
    model_class: Type[PreTrainedModel],
    device: str = "cuda",
    dtype: Optional[Union[str, torch.dtype]] = torch.bfloat16,
    strict: bool = False,
    config_kwargs: Optional[Dict[str, Any]] = None,
    model_prefix: str = '',
    **kwargs
) -> PreTrainedModel:
    """
    Load Hugging Face pretrained model from model path and load weights from safetensors files or pytorch_model.bin file.
    Priority: safetensors files > pytorch_model.bin
    
    Args:
        model_path: Model directory path
        model_class: Model class, e.g., OmniModel
        device: Device, default is "cuda"
        dtype: Data type, default is torch.bfloat16
        strict: Whether to strictly match parameter names, default is False (skip non-existent parameters)
        config_kwargs: Additional parameters passed to model configuration
        model_prefix: Prefix to remove from state_dict keys
        **kwargs: Additional parameters passed to model_class.from_pretrained
        
    Returns:
        Model instance with loaded weights
    """
    # Set default config parameters
    if config_kwargs is None:
        config_kwargs = {}
    
    # Try to load config
    try:
        config = PretrainedConfig.from_pretrained(model_path, **config_kwargs)
        print("Config object type:", type(config))
        # print("Config content:", config)
    except Exception as e:
        print(f"Warning: Failed to load config from {model_path}, will use default config: {e}")
        config = None
    
    # Create model instance
    try:
        if config is None or not hasattr(model_class, 'config_class') or not hasattr(model_class.config_class, 'from_pretrained'):
            model = model_class.from_pretrained(model_path, state_dict=None, **kwargs)
        else:
            # Try using model-specific config_class
            print(f"Using model-specific config class: {model_class.config_class.__name__}")
            model_specific_config = model_class.config_class.from_pretrained(model_path)
            model = model_class(config=model_specific_config, **kwargs)

    except Exception as e:
        print(f"Warning: Error creating model instance: {e}")
        print("Trying to instantiate model class directly...")
        model = model_class()
    
    # Move model to specified device and data type
    model = model.to(device)
    if isinstance(dtype, str):
        if dtype.lower() == "bfloat16" or dtype.lower() == "bf16":
            dtype = torch.bfloat16
        elif dtype.lower() == "float16" or dtype.lower() == "fp16":
            dtype = torch.float16
        elif dtype.lower() == "float32" or dtype.lower() == "fp32":
            dtype = torch.float32
    model = model.to(dtype)
    
    # First try to load from safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    state_dict = None
    
    if safetensors_files:
        print(f"Found {len(safetensors_files)} safetensors files: {safetensors_files}")
        try:
            # Merge all safetensors files into one state_dict
            state_dict = {}
            for file_path in safetensors_files:
                print(f"Loading weights from safetensors file: {file_path}")
                file_state_dict = load_file(file_path, device=device)
                state_dict.update(file_state_dict)
            print(f"Successfully loaded weights from safetensors files")
        except Exception as e:
            print(f"Warning: Error loading safetensors files: {e}")
            state_dict = None
    
    # If no safetensors files or loading failed, try to load from pytorch_model.bin
    if state_dict is None:
        bin_file_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(bin_file_path):
            print(f"Found weights file: {bin_file_path}")
            try:
                state_dict = torch.load(bin_file_path, map_location=device)
                print(f"Successfully loaded weights from pytorch_model.bin")
            except Exception as e:
                print(f"Warning: Error loading {bin_file_path}: {e}")
                raise RuntimeError(f"Failed to load weights from {bin_file_path}") from e
        else:
            raise FileNotFoundError(f"No weight files found in {model_path} directory")
    
    # Print state dict keys for debugging
    original_keys = set(state_dict.keys())
    # print(f"state dict: {', '.join(sorted(original_keys))}")
    original_len = len(state_dict)
    
    # Remove model_prefix prefix from model parameter file
    if len(model_prefix):
        state_dict = {k.replace(model_prefix + '.', ''): v for k, v in state_dict.items()}
    else:
        print(f"base_model_prefix: {model.base_model_prefix}")
        if model.base_model_prefix:
            state_dict = {k.replace(model.base_model_prefix + '.', ''): v for k, v in state_dict.items()}
    
    # If not strict matching, filter parameters that exist in the model
    if not strict:
        model_keys = set(model.state_dict().keys())
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
        filtered_len = len(filtered_state_dict)
        if filtered_len < original_len:
            print(f"Filtered out {original_len - filtered_len} parameters that don't exist in the model")
    
    # Load model with weights
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        print(f"model parameter load, missing_keys: {', '.join(sorted(missing_keys))}\n unexpected_keys: {', '.join(sorted(unexpected_keys))}")
    except Exception as e:
        raise RuntimeError(f"Error loading weights: {e}")
    
    return model

def load_model_state_dict(model_path: str, model_prefix: str = "", device: str="cuda"):
    # First try to load from safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    state_dict = None
    
    if safetensors_files:
        print(f"Found {len(safetensors_files)} safetensors files: {safetensors_files}")
        try:
            # Merge all safetensors files into one state_dict
            state_dict = {}
            for file_path in safetensors_files:
                print(f"Loading weights from safetensors file: {file_path}")
                file_state_dict = load_file(file_path, device=device)
                state_dict.update(file_state_dict)
            print(f"Successfully loaded weights from safetensors files")
        except Exception as e:
            print(f"Warning: Error loading safetensors files: {e}")
            state_dict = None
    
    # If no safetensors files or loading failed, try to load from pytorch_model.bin
    if state_dict is None:
        bin_file_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(bin_file_path):
            print(f"Found weights file: {bin_file_path}")
            try:
                state_dict = torch.load(bin_file_path, map_location=device)
                print(f"Successfully loaded weights from pytorch_model.bin")
            except Exception as e:
                print(f"Warning: Error loading {bin_file_path}: {e}")
                raise RuntimeError(f"Failed to load weights from {bin_file_path}") from e
        else:
            raise FileNotFoundError(f"No weight files found in {model_path} directory")
    
    if model_prefix:
        state_dict = {k.replace(model_prefix + '.', ''): v for k, v in state_dict.items()}

    return state_dict

def load_weights_from_safetensors_helper(model_path: str, key_words_list: list, device="cpu"):
    """
    从safetensors文件中加载权重，并根据关键词列表分配到不同的模型。

    :param model_path: 模型文件路径
    :param key_words_list: 关键词列表，用于筛选权重
    :param device: 张量加载的设备
    :return: 分配好的状态字典列表
    """
    # 加载 index.json 文件
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
       safetensors_path = os.path.join(model_path, "model.safetensors")
       state_dicts = [defaultdict(dict) for _ in key_words_list]
       with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        # 遍历所有张量键
        for key in f.keys():
            # 检查键是否以关键词列表中的某个词为前缀
            for i, key_word in enumerate(key_words_list):
                if key.startswith(key_word):
                    # 提取张量并存储到状态字典中
                    new_key = key[len(key_word):]
                    state_dicts[i][new_key] = f.get_tensor(key).to(device)
        return state_dicts
    
    try:
        with open(index_path, 'r') as f:
            index_data = json.load(f)
    except FileNotFoundError:
        raise Exception(f"Index file not found at {index_path}")

    weight_map = index_data.get("weight_map", {})
    
    # 创建一个字典，以文件名为键，存储相关的键列表
    file_to_keys = defaultdict(list)
    for key, file_name in weight_map.items():
        for key_words in key_words_list:
            if key.startswith(key_words):
                file_to_keys[file_name].append(key)

    # 准备状态字典
    state_dicts = [defaultdict(dict) for _ in key_words_list]

    # 遍历每个文件，一次性加载所有相关键的权重
    for file_name, keys in file_to_keys.items():
        safetensors_path = os.path.join(model_path, file_name)
        try:
            with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                for key in keys:
                    # 找到与键对应的模型
                    for i, key_words in enumerate(key_words_list):
                        if key.startswith(key_words):
                            # 去掉前缀
                            new_key = key[len(key_words):]
                            state_dicts[i][new_key] = f.get_tensor(key).to(device)
        except Exception as e:
            raise Exception(f"Error loading weights from {safetensors_path}: {e}")

    return state_dicts
import re

def generate_split_prompts(text):
    prompts = []
    cur_prompt = ""
    for _text in split_string_with_punctuation_merged(text):
        cur_prompt += _text
        if len(cur_prompt) > 10:
            prompts.append(cur_prompt)
            cur_prompt = ""
    if cur_prompt:
        prompts.append(cur_prompt)
    return prompts
def split_string_with_punctuation_merged(s):
    # 正则表达式匹配任意标点符号
    pattern = r'([:,;!?，。；：！？])'
    
    # 查找所有标点符号的位置
    punctuation_positions = [(m.start(0), m.group(0)) for m in re.finditer(pattern, s)]
    
    # 根据标点符号的位置分割字符串，并合并标点到前一个子字符串
    substrings = []
    last_index = 0
    for pos, punct in punctuation_positions:
        # 添加标点前的子字符串和标点本身
        substrings.append(s[last_index:pos] + punct)
        last_index = pos + len(punct)
    # 添加最后一个标点之后的子字符串（如果有的话）
    if last_index < len(s):
        substrings.append(s[last_index:])
    
    return substrings

def probe():
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename
    line = frame.f_lineno
    magic_num = hash(f"{filename}_{line}") % int(1e4) / 1e4
    x = torch.randn(10, 10, device="cuda") / magic_num
    y = torch.randn(10, 10, device="cuda") / magic_num
    z = torch.mm(x, y)
    torch.cuda.synchronize()
    print(f"============ {magic_num=}, {z.flatten()[0]:.0f} probe ok. {filename}:{line} ============",flush=True)
    return True
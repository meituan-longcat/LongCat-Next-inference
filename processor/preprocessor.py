import os

import torch
import numpy as np
from PIL import Image
from typing import Optional, List, Union
from transformers import AutoConfig, AutoTokenizer

from utils.model_utils import load_weights_from_safetensors_helper
from processor.flash_omni.modeling_longcat_oe import LongcatModel, LongcatAudioTokenizer
from processor.flash_omni.processor_omni import OmniMMProcessor, OmniProcessorOutput

import random



def seed_everything(seed: int) -> int:
    print(f"\033[31m[============ Seed Everything {seed=} ============]\033[0m")
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
class PreProcessor:
    def __init__(self, configs, rank=0, oe_delegate_fn=None):
        torch.cuda.set_device(rank)
        self.device = "cuda"
        MODEL_PATH = os.environ.get("NMM_INFER_MODEL_ROOT")
        model_path = configs.get("model-path", MODEL_PATH)
        if model_path=="${NMM_INFER_MODEL_ROOT}":
            model_path = MODEL_PATH
        print("model_path",model_path)
        self.model_path = model_path
        image_model_path = configs.get("image-model-path", model_path)
        config_path = configs.get("config-path", os.path.join(model_path, "nmm_infer"))
        print(f"\033[32m[============ PreProcessor Set {rank=}, {config_path=} ============]\033[0m")
        
        
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # 初始化模型
        self.model = LongcatModel(config=config, oe_delegate_fn=oe_delegate_fn)

        # 初始化 audio tokenizer
        self.audio_tokenizer = LongcatAudioTokenizer(config)
        audio_weights_path = configs.get("audio-tokenizer-path", model_path)
        state_dicts = load_weights_from_safetensors_helper(audio_weights_path, ["model.audio_tokenizer."], "cpu")
        self.audio_tokenizer_path = audio_weights_path.strip().rstrip("/")
        self.audio_tokenizer.load_state_dict(state_dicts[0], strict=True)
        self.audio_tokenizer.to("cuda").to(torch.bfloat16)
        
        self.model.load_weights_from_safetensors(model_path, image_model_path, "cpu")

        self.model.to("cuda").to(torch.bfloat16)
        self.model.eval()
        self.processor = OmniMMProcessor(self.tokenizer, config, training=False, relative_path='')
        self.omni_processor_output_cls = OmniProcessorOutput
    
    
    def process_from_raw_input_new(self, json_config):
        question = json_config["question"]
        multimodal_input = json_config.get("prompt", question) # 不设置prompt时直接把question进入processor，image，asr都是这种形式
        print("multimodal_input",multimodal_input)
        processed = self.processor([multimodal_input])
        task = json_config.get("task","image")
        print(f"{task=}")
        
        # 准备输入
        if task == "image":
            input_ids = processed.input_ids.to(device=self.device)
            # torch.save(input_ids, "input_ids_ckpt_18988.pt")
            attention_mask = processed.attention_mask.to(device=self.device)
            images = [img.to(device=self.device) for img in processed.images] if processed.images is not None else None
            if hasattr(processed, 'patch_nums') and processed.patch_nums is not None:
                patch_nums = processed.patch_nums.to(device=self.device) if hasattr(processed.patch_nums, 'to') else processed.patch_nums
            else:
                patch_nums = None
                
            # 处理images_grid
            if hasattr(processed, 'images_grid') and processed.images_grid is not None:
                images_grid = processed.images_grid
            else:
                images_grid = None
            embedding = self.model.forward(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                images=images,
                                                patch_nums=patch_nums,
                                                images_grid=images_grid)
        elif task == "audio":
                
            input_ids=processed.input_ids.cuda()
            audios = processed.audios.cuda() if processed.audios is not None else None
            audiotext_ids = processed.audiotext_ids.cuda() if processed.audiotext_ids is not None else None
            encoder_length = processed.encoder_length.cuda() if processed.encoder_length is not None else None
            bridge_length = processed.bridge_length.cuda() if processed.bridge_length is not None else None
            audio_tokens = None
            if audios is None or len(audios) == 0:
                audio_tokens = None
                print("audio_tokens is None")
            else:
                audio_tokens = self.audio_tokenizer.forward(audios, encoder_length=encoder_length, bridge_length=bridge_length)
                
                audio_tokens = audio_tokens.cuda()
            embedding = self.model.forward(input_ids=input_ids,
                            audiotext_ids=audiotext_ids,
                            audios_tokens=audio_tokens)
         
        return embedding, input_ids

# from mllminfer.src.core.utils.singleton import Singleton
import os
from transformers import AutoConfig
class Singleton:

    _instances = {}

    @classmethod
    def init(cls, name, initializer):
        if name in cls._instances:
            raise ValueError(f"Singleton '{name}' already exists")
        inst = initializer()
        cls._instances[name] = inst
        print(f"\033[32m[Singleton init {name} for {os.getpid()}]\033[0m")
        return cls._instances[name]

    @classmethod
    def has(cls, name):
        return name in cls._instances

    @classmethod
    def get(cls, name):
        if name not in cls._instances:
            raise ValueError(f"Singleton '{name}' not found")
        return cls._instances[name]

    @classmethod
    def delete(cls, name):
        if name in cls._instances:
            del cls._instances[name]

class SpecialTokens:
    def __init__(self, hf_path):
        print(f"{hf_path=}")
        try:
            config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
        except Exception as e:
            print(f"加载配置失败: {e}")
            raise

        visual_config = getattr(config, "visual_config", None)
        audio_config = getattr(config, "audio_config", None)
        if visual_config is None or audio_config is None:
            print("注意！可能没有成功从config里加载SpecialTokens")
        try:
            self.IMAGE_START_TOKEN_ID = visual_config.image_start_token_id # 131106
            self.IMAGE_END_TOKEN_ID = visual_config.image_end_token_id # 131107
            self.IMAGE_PAD_TOKEN_ID = visual_config.image_pad_token_id # 131108
            self.IMAGE_NEWLINE_TOKEN_ID = visual_config.image_line_token_id # 131109
            self.IMAGE_TOKEN_SIZE_START_TOKEN_ID = 131090
            self.IMAGE_TOKEN_SIZE_END_TOKEN_ID = 131091
            self.AUDIO_PAD_TOKEN_ID = audio_config.audio_pad_token_id
            self.AUDIOTEXT_START_TOKEN_ID = audio_config.audiotext_start_token_id
            self.AUDIOTEXT_PAD_TOKEN_ID = audio_config.audiotext_pad_token_id
            self.AUDIO_GEN_START_TOKEN_ID = audio_config.audiogen_start_token_id
            self.AUDIO_GEN_END_TOKEN_ID = audio_config.audiogen_end_token_id
            self.EOS_ID = config.eos_token_id
            self.MULTIMODAL_SPECIAL_TOKEN_LIST = getattr(config, "multimodal_special_token_list", [])
            self.AUDIO_END_FLAG_ID = audio_config.vq_config.codebook_sizes[0]
        except AttributeError as e:
            print(f"配置文件中缺少关键字段: {e}")
            raise


SPT_KEY = "longcat_o_moe3b_spt"


def init_spt(hf_path: str) -> SpecialTokens:
    spec = SpecialTokens(hf_path=hf_path)
    Singleton.init(SPT_KEY, lambda: spec)


def get_spt() -> SpecialTokens:
    return Singleton.get(SPT_KEY)

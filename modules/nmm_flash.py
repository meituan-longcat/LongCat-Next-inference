from typing import Optional, Callable, Iterable, Tuple

import torch
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.models.longcat_flash import (
    FLASHForCausalLM, FLASHConfig
)
from modules.nmm_sample import NmmSample
from sglang.srt.utils import get_colorful_logger
logger = get_colorful_logger(__name__)

class NmmFlashForCausalLM(FLASHForCausalLM):
    def __init__(
        self,
        config: FLASHConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__(config, quant_config=quant_config)
        self.sampler = NmmSample(config=config, language_model=self)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """
        重写 load_weights 方法，过滤掉 audio_output_layer 和 audio_embed 的参数。
        这些参数已经在 __init__ 中通过 load_state_dict 加载，不应该被父类的 load_weights 处理。
        """
         # 过滤掉 audio 相关的权重
        skip_prefixes = [
            "audio_head.",
            "model.audio_tokenizer.",
            "visual_head.",
            "model.visual_tokenizer.",
            "model.audio_embed_layers.",
        ]
        filtered_weights = []
        for name, weight in weights:
            if any(name.startswith(prefix) for prefix in skip_prefixes):
                continue
            if name in ("model.embed_tokens.weight", "lm_head.weight"):
                # HACK[zhaoxiaoyu17]: 当前词表需要HACK
                print(f"\033[33m[{name=}, {weight.shape=}]\033[0m")
                weight = weight[:131125]
            filtered_weights.append((name, weight))
        
        # 调用父类的 load_weights 方法，传入过滤后的权重
        super().load_weights(iter(filtered_weights))


    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        input_tensor_dict = getattr(forward_batch, f"request_cache_input")
        input_embeds = input_tensor_dict["input_embedding"]
        forward_batch.capture_hidden_mode = CaptureHiddenMode.LAST
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def embedding_lookup(
        self,
        input_ids_list, device, kwargs
    ):
        return self.sampler.embedding_lookup(input_ids_list, device, kwargs)

    def get_logits_output(self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )
    
    def sample(
        self,
        forward_batch: ForwardBatch,
        sample_func: Callable,
        text_logits_output: LogitsProcessorOutput,
    ):
        return self.sampler.sample(forward_batch, sample_func, text_logits_output)
    
    def capture_one_decode(
        self,
        forward_batch: ForwardBatch,
        model_runner, 
        stream
    ):
        self.sampler.capture_one_decode(forward_batch, model_runner, stream)

EntryClass = [NmmFlashForCausalLM]

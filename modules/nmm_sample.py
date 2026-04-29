from typing import Iterable, Optional, Tuple, Callable
import os
import torch
from transformers import AutoConfig
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)

from torch import Tensor
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.models.longcat_flash import (
    FLASHConfig
)
from sglang.global_config import global_config
from .context import LongcatOOverEmbContext
from .input_processor import LongcatOOverEmbInputProcessor
from .output_processor import LongcatOOverEmbOutputProcessor


class NmmSample():
    def __init__(
        self,
        config: FLASHConfig,
        language_model,
    ) -> None:
        config_dict = config.onmi_extra_info
        self.context = LongcatOOverEmbContext(base_lm = language_model, config_dict=config_dict["context"])
        self.input_processor = LongcatOOverEmbInputProcessor(language_model, self.context, config_dict["input_processor"])
        self.output_processor = LongcatOOverEmbOutputProcessor(language_model, self.context, config_dict["output_processor"])
        
    def embedding_lookup(
        self,
        input_ids_list, device, kwargs
    ):
        return self.input_processor.get_emb(input_ids_list)
        
        
    def sample(
        self,
        forward_batch: ForwardBatch,
        sample_func: Callable,
        text_logits_output: LogitsProcessorOutput,
    ):    
        if forward_batch.forward_mode.is_extend():
            self.input_processor.forward_extend(None, None, forward_batch, None)
        
        self.output_processor.forward(forward_batch=forward_batch, text_logits_output=text_logits_output, sample_func=sample_func)
        if self.output_processor.enable_cuda_graph and self.output_processor.replay_cuda_graph:
            self.output_processor.replay_one_config_decode(forward_batch, text_logits_output, sample_func)
        output_text_ids = forward_batch.next_token_ids.clone()
        output_multi_ids = forward_batch.temp_multi_ids.clone()
        output_embeddings = self.input_processor.forward(input_ids=output_text_ids, 
                                                         input_multi_ids=output_multi_ids,
                                                         forward_batch=forward_batch)
        return output_text_ids.reshape(forward_batch.batch_size, ), \
            {
                "output_multi_ids":output_multi_ids, 
                "input_embedding": output_embeddings,
            }

    def capture_one_decode(
        self,
        forward_batch: ForwardBatch,
        model_runner, 
        stream
    ):
        if self.output_processor.enable_cuda_graph:
            self.output_processor.capture_one_config_decode(forward_batch.batch_size, forward_batch, model_runner, stream)

import sys
import copy
import os
import torch
import inspect
import warnings
import numpy as np
import torch.nn as nn
from typing import Optional, Union, List, Callable, Tuple, Dict, Any
import torch.distributed as dist
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torchaudio 
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import (
    ModelOutput,
    GenerationConfig,
    GenerationMode,
    LogitsProcessorList,
    StoppingCriteriaList,
    GenerateOutput, 
    GenerationMixin,
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
    GenerateNonBeamOutput,
    is_deepspeed_zero3_enabled,
    # is_torchdynamo_compiling,
    NEED_SETUP_CACHE_CLASSES_MAPPING,
    QUANT_BACKEND_CLASSES_MAPPING,
    is_hqq_available,
    QuantizedCacheConfig,
    # is_quanto_available,
    DynamicCache,
    EncoderDecoderCache,
    logging
)
from tqdm import tqdm
# from transformers.generation.stopping_criteria import validate_stopping_criteria

@dataclass
class GenerationImageHeadOutput(ModelOutput):
    """
    Outputs of decoder-only generation models, when using non-beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True`):
            Returns the model cache, used to speed up decoding. Different models have a different cache format, check
            the model's documentation. Usually, a [`~cache_utils.Cache`] instance.
        visual_sequences (`torch.LongTensor` of shape `(batch_size, sequence_length,code_book_size)`):
            The generated visual sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
    """

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None
    vision_sequences: Optional[torch.LongTensor] = None
    vision_embeddings: Optional[Tuple[torch.FloatTensor]] = None


logger = logging.get_logger(__name__)
class GenerationImageHead(GenerationMixin):

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        streamer_unit: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        cfg_scale: Optional[float] = 1.0,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        finish_until_max_new_tokens = kwargs.pop("finish_until_max_new_tokens", False)
        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        self._validate_model_kwargs(model_kwargs.copy())
        self._validate_assistant(assistant_model, tokenizer, None)

        # 用于输入image的一些参数、
        model_kwargs['images']      = kwargs.pop("images", None)
        model_kwargs['patch_nums']  = kwargs.pop("patch_nums", None) 
        model_kwargs['images_grid'] = kwargs.pop("images_grid", None)

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # decoder-only models must use left-padding for batched generation.
        # if not self.config.is_encoder_decoder and not is_torchdynamo_compiling():
        if not self.config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config._pad_token_tensor is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config, model_kwargs
            )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, tokenizer)

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        use_dynamic_cache_by_default = False
        if "mamba" in self.__class__.__name__.lower():
            cache_name = "cache_params"
        else:
            cache_name = "past_key_values"
        if generation_config.cache_implementation is not None and (model_kwargs.get(cache_name) is not None):
            raise ValueError(
                f"Passing both `cache_implementation` (used to initialize certain caches) and `{cache_name}` (a "
                "Cache object) is unsupported. Please use only one of the two."
            )
        elif generation_config.cache_implementation is not None:
            if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
                if generation_config.cache_implementation == "static" and not self._supports_static_cache:
                    raise ValueError(
                        "This model does not support `cache_implementation='static'`. Please check the following "
                        "issue: https://github.com/huggingface/transformers/issues/28981"
                    )
                model_kwargs[cache_name] = self._get_cache(
                    generation_config.cache_implementation,
                    getattr(generation_config, "num_beams", 1) * batch_size,
                    generation_config.max_length,
                    model_kwargs,
                )
            elif generation_config.cache_implementation == "quantized":
                if not self._supports_quantized_cache:
                    raise ValueError(
                        "This model does not support the quantized cache. If you want your model to support quantized "
                        "cache, please open an issue."
                    )

                cache_config = (
                    generation_config.cache_config
                    if generation_config.cache_config is not None
                    else QuantizedCacheConfig()
                )
                cache_class = QUANT_BACKEND_CLASSES_MAPPING[cache_config.backend]

                '''
                if cache_config.backend == "quanto" and not is_quanto_available():
                    raise ImportError(
                        "You need to install `quanto` in order to use KV cache quantization with quanto backend. "
                        "Please install it via  with `pip install quanto`"
                    )
                elif cache_config.backend == "HQQ" and not is_hqq_available():
                    raise ImportError(
                        "You need to install `HQQ` in order to use KV cache quantization with HQQ backend. "
                        "Please install it via  with `pip install hqq`"
                    )
                '''

                model_kwargs[cache_name] = cache_class(cache_config)
        # Use DynamicCache() instance by default. This will avoid back and forth from legacy format that
        # keeps copying the cache thus using much more memory
        elif generation_config.cache_implementation is None and self._supports_default_dynamic_cache():
            past = model_kwargs.get(cache_name, None)
            requires_cross_attention_cache = (
                self.config.is_encoder_decoder or model_kwargs.get("encoder_outputs") is not None
            )
            if past is None:
                model_kwargs[cache_name] = (
                    DynamicCache()
                    if not requires_cross_attention_cache
                    else EncoderDecoderCache(DynamicCache(), DynamicCache())
                )
                use_dynamic_cache_by_default = True
            elif isinstance(past, tuple):
                model_kwargs[cache_name] = (
                    DynamicCache.from_legacy_cache(past)
                    if not requires_cross_attention_cache
                    else EncoderDecoderCache.from_legacy_cache(past)
                )
                use_dynamic_cache_by_default = True

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. determine generation mode
        generation_mode = generation_config.get_generation_mode(assistant_model)

        if (streamer is not None or streamer_unit is not None) and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        # 9. prepare stopping criteria
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
        )
        # 10. go into different generation modes
        # 只支持了sample模式
        if generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            # 11. prepare logits warper
        
            # 12. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
            
            return GenerationImageHead._sample(self,
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria= None, # 生成音频的时候不能有stopping_criteria默认在音频eos结束
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                max_new_tokens=kwargs["max_new_tokens"],
                finish_until_max_new_tokens=finish_until_max_new_tokens,
                cfg_scale=cfg_scale,
                **model_kwargs,
            )
        else:
            raise NotImplementedError

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        finish_until_max_new_tokens: bool,
        max_new_tokens: int,
        cfg_scale:float,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        assert stopping_criteria is None
        do_sample = generation_config.do_sample
        batch_size, cur_len = input_ids.shape
        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        NUM_CODEBOOKS       = len(self.config.visual_quantizer_config.codebook_sizes)
        IMAGE_PAD_TOKEN_ID  = self.config.visual_config.image_pad_token_id
        VISION_END_TOKEN_ID = self.config.visual_quantizer_config.codebook_sizes
        vision_sequences = torch.empty(batch_size,0,NUM_CODEBOOKS,device=input_ids.device,dtype=torch.long)
        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
        # print(f"Generate max len {max_length} current len {cur_len}")
        # 老版本的transformers
        # while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length) and cur_len < max_length:
        # if fix_max_max_length
        
        num_new_tokens = 0 
        pbar = tqdm()
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device,) and cur_len < max_length:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # 这里应该放modeling里prepare_inputs_for_generation中，这里先外部处理
            model_inputs['vision_tokens'] = model_kwargs.get('vision_tokens',None)
            # 对于图像编辑来说,首次输入会有这些内容
            model_inputs['images'] = model_kwargs.pop('images',None)
            model_inputs['patch_nums'] = model_kwargs.pop('patch_nums',None)
            model_inputs['images_grid'] = model_kwargs.pop('images_grid',None)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            # 从这里开始引入音频生成逻辑
            # forward pass to get next token
            # 生成单个音频token需要执行码表大小次前向传播
            # from IPython import embed; embed()

            outputs = self(**model_inputs, return_dict=True)
            next_token_ids = torch.zeros(batch_size, NUM_CODEBOOKS,dtype=torch.long).cuda()
            for i in range(NUM_CODEBOOKS):
                logits = self.image_head(outputs.vision_emb_for_infer, next_token_ids, self.model.visual_bridge_model.embedding_layers)
                
                # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
                # (the clone itself is always small)
                if i == 0:
                    next_token_logits = logits[i][:,:].clone()
                else:
                    next_token_logits = logits[i][:,:-1].clone()

                if cfg_scale != 1.0:
                    cond_logits, uncond_logits = next_token_logits.split(batch_size//2, dim=0)
                    next_token_logits = cfg_scale * (cond_logits-uncond_logits) + uncond_logits
                    # pre-process distribution
                    next_token_scores = logits_processor(vision_sequences[:batch_size//2,:,i], next_token_logits)
                else:
                    next_token_scores = logits_processor(vision_sequences[:batch_size//2,:,i], next_token_logits)

                if do_sample:
                    probs = nn.functional.softmax(next_token_scores, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_scores, dim=-1)
                
                if cfg_scale != 1.0:
                    next_tokens = next_tokens.repeat((2,))

                next_token_ids[:,i] = next_tokens
                if return_dict_in_generate and i==0: # 只记录第一个码表的分数
                    if output_scores:
                        scores += (next_token_scores,)
                    if output_logits:
                        raw_logits += (next_token_logits,)
                    if output_attentions:
                        decoder_attentions += (
                            (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                        )
                        if self.config.is_encoder_decoder:
                            cross_attentions += (outputs.cross_attentions,)

                    if output_hidden_states:
                        decoder_hidden_states += (
                            (outputs.decoder_hidden_states,)
                            if self.config.is_encoder_decoder
                            else (outputs.hidden_states,)
                        )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need
            pbar.update(1)
            if finish_until_max_new_tokens:
                unfinished_vision_sequences = num_new_tokens < max_new_tokens
            else:
                # update generated ids, model inputs, and length for next step
                unfinished_vision_sequences = ~(next_token_ids[:,0]==VISION_END_TOKEN_ID[0])
            unfinished_sequences = unfinished_sequences & unfinished_vision_sequences
            this_peer_finished = unfinished_sequences.max() == 0
            
            full_vision_pad_token = torch.full((batch_size, 1), IMAGE_PAD_TOKEN_ID, dtype=torch.long, device=input_ids.device)
            input_ids = torch.cat([input_ids,full_vision_pad_token], dim=-1)
            vision_sequences = torch.cat([vision_sequences, next_token_ids[:, None, :]], dim=1)
            
            if streamer is not None:
                streamer.put(next_token_ids.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            model_kwargs['vision_tokens'] = next_token_ids * unfinished_sequences.long().unsqueeze(-1) # 输入当前音频token,完成了的序列mask掉
            # 在这里支持音频生成position自增 TODO 放到update_model_kwargs_for_generation中
            # if model_kwargs.get('position_ids') is not None:
            #     model_kwargs['position_ids'] = model_kwargs['position_ids'] + 1
            
            cur_len += 1
            num_new_tokens += 1
            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        # 末尾都拼上eos来方便后续处理
        # vision_sequences = torch.cat([vision_sequences, torch.LongTensor(VISION_END_TOKEN_ID).to(input_ids).expand(batch_size,1,-1)], dim=1)
        
        vision_embeddings = []
        for b_idx in range(batch_size):
            vision_embeddings.append(self.model.visual_bridge_model(vision_sequences[b_idx]).unsqueeze(0))
        vision_embeddings = torch.cat(vision_embeddings, dim=0)
        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerationImageHeadOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                    vision_sequences = vision_sequences,
                    vision_embeddings = vision_embeddings,
                )
            else:
                return GenerationImageHeadOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                    vision_sequences = vision_sequences,
                    vision_embeddings = vision_embeddings,
                )
        else:
            return input_ids, vision_sequences, vision_embeddings
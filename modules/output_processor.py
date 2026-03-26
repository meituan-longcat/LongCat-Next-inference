from sglang.global_config import global_config
from .context import LongcatOOverEmbContext
from .state_machine import StateEnum, StateMachine, StateMachineInput
from sglang.srt.managers.schedule_batch import Req
from typing import Any, Dict, List, Optional, Tuple
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode, CaptureHiddenMode
from sglang.global_config import global_config
from sglang.srt.layers.dp_attention import get_attention_tp_rank
# from sglang.srt.models.extensible import capture
from .special_token import init_spt, get_spt
from utils.model_utils import load_weights_from_safetensors_helper
from .image_head import OmniImageHead, OmniAudioHead
from torch.cuda import CUDAGraph, Stream
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import  Tensor
from enum import Enum


class TaskType(Enum):
    """多模态生成任务类型"""
    IMAGE = "image"
    AUDIO = "audio"


# if torch.cuda.is_available():
#     from sgl_kernel import (
#         top_p_sampling_from_probs,
#         top_k_renorm_prob,
#         top_k_top_p_sampling_from_probs,
#         top_p_renorm_prob,
#     )


class LongcatOOverEmbOutputProcessor():
    def __init__(self, base_lm, ctx: LongcatOOverEmbContext, config_dict: Dict[str, Any]) -> None:
        self.base_lm = base_lm
        self.base_model = self.base_lm.model
        self.ctx = ctx
        self.sm_dict = self.ctx.sm_dict
        self.hidden_size = self.ctx.hidden_size
        self.codebook_sizes = self.ctx.codebook_sizes
        self.model_path = self.ctx.model_path
        if self.ctx.visual_enable:
            self.visual_bridge_model = self.ctx.visual_bridge_model
            self.image_head = OmniImageHead(
                hidden_size=self.hidden_size,
                codebook_sizes=self.codebook_sizes,
                image_head_transformer_ffn_scale=self.ctx.config.visual_config.image_head_config.image_head_transformer_ffn_scale,
                image_head_transformer_dims=self.ctx.config.visual_config.image_head_config.image_head_transformer_dims,
                image_head_transformer_layers=self.ctx.config.visual_config.image_head_config.image_head_transformer_layers,
                image_head_enable=config_dict["image_head_enable"],
            )
            image_head_state_dict = load_weights_from_safetensors_helper(self.model_path, ["visual_head."])[0]
            self.image_head.load_state_dict(image_head_state_dict, strict=True)
            self.image_head.to("cuda").to(torch.bfloat16)
        if self.ctx.audio_enable:
            self.audio_head = OmniAudioHead(
                hidden_size=self.hidden_size,
                codebook_sizes=self.ctx.audio_codebook_sizes,
                audio_head_transformer_ffn_scale=self.ctx.config.audio_config.audio_head_transformer_ffn_scale,
                audio_head_transformer_layers=self.ctx.config.audio_config.audio_head_transformer_layers,
                audio_head_transformer_dims=self.ctx.config.audio_config.audio_head_transformer_dims,
                audio_head_enable=config_dict["audio_head_enable"]
            )
            audio_head_state_dict = load_weights_from_safetensors_helper(self.model_path, ["audio_head."])[0]
            if self.ctx.use_oe: # flash模型的audiohead的"hidden_proj"命名有不同
                audio_head_state_dict = {k.replace("hidden_in_proj", "hidden_proj"): v for k, v in audio_head_state_dict.items()}
            self.audio_head.load_state_dict(audio_head_state_dict, strict=True)
            self.audio_head.to("cuda").to(torch.bfloat16)
        self.enable_cuda_graph = config_dict.get("enable_cuda_graph", False)
        self.replay_cuda_graph = False
        if self.enable_cuda_graph:
            self._inited_for_decode_graphs = False
            self.init_for_decode_graphs()
            self._has_caputre = False
    
    def init_for_decode_graphs(self):
        if self._inited_for_decode_graphs:
            return
        self._inited_for_decode_graphs = True
        self.num_multi_ids = self.ctx.num_multi_ids
        self.cuda_graph_max_bs = global_config.server_args.cuda_graph_max_bs
        with torch.device("cuda"):
            self.persistent_output_ids = torch.full(
                (self.cuda_graph_max_bs, self.num_multi_ids),
                fill_value=-888881,
                dtype=torch.int64,
            )
        self.graphs: Dict[int, CUDAGraph] = {}

    def forward(
        self,
        input_ids: Tensor = None,
        positions: Tensor = None,
        forward_batch: ForwardBatch = None,
        output_hidden_states: Tensor = None,
        text_logits_output = None,
        sample_func = None,
    ):
        self.replay_cuda_graph = False
        forward_batch.capture_hidden_mode = CaptureHiddenMode.LAST
        forward_batch.next_token_ids = None
        
        sample_hidden_states = text_logits_output.hidden_states
        sample_hidden_states = sample_hidden_states.reshape(forward_batch.batch_size, 1, sample_hidden_states.shape[-1])
        
        for req in forward_batch.reqs:
            req.output_extra_info = {}
        
        NUM_CODEBOOKS = len(self.codebook_sizes)
        forward_batch.temp_multi_ids = torch.full(
            (forward_batch.batch_size, NUM_CODEBOOKS),
            -999997,
            dtype=torch.int64,
            device=sample_hidden_states.device,
        )
        
        # 如果 batch 中全是生文任务，不需要执行 depth-transformer，直接返回结果即可。
        if self.ctx.check_text_generation_only_in_batch(forward_batch):
            text_ids, _ = sample_func(text_logits_output, forward_batch)
            forward_batch.next_token_ids = text_ids
            return text_logits_output


        # outputprocess 开启 cudagraph 并且有 capture 后才能跳过
        # 注意：CFG 需要根据 batch 动态构造 cond/uncond 配对索引，为保证正确性，这里禁用 cudagraph replay。
        if (
            forward_batch.forward_mode == ForwardMode.DECODE
            and self.enable_cuda_graph
            and self._has_caputre
        ):
            self.replay_cuda_graph = True
        else:
            top_k, top_p, temperature, repetition_penalty, past_multi_ids, cfg_scale = self.process_forward_batch_requests(
                forward_batch, sample_hidden_states.device, multi_text_flag=False)
            text_indices, image_indices, audio_indices, cond_indices, uncond_indices = self.ctx.collect_gen_type_indices(forward_batch)
            # print("!!index:",text_indices, image_indices, audio_indices, cond_indices, uncond_indices)
            cfg_pair_indices = (cond_indices, uncond_indices)
            output_multi_ids = forward_batch.temp_multi_ids.clone()
            if len(image_indices) > 0:
                output_multi_ids[image_indices] = self.depth_transformer_forward_new(TaskType.IMAGE, len(image_indices), top_k[image_indices], 
                                                            top_p[image_indices], temperature[image_indices], repetition_penalty[image_indices], past_multi_ids[image_indices], 
                                                            sample_hidden_states[image_indices], cfg_scale[image_indices], cfg_pair_indices)
            if len(audio_indices) > 0:
                output_multi_ids[audio_indices] = self.depth_transformer_forward_new(TaskType.AUDIO, len(audio_indices), top_k[audio_indices], 
                                                            top_p[audio_indices], temperature[audio_indices], repetition_penalty[audio_indices], past_multi_ids[audio_indices], 
                                                            sample_hidden_states[audio_indices], cfg_scale[audio_indices], cfg_pair_indices)
            self.post_process(text_logits_output, input_ids, output_multi_ids, forward_batch)
        
        return text_logits_output
    
    def post_process(
        self,
        text_logits_output: Tensor,
        input_ids: Tensor,
        output_multi_ids: Tensor,
        forward_batch: ForwardBatch
    ):

        NUM_CODEBOOKS = len(self.codebook_sizes)
        tmp_multi_ids = output_multi_ids.reshape(forward_batch.batch_size, NUM_CODEBOOKS)

        top_k, top_p, temperature, repetition_penalty, past_ids, cfg_scale = self.process_forward_batch_requests(
                forward_batch, output_multi_ids.device, multi_text_flag=True)
        text_ids = self.sample(text_logits_output.next_token_logits, top_k, top_p, temperature, repetition_penalty, past_ids)

        
        def process_req(req_idx):
            req: Req = forward_batch.reqs[req_idx]
            sm: StateMachine = self.sm_dict[req.rid]
            
            if sm.get_state() == StateEnum.GEN_IMAGE_STAGE:
                token_w = req.input_extra_infos[0].get("token_w", None)
                # token_h = req.input_extra_infos[0].get("token_h", None)
                image_ids = output_multi_ids.tolist()[req_idx]
                sm_input = StateMachineInput(multi_ids=image_ids)
                sm.process(sm_input)
                if token_w and sm.context.gen_step%(token_w+1)==0:
                    # 推理完一行图像，插入<img_newline>标识换行，同时将当前多模id填成无效值
                    text_ids[req_idx] = get_spt().IMAGE_NEWLINE_TOKEN_ID
                    tmp_multi_ids[req_idx].fill_(-999997)
                else:
                    text_ids[req_idx] = get_spt().IMAGE_PAD_TOKEN_ID
                # print(f"{req.img_run_cnt=},{sm.context.gen_step=},{req.token_w=}")
            elif sm.get_state() == StateEnum.GEN_AUDIO_STAGE:
                gen_step = sm.context.gen_step # 这里是当前步的step从0开始
                # 切换状态
                audio_ids = tmp_multi_ids.tolist()[req_idx]
                sm_input = StateMachineInput(multi_ids=audio_ids)
                sm.process(sm_input)
                
                orig_input_ids = req.input_extra_infos[0]["orig_input_ids"]
                delay = req.input_extra_infos[0]["delay"]  # 0/float(inf)
                
                # 音频控制
                if not sm.context.audio_start:
                    # 预训练当前步数 <= delay（即sm.context.audio_start = False）时没有音频，
                    # 体现为delay0只有第0步没有音频，delayinf在复述阶段及下一步一直没有音频
                    tmp_multi_ids[req_idx].fill_(-99997)     
                # 文本控制
                if not sm.context.text_end:
                    if text_ids[req_idx] == get_spt().AUDIOTEXT_PAD_TOKEN_ID:
                        # 出现第一个pad表示生文结束, 同时记录delay用于控制ats
                        sm.context.text_end = True
                        delay = min(delay, gen_step) # delay=0时无效，delayinf时记录当前步（仅一次）

                else:
                    text_ids[req_idx] = get_spt().AUDIOTEXT_PAD_TOKEN_ID
                # ext_ids控制
                if gen_step == delay:
                    # delay 0 的第0步就给ats；delay inf在文本结束后一步给ats
                    req.input_extra_infos[0]["ext_ids"] = get_spt().AUDIOTEXT_START_TOKEN_ID
                    sm.context.audio_start = True # 允许下一步开始生成音频
                elif sm.get_state()!=StateEnum.NEXT_AUDIO_STAGE:
                    # 没结束的情况下都填pad
                    req.input_extra_infos[0]["ext_ids"] = get_spt().AUDIOTEXT_PAD_TOKEN_ID
                else:
                    # 结束时填ate
                    req.input_extra_infos[0]["ext_ids"] = get_spt().AUDIO_GEN_END_TOKEN_ID
                # print(f"{req.input_extra_infos[0]['ext_ids']=}, {text_ids[req_idx]=},{output_multi_ids=}", req.input_extra_infos[0]["delay"])

                
            elif sm.get_state() == StateEnum.GEN_TEXT_STAGE:
                # TODO: 如果是生文的case，暂不更新 state machine 状态？
                req.img_run_cnt = 0
                req.pending_img_newline = False
                tmp_multi_ids[req_idx].fill_(-999997)
            elif sm.get_state() == StateEnum.NEXT_AUDIO_STAGE:
                sm_input = StateMachineInput(text_id=text_ids[req_idx])
                sm.process(sm_input)
                # 当前步没有音频
                tmp_multi_ids[req_idx].fill_(-999997)
            elif sm.get_state() == StateEnum.ABORT:
                req.img_run_cnt = 0
                req.pending_img_newline = False
                req.to_abort = True
                req.to_abort_message = f"StateEnum.ABORT"
                text_ids[req_idx] = -3000001
        
        for req_idx in range(len(forward_batch.reqs)):
            process_req(req_idx)
        
        forward_batch.temp_multi_ids.copy_(tmp_multi_ids)
        # next_token_ids 非空会跳过 FluentLLM 本身负责的采样
        forward_batch.next_token_ids = text_ids

    def depth_transformer_forward_new(self,
                                  task: TaskType,
                                  batch_size,
                                  top_k: Tensor,
                                  top_p: Tensor,
                                  temperature: Tensor,
                                  repetition_penalty: Tensor,
                                  past_multi_ids: Tensor,
                                  output_hidden_states: Tensor,
                                  cfg_scale: Tensor = None,
                                  cfg_pair_indices: Tuple[List[int], List[int]] = (None,None)):
        output_hidden_states = output_hidden_states.reshape(batch_size, -1, output_hidden_states.shape[-1])
        vision_emb_for_infer = output_hidden_states[:,-1,:]
        cond_indices, uncond_indices = cfg_pair_indices
        if cond_indices is not None:
            cfg_cond_idx = torch.tensor(cond_indices, dtype=torch.long, device=output_hidden_states.device)
            cfg_uncond_idx = torch.tensor(uncond_indices, dtype=torch.long, device=output_hidden_states.device)
        NUM_CODEBOOKS = len(self.codebook_sizes)
        if task == TaskType.AUDIO:
            next_token_ids = torch.zeros(batch_size, NUM_CODEBOOKS, dtype=torch.long, device="cuda")
            for i in range(NUM_CODEBOOKS):
                logits = self.audio_head(vision_emb_for_infer, next_token_ids, self.ctx.audio_embed_layers, batch_size)[i]
                next_token_ids[:,i] = self.sample(logits, top_k, top_p, temperature, repetition_penalty, past_multi_ids[:,:,i])
        if task == TaskType.IMAGE:
            next_token_ids = torch.zeros(batch_size, NUM_CODEBOOKS, dtype=torch.long, device="cuda")
            for i in range(NUM_CODEBOOKS):
                logits = self.image_head(vision_emb_for_infer, next_token_ids, self.visual_bridge_model.embedding_layers, batch_size, i)
                if cond_indices is not None:
                    cond_logits = logits[cfg_cond_idx]
                    uncond_logits = logits[cfg_uncond_idx]
                    guided_logits = (cfg_scale[cfg_cond_idx] * (cond_logits - uncond_logits)).to(logits.dtype) + uncond_logits
                    logits[cfg_cond_idx] = guided_logits
                    logits[cfg_uncond_idx] = guided_logits
                logits[:, self.codebook_sizes[i]] = torch.finfo(logits.dtype).min
                next_token_ids[:,i] = self.sample(logits, top_k, top_p, temperature, repetition_penalty, past_multi_ids[:,:,i])
                # CFG 模式下保证 cond/uncond 采样 token 完全一致，否则后续 codebook 会发生分叉
                if cond_indices is not None:
                    next_token_ids[cfg_uncond_idx, i] = next_token_ids[cfg_cond_idx, i]
                
        return next_token_ids
    
    def process_forward_batch_requests(self, forward_batch:ForwardBatch, device, multi_text_flag=False):
        """
        处理 forward_batch.reqs 的参数并返回相关张量。

        Args:
            forward_batch: ForwardBatch实例
            device: PyTorch设备
            multi_text_flag: 是否用input_extro_info中的采样参数处理文本，默认false

        Returns:
            top_k, top_p, temperature, repetition_penalty, past_ids, cfg_pair_indices
        """
        top_k = []
        top_p = []
        temperature = []
        repetition_penalty = []
        past_ids = []
        cfg_scale = []
        if multi_text_flag:
            # 用input_extro_info中的采样参数处理文本
            for req in forward_batch.reqs:
                multi_params = req.input_extra_infos[0].get("multi_sampling_params", {})
                top_k.append(multi_params.get("top_k", req.sampling_params.top_k))
                top_p.append(multi_params.get("top_p", req.sampling_params.top_p))
                temperature.append(multi_params.get("temperature", req.sampling_params.temperature))
                repetition_penalty.append(multi_params.get("repetition_penalty", req.sampling_params.repetition_penalty))
                if req.output_ids and len(req.output_ids) > 0:
                    past_ids.append(torch.tensor(req.output_ids, device=device))
                else:
                    # 如果 output_ids 没有数据，填充一个空形状为 [0] 的 Tensor
                    past_ids.append(torch.empty((0), device=device, dtype=torch.int64))
        else:
            # 用reqs中的采样参数处理多模id
            for req in forward_batch.reqs:
                top_k.append(req.sampling_params.top_k)
                top_p.append(req.sampling_params.top_p)
                temperature.append(req.sampling_params.temperature)
                repetition_penalty.append(req.sampling_params.repetition_penalty)
                custom_params = getattr(req.sampling_params, "custom_params", None) or {}
                cfg_scale.append(custom_params.get("cfg_scale", 1.0))
                if req.output_multi_ids and len(req.output_multi_ids) > 0:
                    # 过滤掉包含负数的行
                    filtered_ids = [
                        row for row in req.output_multi_ids 
                        if all(x >= 0 for x in row)  # 保留所有元素 >= 0 的行
                    ]
                    filtered_tensor = torch.tensor(filtered_ids, device=device) if filtered_ids else torch.empty((0, 8), device=device, dtype=torch.int64)
                    past_ids.append(filtered_tensor)
                else:
                    # 如果 output_multi_ids 没有数据，填充一个空形状为 [0, 8] 的 Tensor
                    past_ids.append(torch.empty((0, 8), device=device, dtype=torch.int64))

        top_k = torch.tensor(top_k, device=device).unsqueeze(-1)
        top_p = torch.tensor(top_p, device=device).unsqueeze(-1)
        temperature = torch.tensor(temperature, device=device).unsqueeze(-1)
        temperature = torch.where(temperature == 0.0, torch.ones_like(temperature, device=device), temperature)
        repetition_penalty = torch.tensor(repetition_penalty, device=device).unsqueeze(-1)
        past_ids = pad_sequence(past_ids, batch_first=True, padding_value=0)  # 填充到一样长 多模[bs,len,8]/文本[bs,len]
        cfg_scale = torch.tensor(cfg_scale, dtype=torch.float32, device=device).unsqueeze(-1)
        # print(top_k, top_p, temperature, repetition_penalty)
        return top_k, top_p, temperature, repetition_penalty, past_ids, cfg_scale
    
    def sample(self, logits: Tensor, top_k: Tensor, top_p: Tensor, temperature: Tensor,
               repetition_penalty: Tensor, output_multi_ids: Tensor):
        next_token_logits = logits[:,:].clone()
        next_token_logits = self.process_repetition_penalty(output_multi_ids, next_token_logits, repetition_penalty)
        next_token_logits.div_(temperature)
        probs = F.softmax(next_token_logits, dim=-1)
        # # TODO: 当前算子可能存在topk出相等的多于k值的logits, 导致采样有随机性
        # probs = top_k_renorm_prob(probs, top_k)
        # next_tokens = top_p_sampling_from_probs(probs, top_p)
        # next_tokens = top_k_top_p_sampling_from_probs(probs, top_k, top_p)
        # next_tokens = torch.argmax(next_token_logits, dim=-1)
        next_tokens = top_k_top_p_min_p_sampling_from_probs_torch(
                        probs = probs,
                        top_ks = top_k,
                        top_ps = top_p,
                        min_ps = top_p,  # 因为不用所以随便传一个
                        need_min_p_sampling = False,
                    )
        return next_tokens
    
    def process_repetition_penalty(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, penalty:torch.FloatTensor) -> torch.FloatTensor:
        score = torch.gather(scores, 1, input_ids)

        # if score < 0 then repetition penalty has to be multiplied to reduce the token probabilities
        score = torch.where(score < 0, score * penalty, score / penalty)
        score = score.to(scores.dtype)

        scores_processed = scores.scatter(1, input_ids, score)
        return scores_processed
    
    def forward_to_capture(self,
                        bs: int,
                        top_k: Tensor = None,
                        top_p: Tensor = None,
                        temperature: Tensor = None,
                        repetition_penalty: Tensor = None,
                        past_multi_ids: Tensor = None,
                        output_hidden_states: Tensor = None):
        return self.depth_transformer_forward(bs,  top_k, top_p, temperature, repetition_penalty, past_multi_ids, output_hidden_states)
    
    def capture_one_config_decode(
        self,
        bs: int,
        capture_batch: ForwardBatch,
        model_runner: ModelRunner,
        stream: Stream,
    ) -> None:
        self.ctx.init_for_decode_graphs()
        if not self.enable_cuda_graph:
            return
        self._has_caputre = True
        output_hidden_states = self.ctx.decode_output_hidden_states[:bs]
        top_k = self.ctx.top_k[:bs]
        top_p = self.ctx.top_p[:bs]
        repetition_penalty = self.ctx.repetition_penalty[:bs]
        past_multi_ids = self.ctx.past_multi_ids[:bs,:,:]
        temperature = self.ctx.temperature[:bs]
        def dep_former_fn():
            raw_multi_ids : torch.Tensor = self.forward_to_capture(bs, top_k, top_p, temperature, repetition_penalty, past_multi_ids, output_hidden_states)
            self.persistent_output_ids[:bs].copy_(raw_multi_ids)
            return None

        graph, out = capture(fn=dep_former_fn, stream=stream, model_runner=model_runner)
        self.graphs[bs] = graph

        return

    def replay_one_config_decode(self, bs: int, replay_batch: ForwardBatch) -> None:
        real_bs = len(replay_batch.seq_lens)
        assert replay_batch.input_ids.shape[0] == real_bs

        logits_output: LogitsProcessorOutput = self.forward(
            input_ids=replay_batch.input_ids,
            positions=replay_batch.positions,
            forward_batch=replay_batch,
            output_hidden_states=self.ctx.decode_output_hidden_states[:real_bs],
        )
        self.ctx.decode_next_token_logits[:real_bs].copy_(logits_output.next_token_logits)
        self.ctx.decode_output_hidden_states[:real_bs].copy_(logits_output.hidden_states)

        top_k, top_p, temperature, repetition_penalty, past_multi_ids, cfg_scale = self.process_forward_batch_requests(
                replay_batch, self.ctx.decode_next_token_logits.device, multi_text_flag=False)
        self.ctx.top_k[:real_bs].copy_(top_k)
        self.ctx.top_p[:real_bs].copy_(top_p)
        self.ctx.temperature[:real_bs].copy_(temperature)
        self.ctx.repetition_penalty[:real_bs].copy_(repetition_penalty)
        self.ctx.cfg_scale[:real_bs].copy_(cfg_scale)
        past_multi_ids = past_multi_ids[:,-self.ctx.max_seq_len:,:]
        len_ids = past_multi_ids.shape[1]
        self.ctx.past_multi_ids[:real_bs,:len_ids,:].copy_(past_multi_ids)
        
        # forward 方法中判断是否需要 replay cuda graph
        if self.enable_cuda_graph and self.replay_cuda_graph:
            self.graphs[bs].replay()
            real_output_ids = self.persistent_output_ids[:real_bs]

            # 每一条请求做后处理
            self.post_process(
                logits_output,
                None,
                real_output_ids, # [bs, head_num]
                replay_batch,
            )

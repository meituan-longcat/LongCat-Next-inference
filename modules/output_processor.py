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

        self.enable_tp = False
        
        if self.ctx.visual_enable:
            self.visual_bridge_model = self.ctx.visual_bridge_model
            self.image_head = OmniImageHead(
                hidden_size=self.hidden_size,
                codebook_sizes=self.codebook_sizes,
                image_head_transformer_ffn_scale=self.ctx.config.visual_config.image_head_config.image_head_transformer_ffn_scale,
                image_head_transformer_dims=self.ctx.config.visual_config.image_head_config.image_head_transformer_dims,
                image_head_transformer_layers=self.ctx.config.visual_config.image_head_config.image_head_transformer_layers,
                image_head_enable=config_dict["image_head_enable"],
                enable_tp=self.enable_tp,
            )
            image_head_state_dict = load_weights_from_safetensors_helper(self.model_path, ["visual_head."])[0]
            # Load with TP-aware weight loading
            self.image_head.load_tp_state_dict(image_head_state_dict)
            self.image_head.to("cuda").to(torch.bfloat16)
        if self.ctx.audio_enable:
            self.audio_head = OmniAudioHead(
                hidden_size=self.hidden_size,
                codebook_sizes=self.ctx.audio_codebook_sizes,
                audio_head_transformer_ffn_scale=self.ctx.config.audio_config.audio_head_transformer_ffn_scale,
                audio_head_transformer_layers=self.ctx.config.audio_config.audio_head_transformer_layers,
                audio_head_transformer_dims=self.ctx.config.audio_config.audio_head_transformer_dims,
                audio_head_enable=config_dict["audio_head_enable"],
                enable_tp=self.enable_tp,
            )
            audio_head_state_dict = load_weights_from_safetensors_helper(self.model_path, ["audio_head."])[0]
            if self.ctx.use_oe: # flash模型的audiohead的"hidden_proj"命名有不同
                audio_head_state_dict = {k.replace("hidden_in_proj", "hidden_proj"): v for k, v in audio_head_state_dict.items()}
            # Load with TP-aware weight loading
            self.audio_head.load_tp_state_dict(audio_head_state_dict)
            self.audio_head.to("cuda").to(torch.bfloat16)
        image_total_params = sum(p.numel() for p in self.image_head.parameters())
        audio_total_params = sum(p.numel() for p in self.audio_head.parameters())
        print(f"总参数量: {audio_total_params=}, {image_total_params=}")
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
            self.audio_output_ids = torch.full(
                (self.cuda_graph_max_bs, self.num_multi_ids),
                fill_value=-888881,
                dtype=torch.int64,
            )
            self.image_output_ids = torch.full(
                (self.cuda_graph_max_bs, self.num_multi_ids),
                fill_value=-888881,
                dtype=torch.int64,
            )
        self.graphs: Dict[int, CUDAGraph] = {}
        self.audio_graph: Dict[int, CUDAGraph] = {}
        self.image_cfg_graph: Dict[int, CUDAGraph] = {}
        # 记录已经capture的batch size
        self.captured_audio_bs: List[int] = []
        self.captured_image_bs: List[int] = []
        # 为audio和image创建独立的CUDA stream，以支持并行replay
        self.audio_stream = Stream()
        self.image_stream = Stream()

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
                forward_batch, sample_hidden_states.device)
            text_indices, image_indices, audio_indices, image_cfg_indices = self.ctx.collect_gen_type_indices(forward_batch)
            if len(image_indices) == len(image_cfg_indices):
                # 当生图全是cfg生图时，直接使用cfg的索引
                image_indices = image_cfg_indices
            else:
                assert len(image_cfg_indices)==0, "不支持非cfg和cfg混用"
            output_multi_ids = forward_batch.temp_multi_ids.clone()
            if len(image_indices) > 0:
                output_multi_ids[image_indices] = self.depth_transformer_forward(TaskType.IMAGE, len(image_indices), top_k[image_indices], 
                                                            top_p[image_indices], temperature[image_indices], repetition_penalty[image_indices], past_multi_ids[image_indices], 
                                                            sample_hidden_states[image_indices], cfg_scale[image_indices], len(image_cfg_indices)!=0)
            if len(audio_indices) > 0:
                output_multi_ids[audio_indices] = self.depth_transformer_forward(TaskType.AUDIO, len(audio_indices), top_k[audio_indices], 
                                                            top_p[audio_indices], temperature[audio_indices], repetition_penalty[audio_indices], past_multi_ids[audio_indices], 
                                                            sample_hidden_states[audio_indices], cfg_scale[audio_indices], False)
            self.post_process(text_logits_output, input_ids, output_multi_ids, forward_batch, sample_func)
        
        return text_logits_output
    
    def post_process(
        self,
        text_logits_output: Tensor,
        input_ids: Tensor,
        output_multi_ids: Tensor,
        forward_batch: ForwardBatch,
        sample_func = None
    ):

        NUM_CODEBOOKS = len(self.codebook_sizes)
        tmp_multi_ids = output_multi_ids.reshape(forward_batch.batch_size, NUM_CODEBOOKS)
        
        text_ids, _ = sample_func(text_logits_output, forward_batch)
        # 消耗随机数以对齐随机状态
        _ = torch.rand(text_logits_output.next_token_logits.shape[0], device="cuda")

        
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

    def depth_transformer_forward(self,
                                  task: TaskType,
                                  batch_size,
                                  top_k: Tensor,
                                  top_p: Tensor,
                                  temperature: Tensor,
                                  repetition_penalty: Tensor,
                                  past_multi_ids: Tensor,
                                  output_hidden_states: Tensor,
                                  cfg_scale: Tensor = None,
                                  cfg_enable: bool = False):
        output_hidden_states = output_hidden_states.reshape(batch_size, -1, output_hidden_states.shape[-1])
        vision_emb_for_infer = output_hidden_states[:,-1,:]
            
        NUM_CODEBOOKS = len(self.codebook_sizes)
        if task == TaskType.AUDIO:
            next_token_ids = torch.zeros(batch_size, NUM_CODEBOOKS, dtype=torch.long, device="cuda")
            for i in range(NUM_CODEBOOKS):
                logits = self.audio_head(vision_emb_for_infer, next_token_ids, self.ctx.audio_embed_layers, batch_size, i)
                next_token_ids[:,i] = self.sample(logits, top_k, top_p, temperature, repetition_penalty, past_multi_ids[:,:,i])
        if task == TaskType.IMAGE:
            if cfg_enable:
                cfg_cond_idx = torch.arange(0, batch_size, 2, dtype=torch.long, device=output_hidden_states.device)
                cfg_uncond_idx = torch.arange(1, batch_size, 2, dtype=torch.long, device=output_hidden_states.device)
            next_token_ids = torch.zeros(batch_size, NUM_CODEBOOKS, dtype=torch.long, device="cuda")
            for i in range(NUM_CODEBOOKS):
                logits = self.image_head(vision_emb_for_infer, next_token_ids, self.visual_bridge_model.embedding_layers, batch_size, i)
                if cfg_enable:
                    cond_logits = logits[cfg_cond_idx]
                    uncond_logits = logits[cfg_uncond_idx]
                    guided_logits = (cfg_scale[cfg_cond_idx] * (cond_logits - uncond_logits)).to(logits.dtype) + uncond_logits
                    logits[cfg_cond_idx] = guided_logits
                    logits[cfg_uncond_idx] = guided_logits
                logits[:, self.codebook_sizes[i]] = torch.finfo(logits.dtype).min
                next_token_ids[:,i] = self.sample(logits, top_k, top_p, temperature, repetition_penalty, past_multi_ids[:,:,i])
                # CFG 模式下保证 cond/uncond 采样 token 完全一致，否则后续 codebook 会发生分叉
                if cfg_enable:
                    next_token_ids[cfg_uncond_idx, i] = next_token_ids[cfg_cond_idx, i]
                
        return next_token_ids
    
    
    def process_forward_batch_requests(self, forward_batch:ForwardBatch, device):
        """
        处理 forward_batch.reqs 的参数并返回相关张量。

        Args:
            forward_batch: ForwardBatch实例
            device: PyTorch设备

        Returns:
            top_k, top_p, temperature, repetition_penalty, past_ids, cfg_scale
        """
        top_k = []
        top_p = []
        temperature = []
        repetition_penalty = []
        past_ids = []
        cfg_scale = []
        # 用input_extro_info中的采样参数处理
        for req in forward_batch.reqs:
            multi_params = req.input_extra_infos[0].get("multi_sampling_params", {})
            top_k.append(multi_params.get("top_k", req.sampling_params.top_k))
            top_p.append(multi_params.get("top_p", req.sampling_params.top_p))
            temperature.append(multi_params.get("temperature", req.sampling_params.temperature))
            repetition_penalty.append(multi_params.get("repetition_penalty", req.sampling_params.repetition_penalty))
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
        
        # 音频任务使用独立的 audio buffer 和 audio_stream
        output_hidden_states_audio = self.ctx.hidden_states_audio[:bs]
        top_k_audio = self.ctx.top_k_audio[:bs]
        top_p_audio = self.ctx.top_p_audio[:bs]
        repetition_penalty_audio = self.ctx.repetition_penalty_audio[:bs]
        past_multi_ids_audio = self.ctx.past_multi_ids_audio[:bs,:,:]
        temperature_audio = self.ctx.temperature_audio[:bs]
        cfg_scale_audio = self.ctx.cfg_scale_audio[:bs]
        
        def dep_former_fn_audio():
            raw_multi_ids : torch.Tensor = self.depth_transformer_forward(TaskType.AUDIO, bs, top_k_audio, top_p_audio, 
                                                temperature_audio, repetition_penalty_audio, past_multi_ids_audio, output_hidden_states_audio)
            self.audio_output_ids[:bs].copy_(raw_multi_ids)
            return None

        # 使用独立的 audio_stream 进行 capture
        graph, out = capture(fn=dep_former_fn_audio, stream=self.audio_stream, model_runner=model_runner)
        self.audio_graph[bs] = graph
        self.captured_audio_bs.append(bs)
        
        # 图像任务使用独立的 image buffer 和 image_stream
        output_hidden_states_image = self.ctx.hidden_states_image[:bs]
        top_k_image = self.ctx.top_k_image[:bs]
        top_p_image = self.ctx.top_p_image[:bs]
        repetition_penalty_image = self.ctx.repetition_penalty_image[:bs]
        past_multi_ids_image = self.ctx.past_multi_ids_image[:bs,:,:]
        temperature_image = self.ctx.temperature_image[:bs]
        cfg_scale_image = self.ctx.cfg_scale_image[:bs]
        
        def dep_former_fn_image():
            raw_multi_ids : torch.Tensor = self.depth_transformer_forward(TaskType.IMAGE, bs, top_k_image, top_p_image, 
                                                temperature_image, repetition_penalty_image, past_multi_ids_image, output_hidden_states_image,
                                                cfg_scale_image, cfg_enable=True)
            self.image_output_ids[:bs].copy_(raw_multi_ids)
            return None
        if bs%2==0:
            # 只capture 偶数bs，使用独立的 image_stream 进行 capture
            graph, out = capture(fn=dep_former_fn_image, stream=self.image_stream, model_runner=model_runner)
            self.image_cfg_graph[bs] = graph
            self.captured_image_bs.append(bs)
        return
    
    def _find_replay_bs(self, actual_bs: int, captured_bs_list: List[int]) -> Optional[int]:
        """
        找到合适的batch size进行replay
        向上取整到已capture的bs，如果没有足够大的则返回None
        
        Args:
            actual_bs: 实际需要的batch size
            captured_bs_list: 已capture的batch size列表
        
        Returns:
            合适的batch size，如果没有找到则返回None
        """
        if actual_bs in captured_bs_list:
            return actual_bs
        
        # 找到大于等于actual_bs的最小已capture的bs
        suitable_bs = None
        for captured_bs in captured_bs_list:
            if captured_bs >= actual_bs:
                if suitable_bs is None or captured_bs < suitable_bs:
                    suitable_bs = captured_bs
        
        return suitable_bs

    def replay_one_config_decode(self, replay_batch: ForwardBatch, logits_output, sample_func) -> None:
        real_bs = len(replay_batch.seq_lens)
        assert replay_batch.input_ids.shape[0] == real_bs

        self.ctx.decode_next_token_logits[:real_bs].copy_(logits_output.next_token_logits)
        self.ctx.decode_output_hidden_states[:real_bs].copy_(logits_output.hidden_states)

        top_k, top_p, temperature, repetition_penalty, past_multi_ids, cfg_scale = self.process_forward_batch_requests(
                replay_batch, self.ctx.decode_next_token_logits.device)
        past_multi_ids = past_multi_ids[:,-self.ctx.max_seq_len:,:]
        len_ids = past_multi_ids.shape[1]
        
        text_indices, image_indices, audio_indices, image_cfg_indices = self.ctx.collect_gen_type_indices(replay_batch)
        if len(image_indices) == len(image_cfg_indices):
            # 当生图全是cfg生图时，直接使用cfg的索引
            image_indices = image_cfg_indices
        else:
            assert False, "graph 不支持非cfg生图"
        output_multi_ids = replay_batch.temp_multi_ids.clone()
        
        # 准备数据并记录需要replay的graph
        image_replay_bs = None
        audio_replay_bs = None
        
        # 生图任务（CFG模式）：准备数据
        if len(image_indices) > 0:
            num_image = len(image_indices)
            # 找到合适的batch size进行replay（向上取整到已capture的bs）
            image_replay_bs = self._find_replay_bs(num_image, self.captured_image_bs)
            if image_replay_bs is None:
                raise ValueError(f"No captured graph found for image batch size {num_image}. Captured sizes: {self.captured_image_bs}")
            
            # 将数据按 image_cfg_indices 顺序重排后填入独立的 image buffer
            self.ctx.hidden_states_image[:num_image].copy_(logits_output.hidden_states[image_indices])
            self.ctx.top_k_image[:num_image].copy_(top_k[image_indices])
            self.ctx.top_p_image[:num_image].copy_(top_p[image_indices])
            self.ctx.temperature_image[:num_image].copy_(temperature[image_indices])
            self.ctx.repetition_penalty_image[:num_image].copy_(repetition_penalty[image_indices])
            self.ctx.cfg_scale_image[:num_image].copy_(cfg_scale[image_indices])
            self.ctx.past_multi_ids_image[:num_image,:len_ids,:].copy_(past_multi_ids[image_indices])
        
        # 音频任务：准备数据
        if len(audio_indices) > 0:
            num_audio = len(audio_indices)
            # 找到合适的batch size进行replay（向上取整到已capture的bs）
            audio_replay_bs = self._find_replay_bs(num_audio, self.captured_audio_bs)
            if audio_replay_bs is None:
                raise ValueError(f"No captured graph found for audio batch size {num_audio}. Captured sizes: {self.captured_audio_bs}")
            
            # 将数据按 audio_indices 顺序重排后填入独立的 audio buffer
            self.ctx.hidden_states_audio[:num_audio].copy_(logits_output.hidden_states[audio_indices])
            self.ctx.top_k_audio[:num_audio].copy_(top_k[audio_indices])
            self.ctx.top_p_audio[:num_audio].copy_(top_p[audio_indices])
            self.ctx.temperature_audio[:num_audio].copy_(temperature[audio_indices])
            self.ctx.repetition_penalty_audio[:num_audio].copy_(repetition_penalty[audio_indices])
            self.ctx.cfg_scale_audio[:num_audio].copy_(cfg_scale[audio_indices])
            self.ctx.past_multi_ids_audio[:num_audio,:len_ids,:].copy_(past_multi_ids[audio_indices])
        
        # 并行执行 graph replay（在不同的stream上）
        if image_replay_bs is not None:
            # 在 image_stream 上执行 image graph replay
            with torch.cuda.stream(self.image_stream):
                self.image_cfg_graph[image_replay_bs].replay()
        
        if audio_replay_bs is not None:
            # 在 audio_stream 上执行 audio graph replay
            with torch.cuda.stream(self.audio_stream):
                self.audio_graph[audio_replay_bs].replay()
        
        # 同步两个stream，确保replay完成
        if image_replay_bs is not None:
            self.image_stream.synchronize()
            num_image = len(image_indices)
            output_multi_ids[image_indices] = self.image_output_ids[:num_image]
        
        if audio_replay_bs is not None:
            self.audio_stream.synchronize()
            num_audio = len(audio_indices)
            output_multi_ids[audio_indices] = self.audio_output_ids[:num_audio]

        # 每一条请求做后处理
        self.post_process(
            logits_output,
            None,
            output_multi_ids, # [bs, head_num]
            replay_batch,
            sample_func
        )

g_graph_memory_pool = None
def capture(fn: callable, stream, model_runner: ModelRunner, num_warmups=5):
    graph = torch.cuda.CUDAGraph()

    for _ in range(num_warmups):
        torch.cuda.synchronize()
        model_runner.tp_group.barrier()
        fn()

    torch.cuda.synchronize()
    model_runner.tp_group.barrier()
    global g_graph_memory_pool

    with torch.cuda.graph(graph, pool=g_graph_memory_pool, stream=stream):
        out = fn()

    torch.cuda.synchronize()
    model_runner.tp_group.barrier()

    g_graph_memory_pool = graph.pool()

    return graph, out
# from sglang.srt.models.extensible import ContextBase, register_ext_cls, capture
from sglang.global_config import global_config
from .state_machine import StateEnum, StateMachine, StateMachineInput
from .visual_emb import VisualEmbeddingBridge
from transformers import AutoConfig
from .special_token import init_spt, get_spt
from glob import glob
from safetensors import safe_open
import os
from typing import Any, Dict, List, Optional, Tuple
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.global_config import global_config
from utils.model_utils import load_weights_from_safetensors_helper

class LongcatOOverEmbContext():
    def __init__(self, base_lm, config_dict):
        self.model_path = os.environ.get("NMM_INFER_MODEL_ROOT", None)
        self.config_path = os.path.join(self.model_path, "nmm_infer")
        self.config = AutoConfig.from_pretrained(self.config_path, trust_remote_code=True)
        self.sm_dict = {}
        self.codebook_sizes = self.config.visual_quantizer_config.codebook_sizes
        self.audio_codebook_sizes = self.config.audio_config.vq_config.codebook_sizes
        self.hidden_size = self.config.hidden_size
        self.intermediate_size = self.config.intermediate_size
        self.hidden_act = self.config.hidden_act
        self.rms_norm_eps = self.config.rms_norm_eps
        self.num_multi_ids = config_dict["num_multi_ids"]
        self.visual_enable = config_dict["visual_enable"]
        self.audio_enable = config_dict["audio_enable"]
        self.use_oe = config_dict["use_oe"]
        self.max_seq_len = 2048 # TODO:改成serverargs里的max_seq_len？但是太长了也会影响采样性能
        if self.visual_enable:
            self.visual_bridge_model = VisualEmbeddingBridge(
                codebook_sizes=self.codebook_sizes,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                hidden_act=self.hidden_act,
                rms_norm_eps=self.rms_norm_eps,
            )
            key_words_list = ["model.visual_tokenizer.visual_embedding_layer.pre_buffer.", "model.embed_tokens."]
            state_dicts = load_weights_from_safetensors_helper(self.model_path, key_words_list)    
            emb_state_dicts = {}
            offset = self.config.visual_offset
            for i, codedim in enumerate(self.codebook_sizes):
                # 16384 * 8层
                emb_state_dicts[f"{i}.weight"] = state_dicts[1]["weight"][offset:offset+codedim+1, :]
                offset += codedim
            self.visual_bridge_model.embedding_layers.load_state_dict(emb_state_dicts, strict=True)
            self.visual_bridge_model.transformer_block.load_state_dict(state_dicts[0], strict=True)
            self.visual_bridge_model.to("cuda").to(torch.bfloat16)
        if self.audio_enable:
            self.audio_embed_layers = torch.nn.ModuleList([
                torch.nn.Embedding(codedim+1, self.hidden_size)
                    for i, codedim in enumerate(self.audio_codebook_sizes)
            ])
            if not self.use_oe:
                key_words_list = ["model.audio_embed_layers."]
                emb_state_dicts = load_weights_from_safetensors_helper(self.model_path, key_words_list)[0]
            else: # flash模型的audioemb需要从大emb中切
                key_words_list = ["model.embed_tokens."]
                state_dicts = load_weights_from_safetensors_helper(self.model_path, key_words_list)
                emb_state_dicts = {}
                offset = self.config.audio_offset
                for i, codedim in enumerate(self.audio_codebook_sizes):
                    # 8192,4096,2048, 1024*5, 共8层
                    emb_state_dicts[f"{i}.weight"] = state_dicts[0]["weight"][offset:offset+codedim+1, :]
                    offset += codedim
            self.audio_embed_layers.load_state_dict(emb_state_dicts, strict=True)
            self.audio_embed_layers.to("cuda").to(torch.bfloat16)
        init_spt(self.config_path)
    
    def check_text_generation_only_in_batch(self, forward_batch: ForwardBatch):
        # 检查状态机中，是否所有 req 都是生文任务，是的话前后处理都简单处理即可。
        for req in forward_batch.reqs:
            sm: StateMachine = self.sm_dict[req.rid]
            
            if sm.get_state() != StateEnum.GEN_TEXT_STAGE:
                return False
        return True
    
    def init_for_decode_graphs(self):
        if self._inited_for_decode_graphs:
            return
        self._inited_for_decode_graphs = True
        self.cuda_graph_max_bs = global_config.server_args.cuda_graph_max_bs
        with torch.device("cuda"):
            self.decode_input_hidden_states = torch.full(
                (self.cuda_graph_max_bs, self.hidden_size),
                fill_value=0.0,
                dtype=global_config.dtype,
            )
            self.decode_output_hidden_states = torch.full(
                (self.cuda_graph_max_bs, self.hidden_size),
                fill_value=0.0,
                dtype=global_config.dtype,
            )
            self.decode_next_token_logits = torch.full(
                (self.cuda_graph_max_bs, global_config.model_config.vocab_size),
                fill_value=0.0,
                dtype=torch.float32,
            )
            self.top_k = torch.full(
                (self.cuda_graph_max_bs, 1),
                fill_value=1,
                dtype=torch.int64,
                device='cuda',
            )
            self.top_p = torch.full(
                (self.cuda_graph_max_bs, 1),
                fill_value=1.0,
                dtype=torch.float,
                device='cuda',
            )
            self.temperature = torch.full(
                (self.cuda_graph_max_bs, 1),
                fill_value=1.0,
                dtype=torch.float,
                device='cuda',
            )
            self.repetition_penalty = torch.full(
                (self.cuda_graph_max_bs, 1),
                fill_value=1.0,
                dtype=torch.float,
                device='cuda',
            )
            self.past_multi_ids = torch.full(
                (self.cuda_graph_max_bs, self.max_seq_len, 8),
                fill_value=0,
                dtype=torch.long,
                device='cuda',
            )
            self.cfg_scale = torch.full(
                (self.cuda_graph_max_bs, 1),
                fill_value=1.0,
                dtype=torch.float,
                device='cuda',
            )
            
    def collect_gen_type_indices(self, forward_batch: ForwardBatch) -> Tuple[List[int], List[int], List[int], List[int], List[int]]:
        """Collect indices for text/image/audio generation requests and CFG pairs.

        Returns:
            Tuple of (text_indices, image_indices, audio_indices, cond_indices, uncond_indices)
        """
        text_indices: List[int] = []
        image_indices: List[int] = []
        audio_indices: List[int] = []
        cfg_pair_to_roles: Dict[str, Dict[str, int]] = {}  # pair_id -> {"cond": idx, "uncond": idx}

        for req_idx, req in enumerate(forward_batch.reqs):
            gen_image = req.input_extra_infos[0].get("gen_image", False)
            gen_audio = req.input_extra_infos[0].get("gen_audio", False)

            if gen_image:
                image_indices.append(req_idx)
                custom_params = getattr(getattr(req, "sampling_params", {}), "custom_params", None)
                if isinstance(custom_params, dict):
                    pair_id = custom_params.get("cfg_pair_id", None)
                    role = custom_params.get("cfg_role")
                    if pair_id is not None and role in ("cond", "uncond"):
                        pair_id = str(pair_id)
                        cfg_pair_to_roles.setdefault(pair_id, {})[role] = req_idx
            elif gen_audio:
                audio_indices.append(req_idx)
            else:
                text_indices.append(req_idx)

        # Build cond/uncond indices with consistent ordering by pair_id
        cond_indices: List[int] = []
        uncond_indices: List[int] = []
        for pair_id in sorted(cfg_pair_to_roles.keys()):
            roles = cfg_pair_to_roles[pair_id]
            if "cond" in roles and "uncond" in roles:
                cond_indices.append(roles["cond"])
                uncond_indices.append(roles["uncond"])

        return (text_indices, image_indices, audio_indices, cond_indices, uncond_indices)
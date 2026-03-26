
import threading
import copy
from utils.timeout_lock import TimeoutLock

from sglang.srt.utils import (
    get_colorful_logger,
)
from framework.request_cache.utils import build_rid

logger = get_colorful_logger(__name__)

class RequestCacheInfo:
    def __init__(self, cache_idx: int) -> None:
        self._cache_idx = cache_idx
        self.req_cache = None
        self.input_cache_idx = None
        self.output_cache_idx = None
        self.last_prefill_step = 0
        # self.output_cache_idx_map = {}
        # self.running_rids_dict = {}
        self._lock = TimeoutLock("RequestCacheInfo")
        self.stop = False
        self.interrupt_event = None
        self.interrupt_finish_event = None
    
    def clear(self):
        self.stop = True
        # self._cache_idx = None
        # self.req_cache = None
        # self.input_cache_idx = None
        # self.output_cache_idx = None
        # self.output_cache_idx_map = {}
        # self.running_rids_dict = {}
        # self.last_prefill_step = 0
        # self.interrupt_event = None
        # self.interrupt_finish_event = None
    
    def build_current_rid(self, session_id_hash_id, request_id, prefill_len):
        cur_session_id = self.req_cache.get_global_state("input", self.input_cache_idx, "SESSION_ID")
        if session_id_hash_id != cur_session_id:
            logger.info(f"request_id:{request_id} session changed {session_id_hash_id}->{cur_session_id}")
            # return None
        cur_round_id = self.req_cache.get_global_state("input", self.input_cache_idx, "ROUND_ID")
        # cur_step = self.req_cache.get_global_state("input", self.input_cache_idx, "TOTAL_STEP")
        return build_rid(
            session_id=cur_session_id, 
            round_id=cur_round_id, 
            input_cache_idx=self.input_cache_idx, 
            output_cache_idx=self.output_cache_idx,
            prefill_len=prefill_len,
            request_id=request_id)

    def increase_round_id(self, skip_empty_round=True):
        # self.req_cache.update_global_state("output", self.output_cache_idx,"ROUND_ID", lambda x : x+ 1)
        if skip_empty_round:
            round_id = self.get_round_id()
            round_info = self.get_round_info(round_id+1, start_round_id=round_id)
            # logger.info(f"increase_round_id show {round_id=} {round_info=}")
            if round_info[-1][2] == 0:
                return round_id-1, round_id
        return self.req_cache.update_global_state("input", self.input_cache_idx,"ROUND_ID", lambda x : x+ 1)
    
    def get_round_id(self):
        return self.req_cache.get_global_state("input", self.input_cache_idx, "ROUND_ID")
    
    def set_round_id(self, round_id):
        return self.req_cache.set_global_state("input", self.input_cache_idx, "ROUND_ID", round_id)
    
    def get_session_id(self):
        return self.req_cache.get_global_state("input", self.input_cache_idx, "SESSION_ID")

    def get_input_ids(self, start=0, end=-1):
       return self.req_cache.get_input_ids(self.input_cache_idx, start=start, end=end)

    def append_input(self, input_token_ids, max_new_tokens, input_tensor_dict, kv_paged_base_str=None, step_check=None):
        return self.req_cache.append_input(self.input_cache_idx, input_token_ids, max_new_tokens, input_tensor_dict=input_tensor_dict, kv_paged_base_str=kv_paged_base_str, step_check=step_check)

    # def get_input_ids(self, step_check=None):
    #     return self.req_cache.get_input_ids(self.input_cache_idx, step_check=step_check)
    
    def copy_output_to_input(self, resp_info, output_start_idx, input_start_idx, extend_len, output_ids, clone=True, timeout=0.5):
        return self.req_cache.copy_output_to_input(resp_info, self.output_cache_idx, self.input_cache_idx, output_start_idx, input_start_idx, extend_len, output_ids, clone=clone, timeout=timeout)
    
    # def add_running_req(self, cur_round_id, request_ids):
    #     self.running_rids_dict[cur_round_id] = request_ids
    
    def update_round_info(self, round_id, hash_id=None, input_tokens=None, output_tokens=None, back_tokens=None, step_check=None, step_check_round=None):
        return self.req_cache.update_round_info(self.input_cache_idx, round_id, hash_id=hash_id,
                                                             input_tokens=input_tokens, output_tokens=output_tokens, back_tokens=back_tokens,
                                                             step_check=step_check, step_check_round=step_check_round)
    
    def get_round_info(self, end_round_id, start_round_id=0):
        return self.req_cache.get_round_info(self.input_cache_idx, end_round_id, start_round_id=start_round_id)
    
    def get_input_buf_idx(self):
        return self.input_cache_idx * self.req_cache.max_seq_len

    def set_step_and_round_info(self, round_id, step, round_info, diff_round_start_idx):
        #  self.req_cache.set_global_state("output", self.output_cache_idx,"ROUND_ID", round_id)
        self.req_cache.set_global_state("input", self.input_cache_idx, "ROUND_ID", round_id)
        self.req_cache.set_global_state("input", self.input_cache_idx, "TOTAL_STEP", step)
        for new_round_id in range(diff_round_start_idx, round_id + 1):
            (_, hash_id, input_tokens, output_tokens, back_tokens) = round_info[new_round_id]
            self.update_round_info(new_round_id, hash_id=hash_id, input_tokens=input_tokens, output_tokens=output_tokens, back_tokens=back_tokens)

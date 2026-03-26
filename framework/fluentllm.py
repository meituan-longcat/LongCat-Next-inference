import os
os.environ["SGLANG_DISABLED_MODEL_ARCHS"] = "modules"
os.environ["REQUEST_CACHE_MODULE_PATH"] = "framework.request_cache.request_cache"
from datetime import datetime
import torch
import asyncio
from random import randint
import json
import time
from typing import Callable, Dict, Any, List, Optional
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.utils import get_colorful_logger
from sglang.srt.server_args import ServerArgs, prepare_server_args
from framework.request_cache.request_cache import RequestCache, hash_to_int, EMBEDDING_CACHE_KEY, EMBEDDING_SUB_KEY
from utils.config_utils import dict_to_cli_args

from utils.timeout_lock import TimeoutLock

logger = get_colorful_logger(__name__)

class RequstCountLimitLock:
    """请求计数限制锁 - 支持上下文管理器"""
    def __init__(self, backend, session_id) -> None:
        self.backend = backend
        self.session_id = session_id
        # already in lock
        if self.session_id not in self.backend.request_count_map:
            self.backend.request_count_map[self.session_id] = 0
        self.backend.request_count_map[self.session_id] += 1
    
    def __enter__(self):
        """进入上下文"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时自动释放"""
        with self.backend.request_count_limit_lock:
            if self.session_id in self.backend.request_count_map:
                self.backend.request_count_map[self.session_id] -= 1
                if self.backend.request_count_map[self.session_id] <= 0:
                    del self.backend.request_count_map[self.session_id]
        return False  # 不抑制异常

class FluentLlmBackendException(Exception):
    def __init__(self, message, data=None):
        super().__init__(message)
        self.status = message
        self.data = data

class FluentLlmBackend:

    def __init__(self, backend_params: dict):
        mllm_infer_llm_backend_params = backend_params.pop("mllm_infer_llm_backend_params", {})
        request_cache_config = mllm_infer_llm_backend_params.pop("request_cache_config", {"request_cache_size": 0})
        
        cli_args = dict_to_cli_args(backend_params)
        logger.info(f"\033[35m[============单机启动: {cli_args=}============]\033[0m")
        
        server_args: ServerArgs = prepare_server_args(cli_args)
        self.server_args = server_args
        with open(os.path.join(server_args.model_path, "config.json"), 'r', encoding='utf-8') as file:
            self.model_config = json.load(file)
        
        self.interrupt_need_wait = mllm_infer_llm_backend_params.get("interrupt_need_wait", False)
        
        self.request_cache = None
        if request_cache_config["request_cache_size"] > 0:
            server_args.request_cache_size = request_cache_config["request_cache_size"]
            request_cache_config_str = json.dumps(request_cache_config)
            request_cache_config_str = request_cache_config_str.replace('${NMMINFER_MODEL_HIDDEN_SIZE}', f"{self.model_config.get('hidden_size', 0)}")
            server_args.request_cache_config = request_cache_config_str
            os.environ["REQUEST_CACHE_SUFFIX"] = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.request_cache = RequestCache.get_instance(self.server_args, gpu_id=-1, run_device="cpu", create=True)

        self.engine = Engine(server_args=server_args)
        self.request_count_limit_lock = TimeoutLock("request_count_limit_lock")
        self.request_count_map = {}
        self.request_count_limit = self.server_args.max_running_requests * 2
        if self.request_cache is not None:
            self.request_count_limit = min(self.request_count_limit, request_cache_config["request_cache_size"])
        


    def _acquire_request_cache_limit_lock(self, session_id):
        with self.request_count_limit_lock:
            if len(self.request_count_map) > self.request_count_limit:
                return None
            return RequstCountLimitLock(self, session_id)

    def _match_round_info(self, cache_info, all_round_info_raw, check_input=True, check_ouput=True, update_round_info=False):
        # import pdb; pdb.set_trace()
        all_round_info = []
        for round_info_item in all_round_info_raw:
            check_hash_id = round_info_item[0]
            input_token_len = round_info_item[1]
            output_token_len = round_info_item[2]
            role = round_info_item[3] if len(round_info_item) > 3 else "user"
            all_round_info.append((hash_to_int(f"{check_hash_id}", "int64"), input_token_len, output_token_len, role))
        # 这里乘 2 是因为 round_info 中可能存在无效数据（total_token=0），所以需要加大 round_info 取值区间，尽可能
        # 多的查找能否匹配。
        round_info = cache_info.get_round_info(2 * len(all_round_info))
        input_round_idx = 0
        prefix_sum_len = 0
        matched_cache_round_idx = 0
        for (round_id, hash_id, input_tokens, output_tokens, back_tokens) in round_info:
            if input_round_idx >= len(all_round_info):
                break
            total_token = input_tokens + output_tokens + back_tokens
            if total_token == 0:
                # round_info 的值可能是 [0, -3783325983370997026, 6, 0, -6] 这种，这种 cache 表示已经废弃了，
                # 此时真正能匹配的 round_info 值是比 all_round_info 少的，会导致实际上能匹配到的场景，但是因为对
                # 取到的 cache 中的 round_info 数量太少（len(all_round_info）限制)，导致没能匹配上。
                # 真正要解决，应该把 [0, -3783325983370997026, 6, 0, -6] 这种占着 round_info 位置，但是实际上
                # 没有 cache 的数据清除掉，保证 round_info 中的数据都是可用的。
                logger.warning(f"Useless data with total_token=0 appeared in round_info")
                continue
            (check_hash_id, input_token_len, output_token_len, role) = all_round_info[input_round_idx]
            if hash_id == check_hash_id and \
               (input_tokens == input_token_len if check_input else True) and \
               (output_tokens == output_token_len if check_ouput else True):
                if update_round_info:
                    all_round_info[input_round_idx] = (check_hash_id, input_tokens, output_tokens, role)
                input_round_idx += 1
                prefix_sum_len += total_token
                matched_cache_round_idx += 1
                
                if output_tokens > 0:
                    while input_round_idx < len(all_round_info):
                        next_item = all_round_info[input_round_idx]
                        next_role = next_item[3]
                        if next_role == "assistant":
                            input_round_idx += 1
                        else:
                            break
                continue
            else:
                break
        logger.info(f"_match_round_info: round_info:{round_info} all_round_info:{all_round_info}, input_round_idx:{input_round_idx}, matched_cache_round_idx:{matched_cache_round_idx}, prefix_sum_len:{prefix_sum_len}")
        return all_round_info, prefix_sum_len, input_round_idx, matched_cache_round_idx

    def match_round_info(
        self,
        session_id,
        all_round_info, #[(hash_id, input_token_len, output_token_len)]
        check_input=True,
        check_ouput=True,
        update_round_info=False,
    ):
        session_id_hash_id, cache_info = self.request_cache.get(session_id, get_without_create = True)
        if cache_info is None:
            return [], 0, 0, 0
        with cache_info._lock:
            if cache_info.stop:
                return [], 0, 0, 0
            return self._match_round_info(cache_info, all_round_info, check_input=check_input, check_ouput=check_ouput, update_round_info=update_round_info)

    async def generate(
        self,
        session_id,
        input_ids,
        input_tensor_dict: Optional[Dict[str, torch.Tensor]] = {},
        input_tensor_dict_free_func: Optional[Callable] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        input_extra_infos: Optional[Dict[str, Any]] = None,
        stream=True,
        step: int = None,
        new_round_info=None, #[(hash_id, input_token_len, output_token_len)]
        new_round_id=None,
        all_round_info=None, #[(hash_id, input_token_len, output_token_len)]
        need_decode: bool = True,
        chunk_size=128,
    ):
        cache_info = None
        interrupt_finish_event = None
        running_rids = []
        try:
            if not need_decode:
                sampling_params["max_new_tokens"] = 1
            cur_time = time.time()
            if self.request_cache is None:
                async_gen = await self.engine.async_generate(
                    input_ids=input_ids,
                    sampling_params=sampling_params,
                    input_extra_infos=input_extra_infos,
                    stream=stream,
                    rid=session_id,
                )
                last_time = time.time()
                async for output in async_gen:
                    yield output
                    cur_time = time.time()
                    last_time = cur_time
            else:
                max_new_tokens = sampling_params["max_new_tokens"]
                if max_new_tokens > self.server_args.request_max_output_len:
                    f'max_new_tokens not support {max_new_tokens} > {self.server_args.request_max_output_len}'
                    raise FluentLlmBackendException("error", 
                        {
                            "__backend_msg": f'max_new_tokens not support {max_new_tokens} > {self.server_args.request_max_output_len}'
                        }
                    )
                if step is not None and step >= 0:
                    if step + max_new_tokens > self.server_args.request_max_input_len:
                        raise FluentLlmBackendException("error", 
                            {
                                "__backend_msg":  f'step not support {step} + {max_new_tokens} > {self.server_args.request_max_input_len}'
                            }
                        )
                request_count_lock = self._acquire_request_cache_limit_lock(session_id)
                if request_count_lock is None:
                    raise FluentLlmBackendException("error", 
                        {
                            "__backend_msg": "too many requests"
                        }
                    )
                def check_cache_status(session_id, cache_info, session_id_hash_id, cur_round_id, step):
                    if cache_info.stop:
                        # not delete any more, use lru delete
                        # running_rids.extend(self.request_cache.delete(session_id, cache_info, free_input=True, free_output=True))
                        raise FluentLlmBackendException("error", 
                            {
                                "step": step,
                                "task_abort": True,
                                "__backend_msg": "task stoped for cache info stop",
                            }
                        )
                    cur_round_id_new = cache_info.get_round_id()
                    cur_session_id_new = cache_info.get_session_id()
                    if cur_round_id != cur_round_id_new or session_id_hash_id != cur_session_id_new:
                        logger.warning(f"{session_id} {session_id_hash_id} round changed {cur_round_id} -> {cur_round_id_new} {session_id_hash_id}->{cur_session_id_new}")
                        # not delete any more, use lru delete
                        # self.request_cache.delete(session_id, cache_info, free_input=False, free_output=True, cur_round_id=cur_round_id)
                        # not abort let sglang engine write to radix tree?
                        raise FluentLlmBackendException("error", 
                            {
                                "step": step,
                                "task_abort": True,
                                "__backend_msg": "task round changed or abort",
                            }
                        )
                async def interruptible_generator(gen, interrupt_event, interrupt_finish_event, step):
                    next_val_task = None
                    interrupt_wait_task = None
                    try:
                        iterator = gen.__aiter__()
                        # 在循环外创建 interrupt_wait_task，避免重复创建
                        interrupt_wait_task = asyncio.create_task(interrupt_event.wait())
                        while True:
                            if next_val_task is None:
                                next_val_task = asyncio.create_task(iterator.__anext__())
                            
                            wait_tasks = [next_val_task, interrupt_wait_task]
                            
                            done, pending = await asyncio.wait(
                                wait_tasks,
                                return_when=asyncio.FIRST_COMPLETED
                            )
                            
                            if interrupt_wait_task in done:
                                next_val_task.cancel()
                                try:
                                    await next_val_task
                                except asyncio.CancelledError:
                                    pass
                                except Exception:
                                    pass
                                logger.debug(f"interruptible_generator {session_id} interrupt_finish_event.set()")
                                interrupt_finish_event.set()
                                raise FluentLlmBackendException("error", {
                                    "step": step,
                                    "task_abort": True,
                                    "__backend_msg": "task interrupted by user",
                                })
                                
                            if next_val_task in done:
                                try:
                                    yield next_val_task.result()
                                    next_val_task = None
                                except StopAsyncIteration:
                                    break
                                except Exception as e:
                                    raise e
                    finally:
                        if next_val_task and not next_val_task.done():
                            next_val_task.cancel()
                        if interrupt_wait_task and not interrupt_wait_task.done():
                            interrupt_wait_task.cancel()

                
                new_input_ids_len = len(input_ids)
                cache_offset = -1
                with request_count_lock:
                    await self.interrupt(session_id, wait_generate_return=False)
                    session_id_hash_id, cache_info = self.request_cache.get(session_id)
                    with cache_info._lock:
                        interrupt_event, interrupt_finish_event = asyncio.Event(), asyncio.Event()
                        cache_info.interrupt_event = interrupt_event
                        cache_info.interrupt_finish_event = interrupt_finish_event
                        if cache_info.stop:
                            # not delete any more, use lru delete
                            # running_rids.extend(self.request_cache.delete(session_id, cache_info, free_input=True, free_output=True))
                            raise FluentLlmBackendException("error", 
                                {
                                    "__backend_msg": "task stoped for cache info stop",
                                }
                            )
                        # import pdb; pdb.set_trace()
                        if all_round_info is not None:
                            assert step is None
                            assert new_round_info is None
                            all_round_info, prefix_sum_len, input_round_idx, matched_cache_round_idx = self._match_round_info(cache_info, all_round_info)
                            step_check = prefix_sum_len
                            cache_info.set_round_id(matched_cache_round_idx)
                            new_round_info = all_round_info[input_round_idx:]
                            # logger.info(f"match new_round_info: {new_round_info}, input_round_idx:{input_round_idx}")
                        else:
                            if new_round_info is None:
                                new_round_info = [(None, new_input_ids_len, 0, "user")]
                            assert len(new_round_info) > 0
                            prefix_sum_len = 0
                            step_check = step
                            if new_round_id is not None:
                                cache_info.set_round_id(new_round_id)
                        prefill_len = step_check

                        for input_round_idx, round_info_item in enumerate(new_round_info):
                            hash_id = round_info_item[0]
                            input_token_len = round_info_item[1]
                            output_token_len = round_info_item[2]
                            if hash_id is None or hash_id == 0 or hash_id == hash_to_int("None", "int64"):
                                hash_id = hash_to_int(f"{session_id}_{cur_time*1000}", "int64")
                            if new_round_id is not None and input_round_idx == 0:
                                # 指定 new_round_id 后会调用 set_round_id 设置 round_id，第一次不需要自动累加 round_id 了
                                cur_round_id = new_round_id
                            else:
                                _, cur_round_id = cache_info.increase_round_id()
                            token_len = input_token_len + output_token_len
                            input_tensor_dict_new = {}
                            for k, v in input_tensor_dict.items():
                                input_tensor_dict_new[k] = v[prefix_sum_len:prefix_sum_len+token_len]
                            prefill_len, step_check = cache_info.append_input(input_ids[prefix_sum_len:prefix_sum_len+token_len], 
                                                                    max_new_tokens if input_round_idx == len(new_round_info) - 1 else 0, 
                                                                    input_tensor_dict=input_tensor_dict_new, 
                                                                    step_check=step_check, 
                                                                    kv_paged_base_str=f"{hash_id}")
                            if prefill_len is None:
                                raise FluentLlmBackendException("error", 
                                    {
                                        "__backend_msg": "append_input error"
                                    }
                                )
                            cache_info.update_round_info(cur_round_id, hash_id=hash_id, 
                                                        input_tokens=input_token_len,
                                                        output_tokens=output_token_len,
                                                        step_check=step_check, step_check_round=cur_round_id-1)
                            if step_check is not None:
                                step_check += token_len
                            prefix_sum_len += token_len
                        
                        if input_tensor_dict_free_func is not None:
                            input_tensor_dict_free_func(input_tensor_dict)
                            input_tensor_dict_free_func = None
                        if prefill_len + max_new_tokens > self.server_args.request_max_input_len:
                            cache_info.stop = True
                            # not delete any more, use lru delete
                            # running_rids.extend(self.request_cache.delete(session_id, cache_info, free_input=True, free_output=True))
                            logger.info(f'prefill_len too long not support {prefill_len} + {max_new_tokens} > {self.server_args.request_max_input_len}')
                            raise FluentLlmBackendException("error", 
                                {
                                    "__backend_msg": "input too long"
                                }
                            )
                        # logger.info(f"{session_id} {session_id_hash_id} generate: {len(input_ids)} {len(fill_input_ids)} {fill_input_ids}") # {input_ids} {fill_input_ids}
                        if not need_decode:
                            if prefill_len - cache_info.last_prefill_step < chunk_size:
                                raise FluentLlmBackendException("success", 
                                    {
                                        "step": prefill_len,
                                        "prefill_down": True,
                                        "__backend_msg": "prefill_down type 1",
                                    }
                                )
                            cache_info.last_prefill_step = prefill_len
                        new_request_id = cache_info.build_current_rid(session_id_hash_id, session_id, prefill_len)
                        running_rids.append(new_request_id)
                        # cache_info.add_running_req(cur_round_id, [new_request_id])
                        cache_offset = cache_info.get_input_buf_idx()
                    dp_room = 0
                    bootstrap_room = hash_to_int(new_request_id, "uint64")
                    
                    decode_start_step = prefill_len
                    
                    async_gen = await self.engine.async_generate(
                        input_ids=[0, 0],
                        sampling_params=sampling_params,
                        input_extra_infos=input_extra_infos,
                        stream=stream,
                        rid=new_request_id,
                        bootstrap_room=bootstrap_room,
                        bootstrap_host=None,
                        bootstrap_port=None,
                        data_parallel_rank=dp_room,
                    )

                    last_output_time = time.time()
                    async for output in interruptible_generator(async_gen, interrupt_event, interrupt_finish_event, decode_start_step):
                        output_list = []
                    
                        if not need_decode:
                            raise FluentLlmBackendException("success", 
                                {
                                    "step": prefill_len,
                                    "prefill_down": True,
                                    "__backend_msg": "prefill_down type 2",
                                }
                            )
                        # logger.info(f"{session_id} {session_id_hash_id} llm generate: output:{output}")
                        with cache_info._lock:
                            check_cache_status(session_id, cache_info, session_id_hash_id, cur_round_id, decode_start_step)
                            finish_reasion_type = ""
                            finish_reasion_msg = ""
                            if output["meta_info"]["finish_reason"] is not None:
                                finish_reasion_type = output["meta_info"]["finish_reason"]["type"]
                                if "message" in output["meta_info"]["finish_reason"]:
                                    finish_reasion_msg = output["meta_info"]["finish_reason"]["message"]
                            if finish_reasion_type == "abort" and finish_reasion_msg.startswith("[REQUEST_CACHE_ERROR]"):  
                                raise FluentLlmBackendException("error", 
                                    {
                                        "step": decode_start_step,
                                        "task_abort": True,
                                        "__backend_msg": "task round changed or abort",
                                    }
                                )

                            completion_tokens = output["meta_info"]["completion_tokens"]
                            # output_ids = output["output_ids"]
                            # if decode_start_step == prefill_len:
                            #     output_ids = output_ids[-completion_tokens:]
                            # logger.info(f"generate {session_id} {session_id_hash_id} out:{output} {output_ids} {cache_info.input_cache_idx} {cache_info.output_cache_idx}")
                            # output_len = len(output_ids)
                            output_len = completion_tokens - (decode_start_step - prefill_len)
                            try:
                                output_tensor_dict = cache_info.copy_output_to_input(output, decode_start_step-prefill_len, decode_start_step, output_len, None)
                            except Exception as e:
                                logger.error(f"[Decode] copy_output_to_input failed: {e}", exc_info=True)
                                raise FluentLlmBackendException("error", 
                                {
                                    "step": decode_start_step,
                                    "__backend_msg": f"copy_output_to_input error: {e}"
                                }
                            )
                            cache_info.update_round_info(cur_round_id, output_tokens=decode_start_step-prefill_len+output_len)
                            # not delete any more, use lru delete
                            # if finish_reasion_type != "":
                            #     self.request_cache.delete(session_id, cache_info, free_input=False, free_output=True, cur_round_id=cur_round_id)
                            if output["meta_info"]["prompt_tokens"] != prefill_len:
                                logger.warning(f"session:{session_id} prompt_tokens != prefill_len, {prefill_len}, {output}")
                            # round_info = None
                            # round_info = cache_info.get_round_info(cur_round_id + 1)
                            # logger.info(f"show round_info:{round_info}")
                            output_ids = output_tensor_dict.pop("__system_input_ids").view(-1).tolist()
                            if output_ids[-1] != output["output_ids"][-1]:
                                logger.warning(f"session:{session_id} output_ids dismatch, {output_ids}, {output}")
                            output_extra_info = output.get("output_extra_info", {})

                            for output_idx in range(output_len):
                                output_tensor_dict_part = {}
                                for k, v in output_tensor_dict.items():
                                    output_tensor_dict_part[k] = v[output_idx:output_idx+1]
                                output_extra_info_cur = get_full_output_infos_from_resp(output_extra_info, decode_start_step-prefill_len, decode_start_step-prefill_len+1, len_1_return_dict=True)
                                decode_start_step += 1
                                output_dict = {
                                    "text": output.get("text", ""),
                                    "output_tensor_dict": output_tensor_dict_part,
                                    "step": decode_start_step,
                                    # "round_info": round_info,
                                    "output_ids": output_ids[output_idx:output_idx+1],
                                    # "output_multi_ids": output["output_multi_ids"], # 应该从output_tensor_dict_part取
                                    "meta_info" : {
                                        "completion_tokens": decode_start_step,
                                        "prompt_tokens": output["meta_info"]["prompt_tokens"],
                                        "cached_tokens": output["meta_info"]["cached_tokens"],
                                        "finish_reason": output["meta_info"]["finish_reason"] if output_idx == output_len - 1 else None,
                                    },
                                    "output_extra_info": output_extra_info_cur,
                                }
                                output_list.append(output_dict)
                            cache_info.last_prefill_step = decode_start_step
                            # logger.info(f"{session_id} {session_id_hash_id} generate: output:{output} {output_dict}")
                            # output={'text': '\nA. It means that the Cloud is more efficient.\nB. It means that the Cloud is more secure.\nC. It means that the Cloud is more accessible.\nD. It means that the Cloud is more reliable.\nE. It means that the Cloud is more powerful.\nF. It means that the Cloud is more sustainable.\nG. It means that the Cloud is more efficient.\nH. It means that the Cloud is more secure.\nI. It means that the Cloud is more accessible.\nJ. It means that the Cloud is more reliable.\nK. It means that the', 'output_ids': [254], 'meta_info': {'id': 'SESSION::8196614605186928355::IN_CACHE_IDX::1::OUT_CACHE_IDX::26::ROUND::45::PRFILL::377::RID::[826644]', 'finish_reason': None, 'prompt_tokens': 377, 'completion_tokens': 127, 'cached_tokens': 0}, 'output_extra_info': {'decode_prefix_len': 0}}
                        cur_time = time.time()
                        last_output_time = cur_time
                        for part_out in output_list:
                            yield part_out
        except FluentLlmBackendException as e:
            if e.status == "error":
                logger.info(f"FluentLlmBackendException: {session_id} {e} {e.data}")
                raise RuntimeError(e.data)
            else:
                yield e.data
                return
        except Exception as e:
            logger.info(f"FluentLlmBackend Exception {e}", exc_info=True)
            raise FluentLlmBackendException("error", 
                {
                    "__backend_msg": "task stoped for exception: " + str(e),
                }
            )
        finally:
            if interrupt_finish_event is not None:
                interrupt_finish_event.set()
            for rid in running_rids:
                self.engine.tokenizer_manager.abort_request(rid)
            if input_tensor_dict_free_func is not None:
                input_tensor_dict_free_func(input_tensor_dict)
                input_tensor_dict_free_func = None
            # if cache_info is not None:
            #     running_rids = self.request_cache.delete(session_id, cache_info, free_input=False, free_output=True)
            #     for rid in running_rids:
            #         self.engine.tokenizer_manager.abort_request(rid)
            #     # del cache_info
            #     cache_info = None

    async def get_embedding(self, text_list=None, input_ids_list=None):
        # embedding_lookup_pre_token = []
        # if self.embedding_lookup_add_pre_token_len and prefill_len > 0:
        #     embedding_lookup_pre_token = cache_info.get_input_ids(max(0, prefill_len - self.embedding_lookup_add_pre_token_len), prefill_len)
        # aux_info = {"multi_ids": multi_ids}
        cur_time = time.time()
        tmp_cache_session_id = f"{cur_time}_{randint(-2**63, 2**63-1)}"
        tmp_cache_name = EMBEDDING_CACHE_KEY
        tensor_buffer_dict, tmp_cache_cache_idx, tmp_cache_max_len = self.request_cache.get_tmp_cache(tmp_cache_name, tmp_cache_session_id)

        rid = self.request_cache.embedding_lookup_build_rid(0, tmp_cache_cache_idx, len(input_ids_list))
        ret = await self.engine.embedding_lookup(rid, input_ids_list=input_ids_list, text_list=text_list)
        logger.info(f"ret_list: {ret}")
        def relase_callback():
            self.request_cache.delete_tmp_cache(tmp_cache_name, tmp_cache_session_id)
        if len(ret) == 0 or "tensor_len" not in ret or len(ret["tensor_len"]) == 0:
            relase_callback()
            raise RuntimeError(f"embedding_lookup error")
        tensor_list = []
        tensor_buffer = tensor_buffer_dict[EMBEDDING_SUB_KEY]
        pre_tensor_len = 0
        for tensor_len in ret["tensor_len"]:
            tensor_list.append(tensor_buffer[pre_tensor_len:pre_tensor_len + tensor_len])
            pre_tensor_len += tensor_len
        return tensor_list, relase_callback

                
    def abort(self, session_id):
        # logger.info(f"session_id:{session_id} call abort")
        if self.request_cache is not None:
            session_id_hash_id, cache_info = self.request_cache.get(session_id, get_without_create = True)
            if cache_info is None:
                return
            # logger.info(f"session_id:{session_id} session_id_hash_id:{session_id_hash_id} abort")
            with cache_info._lock:
                logger.info(f"session_id:{session_id} session_id_hash_id:{session_id_hash_id} abort in lock")
                if cache_info.stop:
                    return
                cache_info.stop = True
                cache_info.increase_round_id()
            #     running_rids = self.request_cache.delete(session_id, cache_info, free_input=True, free_output=True)
            # for rid in running_rids:
            #     self.engine.tokenizer_manager.abort_request(rid)
        else:
            self.engine.tokenizer_manager.abort_request(session_id)


    async def interrupt(self, session_id, wait_generate_return=None):
        # logger.info(f"session_id:{session_id} call interrupt")
        if self.request_cache is not None:
            session_id_hash_id, cache_info = self.request_cache.get(session_id, get_without_create = True)
            if cache_info is None:
                return
            # logger.info(f"session_id:{session_id} session_id_hash_id:{session_id_hash_id} interrupt")
            interrupt_finish_event = None
            with cache_info._lock:
                logger.info(f"session_id:{session_id} session_id_hash_id:{session_id_hash_id} interrupt in lock")
                if cache_info.stop:
                    return
                cache_info.increase_round_id()
                # running_rids = self.request_cache.delete(session_id, cache_info, free_input=False, free_output=True)
                if cache_info.interrupt_event is not None:
                    cache_info.interrupt_event.set()
                    logger.debug(f"interrupt interrupt_event.set() AAAA {id(cache_info.interrupt_event)}")
                if wait_generate_return is not None:
                    if wait_generate_return:
                        interrupt_finish_event = cache_info.interrupt_finish_event
                elif self.interrupt_need_wait:
                    interrupt_finish_event = cache_info.interrupt_finish_event
            # not abort let sglang engine write to radix tree
            # for rid in running_rids:
            #     self.engine.tokenizer_manager.abort_request(rid)
            if interrupt_finish_event is not None:
                await interrupt_finish_event.wait()
                time.sleep(0.002)
                logger.debug(f"interrupt interrupt_finish_event.wait() BBBB ")
        else:
            self.engine.tokenizer_manager.abort_request(session_id)
            
            
def get_full_output_infos_from_resp(output_extra_info, start_idx, end_idx, len_1_return_dict=True):
    full_outputs_info = output_extra_info.get("full_outputs_info", {})
    if len(full_outputs_info) == 0:
        return {}
    output_infos_name_idx_map = full_outputs_info["name_idx_map"]
    assert "output_idx" in output_infos_name_idx_map and output_infos_name_idx_map["output_idx"] == 0, f"output_infos_name_idx_map check output_idx error, {output_extra_info=}"

    full_outputs_info_value = full_outputs_info["value"]
    assert full_outputs_info_value[0][0] <= start_idx and full_outputs_info_value[-1][0] >= end_idx - 1, f"get_full_output_infos_from_resp check output_idx_list error: {output_extra_info=} {start_idx=} {end_idx=}"

    output_idx_list_start_idx = 0
    for i, info_value in enumerate(full_outputs_info_value):
        if info_value[0] == start_idx:
            output_idx_list_start_idx = i
            break
    lenth = end_idx - start_idx
    output_idx_list_end_idx = output_idx_list_start_idx + lenth
    assert full_outputs_info_value[output_idx_list_end_idx - 1][0] == end_idx - 1, f"get_full_output_infos_from_resp check output_idx_list error2: {output_extra_info=} {start_idx=} {end_idx=}"
    ret_dict = {}
    for name, i in output_infos_name_idx_map.items():
        ret_dict[name] = []
    for name, i in output_infos_name_idx_map.items():
        for idx in range(output_idx_list_start_idx, output_idx_list_end_idx):
            ret_dict[name].append(full_outputs_info_value[idx][i])
    if lenth == 1 and len_1_return_dict:
        for name, i in output_infos_name_idx_map.items():
            ret_dict[name] = ret_dict[name][0]
    return ret_dict
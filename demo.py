
# Standard Library
import argparse
import asyncio
import json
import os
import random
import re
import traceback
from typing import Dict, List, Optional

# Third Party
import numpy as np
import torch
import yaml
from sglang.srt.utils import random_uuid

# Local
from framework.fluentllm import FluentLlmBackend
from modules.special_token import get_spt, init_spt
from processor.postprocessor import PostProcessor
from processor.preprocessor import PreProcessor
from utils.async_utils import sync_run_async


# ==============================================================================
# Utility Functions
# ==============================================================================

def seed_everything(seed: int) -> int:
    print(f"\033[31m[============ Seed Everything {seed=} ============]\033[0m")
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


# ==============================================================================
# Main Ensemble Class
# ==============================================================================

class NmmEnsemble:
    """多模态推理集成类
    
    整合 LLM 后端、预处理器和后处理器，提供端到端的多模态推理能力。
    
    Attributes:
        lm: FluentLLM 后端实例
        encoder: 预处理器实例
        decoder: 后处理器实例
        multi_save: 多模态输出缓存
    """
    
    def __init__(self, configs: Dict):
        # 初始化 LLM 后端
        try:
            if configs["backend_params"]["model-path"] == "${NMM_INFER_MODEL_ROOT}":
                configs["backend_params"]["model-path"] = os.getenv("NMM_INFER_MODEL_ROOT")
            self.lm = FluentLlmBackend(configs["backend_params"])
        except Exception as e:
            print(f"初始化 LLM 后端失败: {e}")
            raise
        
        # 初始化特殊 token
        init_spt(os.path.join(configs["backend_params"]["model-path"], "nmm_infer"))
        
        # 初始化预处理器（带 embedding 查询回调）
        oe_delegate_fn = lambda input_ids: sync_run_async(
            self.lm.get_embedding(input_ids_list=input_ids)
        )
        self.encoder = PreProcessor(
            configs=configs["multimodal_params"],
            rank=0,
            oe_delegate_fn=oe_delegate_fn
        )
        
        # 初始化后处理器
        self.decoder = PostProcessor(
            configs=configs["multimodal_params"],
            encoder=self.encoder
        )
        
        # 多模态输出缓存
        self.multi_save: Dict[str, torch.Tensor] = {}
        
        print("NmmEnsemble 初始化完成!")

    def need_to_generate_multi(self, input_ids: Optional[torch.Tensor] = None) -> tuple:
        gen_audio = False
        gen_image = False
        
        if input_ids is not None:
            last_token = input_ids.flatten()[-1].item()
            if last_token == get_spt().IMAGE_START_TOKEN_ID:
                gen_image = True
            elif last_token == get_spt().AUDIO_GEN_START_TOKEN_ID:
                gen_audio = True
        
        return gen_audio, gen_image

    async def generate_new(
        self,
        raw_input: Dict,
        request_id: Optional[str] = None,
        sampling_params: Optional[Dict] = None,
        multi_sampling_params: Optional[Dict] = None,
        delay: int = 0,
        token_w: Optional[int] = None,
        file_path: str = "",
        enable_gen_multi: bool = True,
        input_extra_infos: Optional[Dict] = None,
        step: int = 0
    ) -> tuple:
        if sampling_params is None:
            sampling_params = {}
        if multi_sampling_params is None:
            multi_sampling_params = {}
        if input_extra_infos is None:
            input_extra_infos = {}
        
        # 预处理：编码输入
        emb, input_ids = self.encoder.process_from_raw_input_new(raw_input)
        emb: torch.Tensor = emb.reshape(-1, emb.shape[-1])
        
        orig_input_ids = self.encoder.tokenizer.encode(
            raw_input["question"],
            add_special_tokens=False,
            return_tensors='pt'
        )[0]
        
        # 判断多模态生成类型
        gen_audio, gen_image = self.need_to_generate_multi(input_ids=input_ids)
        input_ids = input_ids.view(-1).tolist()
        
        # 构建额外输入信息
        _input_extra_infos = {
            "gen_image": gen_image,
            "gen_audio": gen_audio,
            "orig_input_ids": orig_input_ids,
            "delay": delay,
            "multi_sampling_params": multi_sampling_params,
            "token_w": token_w,
        }
        input_extra_infos.update(_input_extra_infos)
        
        # 流式生成
        output_ids = []
        output_multi_ids = []

        async for res in self.lm.generate(
            request_id,
            input_ids,
            input_tensor_dict={"input_embedding": emb},
            sampling_params=sampling_params,
            input_extra_infos=input_extra_infos,
            stream=True,
            step=step
        ):
            output_ids.extend(res["output_ids"])
            multi_ids = res.get("output_tensor_dict", {}).get("output_multi_ids", None)
            if multi_ids is not None:
                output_multi_ids.append(multi_ids.squeeze().tolist())
        
        # 解码文本
        text = self.encoder.tokenizer.decode(output_ids[:-1], skip_special_tokens=True)
        # print(f"生成文本: {text}")
        # print(f"多模态输出: {output_multi_ids}")
        
        # 处理多模态输出
        multi_data = torch.tensor(output_multi_ids)
        
        if gen_audio:
            # 音频生成：按分隔符分段
            multi_data = torch.cat(
                (multi_data, torch.tensor([[8192] * 8])),
                dim=0
            )
            split_indices = (multi_data[:, 0] < 0).nonzero(as_tuple=True)[0]
            
            start = 0
            segments = []
            for idx in split_indices:
                if start < idx:
                    segments.append(multi_data[start:idx])
                start = idx + 1
            if start < len(multi_data):
                segments.append(multi_data[start:])
            multi_data = segments
            
        elif gen_image:
            # 图像生成：过滤无效行
            multi_data = multi_data[multi_data[:, 0] >= 0]
        
        # 后处理：生成最终输出文件
        file_name = request_id.split("-")[-1]
        if file_path:
            file_name = file_path.rsplit('.', maxsplit=1)[0]
        
        # 确定生成的文件名
        generated_file = None
        if enable_gen_multi and gen_image or gen_audio:
            self.decoder.decode_multi(
                multi_data,
                file_name,
                gen_image,
                gen_audio,
                token_w,token_w
            )
            if gen_image:
                generated_file = f"{file_name}.png"
            elif gen_audio:
                generated_file = f"{file_name}.wav"
        
        return text, output_multi_ids, res, generated_file


# ==============================================================================
# Test Cases
# ==============================================================================

def _create_uncond_case(cond_case: Dict) -> Dict:
    question = cond_case["question"]
    # 提取 img_token_size 和 img_start 作为无条件输入
    match = re.search(r'(<longcat_img_token_size>\d+ \d+</longcat_img_token_size><longcat_img_start>)', question)
    uncond_question = match.group(1) if match else '<longcat_img_start>'
    
    return {
        "question": uncond_question,
        "task": cond_case.get("task"),
        "token_w": cond_case.get("token_w"),
        "cfg_scale": cond_case.get("cfg_scale"),
        "cfg_role": "uncond",
        "request_id": cond_case.get("request_id"),
        "sampling_params": cond_case.get("sampling_params", {}),
    }


# 任务类型常量
TASK_IMG_GEN = "img_gen"           # Image Generation
TASK_IMG_UND = "img_und"           # Image Understanding
TASK_AUD_2_TXT = "aud_2_txt"       # Audio-to-Text
TASK_SPK_SYN = "spk_syn"           # Speech Synthesis
TASK_AUD_2_AUD = "aud_2_aud"       # Audio-to-Audio

ALL_TASKS = [
    TASK_IMG_GEN,
    TASK_IMG_UND,
    TASK_AUD_2_TXT,
    TASK_SPK_SYN,
    TASK_AUD_2_AUD,
]

# 测试用例配置文件路径
TEST_CASES_FILE = os.path.join(os.path.dirname(__file__), "example", "test_cases.yaml")


def load_test_cases() -> Dict[str, Dict]:
    """从 YAML 文件加载测试用例配置"""
    with open(TEST_CASES_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_test_cases_by_task(task: str, test_cases_config: Dict[str, Dict]) -> List[Dict]:
    """根据任务类型获取对应的测试用例"""
    if task in test_cases_config:
        case = test_cases_config[task].copy()
        # 处理 YAML 中的多行字符串，去除多余空白
        if "question" in case:
            case["question"] = " ".join(case["question"].split())
        return [case]
    return []


def get_test_cases(tasks: Optional[List[str]] = None) -> List[Dict]:
    """获取指定任务类型的测试用例
    
    Args:
        tasks: 任务类型列表，为 None 时返回全部用例
    
    Returns:
        测试用例列表
    """
    test_cases_config = load_test_cases()
    
    if tasks is None:
        tasks = ALL_TASKS
    
    test_cases = []
    for task in tasks:
        test_cases.extend(get_test_cases_by_task(task, test_cases_config))
    
    return test_cases


# ==============================================================================
# Test Runner
# ==============================================================================

async def run_test(
    ensemble: NmmEnsemble,
    is_sequential: bool = True,
    output_dir: str = "output",
    tasks: Optional[List[str]] = None,
) -> None:
    test_cases = get_test_cases(tasks)

    # 默认采样参数
    sampling_params = {
        "temperature": 0.5,
        "max_new_tokens": 2048,
        "top_p": 0.85,
        "top_k": 5,
        "repetition_penalty": 1.3,
        "ignore_eos": False,
        "stream_interval": 1,
    }
    
    multi_sampling_params = {
        "temperature": 0.2,
        "top_p": 0.85,
        "top_k": 20,
        "repetition_penalty": 1.1,
    }

    async def process_case(case: Dict, index: int) -> Optional[Dict]:
        """处理单个测试用例"""
        # 结果变量
        text = ""
        generated_file = None
        error_msg = None
        
        # CFG 相关配置
        cfg_scale = case.get("cfg_scale", None)
        cfg_role = case.get("cfg_role", None)
        
        try:
            request_id = case.get("request_id", random_uuid() + f"-case{index}")
            delay = case.get("delay", float('inf'))
            token_w = case.get("token_w", None)
            enable_gen_multi = True
            
            # 合并采样参数：case 提供的参数覆盖默认参数
            case_sampling_params = sampling_params.copy()
            if "sampling_params" in case:
                case_sampling_params.update(case["sampling_params"])
            if token_w:
                case_sampling_params["max_new_tokens"] += (token_w - 1)
            
            # 合并多模态采样参数：case 提供的参数覆盖默认参数
            case_multi_sampling_params = multi_sampling_params.copy()
            if "multi_sampling_params" in case:
                case_multi_sampling_params.update(case["multi_sampling_params"])
            
            input_extra_infos = {}
            
            if cfg_scale:
                case_sampling_params["custom_params"] = {
                    "cfg_scale": cfg_scale,
                    "cfg_pair_id": f"{request_id}pair",
                    "cfg_role": cfg_role,
                }
                request_id += cfg_role
                
                if cfg_role == 'uncond':
                    enable_gen_multi = False
                    
                input_extra_infos = {
                    "group_name": "CFG_test",
                    "group_size": 2,
                }
            
            # 构建输出文件路径前缀
            file_path = os.path.join(output_dir, f"case{index}")
            
            # 执行生成
            text, output_multi_ids, output, generated_file = await ensemble.generate_new(
                raw_input=case,
                request_id=request_id,
                sampling_params=case_sampling_params,
                multi_sampling_params=case_multi_sampling_params,
                delay=delay,
                token_w=token_w,
                file_path=file_path,
                enable_gen_multi=enable_gen_multi,
                input_extra_infos=input_extra_infos,
            )
            print(f"测试案例 {index}: {text}")
            
        except Exception as e:
            print(f"测试案例 {index} 失败: {e}")
            print(traceback.format_exc())
        finally:
            # CFG uncond 分支不记录结果
            if cfg_scale and cfg_role == 'uncond':
                return None
            
            # 构建结果字典
            result = {
                "input": case.get("question", ""),
                "output_text": text,
                "generated_file": generated_file,
            }
            return result

    # 展开测试用例：为 CFG 条件分支自动创建无条件分支
    expanded_cases = []
    for case in test_cases:
        expanded_cases.append(case)
        # 如果是 CFG 条件分支，自动创建无条件分支
        if case.get("cfg_scale") and case.get("cfg_role") == "cond":
            uncond_case = _create_uncond_case(case)
            expanded_cases.append(uncond_case)

    # 执行模式
    results = []
    if is_sequential:
        # 顺序执行，但 CFG cond/uncond 对必须一起执行
        i = 0
        while i < len(expanded_cases):
            case = expanded_cases[i]
            next_case = expanded_cases[i + 1] if i + 1 < len(expanded_cases) else None
            
            # 检查是否为 CFG 对（当前是 cond，下一个是 uncond）
            is_cfg_pair = (
                case.get("cfg_scale") and case.get("cfg_role") == "cond" and
                next_case and next_case.get("cfg_role") == "uncond"
            )
            
            if is_cfg_pair:
                # CFG 对一起并发执行
                batch_results = await asyncio.gather(
                    process_case(case, i),
                    process_case(next_case, i + 1)
                )
                results.extend([r for r in batch_results if r is not None])
                i += 2
            else:
                # 非 CFG 用例单独执行
                result = await process_case(case, i)
                if result is not None:
                    results.append(result)
                i += 1
    else:
        # 并发执行（批量）
        batch_size = 8
        for i in range(0, len(expanded_cases), batch_size):
            batch_cases = expanded_cases[i:i + batch_size]
            batch_results = await asyncio.gather(
                *(process_case(case, j) for j, case in enumerate(batch_cases, start=i))
            )
            results.extend([r for r in batch_results if r is not None])  # 过滤掉 CFG uncond 分支
    
    # 写入结果 JSON 文件
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存至: {json_path}")


# ==============================================================================
# Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NMM 多模态推理测试脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-s", "--sequential",
        action="store_true",
        help="顺序执行测试用例（默认并发执行）"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="output",
        help="文件输出目录（默认为 output）"
    )
    parser.add_argument(
        "-t", "--tasks",
        type=str,
        nargs="+",
        choices=ALL_TASKS,
        default=None,
        help=(
            "指定要执行的任务类型，可选: "
            "img_gen(Image Generation), "
            "img_und(Image Understanding), "
            "aud_2_txt(Audio-to-Text), "
            "spk_syn(Speech Synthesis), "
            "aud_2_aud(Audio-to-Audio)。"
            "默认执行全部任务"
        )
    )
    parser.add_argument(
        "-m", "--model-path",
        type=str,
        required=True,
        help="模型路径，将设置为 NMM_INFER_MODEL_ROOT 环境变量"
    )
    args = parser.parse_args()

    # 检查模型路径是否存在
    if not os.path.isdir(args.model_path):
        raise ValueError(f"模型路径不存在: {args.model_path}")
    
    # 设置模型路径环境变量
    os.environ["NMM_INFER_MODEL_ROOT"] = args.model_path

    # 加载配置
    yaml_path = 'nmm_pf.yaml'
    
    with open(yaml_path, "r", encoding="utf8") as f:
        configs = yaml.safe_load(f)
    
    # 初始化模型
    ensemble = NmmEnsemble(configs)

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行测试
    asyncio.run(
        run_test(
            ensemble=ensemble,
            is_sequential=args.sequential,
            output_dir=args.output_dir,
            tasks=args.tasks,
        )
    )


if __name__ == "__main__":
    seed_everything(42)
    main()

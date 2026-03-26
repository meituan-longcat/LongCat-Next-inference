from sglang.srt.utils import (
    get_colorful_logger,
)
import subprocess
import os
import time
import yaml
import re
import json
from typing import Dict, Any, Union
from datetime import datetime

logger = get_colorful_logger(__name__)

def exec_cmd(command: str, verbose=False) -> str:
    if verbose:
        logger.info(f"\033[35m[Run: [{command}]]\033[0m")
    try:
        result = subprocess.run(
            command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        result = result.stdout.strip()
        return result

    except subprocess.CalledProcessError as e:
        error_msg = f"{e.returncode=}: {e.stderr.strip()}"
        raise RuntimeError(error_msg) from e


_MLLMINFER_CODE_ROOT = None


def get_mllminfer_root():
    global _MLLMINFER_CODE_ROOT
    if _MLLMINFER_CODE_ROOT:
        return _MLLMINFER_CODE_ROOT

    current_file_path = os.path.abspath(__file__)
    _MLLMINFER_CODE_ROOT = os.path.abspath(os.path.join(current_file_path, "../../../../"))
    print(f"\033[34m[{_MLLMINFER_CODE_ROOT=}]\033[0m")
    return _MLLMINFER_CODE_ROOT


def get_mllminfer_rel_path(relative_path):
    root_path = get_mllminfer_root()
    abs_path = os.path.join(root_path, relative_path)
    abs_path = os.path.abspath(abs_path)
    return abs_path


def replace_in_string(text: str, replacements: Dict[str, str]) -> str:
    def replace_match(match):
        var_name = match.group(1)
        old_value = match.group(0)
        new_value = replacements.get(var_name, old_value)
        logger.info(f"\033[34m[{old_value} -> {new_value}]\033[0m")
        return new_value

    pattern = r"\$\{([^}]+)\}"
    result = re.sub(pattern, replace_match, text)
    return result


def replace_in_nested_dict(data: Any, replacements: Dict[str, str]) -> Any:
    if isinstance(data, dict):
        return {key: replace_in_nested_dict(value, replacements) for key, value in data.items()}
    elif isinstance(data, list):
        return [replace_in_nested_dict(item, replacements) for item in data]
    elif isinstance(data, str):
        return replace_in_string(data, replacements)
    else:
        return data


def replace_yaml_with_envs(input_path, output_path) -> str:
    with open(input_path, "r", encoding="utf8") as f:
        data = yaml.safe_load(f)
    replacements = dict(os.environ)
    processed_data = replace_in_nested_dict(data, replacements)
    with open(output_path, "w", encoding="utf8") as f:
        yaml.dump(processed_data, f)
    return processed_data


def replace_json_with_envs(input_path, output_path) -> str:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    replacements = dict(os.environ)
    processed_data = replace_in_nested_dict(data, replacements)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    return processed_data


def get_timestamp(us=False):
    if us:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    else:
        return time.strftime("%Y%m%d_%H%M%S", time.localtime())

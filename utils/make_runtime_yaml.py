import os
import sys
import argparse
import yaml
import json
from utils.misc_utils import get_timestamp, replace_yaml_with_envs, replace_json_with_envs


def make_runtime_yaml(raw_yaml_path, output_dir="/tmp"):
    temp_dir = os.path.join(output_dir, f"mllminfer_runtime_{get_timestamp()}")
    os.makedirs(temp_dir, exist_ok=True)
    runtime_yaml_path = os.path.join(temp_dir, "main.yaml")
    runtime_configs = replace_yaml_with_envs(raw_yaml_path, runtime_yaml_path)

    if "json-model-override-args" in runtime_configs["backend_params"]:
        json_config = runtime_configs["backend_params"]["json-model-override-args"]
        if "omni_architectures" in json_config:
            json_config["omni_architectures"] = [json_config["omni_architectures"]]
        runtime_configs["backend_params"]["json-model-override-args"] = json.dumps(runtime_configs["backend_params"]["json-model-override-args"])

    
    if "multimodal_params" in runtime_configs and "processor-file" in runtime_configs["multimodal_params"]:
        runtime_audio_processor_path = os.path.join(temp_dir, "audio_preprocessor_config.json")
        input_audio_processor_path = runtime_configs["multimodal_params"]["processor-file"]
        real_audio_processor_configs = replace_json_with_envs(input_audio_processor_path, runtime_audio_processor_path)
        runtime_configs["multimodal_params"]["processor-file"] = runtime_audio_processor_path

    with open(runtime_yaml_path, "w", encoding="utf8") as f:
        yaml.dump(runtime_configs, f)

    return runtime_yaml_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process main YAML file and generate runtime YAML files.")
    parser.add_argument("--raw_yaml_path", required=True, help="Path to the main YAML file.")
    parser.add_argument("--output_dir", default="/tmp", help="Output directory for generated files. Default is /tmp.")

    args = parser.parse_args()

    raw_yaml_path = args.raw_yaml_path
    output_dir = args.output_dir

    output_main_path = make_runtime_yaml(raw_yaml_path, output_dir)
    print(f"{output_main_path}")

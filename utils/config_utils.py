import json

def dict_to_cli_args(params_dict):
    cli_args = []
    for arg_name, arg_value in params_dict.items():
        if arg_value is None:
            cli_args.append(f"--{arg_name}")
        else:
            if isinstance(arg_value, list):
                cli_args.extend([f"--{arg_name}"])
                cli_args.extend([str(_) for _ in arg_value])
            elif isinstance(arg_value, dict):
                # 字典类型需要转换为 JSON 字符串，确保使用双引号
                cli_args.extend([f"--{arg_name}", json.dumps(arg_value)])
            else:
                cli_args.extend([f"--{arg_name}", str(arg_value)])
    return cli_args

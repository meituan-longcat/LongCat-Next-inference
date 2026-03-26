#!/bin/bash

CURRENT_FILE="$(realpath "${BASH_SOURCE[0]}")"
CURRENT_DIR="$(dirname "$CURRENT_FILE")"
export MLLMINFER_CODE_ROOT="$(readlink -f "$CURRENT_DIR/")"

# ============ FluentLLM ============
export PYTHONPATH=${MLLMINFER_CODE_ROOT}/:$PYTHONPATH

export FLUENTLLM_HOME="${MLLMINFER_CODE_ROOT}/fluentllm"
export PYTHONPATH=${FLUENTLLM_HOME}/python:${PYTHONPATH}
export PYTHONPATH="${FLUENTLLM_HOME}/3rdparty/eps/python/":$PYTHONPATH

export LD_PRELOAD=/usr/lib64/libcuda.so

# ============自定义变量============
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# export NCCL_DEBUG=WARN
# export EPS_DEBUG_LEVEL="VERBOSE"
export SGLANG_BLOCK_NONZERO_RANK_CHILDREN=0
export SGLANG_ENABLE_TORCH_COMPILE=0
export SYNC_TOKEN_IDS_ACROSS_TP=1
export SGLANG_SET_CPU_AFFINITY=0
export TOKENIZERS_PARALLELISM=false



# ============ 其他操作 ============
rm -rf /dev/shm/*

echo -e "\033[32m============ mllminfer 环境变量配置完成. ============\033[0m"

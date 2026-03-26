
CURRENT_FILE="$(realpath "${BASH_SOURCE[0]}")"
CURRENT_DIR="$(dirname "$CURRENT_FILE")"
export MLLMINFER_CODE_ROOT="$(readlink -f "$CURRENT_DIR/../../../../")"

# ============ FluentLLM ============
export PYTHONPATH=${MLLMINFER_CODE_ROOT}/:$PYTHONPATH
export EPS_HOME="/home/fluentllm/3rdparty/eps"
export PYTHONPATH=${EPS_HOME}/python/:$PYTHONPATH

export FLUENTLLM_HOME="${MLLMINFER_CODE_ROOT}/fluentllm"
export PYTHONPATH=${FLUENTLLM_HOME}/python:${PYTHONPATH}

export LD_PRELOAD=/usr/lib64/libcuda.so
# python3 processor/preprocessor.py
python3 processor/postprocessor.py
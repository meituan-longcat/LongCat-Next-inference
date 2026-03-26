export BASE_PYTHON="${HOME}/nmm-oss-env/bin/python3.11"
export USER_ENV_PATH="${HOME}/nmm-ext-env"

if [ ! -f "${BASE_PYTHON}" ]; then
    echo -e "基础环境不存在: ${BASE_PYTHON}"
    exit 1
fi

create_user_env() {
    echo -e "创建用户环境: ${USER_ENV_PATH}"
    uv venv "${USER_ENV_PATH}" \
        --python "${BASE_PYTHON}" \
        --system-site-packages
    echo -e "用户环境创建成功"
}

if [ -d "${USER_ENV_PATH}" ] && [ -f "${USER_ENV_PATH}/bin/activate" ]; then
    echo -e "\033[32m[============ 用户环境已存在: ${USER_ENV_PATH} ============]\033[0m"
else
    echo -e "\033[31m[============ 用户环境不存在，开始创建... ============]\033[0m"
    create_user_env
fi

source "${USER_ENV_PATH}/bin/activate"
CURRENT_PYTHON_VERSION=$(python --version 2>&1)
echo -e "\033[32m[============ 当前 Python 版本: ${CURRENT_PYTHON_VERSION} ============]\033[0m"
export PYTHONPATH="${HOME}/nmm-oss-env/lib/python3.11/site-packages:$PYTHONPATH"
export PYTHONPATH="${USER_ENV_PATH}/lib/python3.11/site-packages:$PYTHONPATH"

export PYTHONIOENCODING=utf-8
export LD_LIBRARY_PATH=/root/.local/share/uv/python/cpython-3.11.12-linux-x86_64-gnu/lib/:$LD_LIBRARY_PATH
# ============ sglang-FluentLLM ============

git submodule update --init --recursive
cd fluentllm

pip3 install uv==0.7.2
uv venv ${HOME}/nmm-oss-env --python 3.11 --seed

source ${HOME}/nmm-oss-env/bin/activate
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && . "$HOME/.cargo/env"

pip3 install -e "./python[cuda_sm90]" --no-cache-dir

pip install --upgrade certifi
export SSL_CERT_FILE=$(python3 -m certifi)

sh clean_setup.sh sm90

cd -
source create_env.sh
sh install.sh
chmod -R 777 ${HOME}/nmm-oss-env
chmod -R 777 ${HOME}/nmm-ext-env
# ============ nmm ============
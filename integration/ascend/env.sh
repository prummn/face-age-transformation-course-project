export B=/data/home/1120231322/scow/face-age
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || source /usr/local/Ascend/cann/set_env.sh
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then source /usr/local/Ascend/nnal/atb/set_env.sh; fi
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HOME=$B/hf_cache TORCH_HOME=$B/torch_cache
if [ -f $B/venv/bin/activate ]; then source $B/venv/bin/activate; fi

# face-age-web runtime env (Ascend CANN + own venv layered on face-age venv via .pth)
export W=/data/home/1120231322/scow/face-age-web
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || source /usr/local/Ascend/cann/set_env.sh
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then source /usr/local/Ascend/nnal/atb/set_env.sh; fi
export TORCH_HOME=/data/home/1120231322/scow/face-age/torch_cache
export GRADIO_ANALYTICS_ENABLED=False GRADIO_TEMP_DIR=$W/tmp HF_HUB_OFFLINE=1
source $W/venv/bin/activate

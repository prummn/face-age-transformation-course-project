#!/usr/bin/env bash
# Start only this integration service; leave training and the previous Gradio app intact.
set -euo pipefail
TASK_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$TASK_DIR"
mkdir -p logs
if [[ -f backend.pid ]]; then
  TASK_PID=$(cat backend.pid)
  if [[ "$TASK_PID" =~ ^[0-9]+$ ]] && kill -0 "$TASK_PID" 2>/dev/null &&
     tr '\0' ' ' < "/proc/$TASK_PID/cmdline" | grep -Fq "$TASK_DIR/backend.py"; then
    echo "整合服务已启动，PID=$TASK_PID"
    exit 0
  fi
fi
if (echo > /dev/tcp/127.0.0.1/8765) 2>/dev/null; then
  echo "端口 8765 已占用；请检查现有服务，未停止任何进程。" >&2
  exit 3
fi
# Existing CANN env files refer to unset variables; disable nounset while sourcing.
set +u
source /data/home/1120231322/scow/face-age-web/env.sh
set -u
if ! TASK_NPU_STATUS=$(npu-smi info 2>/dev/null) ||
   ! grep -q 'No running processes found in NPU 0 ' <<< "$TASK_NPU_STATUS"; then
  echo "NPU 0 正在使用或状态不可读；未抢占现有任务。" >&2
  exit 2
fi
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
nohup setsid python -u "$TASK_DIR/backend.py" --port 8765 > logs/backend.log 2>&1 < /dev/null &
echo $! > backend.pid
echo "整合服务正在加载，PID=$(cat backend.pid)，日志：$TASK_DIR/logs/backend.log"

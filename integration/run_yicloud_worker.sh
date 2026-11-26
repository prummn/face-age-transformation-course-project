#!/usr/bin/env bash
# Dedicated platform allocation; never install into the image's shared Python.
set -euo pipefail
repo_dir="$1"
run_dir="$2"
cd "$repo_dir"
exec >> "$run_dir/worker.log" 2>&1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export FACE_AGE_BASE="$repo_dir/deployment/model-files/face-age"
export FACE_AGE_DEVICE=cuda:0
export FACE_AGE_QUALITY="${FACE_AGE_QUALITY:-original}"
if [[ "${FACE_AGE_STABLE_NUMERICS:-0}" == 1 ]]; then
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
fi
export TORCH_HOME="$FACE_AGE_BASE/torch_cache"
export HF_HUB_OFFLINE=1
# Only the SHA256-verified, trusted legacy SAM checkpoint is loaded here.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
date -Is
nvidia-smi
python3 - "$run_dir" <<'PY'
import json, os, socket, sys, torch
from pathlib import Path
if not torch.cuda.is_available():
    raise RuntimeError('Platform allocation did not provide a CUDA GPU')
info = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0), memory_bytes=torch.cuda.get_device_properties(0).total_memory,
            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), hostname=socket.gethostname())
Path(sys.argv[1], 'device.json').write_text(json.dumps(info, indent=2))
print(json.dumps(info), flush=True)
PY
# Dependency staging and full SHA checks finish on the workspace before this flag.
for ((attempt=0; attempt<360; attempt++)); do
    [[ -f "$repo_dir/deployment/assets-ready.json" ]] && break
    sleep 5
done
test -f "$repo_dir/deployment/assets-ready.json"
python3 -m venv --system-site-packages /tmp/face-age-demo-venv
task_python=/tmp/face-age-demo-venv/bin/python
"$task_python" -m pip install --no-index --no-deps "$repo_dir"/deployment/wheels/*.whl
"$task_python" - "$run_dir" <<'PY'
import importlib.metadata, json, sys, subprocess
from pathlib import Path
names = ['torch','torchvision','timm','fastapi','uvicorn','numpy','Pillow','opencv-python-headless','pyarrow']
Path(sys.argv[1], 'packages.json').write_text(json.dumps({n:importlib.metadata.version(n) for n in names}, indent=2))
subprocess.run([sys.executable, '-m', 'pip', 'check'], check=False)
PY
ssh -N -T -p 18822 -i "$repo_dir/deployment/ssh/tunnel_ed25519" \
    -o BatchMode=yes -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$repo_dir/deployment/ssh/known_hosts" \
    -R 127.0.0.1:18765:127.0.0.1:8765 root@172.16.25.78 &
tunnel_pid=$!
"$task_python" -u "$repo_dir/integration/backend.py" --port 8765 >> "$run_dir/backend.log" 2>&1 &
backend_pid=$!
cleanup() { kill "$backend_pid" "$tunnel_pid" 2>/dev/null || true; }
trap cleanup EXIT TERM INT
# Either component ending makes the task terminate instead of leaking a GPU.
wait -n "$backend_pid" "$tunnel_pid"

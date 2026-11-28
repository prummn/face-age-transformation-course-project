#!/usr/bin/env bash
set -euo pipefail
repo="$1"
run="$2"
cd "$repo"
exec >> "$run/worker.log" 2>&1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export FACE_AGE_REPO="$repo" FACE_AGE_EXPERIMENT_ALLOCATION=1 HF_HUB_OFFLINE=1
export TORCH_HOME="$repo/deployment/model-files/face-age/torch_cache"
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -m venv --system-site-packages /tmp/face-age-sfe-conditioned
py=/tmp/face-age-sfe-conditioned/bin/python
"$py" -m pip install --no-index --no-deps "$repo"/deployment/wheels/{numpy-1.26.4-*,opencv_python_headless-4.11.0.86-*,scipy-1.15.3-*,lpips-0.1.4-*,timm-1.0.22-*,safetensors-0.6.2-*,huggingface_hub-0.36.0-*}.whl
"$py" "$run/source/sfe_conditioned_study.py" worker "$run"

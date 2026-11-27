#!/usr/bin/env bash
set -euo pipefail
repo="$1"
run="$2"
cd "$repo"
exec >> "$run/worker.log" 2>&1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export FACE_AGE_REPO="$repo" FACE_AGE_EXPERIMENT_ALLOCATION=1 HF_HUB_OFFLINE=1
if [[ "${FACE_AGE_EXPERIMENT_DEVICE:-cuda:0}" != "cpu" ]]; then
  nvidia-smi --query-gpu=name,memory.total --format=csv
fi
python3 -m venv --system-site-packages /tmp/face-age-sfe
py=/tmp/face-age-sfe/bin/python
"$py" -m pip install --no-index --no-deps "$repo"/deployment/wheels/opencv_python_headless-4.11.0.86-*.whl
"$py" "$run/source/sfe_study.py" worker "$run"

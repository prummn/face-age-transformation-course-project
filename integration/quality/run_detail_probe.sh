#!/usr/bin/env bash
set -euo pipefail
repo_dir="$1"
run_dir="$2"
previous_results="$3"
cd "$repo_dir"
exec >> "$run_dir/worker.log" 2>&1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export FACE_AGE_BASE="$repo_dir/deployment/model-files/face-age"
export FACE_AGE_DEVICE=cuda:0 HF_HUB_OFFLINE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
date -Is
nvidia-smi
python3 -m venv --system-site-packages /tmp/face-age-detail-venv
task_python=/tmp/face-age-detail-venv/bin/python
"$task_python" -m pip install --no-index --no-deps "$repo_dir"/deployment/wheels/*.whl
"$task_python" -u "$repo_dir/integration/quality/probe_inversion_detail.py" \
  --previous-results "$previous_results" --output "$run_dir/results" --steps 120

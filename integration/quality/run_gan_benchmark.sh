#!/usr/bin/env bash
set -euo pipefail
repo_dir="$1"
run_dir="$2"
mode="${3:-all}"
cd "$repo_dir"
exec >> "$run_dir/worker.log" 2>&1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export FACE_AGE_REPO="$repo_dir" FACE_AGE_BASE="$repo_dir/deployment/model-files/face-age"
export FACE_AGE_DEVICE=cuda:0 HF_HUB_OFFLINE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCH_HOME="$FACE_AGE_BASE/torch_cache"
date -Is
nvidia-smi
python3 -m venv --system-site-packages /tmp/face-age-gan-benchmark-venv
task_python=/tmp/face-age-gan-benchmark-venv/bin/python
"$task_python" -m pip install --no-index --no-deps "$repo_dir"/deployment/wheels/*.whl
if [[ "$mode" == all ]]; then
  "$task_python" -u "$run_dir/source/benchmark_gan.py" \
    --manifest "$run_dir/diagnostic-manifest.json" --output "$run_dir/sam" --steps 120
elif [[ "$mode" != cusp-only ]]; then
  echo "Unknown benchmark mode: $mode" >&2
  exit 2
fi
# Separate interpreter: model namespaces and CUDA allocations cannot collide.
"$task_python" -u "$run_dir/source/benchmark_gan.py" \
  --manifest "$run_dir/diagnostic-manifest.json" --output "$run_dir/cusp" --cusp

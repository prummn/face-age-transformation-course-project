#!/usr/bin/env bash
set -euo pipefail
repo="$1"
run="$2"
steps="$3"
cd "$repo"
exec >> "$run/worker.log" 2>&1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export FACE_AGE_REPO="$repo" FACE_AGE_BASE="$repo/deployment/model-files/face-age"
export FACE_AGE_DEVICE=cuda:0 FACE_AGE_EXPERIMENT_ALLOCATION=1 HF_HUB_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TORCH_HOME="$FACE_AGE_BASE/torch_cache"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
date -Is
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -m venv --system-site-packages /tmp/face-age-highres-texture
py=/tmp/face-age-highres-texture/bin/python
"$py" -m pip install --no-index --no-deps "$repo"/deployment/wheels/*.whl
"$py" - "$run" <<'PY'
import hashlib,json
from pathlib import Path
import sys
paths=json.loads((Path(sys.argv[1])/'experiment.json').read_text())['frozen_inputs']
for filename,expected in paths.items():
    digest=hashlib.sha256()
    with Path(filename).open('rb') as source:
        for block in iter(lambda:source.read(8*1024*1024),b''):digest.update(block)
    if digest.hexdigest()!=expected:raise ValueError('Frozen input changed: '+filename)
print('HASHES_VERIFIED',len(paths),flush=True)
PY
read -r min_resolution generator_lr adversarial_weight < <("$py" -c 'import json,sys;v=json.load(open(sys.argv[1]));print(v.get("min_resolution",256),v.get("generator_lr",2e-5),v.get("adversarial_weight",.02))' "$run/experiment.json")
"$py" "$run/source/train_highres_texture.py" "$run" --arm control --steps "$steps" --execute \
  --min-resolution "$min_resolution" --generator-lr "$generator_lr" --adversarial-weight "$adversarial_weight"
"$py" "$run/source/train_highres_texture.py" "$run" --arm patch_gan --steps "$steps" --execute \
  --min-resolution "$min_resolution" --generator-lr "$generator_lr" --adversarial-weight "$adversarial_weight"
"$py" "$run/source/evaluate_highres_texture.py" "$run"
date -Is
printf 'Both highres arms and native paired evaluation complete\n' > "$run/DONE"

#!/usr/bin/env bash
set -euo pipefail
repo="$1"
run="$2"
cd "$repo"
exec >> "$run/worker.log" 2>&1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export FACE_AGE_REPO="$repo" FACE_AGE_BASE="$repo/deployment/model-files/face-age"
export FACE_AGE_DEVICE=cuda:0 FACE_AGE_EXPERIMENT_ALLOCATION=1 HF_HUB_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TORCH_HOME="$FACE_AGE_BASE/torch_cache"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
date -Is
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -m venv --system-site-packages /tmp/face-age-saved-evaluation
py=/tmp/face-age-saved-evaluation/bin/python
"$py" -m pip install --no-index --no-deps "$repo"/deployment/wheels/*.whl
"$py" - "$run" <<'PY'
import hashlib,json
from pathlib import Path
import sys
paths=json.loads((Path(sys.argv[1])/'experiment.json').read_text())['frozen_inputs']
for filename,expected in paths.items():
    digest=hashlib.sha256()
    with Path(filename).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    if digest.hexdigest()!=expected:
        raise ValueError('Frozen input changed: '+filename)
print('HASHES_VERIFIED',len(paths),flush=True)
PY
"$py" "$run/source/evaluate_gan_training.py" "$run" --routes calibrated_encoder target_encoder \
  --native-cohort new-hd-holdout --native-cohort new_appa_asian --redetect-output
date -Is
printf 'Saved-checkpoint evaluation complete; training and W+ reused\n' > "$run/DONE"

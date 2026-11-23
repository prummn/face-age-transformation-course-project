#!/usr/bin/env bash
set -euo pipefail
repo="$1"
run="$2"
control="$3"
cd "$repo"
exec >> "$run/worker.log" 2>&1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export FACE_AGE_REPO="$repo" FACE_AGE_BASE="$repo/deployment/model-files/face-age"
export FACE_AGE_DEVICE=cuda:0 FACE_AGE_EXPERIMENT_ALLOCATION=1 HF_HUB_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TORCH_HOME="$FACE_AGE_BASE/torch_cache"
date -Is
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -m venv --system-site-packages /tmp/face-age-source-age
py=/tmp/face-age-source-age/bin/python
"$py" -m pip install --no-index --no-deps "$repo"/deployment/wheels/*.whl
"$py" "$run/source/train_gan_reconstruction.py" --train-manifest "$run/manifests/train.jsonl" \
  --eval-manifest "$run/manifests/eval.jsonl" --output "$run/calibrated_encoder" --arm encoder \
  --steps 1000 --batch-size 2 --seed 20261123 --lr 2e-5 --same-id-weight 1 \
  --target-id-weight .5 --pair-conditions-by-stratum --execute
"$py" - "$run" "$control" <<'PY'
import json
from pathlib import Path
import sys

run, control = map(Path, sys.argv[1:])
def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

a = rows(control/'expanded_encoder/train.jsonl')
b = rows(run/'calibrated_encoder/train.jsonl')
old = rows(control/'manifests/train.jsonl')
new = rows(run/'manifests/train.jsonl')
assert len(a) == len(b) == 1000 and len(old) == len(new) == 1182
assert all(x == y | {'source_age':x['source_age']} for x,y in zip(old,new))
changed = 0
for x,y in zip(a,b):
    assert all(x[k] == y[k] for k in ('step','indices','target','flips','strata','collections'))
    assert all(abs(value*100-old[i]['source_age']) < 1e-4 for value,i in zip(x['source'],x['indices']))
    assert all(abs(value*100-new[i]['source_age']) < 1e-4 for value,i in zip(y['source'],y['indices']))
    changed += sum(s != t for s,t in zip(x['source'],y['source']))
assert changed
(run/'PAIRED_SAMPLING_VERIFIED').write_text(f'1000 steps, 2000 identical draws, {changed} changed ages\n')
PY
"$py" "$run/source/evaluate_gan_training.py" "$run" --routes calibrated_encoder \
  --native-cohort new_appa_asian --redetect-output
printf 'Calibration arm and paired evaluation complete\n' > "$run/DONE"

#!/usr/bin/env bash
set -euo pipefail
repo_dir="$1"
run_dir="$2"
data_dir="$3"
cd "$repo_dir"
exec >> "$run_dir/worker.log" 2>&1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export FACE_AGE_REPO="$repo_dir" FACE_AGE_BASE="$repo_dir/deployment/model-files/face-age"
export FACE_AGE_DEVICE=cuda:0 FACE_AGE_EXPERIMENT_ALLOCATION=1
export HF_HUB_OFFLINE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TORCH_HOME="$FACE_AGE_BASE/torch_cache"
date -Is
nvidia-smi
python3 -m venv --system-site-packages /tmp/face-age-paired-training-venv
task_python=/tmp/face-age-paired-training-venv/bin/python
"$task_python" -m pip install --no-index --no-deps "$repo_dir"/deployment/wheels/*.whl
if [[ ! -f "$run_dir/manifests/receipt.json" ]]; then
  "$task_python" -u "$run_dir/source/annotate_gan_training.py" "$data_dir" "$run_dir/manifests"
fi
for arm in encoder encoder_generator; do
  "$task_python" -u "$run_dir/source/train_gan_reconstruction.py" \
    --train-manifest "$run_dir/manifests/train.jsonl" --eval-manifest "$run_dir/manifests/eval.jsonl" \
    --output "$run_dir/$arm" --arm "$arm" --steps 100 --batch-size 2 --seed 20261113 --execute
done
"$task_python" -u "$run_dir/source/evaluate_gan_training.py" "$run_dir"
"$task_python" - "$run_dir" <<'PY'
import json,sys
from pathlib import Path
run=Path(sys.argv[1])
a=[json.loads(x) for x in (run/'encoder/train.jsonl').read_text().splitlines()]
b=[json.loads(x) for x in (run/'encoder_generator/train.jsonl').read_text().splitlines()]
assert len(a)==len(b)==100
assert all(all(x[k]==y[k] for k in ['indices','flips','source','target']) for x,y in zip(a,b))
(run/'PAIRED_SAMPLING_VERIFIED').write_text('100 identical batches/flips/conditions verified\n')
(run/'DONE').write_text('Training and evaluation complete; no online promotion\n')
print('EXPERIMENT_DONE',flush=True)
PY

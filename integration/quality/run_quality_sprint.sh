#!/usr/bin/env bash
set -euo pipefail
repo_dir="$1"
run_dir="$2"
cd "$repo_dir"
exec >> "$run_dir/worker.log" 2>&1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export FACE_AGE_REPO="$repo_dir" FACE_AGE_BASE="$repo_dir/deployment/model-files/face-age"
export FACE_AGE_DEVICE=cuda:0 FACE_AGE_EXPERIMENT_ALLOCATION=1
export HF_HUB_OFFLINE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TORCH_HOME="$FACE_AGE_BASE/torch_cache"
date -Is
nvidia-smi
python3 -m venv --system-site-packages /tmp/face-age-quality-sprint-venv
task_python=/tmp/face-age-quality-sprint-venv/bin/python
"$task_python" -m pip install --no-index --no-deps "$repo_dir"/deployment/wheels/*.whl
"$task_python" - "$run_dir" <<'PY'
import json,sys,torch
from pathlib import Path
assert torch.cuda.device_count()==1
p=torch.cuda.get_device_properties(0)
Path(sys.argv[1],'device.json').write_text(json.dumps(dict(device=p.name,total_bytes=p.total_memory,torch=torch.__version__,cuda=torch.version.cuda),indent=2)+'\n')
PY
"$task_python" -u "$run_dir/source/refine_identity_latents.py" "$run_dir"
for arm in recon_encoder recon_generator; do
  scope=encoder
  if [[ "$arm" == recon_generator ]]; then scope=encoder_generator; fi
  "$task_python" -u "$run_dir/source/train_gan_reconstruction.py" \
    --train-manifest "$run_dir/manifests/train.jsonl" --eval-manifest "$run_dir/manifests/eval.jsonl" \
    --output "$run_dir/$arm" --arm "$scope" --steps 800 --batch-size 2 --seed 20261113 \
    --lr 0.00002 --same-id-weight 1.0 --target-id-weight 0.5 \
    --generator-scope mid_convs --generator-lr 0.00001 --execute
done
"$task_python" - "$run_dir" <<'PY'
import json,sys
from pathlib import Path
run=Path(sys.argv[1])
logs=[[json.loads(x) for x in (run/a/'train.jsonl').read_text().splitlines()] for a in ['recon_encoder','recon_generator']]
assert all(len(log)==800 for log in logs)
assert all(all(a[k]==b[k] for k in ['indices','flips','source','target']) for a,b in zip(*logs))
(run/'PAIRED_SAMPLING_VERIFIED').write_text('800 identical batches/flips/conditions verified\n')
PY
"$task_python" -u "$run_dir/source/evaluate_gan_training.py" "$run_dir" --routes fine_direct previous_identity recon_encoder recon_generator
"$task_python" - "$run_dir" <<'PY'
import json,sys
from pathlib import Path
run=Path(sys.argv[1]);report=json.loads((run/'evaluation/report.json').read_text())
assert len(report['records'])==104 and len(report['outputs'])==2080
(run/'DONE').write_text('Quality sprint complete; GPU exits; no automatic promotion\n')
print('QUALITY_SPRINT_DONE',flush=True)
PY

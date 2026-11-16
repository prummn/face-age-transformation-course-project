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
python3 -m venv --system-site-packages /tmp/face-age-identity-pilot-venv
task_python=/tmp/face-age-identity-pilot-venv/bin/python
"$task_python" -m pip install --no-index --no-deps "$repo_dir"/deployment/wheels/*.whl
"$task_python" - "$run_dir" <<'PY'
import json,sys,torch
from pathlib import Path
assert torch.cuda.device_count()==1, 'One visible GPU required'
props=torch.cuda.get_device_properties(0)
Path(sys.argv[1],'device.json').write_text(json.dumps(dict(device=props.name,total_bytes=props.total_memory,torch=torch.__version__,cuda=torch.version.cuda),indent=2)+'\n')
PY
for arm in id_control id_strong; do
  coefficient=0.1
  if [[ "$arm" == id_strong ]]; then coefficient=0.5; fi
  "$task_python" -u "$run_dir/source/train_gan_reconstruction.py" \
    --train-manifest "$run_dir/manifests/train.jsonl" --eval-manifest "$run_dir/manifests/eval.jsonl" \
    --output "$run_dir/$arm" --arm encoder --steps 500 --batch-size 2 --seed 20261113 \
    --target-id-weight "$coefficient" --execute
done
"$task_python" - "$run_dir" <<'PY'
import json,sys
from pathlib import Path
run=Path(sys.argv[1])
arms=[[json.loads(x) for x in (run/a/'train.jsonl').read_text().splitlines()] for a in ['id_control','id_strong']]
assert all(len(x)==500 for x in arms)
assert all(all(a[k]==b[k] for k in ['indices','flips','source','target']) for a,b in zip(*arms))
(run/'PAIRED_SAMPLING_VERIFIED').write_text('500 identical batches/flips/conditions verified\n')
PY
"$task_python" -u "$run_dir/source/evaluate_gan_training.py" "$run_dir" --routes fine_direct id_control id_strong
"$task_python" - "$run_dir" <<'PY'
import json,sys
from pathlib import Path
run=Path(sys.argv[1]);report=json.loads((run/'evaluation/report.json').read_text())
assert len(report['records'])==104 and len(report['outputs'])==1560
assert all(x['sface_cosine'] is not None for x in report['outputs'])
(run/'DONE').write_text('Identity ablation complete; GPU task exits; no online promotion\n')
print('IDENTITY_PILOT_DONE',flush=True)
PY

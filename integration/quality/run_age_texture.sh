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
python3 -m venv --system-site-packages /tmp/face-age-age-texture-venv
task_python=/tmp/face-age-age-texture-venv/bin/python
"$task_python" -m pip install --no-index --no-deps "$repo_dir"/deployment/wheels/*.whl
"$task_python" -u "$run_dir/source/refine_age_texture.py" "$run_dir"
"$task_python" - "$run_dir" <<'PY'
import json,sys
from pathlib import Path
run=Path(sys.argv[1]);r=json.loads((run/'age-texture/report.json').read_text())
assert len(r['records'])==12 and len(r['outputs'])==240
(run/'DONE').write_text('Age/texture experiment complete; GPU exits; accepted baseline unchanged\n')
print('AGE_TEXTURE_DONE',flush=True)
PY

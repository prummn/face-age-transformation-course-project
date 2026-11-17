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
python3 -m venv --system-site-packages /tmp/face-age-texture-validation-venv
task_python=/tmp/face-age-texture-validation-venv/bin/python
"$task_python" -m pip install --no-index --no-deps "$repo_dir"/deployment/wheels/*.whl
"$task_python" -u "$run_dir/source/validate_texture_quality.py" "$run_dir"
export FACE_AGE_QUALITY="$("$task_python" -c 'import json,sys;print(json.load(open(sys.argv[1]))["candidate_mode"])' "$run_dir/experiment.json")"
"$task_python" -u "$run_dir/source/backend.py" --port 8765 >> "$run_dir/quality-backend.log" 2>&1 &
backend_pid=$!
trap 'kill "$backend_pid" 2>/dev/null || true' EXIT
"$task_python" -u "$run_dir/source/verify_quality_backend.py" "$run_dir"
kill "$backend_pid"
wait "$backend_pid" || true
trap - EXIT
"$task_python" - "$run_dir" <<'PY'
import json,sys
from pathlib import Path
run=Path(sys.argv[1]);r=json.loads((run/'age-texture/report.json').read_text())
assert len(r['records'])==11 and len(r['outputs'])==110
(run/'DONE').write_text('Texture candidate additional cases and real API verified; GPU exits\n')
print('TEXTURE_VALIDATION_DONE',flush=True)
PY

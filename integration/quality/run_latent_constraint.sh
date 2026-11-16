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
python3 -m venv --system-site-packages /tmp/face-age-latent-constraint-venv
task_python=/tmp/face-age-latent-constraint-venv/bin/python
"$task_python" -m pip install --no-index --no-deps "$repo_dir"/deployment/wheels/*.whl
previous_run="$("$task_python" -c 'import json,sys;print(json.load(open(sys.argv[1]))["previous_run"])' "$run_dir/experiment.json")"
"$task_python" -u "$run_dir/source/refine_identity_latents.py" "$run_dir" --cached-run "$previous_run" --edit-steps 180
export FACE_AGE_QUALITY=constrained
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
run=Path(sys.argv[1]);report=json.loads((run/'latent-constrained/report.json').read_text())
assert len(report['records'])==12 and len(report['outputs'])==300
(run/'DONE').write_text('Constrained per-input latent refinement done; GPU exits\n')
print('LATENT_CONSTRAINT_DONE',flush=True)
PY

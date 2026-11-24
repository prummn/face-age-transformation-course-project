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
python3 -m venv --system-site-packages /tmp/face-age-matched-latent
py=/tmp/face-age-matched-latent/bin/python
"$py" -m pip install --no-index --no-deps "$repo"/deployment/wheels/*.whl
"$py" - "$run" <<'PY'
import hashlib,json,sys
from pathlib import Path
for filename,expected in json.loads((Path(sys.argv[1])/'experiment.json').read_text())['frozen_inputs'].items():
    h=hashlib.sha256()
    with Path(filename).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    if h.hexdigest()!=expected:raise ValueError('Frozen input changed: '+filename)
print('FROZEN_INPUTS_VERIFIED',flush=True)
PY
"$py" "$run/source/matched_latent_study.py" fit "$run"
"$py" "$run/source/train_highres_texture.py" "$run" --arm control --steps 1000 --execute --min-resolution 64 --generator-lr 5e-5 --adversarial-weight 0 --no-input-flip --native-reference-input --latent-source encoder --output-name direct_latent
"$py" "$run/source/train_highres_texture.py" "$run" --arm control --steps 1000 --execute --min-resolution 64 --generator-lr 5e-5 --adversarial-weight 0 --no-input-flip --native-reference-input --latent-source wplus --output-name matched_wplus
"$py" "$run/source/evaluate_highres_texture.py" "$run"
date -Is
printf 'Matched latent paired experiment and fixed-code evaluation complete\n' > "$run/DONE"

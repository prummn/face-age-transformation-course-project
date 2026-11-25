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
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
python3 -m venv --system-site-packages /tmp/face-age-direction-match
py=/tmp/face-age-direction-match/bin/python
"$py" -m pip install --no-index --no-deps "$repo"/deployment/wheels/*.whl
"$py" "$run/source/original_delta_matched_study.py" verify "$run"
"$py" "$run/source/original_delta_matched_study.py" fit "$run"
common=(--arm patch_gan --execute --min-resolution 64 --generator-lr 5e-5 --adversarial-weight .1 --no-input-flip --native-reference-input --latent-source wplus --age-preservation .5 --patch-edge 256 --teacher-size 32 --teacher-weight .05)
# Probe both code schemas and GAN/R1 gradients, then restart each arm identically.
"$py" "$run/source/train_highres_texture.py" "$run" --steps 2 --output-name direct_latent --cache-dir train-codes-ft "${common[@]}"
"$py" "$run/source/train_highres_texture.py" "$run" --steps 2 --output-name matched_wplus --cache-dir train-codes-original "${common[@]}"
"$py" "$run/source/train_highres_texture.py" "$run" --steps 2000 --output-name control --cache-dir train-codes-ft "${common[@]}"
"$py" "$run/source/train_highres_texture.py" "$run" --steps 2000 --output-name patch_gan --cache-dir train-codes-original "${common[@]}"
"$py" "$run/source/evaluate_highres_texture.py" "$run"
date -Is
printf 'Direction-matched pair trained and35case evaluation complete\n' > "$run/DONE"

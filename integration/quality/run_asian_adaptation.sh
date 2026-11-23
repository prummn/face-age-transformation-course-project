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
date -Is
nvidia-smi
python3 -m venv --system-site-packages /tmp/face-age-asian-adaptation
py=/tmp/face-age-asian-adaptation/bin/python
"$py" -m pip install --no-index --no-deps "$repo"/deployment/wheels/*.whl
"$py" "$run/source/annotate_gan_training.py" "$run/data" "$run/manifests"
"$py" - "$run" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); names={json.loads(s)['name'] for s in (p/'data/control.jsonl').read_text().splitlines()}
lines=[s for s in (p/'manifests/train.jsonl').read_text().splitlines() if json.loads(s)['name'] in names]
assert len(names)==len(lines)
(p/'manifests/control.jsonl').write_text('\n'.join(lines)+'\n')
PY
for route in control_encoder expanded_encoder expanded_generator; do
  manifest="$run/manifests/train.jsonl"
  arm=encoder
  generator=()
  if [[ "$route" == control_encoder ]]; then manifest="$run/manifests/control.jsonl"; fi
  if [[ "$route" == expanded_generator ]]; then
    arm=encoder_generator
    generator=(--generator-scope mid_convs --generator-lr 1e-5)
  fi
  "$py" "$run/source/train_gan_reconstruction.py" --train-manifest "$manifest" \
    --eval-manifest "$run/manifests/eval.jsonl" --output "$run/$route" --arm "$arm" \
    --steps 1000 --batch-size 2 --seed 20261123 --lr 2e-5 --same-id-weight 1 \
    --target-id-weight .5 --pair-conditions-by-stratum --execute "${generator[@]}"
done
"$py" - "$run" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); arms=['control_encoder','expanded_encoder','expanded_generator']
a,b,c=[[json.loads(s) for s in (p/r/'train.jsonl').read_text().splitlines()] for r in arms]
assert len(a)==len(b)==len(c)==1000
assert all(all(x[k]==y[k]==z[k] for k in ['target','flips','strata']) for x,y,z in zip(a,b,c))
assert all(all(y[k]==z[k] for k in ['indices','source','collections']) for y,z in zip(b,c))
(p/'PAIRED_SAMPLING_VERIFIED').write_text('1000 steps; strata/flips/targets fixed; B/C identical inputs and source conditions\n')
PY
"$py" "$run/source/evaluate_gan_training.py" "$run" --routes mixed fine_direct control_encoder expanded_encoder expanded_generator \
  --native-cohort new_appa_asian --redetect-output
"$py" "$run/source/summarize_asian_adaptation.py" "$run"
printf 'Three paired arms and evaluation complete\n' > "$run/DONE"

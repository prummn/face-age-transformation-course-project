"""Submit one bounded GPU evaluation, preserving all trained weights and old outputs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from probe_inversion import sha256, write_json

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'integration'))
from deploy_yicloud import IMAGE,client


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    run=parser.parse_args().run.resolve()
    experiment=json.loads((run/'experiment.json').read_text())
    if experiment['profile']!='conditional_age_preserved_texture' or not (run/'DONE').is_file():raise ValueError('Completed conditional study required')
    for filename,digest in experiment['frozen_inputs'].items():
        if sha256(filename)!=digest:raise ValueError('Frozen study input changed: '+filename)
    job,ctx=client()
    active=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames='face-age',Self=True,Limit=100,Phases='Pending,Running,Inqueue,Starting,Stopping'))
    if active.Items:raise ValueError('One GPU limit: a project task is already active')
    recovery=run/'evaluation-recovery';recovery.mkdir(exist_ok=False);source=recovery/'source';source.mkdir()
    for name in ['evaluate_highres_texture.py','probe_inversion.py','deterministic_quality_ops.py','latent_quality.py','source_color.py']:
        original=Path(__file__).with_name(name) if name=='evaluate_highres_texture.py' else run/'source'/name
        shutil.copy2(original,source/name)
    for arm in ('control','patch_gan'):
        completed=json.loads((run/arm/'completed.json').read_text())
        if sha256(run/arm/'generator_patch.pt')!=completed['patch_sha256']:raise ValueError('Trained patch changed')
    write_json(recovery/'provenance.json',dict(reason='Evaluate all conditions with baseline batch5 to verify zero-gate PNG identity; prior batch1 outputs are retained',
        original_evaluation_sha256=sha256(run/'evaluation/report.json'),source_sha256={p.name:sha256(p) for p in source.iterdir()},
        training_repeated=False,created=datetime.now(timezone.utc).isoformat()))
    worker=recovery/'worker.sh'
    worker.write_text('''#!/usr/bin/env bash
set -euo pipefail
repo="$1"; run="$2"
cd "$repo"
exec >> "$run/evaluation-recovery/worker.log" 2>&1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export FACE_AGE_REPO="$repo" FACE_AGE_BASE="$repo/deployment/model-files/face-age"
export FACE_AGE_DEVICE=cuda:0 FACE_AGE_EXPERIMENT_ALLOCATION=1 HF_HUB_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TORCH_HOME="$FACE_AGE_BASE/torch_cache" CUBLAS_WORKSPACE_CONFIG=:4096:8
date -Is
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -m venv --system-site-packages /tmp/face-age-conditional-reeval
py=/tmp/face-age-conditional-reeval/bin/python
"$py" -m pip install --no-index --no-deps "$repo"/deployment/wheels/*.whl
"$py" "$run/evaluation-recovery/source/evaluate_highres_texture.py" "$run" --output-name evaluation-batch5
date -Is
printf 'Same-shape evaluation complete\\n' > "$run/evaluation-recovery/DONE"
''')
    name='face-age-conditional-reeval-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    request=dict(Name=name,Project='fdj-infra',ZoneId='cn-shanghai-a',QuotaGroup='fdj_gpu_taskpool',Priority='3',UsePrivateMachine=False,AutoDeleteDuration='168h',
        TaskSpecs={'eval':dict(Command=['bash',str(worker),str(ROOT),str(run)],Env={},Image=IMAGE,Replicas=1,CPU='4',Memory='48Gi',GPU='1',
        Internet=False,MaxRunningDuration='10m',MaxWaitDuration='10m',RestartPolicy='Never',MountGPFS='gpfs://gpfs2/haobinzhang:/root')})
    write_json(recovery/'request.json',request)
    specs={k:job.models.CreateJobReqTaskSpec(**v) for k,v in request['TaskSpecs'].items()}
    try:result=job.create_job(ctx,job.models.CreateJobReq(**(request|{'TaskSpecs':specs})))
    except Exception:
        found=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames=name,Self=True,Limit=100))
        exact=[x for x in found.Items or [] if x.Name==name]
        if not exact:raise
        result=exact[0]
    receipt=dict(job_id=result.JobId,name=name,run_dir=str(run),gpus=1,training_repeated=False)
    write_json(recovery/'receipt.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()

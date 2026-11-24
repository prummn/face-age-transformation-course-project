"""Freeze and submit a small SAM delta or pretrained-discriminator diagnostic."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from probe_inversion import sha256,write_json

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'integration'))
from deploy_yicloud import client,IMAGE


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('--submit',action='store_true');p.add_argument('--profile',choices=['prior','original-delta'],default='prior');p.add_argument('--scope',choices=['four','full'],default='four');a=p.parse_args();parent=a.parent.resolve()
    if a.scope=='full' and a.profile!='original-delta':p.error('Full regression is currently limited to the original-delta ablation')
    if not (parent/'DONE').exists() or not (parent/'evaluation-merge.json').exists():raise ValueError('Parent GPU task and regression merge must finish')
    data=json.loads((parent/'evaluation/report.json').read_text());byname={r['name']:r for r in data['records']}
    selection=list(byname.values()) if a.scope=='full' else [byname[n] for n in ('ffhq_64376','ffhq_64919','ffhq_69847','ffhq_60764')]
    if a.scope=='full' and len(selection)!=35:raise ValueError('Expected complete frozen 35-case parent regression')
    duration='35m' if a.scope=='full' else '15m'
    prior=a.profile=='prior';probe='probe_pretrained_realism.py' if prior else 'probe_original_delta.py'
    run=ROOT/'deployment/training-runs'/(('pretrained-prior-' if prior else 'original-delta-')+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'));(run/'source').mkdir(parents=True,exist_ok=False)
    write_json(run/'selection.json',selection)
    for name in ['pretrained_prior_study.py',probe,'train_native_age.py','probe_inversion.py','deterministic_quality_ops.py']:
        shutil.copy2(Path(__file__).with_name(name),run/'source'/name)
    shutil.copy2(ROOT/'integration/source_color.py',run/'source/source_color.py')
    paths=[*list((run/'source').iterdir()),run/'selection.json',parent/'evaluation/report.json']
    runtime=ROOT/'deployment/CUSP'
    if prior:
        paths.extend([ROOT/'deployment/gan-research/stylegan2ada_ffhq_1024.pkl',runtime/'legacy.py'])
        for folder in ('dnnlib','torch_utils'):paths.extend((runtime/folder).rglob('*.py'))
    paths.extend(ROOT/s['path'] for s in json.loads((ROOT/'integration/quality/baselines/sam-wplus-v1.json').read_text())['weights'])
    for row in selection:paths.extend([parent/'evaluation'/row['name']/'fixed_codes.pt',parent/'evaluation'/row['name']/'reference.png'])
    recipe=dict(layers='W+8:18',learning_rate=.01,weights=dict(identity=1,baseline_LPIPS=1,blurred_age_KL=.5,color=.1,delta_L2=.1,pretrained_D=.05),discriminator_frozen=True,generator_frozen=True,minibatch_std_group_size=1) if prior else dict(source_anchor='Identical frozen baseline W+',age_delta='Original SAM encoder E0(t)-E0(s)',raw_route_steps=0,constrained_route_steps=180,layers='W+0:8 correction',learning_rate=.012,weights=dict(identity=.8,age_KL=.5,LPIPS=.6,delta_L2=.05,color=.1),all_models_frozen=True)
    write_json(run/'experiment.json',dict(profile='frozen_pretrained_discriminator_prior' if prior else 'original_sam_delta_ablation',scope=a.scope,parent_run=str(parent),cases=len(selection),steps_per_arm=150 if prior else 180,batch=2 if prior else 4,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        discriminator_runtime_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=runtime,text=True).strip() if prior else None,frozen_inputs={str(path):sha256(path) for path in paths},
        criteria=dict(identity_drop_max=.02,age_span_min_ratio=.9,missing_faces_increase_max=0,native_review_required=True),
        recipe=recipe,
        caveats=['Previously inspected development/regression inputs, not independent generalization.','Optimization loss is not independent evidence of age accuracy or naturalness.','Target age values unchanged; no image blending or global model update.'],resources=dict(gpu=1,cpu=4,memory='32Gi',max_running=duration)))
    print('GAN_DIAGNOSTIC_PREPARED',run,flush=True)
    if not a.submit:return
    job,ctx=client();active=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames='face-age',Self=True,Limit=100,Phases='Pending,Running,Inqueue,Starting,Stopping'))
    if active.Items:raise ValueError('One GPU limit: existing project job is active; prepared inputs retained')
    worker=run/'source/worker.sh';worker.write_text('''#!/usr/bin/env bash
set -euo pipefail
repo="$1";run="$2";cd "$repo"
exec >> "$run/worker.log" 2>&1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export FACE_AGE_REPO="$repo" FACE_AGE_BASE="$repo/deployment/model-files/face-age" FACE_AGE_DEVICE=cuda:0 FACE_AGE_EXPERIMENT_ALLOCATION=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TORCH_HOME="$FACE_AGE_BASE/torch_cache" HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
date -Is
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -m venv --system-site-packages /tmp/face-age-prior-probe
py=/tmp/face-age-prior-probe/bin/python
"$py" -m pip install --no-index --no-deps "$repo"/deployment/wheels/*.whl
"$py" "$run/source/__PROBE__" "$run"
date -Is
'''.replace('__PROBE__',probe))
    name='face-age-'+run.name;request=dict(Name=name,Project='fdj-infra',ZoneId='cn-shanghai-a',QuotaGroup='fdj_gpu_taskpool',Priority='3',UsePrivateMachine=False,AutoDeleteDuration='168h',TaskSpecs={'train':dict(Command=['bash',str(worker),str(ROOT),str(run)],Env={},Image=IMAGE,Replicas=1,CPU='4',Memory='32Gi',GPU='1',Internet=False,MaxRunningDuration=duration,MaxWaitDuration='10m',RestartPolicy='Never',MountGPFS='gpfs://gpfs2/haobinzhang:/root')})
    write_json(run/'request.json',request)
    try:result=job.create_job(ctx,job.models.CreateJobReq(**(request|{'TaskSpecs':{k:job.models.CreateJobReqTaskSpec(**v) for k,v in request['TaskSpecs'].items()}})))
    except Exception:
        existing=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames=name,Self=True,Limit=100));found=[x for x in existing.Items or [] if x.Name==name]
        if not found:raise
        result=found[0]
    write_json(run/'receipt.json',dict(job_id=result.JobId,name=name,run_dir=str(run),gpus=1));print('GAN_DIAGNOSTIC_SUBMITTED',result.JobId,run,flush=True)


if __name__=='__main__':main()

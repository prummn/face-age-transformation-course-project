"""Prepare and submit a full native-data SAM generator study on exactly one GPU."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from probe_inversion import sha256,write_json
from train_native_age import rows

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'integration'))
from deploy_yicloud import client,IMAGE


def prepare(a):
    audit=a.audit.resolve();probe=a.probe
    if not (audit/'DONE').exists():raise ValueError('Finish expanded common grouping first')
    report=json.loads((audit/'audit.json').read_text());data=rows(audit/'train.jsonl')
    data=[r for r in data if r.get('reference_path') and r.get('source_face_short_edge',r.get('native_face_short_edge',0))>=512]
    if probe:
        selected=[]
        for lo,count in [(20,2),(30,1),(40,1),(50,2),(70,2)]:selected.extend([r for r in data if r['age_interval'][0]==lo][:count])
        data=selected
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S');run=ROOT/'deployment/training-runs'/('native-'+('readiness-' if probe else 'full-')+stamp)
    (run/'data').mkdir(parents=True,exist_ok=False);(run/'source').mkdir()
    for split in ('eval','external'):shutil.copy2(audit/(split+'.jsonl'),run/'data'/(split+'.jsonl'))
    (run/'data/train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data))
    shutil.copy2(audit/'audit.json',run/'common-audit.json')
    ref=ROOT/'deployment/training-runs/highres-texture-20261124-104731';available={r['name']:r for split in ('eval','external') for r in rows(run/'data'/(split+'.jsonl'))}
    selection=[available[r['name']] for r in json.loads((ref/'selection.json').read_text())]
    if {r['identity_group'] for r in data}&{r['identity_group'] for r in selection}:raise ValueError('Training and fixed-code groups overlap')
    write_json(run/'selection.json',selection)
    # Freeze a broader25case regression selection before inspecting candidate
    # output. It remains development/regression, not a blinded independent test.
    extra=[];seen={r['name'] for r in selection};groups={r['identity_group'] for r in selection}
    for lo in (70,50,40,30,20):
        pool=[r for r in available.values() if r.get('age_interval',[None])[0]==lo and r.get('reference_path') and r.get('source_face_short_edge',r.get('native_face_short_edge',0))>=512]
        pool.sort(key=lambda r:(r.get('published_demographic_label')!='Asian',r['name']))
        chosen=0
        for r in pool:
            if r['name'] in seen or r['identity_group'] in groups:continue
            extra.append(r);seen.add(r['name']);groups.add(r['identity_group']);chosen+=1
            if chosen==5:break
    write_json(run/'additional-selection.json',extra)
    for name in ['train_native_age.py','native_age_study.py','evaluate_highres_texture.py','deterministic_quality_ops.py','probe_inversion.py']:
        shutil.copy2(Path(__file__).with_name(name),run/'source'/name)
    for name in ['latent_quality.py','source_color.py']:shutil.copy2(ROOT/'integration'/name,run/'source'/name)
    frozen={}
    for spec in json.loads((ROOT/'integration/quality/baselines/sam-wplus-v1.json').read_text())['weights']:
        path=ROOT/spec['path']
        if sha256(path)!=spec['sha256']:raise ValueError('Baseline checkpoint changed')
        frozen[str(path)]=spec['sha256']
    for row in data:
        if sha256(row['reference_path'])!=row['reference_sha256']:raise ValueError('Native input changed')
        frozen[row['reference_path']]=row['reference_sha256']
    for p in [*list((run/'source').iterdir()),*list((run/'data').iterdir()),run/'selection.json',run/'additional-selection.json',ref/'evaluation/report.json']:
        frozen[str(p)]=sha256(p)
    for row in selection:
        p=ref/'evaluation'/row['name']/'fixed_codes.pt';frozen[str(p)]=sha256(p)
    write_json(run/'experiment.json',dict(profile='full_native_age_adaptation',train=len(data),probe=probe,steps=2 if probe else 4000,batch=2,
        train_counts=dict(Counter(str((r['age_interval'],r.get('published_demographic_label'))) for r in data)),
        generator_routes={'native_control':'native_control','native_gan':'native_gan'},fixed_code_run=str(ref),age_conditioned=False,
        frozen_inputs=frozen,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        criteria=dict(identity_drop_max=.02,age_span_min_ratio=.9,missing_faces_increase_max=0,native_review_required=True,no_automatic_promotion=True),
        resources=dict(gpu=1,cpu=4,memory='64Gi',max_running='12m' if probe else '150m'),
        rationale='Use entire expanded native dataset with train-only W+ matching and guaranteed first-epoch coverage; relax texture-preserving teacher from128px/.5to32px/.05; compare extra RGB/highpass256local+central-face GAN. Multiple recipe changes, only GAN is isolated within pair.',
        caveats=['Publisher apparent age and broad demographic labels; Asian70+ remains limited even after expansion.','80/60training cache approximates200/180serving fitting; fixed comparison codes unchanged.','Previously examined regression, not independent blinded population validation.']))
    print('NATIVE_PREPARED',run,flush=True)
    if a.submit:submit(run)


def submit(run):
    run=run.resolve();experiment=json.loads((run/'experiment.json').read_text())
    if (run/'receipt.json').exists():raise ValueError('Already submitted')
    for p,h in experiment['frozen_inputs'].items():
        if sha256(p)!=h:raise ValueError('Frozen input changed: '+p)
    job,ctx=client();active=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames='face-age',Self=True,Limit=100,Phases='Pending,Running,Inqueue,Starting,Stopping'))
    if active.Items:raise ValueError('One GPU limit: another project job is active')
    worker=run/'source/worker.sh';worker.write_text('''#!/usr/bin/env bash
set -euo pipefail
repo="$1";run="$2";steps="$3";cd "$repo"
exec >> "$run/worker.log" 2>&1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export FACE_AGE_REPO="$repo" FACE_AGE_BASE="$repo/deployment/model-files/face-age" FACE_AGE_DEVICE=cuda:0 FACE_AGE_EXPERIMENT_ALLOCATION=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TORCH_HOME="$FACE_AGE_BASE/torch_cache" HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
date -Is
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -m venv --system-site-packages /tmp/face-age-native-full
py=/tmp/face-age-native-full/bin/python
"$py" -m pip install --no-index --no-deps "$repo"/deployment/wheels/*.whl
"$py" - "$run" <<'PY'
import hashlib,json,sys
from pathlib import Path
for filename,expected in json.loads((Path(sys.argv[1])/'experiment.json').read_text())['frozen_inputs'].items():
 h=hashlib.sha256()
 with Path(filename).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 if h.hexdigest()!=expected:raise ValueError('Frozen input changed:'+filename)
print('FROZEN_INPUTS_VERIFIED',flush=True)
PY
"$py" "$run/source/train_native_age.py" cache "$run" --batch 8
"$py" "$run/source/train_native_age.py" train "$run" --arm native_control --steps "$steps" --batch 2
"$py" "$run/source/train_native_age.py" train "$run" --arm native_gan --steps "$steps" --batch 2
if [[ "$steps" -gt 2 ]]; then
 "$py" "$run/source/evaluate_highres_texture.py" "$run"
 "$py" "$run/source/evaluate_highres_texture.py" "$run" --output-name evaluation-additional --selection-name additional-selection.json --fit-fresh-codes
fi
date -Is
printf 'Native cache, paired training and fixed-code evaluation complete\\n' > "$run/DONE"
''')
    write_json(run/'worker-sha256.json',dict(sha256=sha256(worker)))
    name='face-age-'+run.name;request=dict(Name=name,Project='fdj-infra',ZoneId='cn-shanghai-a',QuotaGroup='fdj_gpu_taskpool',Priority='3',UsePrivateMachine=False,AutoDeleteDuration='168h',
        TaskSpecs={'train':dict(Command=['bash',str(worker),str(ROOT),str(run),str(experiment['steps'])],Env={},Image=IMAGE,Replicas=1,CPU='4',Memory='64Gi',GPU='1',Internet=False,
        MaxRunningDuration=experiment['resources']['max_running'],MaxWaitDuration='10m',RestartPolicy='Never',MountGPFS='gpfs://gpfs2/haobinzhang:/root')})
    write_json(run/'request.json',request)
    try:result=job.create_job(ctx,job.models.CreateJobReq(**(request|{'TaskSpecs':{k:job.models.CreateJobReqTaskSpec(**v) for k,v in request['TaskSpecs'].items()}})))
    except Exception:
        found=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames=name,Self=True,Limit=100));exact=[x for x in found.Items or [] if x.Name==name]
        if not exact:raise
        result=exact[0]
    receipt=dict(job_id=result.JobId,name=name,run_dir=str(run),gpus=1);write_json(run/'receipt.json',receipt);print('NATIVE_SUBMITTED',json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True);x=sub.add_parser('prepare');x.add_argument('audit',type=Path);x.add_argument('--probe',action='store_true');x.add_argument('--submit',action='store_true');x=sub.add_parser('submit');x.add_argument('run',type=Path);a=p.parse_args()
    if a.action=='prepare':prepare(a)
    else:submit(a.run)

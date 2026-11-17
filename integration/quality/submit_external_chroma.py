"""Submit one fixed low-frequency chroma W+ experiment after the controls finish."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from probe_inversion import sha256

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'integration'))
from deploy_yicloud import IMAGE,client,save


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--previous-run',type=Path,required=True)
    a=p.parse_args();previous=a.previous_run.resolve()
    assert (previous/'DONE').exists(),'Previous one-GPU experiment must finish first'
    decision=json.loads((previous/'repair-decision.json').read_text())
    assert decision['confirmation_results_inspected'] is False and decision['chroma_weight']==1.0
    reportpath=previous/'external-results/report.json';report=json.loads(reportpath.read_text())
    assert len(report['records'])==32 and len(report['outputs'])==320
    job,ctx=client();active=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames='face-age',Self=True,Limit=100,Phases='Pending,Running,Inqueue,Starting,Stopping'))
    if active.Items:raise RuntimeError('One-card limit: previous project job still active')
    baseline=json.loads((ROOT/'integration/quality/baselines/sam-wplus-v1.json').read_text())
    for r in baseline['weights']:assert sha256(ROOT/r['path'])==r['sha256']
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S');run=ROOT/'deployment/training-runs'/stamp;source=run/'source';source.mkdir(parents=True)
    hashes={}
    for name in ['refine_external_chroma.py','submit_external_chroma.py','probe_inversion.py']:
        path=ROOT/'integration/quality'/name;shutil.copy2(path,source/name);hashes[name]=sha256(path)
    for name in ['latent_quality.py']:
        path=ROOT/'integration'/name;shutil.copy2(path,source/name);hashes[name]=sha256(path)
    runner=source/'run.sh';runner.write_text('''#!/usr/bin/env bash
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
python3 -m venv --system-site-packages /tmp/face-age-chroma-venv
task_python=/tmp/face-age-chroma-venv/bin/python
"$task_python" -m pip install --no-index --no-deps "$repo_dir"/deployment/wheels/*.whl
"$task_python" -u "$run_dir/source/refine_external_chroma.py" "$run_dir"
''');hashes['run.sh']=sha256(runner)
    shutil.copy2(previous/'repair-decision.json',run/'repair-decision.json');shutil.copy2(previous/'annotated-inputs.json',run/'inputs.pre-age.json')
    cache={r['name']:sha256(previous/'external-results'/r['name']/'texture_half_codes.pt') for r in report['records']}
    save(run/'experiment.json',dict(profile='external_chroma',code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        previous_run=str(previous),control_report_sha256=sha256(reportpath),decision_sha256=sha256(previous/'repair-decision.json'),
        chroma_weight=1.0,cache_sha256=cache,steps=180,train_network_weights=False,
        controls='Immutable existing32 baseline and texture controls, same anchors. Correction restarts zero first8 W+ delta, no extra steps on finished edits.',
        change='Only add 1.0*L1 of RGB minus channel mean, pooled to16x16. Preserve luminance freedoms, no source pixels pasted.',
        guardrails=decision['guardrails'],resources=dict(gpus=1,max_running_duration='20m')))
    name='face-age-external-chroma-'+stamp
    payload=dict(Name=name,Project='fdj-infra',ZoneId='cn-shanghai-a',QuotaGroup='fdj_gpu_taskpool',Priority='3',UsePrivateMachine=False,AutoDeleteDuration='168h',
        TaskSpecs={'evaluation':dict(Command=['bash',str(runner),str(ROOT),str(run)],Env={},Image=IMAGE,Replicas=1,CPU='4',Memory='48Gi',GPU='1',Internet=False,
        MaxRunningDuration='20m',MaxWaitDuration='10m',RestartPolicy='Never',MountGPFS='gpfs://gpfs2/haobinzhang:/root')})
    save(run/'request.json',payload);request=dict(payload);request['TaskSpecs']={k:job.models.CreateJobReqTaskSpec(**v) for k,v in payload['TaskSpecs'].items()}
    try:result=job.create_job(ctx,job.models.CreateJobReq(**request))
    except Exception:
        r=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames=name,Self=True,Limit=100));exact=[x for x in r.Items or [] if x.Name==name]
        if not exact:raise
        result=exact[0]
    receipt=dict(job_id=result.JobId,name=name,run_dir=str(run),source_sha256=hashes,submitted_at=datetime.now(timezone.utc).isoformat())
    save(run/'receipt.json',receipt);print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

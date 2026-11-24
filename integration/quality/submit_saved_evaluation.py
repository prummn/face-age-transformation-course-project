"""Recover saved-checkpoint evaluation without repeating training or W+ fitting."""
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
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('original',type=Path)
    a=p.parse_args();original=a.original.resolve()
    done=json.loads((original/'target_encoder/completed.json').read_text())
    if sha256(original/'target_encoder/encoder.pt')!=done['encoder_sha256']:
        raise ValueError('Completed checkpoint digest differs')
    if not (original/'wplus-evaluation/DONE').is_file():
        raise ValueError('Original W+ evaluation must be complete')
    job,ctx=client()
    active=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',
        JobNames='face-age',Self=True,Limit=100,Phases='Pending,Running,Inqueue,Starting,Stopping'))
    if active.Items:raise RuntimeError('One-card limit: another face-age allocation is active')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    run=ROOT/'deployment/training-runs'/('eval-recovery-'+stamp)
    source=run/'source';source.mkdir(parents=True,exist_ok=False)
    for name in ('evaluate_gan_training.py','probe_inversion.py','run_saved_evaluation.sh','submit_saved_evaluation.py'):
        shutil.copy2(Path(__file__).with_name(name),source/name)
    for name in ('manifests','calibrated_encoder','target_encoder','age-guide','data','wplus-evaluation'):
        (run/name).symlink_to(original/name,target_is_directory=True)
    shutil.copy2(original/'data-audit.json',run/'data-audit.json')
    experiment=json.loads((original/'experiment.json').read_text())
    experiment.update(recovery_of=str(original),recovery_reason='Missing optional external_alignment_failures audit field',
        recovery_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    frozen=experiment['frozen_inputs']
    for path in source.iterdir():frozen[str(path)]=sha256(path)
    for route in ('calibrated_encoder','target_encoder'):
        path=original/route/'encoder.pt';frozen[str(path)]=sha256(path)
    save(run/'experiment.json',experiment)
    name='face-age-eval-recovery-'+stamp
    payload=dict(Name=name,Project='fdj-infra',ZoneId='cn-shanghai-a',QuotaGroup='fdj_gpu_taskpool',
        Priority='3',UsePrivateMachine=False,AutoDeleteDuration='168h',TaskSpecs={'evaluation':dict(
        Command=['bash',str(source/'run_saved_evaluation.sh'),str(ROOT),str(run)],Env={},Image=IMAGE,Replicas=1,
        CPU='4',Memory='48Gi',GPU='1',Internet=False,MaxRunningDuration='18m',MaxWaitDuration='10m',
        RestartPolicy='Never',MountGPFS='gpfs://gpfs2/haobinzhang:/root')})
    save(run/'request.json',payload)
    specs={k:job.models.CreateJobReqTaskSpec(**v) for k,v in payload['TaskSpecs'].items()}
    try:
        result=job.create_job(ctx,job.models.CreateJobReq(**(payload|{'TaskSpecs':specs})))
    except Exception:
        found=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames=name,Self=True,Limit=100))
        exact=[x for x in found.Items or [] if x.Name==name]
        if not exact:raise
        result=exact[0]
    receipt=dict(job_id=result.JobId,name=name,run_dir=str(run),gpus=1,original=str(original))
    save(run/'receipt.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()

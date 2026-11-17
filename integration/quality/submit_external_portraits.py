"""Submit one bounded one-GPU evaluation of newly collected test photographs."""
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    a=p.parse_args();data=a.data.resolve();audit=json.loads((data/'audit.json').read_text())
    assert (data/'DONE').exists() and audit['aligned']>=24
    assert sha256(data/'inputs.pre-age.json')==audit['inputs_sha256']
    records=json.loads((data/'inputs.pre-age.json').read_text())
    assert all(not r['training_authorized'] and sha256(r['image_path'])==r['sha256'] for r in records)
    baseline=json.loads((ROOT/'integration/quality/baselines/sam-wplus-v1.json').read_text())
    weights=[]
    for r in baseline['weights']:
        path=ROOT/r['path'];assert sha256(path)==r['sha256'],'Baseline weight changed'
        weights.append(r)
    job,ctx=client()
    active=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames='face-age',Self=True,Limit=100,Phases='Pending,Running,Inqueue,Starting,Stopping'))
    if active.Items:raise RuntimeError('One-card limit; active project jobs: '+','.join(x.JobId for x in active.Items))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S');run=ROOT/'deployment/training-runs'/stamp
    source=run/'source';source.mkdir(parents=True,exist_ok=False);hashes={}
    for name in ['evaluate_external_portraits.py','run_external_portraits.sh','probe_inversion.py','submit_external_portraits.py',
                 'collect_external_portraits.py','prepare_external_portraits.py','prepare_gan_training.py','http_zip.py']:
        path=ROOT/'integration/quality'/name;shutil.copy2(path,source/name);hashes[name]=sha256(path)
    for name in ['latent_quality.py','backend.py','source_color.py']:
        path=ROOT/'integration'/name;shutil.copy2(path,source/name);hashes[name]=sha256(path)
    shutil.copy2(data/'inputs.pre-age.json',run/'inputs.pre-age.json');shutil.copy2(data/'audit.json',run/'data-audit.json')
    save(run/'assets.json',weights)
    save(run/'runtime-source.json',{str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'integration/ascend/src/sam_npu.py',ROOT/'integration/ascend/src/sam_losses.py',ROOT/'integration/ascend/src/faceattr.py']})
    save(run/'experiment.json',dict(profile='external_portraits',code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        inputs=len(records),data=str(data),inputs_sha256=sha256(data/'inputs.pre-age.json'),
        routes=['baseline_v1','texture_half'],source_age='Frozen FairFace pseudo condition; publisher real/apparent ages retained separately.',
        anchor='Serving helper200 steps per input; shared by both routes. No source pixels pasted, no network weights trained.',
        edit='Four age targets jointly optimize180 steps with baseline loss; texture route halves styles8..17 age displacement.',
        review='Inspect diagnostic split first. Confirmation images/results are reserved until repair choice is frozen. No parameter tuning on confirmation.',
        decision='Look for identity failures, apparent aging, source-color drift and glasses/skin/background distortions; shared age scores never establish chronological accuracy.',
        resources=dict(gpus=1,cpu=4,memory='48Gi',max_running_duration='35m'),
        limitations=['APPA official test32 new photographs, not a population sample. Asian is publisher broad category.',
            'Conservative similarity screen does not prove distinct identities; keep repeated-photo groups together.',
            'Publisher face crops may be low resolution;1024 aligned references may upscale. All detection failures retained.']))
    name='face-age-external-portraits-'+stamp
    payload=dict(Name=name,Project='fdj-infra',ZoneId='cn-shanghai-a',QuotaGroup='fdj_gpu_taskpool',Priority='3',UsePrivateMachine=False,
        AutoDeleteDuration='168h',TaskSpecs={'evaluation':dict(Command=['bash',str(source/'run_external_portraits.sh'),str(ROOT),str(run)],Env={},Image=IMAGE,
        Replicas=1,CPU='4',Memory='48Gi',GPU='1',Internet=False,MaxRunningDuration='35m',MaxWaitDuration='10m',RestartPolicy='Never',MountGPFS='gpfs://gpfs2/haobinzhang:/root')})
    save(run/'request.json',payload);request=dict(payload);request['TaskSpecs']={k:job.models.CreateJobReqTaskSpec(**v) for k,v in payload['TaskSpecs'].items()}
    try:result=job.create_job(ctx,job.models.CreateJobReq(**request))
    except Exception:
        r=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames=name,Self=True,Limit=100));exact=[x for x in r.Items or [] if x.Name==name]
        if not exact:raise
        result=exact[0]
    receipt=dict(job_id=result.JobId,name=name,run_dir=str(run),source_sha256=hashes,
        submitted_at=datetime.now(timezone.utc).isoformat(),max_running_duration='35m')
    save(run/'receipt.json',receipt);print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

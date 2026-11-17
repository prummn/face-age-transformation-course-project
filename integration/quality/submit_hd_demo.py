"""Submit one bounded GPU job to render a frozen, source-selected HD demo set."""
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
    data=p.parse_args().data.resolve();selection=json.loads((data/'selection.json').read_text())
    assert (data/'DONE').exists() and selection['samples']==24
    assert sha256(data/'inputs.json')==selection['inputs_sha256']
    records=json.loads((data/'inputs.json').read_text())
    assert all(sha256(r['image_path'])==r['sha256'] for r in records)
    baseline=json.loads((ROOT/'integration/quality/baselines/sam-wplus-v1.json').read_text())
    for r in baseline['weights']:assert sha256(ROOT/r['path'])==r['sha256'],'Baseline asset changed'
    job,ctx=client();active=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames='face-age',Self=True,Limit=100,Phases='Pending,Running,Inqueue,Starting,Stopping'))
    if active.Items:raise RuntimeError('One-card limit; active jobs: '+','.join(x.JobId for x in active.Items))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S');run=ROOT/'deployment/demo-hd'/stamp
    source=run/'source';source.mkdir(parents=True,exist_ok=False);hashes={}
    for name in ['evaluate_hd_demo.py','run_hd_demo.sh','submit_hd_demo.py','collect_hd_demo.py','probe_inversion.py']:
        path=ROOT/'integration/quality'/name;shutil.copy2(path,source/name);hashes[name]=sha256(path)
    for name in ['latent_quality.py','source_color.py']:
        path=ROOT/'integration'/name;shutil.copy2(path,source/name);hashes[name]=sha256(path)
    shutil.copy2(data/'inputs.json',run/'inputs.json');shutil.copy2(data/'selection.json',run/'selection.json')
    save(run/'assets.json',baseline['weights'])
    save(run/'experiment.json',dict(profile='constrained',code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256=hashes,data=str(data),samples=24,targets=['same',20,40,60,75],color_versions=['raw','preserved'],
        selection_uses_generated_results=False,weights_unchanged=True,training=False,
        runtime_source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'integration/ascend/src/sam_npu.py',ROOT/'integration/ascend/src/sam_losses.py',ROOT/'integration/ascend/src/faceattr.py']},
        resources=dict(gpus=1,cpu=4,memory='48Gi',max_running_duration='25m')))
    name='face-age-hd-demo-'+stamp
    payload=dict(Name=name,Project='fdj-infra',ZoneId='cn-shanghai-a',QuotaGroup='fdj_gpu_taskpool',Priority='3',UsePrivateMachine=False,
        AutoDeleteDuration='168h',TaskSpecs={'evaluation':dict(Command=['bash',str(source/'run_hd_demo.sh'),str(ROOT),str(run)],Env={},Image=IMAGE,
        Replicas=1,CPU='4',Memory='48Gi',GPU='1',Internet=False,MaxRunningDuration='25m',MaxWaitDuration='10m',RestartPolicy='Never',MountGPFS='gpfs://gpfs2/haobinzhang:/root')})
    save(run/'request.json',payload);request=dict(payload);request['TaskSpecs']={k:job.models.CreateJobReqTaskSpec(**v) for k,v in payload['TaskSpecs'].items()}
    try:result=job.create_job(ctx,job.models.CreateJobReq(**request))
    except Exception:
        r=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames=name,Self=True,Limit=100));exact=[x for x in r.Items or [] if x.Name==name]
        if not exact:raise
        result=exact[0]
    receipt=dict(job_id=result.JobId,name=name,run_dir=str(run),submitted_at=datetime.now(timezone.utc).isoformat())
    save(run/'receipt.json',receipt);print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

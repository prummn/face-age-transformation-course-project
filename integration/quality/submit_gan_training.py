"""Submit one independent bounded paired GAN pilot; preserve online defaults."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'integration'))
from deploy_yicloud import IMAGE,client,save
from probe_inversion import sha256

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--reuse-manifests',type=Path,help='Reuse exact completed numeric-age annotation after failed pre-step smoke run')
    a=p.parse_args()
    data=a.data.resolve();audit=json.loads((data/'audit.json').read_text())
    if not (data/'DONE').exists() or audit['training']<1000:raise RuntimeError('CPU dataset preparation incomplete')
    if a.reuse_manifests:
        receipt=json.loads((a.reuse_manifests/'receipt.json').read_text())
        if receipt['model_sha256']!=sha256(ROOT/'deployment/model-files/face-age/models/faceattr/convnext_t_fairface.pt'):raise RuntimeError('Age model changed')
        for split in ['train','eval','external']:
            src=data/(split+'.pre-age.jsonl' if split!='external' else 'external.jsonl')
            raw=[json.loads(x) for x in src.read_text().splitlines()]
            aged=[json.loads(x) for x in (a.reuse_manifests/(split+'.jsonl')).read_text().splitlines()]
            if len(raw)!=len(aged) or not all(all(y.get(k)==v for k,v in x.items()) for x,y in zip(raw,aged)):raise RuntimeError('Frozen data differs')
            if sha256(a.reuse_manifests/(split+'.jsonl'))!=receipt['files'][split+'.jsonl']:raise RuntimeError('Annotation hash changed')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S');run=ROOT/'deployment/training-runs'/stamp
    source=run/'source';source.mkdir(parents=True,exist_ok=False);hashes={}
    for name in ['train_gan_reconstruction.py','annotate_gan_training.py','evaluate_gan_training.py',
                 'probe_inversion.py','run_gan_training.sh','submit_gan_training.py',
                 'prepare_gan_training.py','collect_ffhq_training.py','http_zip.py','build_training_review.py']:
        path=Path(__file__).parent/name;shutil.copy2(path,source/name);hashes[name]=sha256(path)
    shutil.copy2(data/'audit.json',run/'data-audit.json')
    if a.reuse_manifests:shutil.copytree(a.reuse_manifests,run/'manifests')
    base=ROOT/'deployment/model-files/face-age'
    weights=[base/'out/ft_l1/encoder_step1000.pt',base/'models/sam/sam_ffhq_aging.pt',
             base/'models/faceattr/convnext_t_fairface.pt',base/'models/irse50/model_ir_se50.pth']
    save(run/'runtime-source.json',{str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'integration/ascend/src/sam_npu.py',ROOT/'integration/ascend/src/sam_losses.py',ROOT/'integration/ascend/src/faceattr.py']})
    save(run/'assets.json',[dict(path=str(p),sha256=sha256(p),bytes=p.stat().st_size) for p in weights])
    save(run/'data-snapshot.json',dict(path=str(data),files={p.name:sha256(p) for p in data.glob('*.jsonl')},audit_sha256=sha256(data/'audit.json')))
    name='face-age-paired-gan-'+stamp
    payload=dict(Name=name,Project='fdj-infra',ZoneId='cn-shanghai-a',QuotaGroup='fdj_gpu_taskpool',Priority='3',
        UsePrivateMachine=False,AutoDeleteDuration='168h',TaskSpecs={'experiment':dict(
        Command=['bash',str(source/'run_gan_training.sh'),str(ROOT),str(run),str(data)],Env={},Image=IMAGE,
        Replicas=1,CPU='8',Memory='64Gi',GPU='1',Internet=False,MaxRunningDuration='1h',
        MaxWaitDuration='30m',RestartPolicy='Never',MountGPFS='gpfs://gpfs2/haobinzhang:/root')})
    save(run/'request.json',payload);job,ctx=client();request=dict(payload)
    request['TaskSpecs']={k:job.models.CreateJobReqTaskSpec(**v) for k,v in payload['TaskSpecs'].items()}
    try:result=job.create_job(ctx,job.models.CreateJobReq(**request))
    except Exception:
        found=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames=name,Self=True,Limit=100))
        exact=[j for j in found.Items or [] if j.Name==name]
        if not exact:raise
        result=exact[0]
    receipt=dict(job_id=result.JobId,name=name,run_dir=str(run),source_sha256=hashes,
        submitted_at=datetime.now(timezone.utc).isoformat(),max_running_duration='1h',
        reused_manifests=str(a.reuse_manifests.resolve()) if a.reuse_manifests else None)
    save(run/'receipt.json',receipt);print(json.dumps(receipt,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

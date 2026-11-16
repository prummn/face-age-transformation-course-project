"""Submit one independent bounded paired GAN pilot; preserve online defaults."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'integration'))
from deploy_yicloud import IMAGE,client,save
from probe_inversion import sha256

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--reuse-manifests',type=Path,help='Reuse exact completed numeric-age annotation after failed pre-step smoke run')
    p.add_argument('--profile',choices=['generator_affines','identity_weight'],default='generator_affines')
    a=p.parse_args()
    if a.profile=='identity_weight' and not a.reuse_manifests:
        raise ValueError('Identity pilot requires the unchanged previously annotated manifests')
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
    job,ctx=client()
    active=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',
        JobNames='face-age',Self=True,Limit=100,Phases='Pending,Running,Inqueue,Starting,Stopping'))
    if active.Items:
        raise RuntimeError('One-card budget: existing project task must finish first: '+','.join(x.JobId for x in active.Items))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S');run=ROOT/'deployment/training-runs'/stamp
    source=run/'source';source.mkdir(parents=True,exist_ok=False);hashes={}
    for name in ['train_gan_reconstruction.py','annotate_gan_training.py','evaluate_gan_training.py',
                 'probe_inversion.py','run_gan_training.sh','submit_gan_training.py',
                 'prepare_gan_training.py','collect_ffhq_training.py','http_zip.py','build_training_review.py',
                 'run_identity_pilot.sh','build_identity_review.py','verify_identity_artifacts.py','verify_identity_review.py']:
        path=Path(__file__).parent/name;shutil.copy2(path,source/name);hashes[name]=sha256(path)
    shutil.copy2(data/'audit.json',run/'data-audit.json')
    if a.reuse_manifests:shutil.copytree(a.reuse_manifests,run/'manifests')
    base=ROOT/'deployment/model-files/face-age'
    weights=[base/'out/ft_l1/encoder_step1000.pt',base/'models/sam/sam_ffhq_aging.pt',
             base/'models/faceattr/convnext_t_fairface.pt',base/'models/irse50/model_ir_se50.pth']
    save(run/'runtime-source.json',{str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'integration/ascend/src/sam_npu.py',ROOT/'integration/ascend/src/sam_losses.py',ROOT/'integration/ascend/src/faceattr.py']})
    save(run/'assets.json',[dict(path=str(p),sha256=sha256(p),bytes=p.stat().st_size) for p in weights])
    save(run/'data-snapshot.json',dict(path=str(data),files={p.name:sha256(p) for p in data.glob('*.jsonl')},audit_sha256=sha256(data/'audit.json')))
    identity=a.profile=='identity_weight'
    if identity:
        save(run/'experiment.json',dict(profile=a.profile,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            hypothesis='Increasing cross-age identity coefficient alone reduces identity drift without suppressing age edits.',
            steps=500,batch_size=2,seed=20261113,lr=1e-5,
            arms={'id_control':.1,'id_strong':.5},routes=['fine_direct','id_control','id_strong'],
            fixed='Same Release initialization, data, seed, sampling, flips, targets, BN, generator and all other loss coefficients.',
            decision=dict(ffhq_cross_sface_gain_over_control=.005,paired_case_bootstrap95_lower_gt=0,
                cross_sface_no_worse_than_release=True,age_span_retention_min=.9,same_lpips_max_regression=.01,
                appa_direction_consistent=True,visual_review_required=True),
            limits=dict(gpus=1,max_running_duration='30m',max_wait_duration='10m',parallel_arms=False),
            caveats=['Repeated evaluation set; exploratory, not a blind generalization claim.',
                     'Age response is measured by the shared FairFace network, not age accuracy.',
                     'APPA and diagnostic portraits are evaluation only. No automatic promotion.']))
    name=('face-age-identity-pilot-' if identity else 'face-age-paired-gan-')+stamp
    runner='run_identity_pilot.sh' if identity else 'run_gan_training.sh'
    duration='30m' if identity else '1h'
    payload=dict(Name=name,Project='fdj-infra',ZoneId='cn-shanghai-a',QuotaGroup='fdj_gpu_taskpool',Priority='3',
        UsePrivateMachine=False,AutoDeleteDuration='168h',TaskSpecs={'experiment':dict(
        Command=['bash',str(source/runner),str(ROOT),str(run),str(data)],Env={},Image=IMAGE,
        Replicas=1,CPU='4' if identity else '8',Memory='32Gi' if identity else '64Gi',GPU='1',Internet=False,MaxRunningDuration=duration,
        MaxWaitDuration='10m' if identity else '30m',RestartPolicy='Never',MountGPFS='gpfs://gpfs2/haobinzhang:/root')})
    save(run/'request.json',payload);request=dict(payload)
    request['TaskSpecs']={k:job.models.CreateJobReqTaskSpec(**v) for k,v in payload['TaskSpecs'].items()}
    try:result=job.create_job(ctx,job.models.CreateJobReq(**request))
    except Exception:
        found=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames=name,Self=True,Limit=100))
        exact=[j for j in found.Items or [] if j.Name==name]
        if not exact:raise
        result=exact[0]
    receipt=dict(job_id=result.JobId,name=name,run_dir=str(run),source_sha256=hashes,
        submitted_at=datetime.now(timezone.utc).isoformat(),max_running_duration=duration,
        reused_manifests=str(a.reuse_manifests.resolve()) if a.reuse_manifests else None)
    save(run/'receipt.json',receipt);print(json.dumps(receipt,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

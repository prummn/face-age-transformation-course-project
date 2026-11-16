"""Submit one independent bounded paired GAN pilot; preserve online defaults."""
import argparse
from datetime import datetime,timezone
import json
import os
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
    p.add_argument('--profile',choices=['generator_affines','identity_weight','quality_sprint','latent_constraint'],default='generator_affines')
    p.add_argument('--previous-run',type=Path)
    a=p.parse_args()
    if a.profile in ['identity_weight','quality_sprint','latent_constraint'] and not a.reuse_manifests:
        raise ValueError('Identity pilot requires the unchanged previously annotated manifests')
    if a.profile=='latent_constraint' and (not a.previous_run or not (a.previous_run/'latent-refinement/DONE').exists()):
        raise ValueError('Constrained refinement requires a completed previous latent run')
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
                 'run_identity_pilot.sh','build_identity_review.py','verify_identity_artifacts.py','verify_identity_review.py',
                 'run_quality_sprint.sh','refine_identity_latents.py','run_latent_constraint.sh','verify_quality_backend.py']:
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
    sprint=a.profile=='quality_sprint'
    constraint=a.profile=='latent_constraint'
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
    if sprint:
        records=[json.loads(x) for split in ['eval','external'] for x in (run/'manifests'/(split+'.jsonl')).read_text().splitlines()]
        selected=[r for r in records if r['name']=='uploaded']
        selected += [r for r in records if r['cohort']=='appa_asian_test'][::2]
        bins=set()
        for r in records:
            if r['cohort']=='ffhq_regression' and tuple(r['age_interval']) not in bins:
                bins.add(tuple(r['age_interval']));selected.append(r)
        assert len(selected)==12
        save(run/'latent-inputs.json',selected)
        previous=ROOT/'deployment/training-runs/20261116-131721/id_strong/encoder.pt'
        assert sha256(previous)=='57b454efeaf2a8d683dfa655001ac610db0c5704b0fde85a5956072a84cbf8fb'
        (run/'previous_identity').mkdir();os.link(previous,run/'previous_identity/encoder.pt')
        save(run/'experiment.json',dict(profile=a.profile,
            code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            priority='Obtain materially better likeness and natural age edits within SAM/StyleGAN2.',
            latent=dict(inputs=12,selection='User + every second fixed APPA case + first FFHQ case per age interval',
                weights_frozen=True,anchor_steps=200,edit_steps=120,no_input_pixels_pasted=True),
            training=dict(steps=800,batch_size=2,encoder_lr=2e-5,generator_lr=1e-5,same_identity=1.,cross_identity=.5,
                paired_arms=['recon_encoder','recon_generator'],generator='Train conv weights/modulation/bias at64..1024; freeze coarse generator, pSp, noise and loss nets.',
                data='Original1054 only; no diagnostic, user or APPA in network training.',
                attribution='Combined reconstruction recipe differs from previous500-step run; paired generator effect is separately identifiable.'),
            limits=dict(gpus=1,sequential=True,max_running_duration='45m',max_wait_duration='10m'),
            selection='Inspect input likeness, same reconstruction, cross-age identity and age response jointly; compare original and previous candidate; no automatic promotion.'))
    if constraint:
        previous=a.previous_run.resolve()
        shutil.copy2(previous/'latent-inputs.json',run/'latent-inputs.json')
        save(run/'experiment.json',dict(profile=a.profile,previous_run=str(previous),
            previous_report_sha256=sha256(previous/'latent-refinement/report.json'),
            code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            reason='Adaptive follow-up to visible texture/color distortions from unrestricted age-KL latent fitting.',
            controls='Same12 inputs, cached anchors, original GAN weights/noise/target conditions; all earlier outputs retained.',
            changes='Only first8 W+ styles optimized; LPIPS to anchored age edits and low-frequency source color; lower ageKL weight.',
            snapshots=[60,180],max_gpus=1,max_running_duration='15m',
            training='No network parameters trained; source image used solely for its own inference optimization.'))
        for name in ['backend.py','latent_quality.py','source_color.py']:
            path=ROOT/'integration'/name;shutil.copy2(path,source/name);hashes[name]=sha256(path)
    name=('face-age-latent-constraint-' if constraint else 'face-age-quality-sprint-' if sprint else 'face-age-identity-pilot-' if identity else 'face-age-paired-gan-')+stamp
    runner='run_latent_constraint.sh' if constraint else 'run_quality_sprint.sh' if sprint else 'run_identity_pilot.sh' if identity else 'run_gan_training.sh'
    duration='15m' if constraint else '45m' if sprint else '30m' if identity else '1h'
    payload=dict(Name=name,Project='fdj-infra',ZoneId='cn-shanghai-a',QuotaGroup='fdj_gpu_taskpool',Priority='3',
        UsePrivateMachine=False,AutoDeleteDuration='168h',TaskSpecs={'experiment':dict(
        Command=['bash',str(source/runner),str(ROOT),str(run),str(data)],Env={},Image=IMAGE,
        Replicas=1,CPU='4' if identity or sprint or constraint else '8',Memory='48Gi' if sprint or constraint else '32Gi' if identity else '64Gi',GPU='1',Internet=False,MaxRunningDuration=duration,
        MaxWaitDuration='10m' if identity or sprint or constraint else '30m',RestartPolicy='Never',MountGPFS='gpfs://gpfs2/haobinzhang:/root')})
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

"""Freeze and submit one bounded paired high-resolution texture experiment."""
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
    p.add_argument('audit',type=Path);p.add_argument('reference_run',type=Path)
    p.add_argument('--steps',type=int,default=500);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--min-resolution',type=int,choices=(64,128,256),default=256)
    p.add_argument('--generator-lr',type=float,default=2e-5)
    p.add_argument('--adversarial-weight',type=float,default=.02)
    p.add_argument('--fixed-code-run',type=Path)
    p.add_argument('--submit-prepared',type=Path)
    a=p.parse_args()
    if not 1<=a.steps<=3000:raise ValueError('Steps must be 1..3000')
    if a.submit_prepared:
        run=a.submit_prepared.resolve();experiment=json.loads((run/'experiment.json').read_text());steps=experiment['steps']
        if (run/'receipt.json').exists():raise ValueError('Run already submitted')
    else:
        audit=a.audit.resolve();ref=a.reference_run.resolve()
        if not (audit/'DONE').exists():raise ValueError('Common group audit must complete')
        audit_info=json.loads((audit/'audit.json').read_text())
        stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S');run=ROOT/'deployment/training-runs'/('highres-texture-'+stamp)
        source=run/'source';source.mkdir(parents=True,exist_ok=False);data=run/'data';data.mkdir()
        for name in ('train_highres_texture.py','evaluate_highres_texture.py','deterministic_quality_ops.py','probe_inversion.py','run_highres_texture.sh','submit_highres_texture.py'):
            shutil.copy2(Path(__file__).with_name(name),source/name)
        for name in ('latent_quality.py','source_color.py'):shutil.copy2(ROOT/'integration'/name,source/name)
        for name,digest in audit_info['files'].items():
            if sha256(audit/name)!=digest:raise ValueError('Audited manifest changed')
            shutil.copy2(audit/name,data/name)
        shutil.copy2(audit/'audit.json',run/'common-audit.json')
        available={r['name']:r for name in ('eval','external') for r in map(json.loads,(data/(name+'.jsonl')).read_text().splitlines())}
        old=json.loads((ref/'wplus-evaluation/report.json').read_text())['records']
        hd=[r for r in old if r['cohort']=='new-hd-holdout']
        selected=[]
        for age_low in (20,30,40,50,70):
            selected.extend([r for r in hd if r['age_interval'][0]==age_low][:2])
        selected.extend([r for r in old if r['cohort']=='new_appa_asian' and r['publisher_age']>=60][:4])
        selected.extend([r for r in old if r['cohort']=='new_appa_asian' and r['publisher_age']<40][:2])
        selected=[available[r['name']] for r in selected]
        if not 12<=len(selected)<=16:raise ValueError('Unexpected pilot selection size')
        save(run/'selection.json',selected)
        frozen={}
        for item in json.loads((ROOT/'integration/quality/baselines/sam-wplus-v1.json').read_text())['weights']:
            path=ROOT/item['path']
            if sha256(path)!=item['sha256']:raise ValueError('Baseline weight changed')
            frozen[str(path)]=item['sha256']
        paths=list(source.iterdir())+list(data.iterdir())+[run/'selection.json',run/'common-audit.json']
        paths.extend(ROOT/'integration/ascend/src'/n for n in ('sam_npu.py','sam_losses.py','faceattr.py'))
        paths.append(ROOT/'deployment/gan-research/face_recognition_sface_2021dec.onnx')
        frozen.update({str(path):sha256(path) for path in paths})
        steps=a.steps
        experiment=dict(profile='paired_native_texture_adapter',steps=steps,run=str(run),min_resolution=a.min_resolution,generator_lr=a.generator_lr,adversarial_weight=a.adversarial_weight,
            fixed_code_run=str(a.fixed_code_run.resolve()) if a.fixed_code_run else None,
            code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            frozen_inputs=frozen,selection_count=len(selected),train=audit_info['strict_counts']['train'],
            reference_run=str(ref),common_audit=str(audit),resources=dict(gpu=1,cpu=4,memory='48Gi',max_running='38m'),
            criteria=dict(identity_drop_max=.02,age_span_min_ratio=.90,missing_faces_increase_max=0,
                native_review_required=True,no_automatic_promotion=True),
            intended_comparison='Release generator vs highres reconstruction control vs identical control with age/region-matched local GAN')
        if a.fixed_code_run:
            previous=a.fixed_code_run.resolve();previous_selection=json.loads((previous/'selection.json').read_text())
            if selected!=previous_selection:raise ValueError('Fixed-code evaluation inputs differ')
            for row in selected:
                code=previous/'evaluation'/row['name']/'fixed_codes.pt'
                if not code.is_file():raise ValueError('Missing prior fixed codes')
                frozen[str(code)]=sha256(code)
            frozen[str(previous/'evaluation/report.json')]=sha256(previous/'evaluation/report.json')
        save(run/'experiment.json',experiment)
    if a.prepare_only:
        print(json.dumps(dict(run=str(run),steps=steps,prepared=True)));return
    job,ctx=client();active=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames='face-age',
        Self=True,Limit=100,Phases='Pending,Running,Inqueue,Starting,Stopping'))
    if active.Items:raise RuntimeError('One-card limit; wait for existing project task to terminate')
    name='face-age-'+run.name
    payload=dict(Name=name,Project='fdj-infra',ZoneId='cn-shanghai-a',QuotaGroup='fdj_gpu_taskpool',Priority='3',UsePrivateMachine=False,
        AutoDeleteDuration='168h',TaskSpecs={'pilot':dict(Command=['bash',str(run/'source/run_highres_texture.sh'),str(ROOT),str(run),str(steps)],
        Env={},Image=IMAGE,Replicas=1,CPU='4',Memory='48Gi',GPU='1',Internet=False,MaxRunningDuration='38m',MaxWaitDuration='10m',
        RestartPolicy='Never',MountGPFS='gpfs://gpfs2/haobinzhang:/root')})
    save(run/'request.json',payload);spec={k:job.models.CreateJobReqTaskSpec(**v) for k,v in payload['TaskSpecs'].items()}
    try:result=job.create_job(ctx,job.models.CreateJobReq(**(payload|{'TaskSpecs':spec})))
    except Exception:
        found=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames=name,Self=True,Limit=100))
        exact=[x for x in found.Items or [] if x.Name==name]
        if not exact:raise
        result=exact[0]
    receipt=dict(job_id=result.JobId,name=name,run_dir=str(run),gpus=1);save(run/'receipt.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()

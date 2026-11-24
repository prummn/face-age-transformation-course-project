"""One-GPU, bounded age-conditioned generator control and local GAN study."""
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
from deploy_yicloud import IMAGE,client


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('matched_run',type=Path)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--submit-prepared',type=Path)
    a=p.parse_args()
    if a.submit_prepared:
        run=a.submit_prepared.resolve()
        if (run/'receipt.json').exists():raise ValueError('Already submitted')
    else:
        prior=a.matched_run.resolve()
        if not (prior/'DONE').exists() or not (prior/'train-codes/DONE').exists():raise ValueError('Matched study must finish')
        stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S');run=ROOT/'deployment/training-runs'/('conditional-texture-'+stamp)
        source=run/'source';source.mkdir(parents=True,exist_ok=False);(run/'data').mkdir()
        for path in (prior/'data').iterdir():shutil.copy2(path,run/'data'/path.name)
        for name in ['common-audit.json','selection.json']:shutil.copy2(prior/name,run/name)
        shutil.copytree(prior/'train-codes',run/'train-codes',copy_function=shutil.copy2)
        names=['train_highres_texture.py','evaluate_highres_texture.py','deterministic_quality_ops.py','probe_inversion.py','submit_conditional_texture.py','run_conditional_texture.sh']
        for name in names:shutil.copy2(Path(__file__).with_name(name),source/name)
        for name in ['latent_quality.py','source_color.py']:shutil.copy2(ROOT/'integration'/name,source/name)
        old=json.loads((prior/'experiment.json').read_text());fixed=Path(old['fixed_code_run']);frozen={}
        for item in json.loads((ROOT/'integration/quality/baselines/sam-wplus-v1.json').read_text())['weights']:
            path=ROOT/item['path']
            if sha256(path)!=item['sha256']:raise ValueError('Baseline changed')
            frozen[str(path)]=item['sha256']
        files=list(source.iterdir())+list((run/'data').iterdir())+list((run/'train-codes').iterdir())+[run/'selection.json',run/'common-audit.json',fixed/'evaluation/report.json']
        files.extend(ROOT/'integration/ascend/src'/name for name in ['sam_npu.py','sam_losses.py','faceattr.py'])
        files.append(ROOT/'deployment/gan-research/face_recognition_sface_2021dec.onnx')
        for row in json.loads((run/'selection.json').read_text()):files.append(fixed/'evaluation'/row['name']/'fixed_codes.pt')
        for row in [json.loads(s) for s in (run/'data/train.jsonl').read_text().splitlines()]:
            for field,key in [('image_path','sha256'),('reference_path','reference_sha256')]:
                if sha256(row[field])!=row[key]:raise ValueError('Training image changed')
                frozen[row[field]]=row[key]
        frozen.update({str(path):sha256(path) for path in files})
        experiment=dict(profile='conditional_age_preserved_texture',steps=1000,min_resolution=64,generator_lr=5e-5,adversarial_weight=.1,
            age_conditioned=True,age_preservation=1.,patch_edge=256,latent_source='wplus',input_flip=False,native_reference_input=True,
            generator_routes={'conditional_control':'control','conditional_gan':'patch_gan'},fixed_code_run=str(fixed),training_code_run=str(prior),
            frozen_inputs=frozen,train=old['train'],selection_count=old['selection_count'],code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            criteria=old['criteria'],resources=dict(gpu=1,cpu=4,memory='48Gi',max_running='25m'),
            generator_condition='theta(target) = theta_base + clamp((target-40)/35,0,1) * (theta_candidate-theta_base); no pixel blending and no target/latent age rescaling',
            shared='Frozen baseline W+ codes, matched sampling/augmentations, native reconstruction, teacher multiview age KL, fixed coarse generator/ToRGB',
            comparison='Identical age-conditioned reconstruction/age protection, with or without local age/region conditional GAN at256px.',
            limitations=['60 training photos and previously inspected15-case panel: exploratory, not independent population validation.','Conditioning+age protection is a new recipe; only GAN contribution isolated within this pair.','Shared FairFace teacher preserves baseline classifier response, not chronological accuracy or guaranteed realism.','No candidate-specific latent fitting.'])
        write_json(run/'experiment.json',experiment);print('PREPARED',run,flush=True)
    if a.prepare_only:return
    for path,digest in json.loads((run/'experiment.json').read_text())['frozen_inputs'].items():
        if sha256(path)!=digest:raise ValueError('Frozen input changed: '+path)
    job,ctx=client();active=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames='face-age',Self=True,Limit=100,Phases='Pending,Running,Inqueue,Starting,Stopping'))
    if active.Items:raise RuntimeError('One GPU limit: another project task is active')
    name='face-age-'+run.name
    request=dict(Name=name,Project='fdj-infra',ZoneId='cn-shanghai-a',QuotaGroup='fdj_gpu_taskpool',Priority='3',UsePrivateMachine=False,AutoDeleteDuration='168h',
        TaskSpecs={'pilot':dict(Command=['bash',str(run/'source/run_conditional_texture.sh'),str(ROOT),str(run)],Env={},Image=IMAGE,Replicas=1,CPU='4',Memory='48Gi',GPU='1',
        Internet=False,MaxRunningDuration='25m',MaxWaitDuration='10m',RestartPolicy='Never',MountGPFS='gpfs://gpfs2/haobinzhang:/root')})
    write_json(run/'request.json',request)
    specs={k:job.models.CreateJobReqTaskSpec(**v) for k,v in request['TaskSpecs'].items()}
    try:result=job.create_job(ctx,job.models.CreateJobReq(**(request|{'TaskSpecs':specs})))
    except Exception:
        found=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames=name,Self=True,Limit=100))
        exact=[x for x in found.Items or [] if x.Name==name]
        if not exact:raise
        result=exact[0]
    receipt=dict(job_id=result.JobId,name=name,run_dir=str(run),gpus=1);write_json(run/'receipt.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()

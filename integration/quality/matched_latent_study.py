"""Bounded encoder-code versus fitted-W+ generator training, using train images only."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from probe_inversion import sha256,write_json

ROOT=Path(__file__).resolve().parents[2]
AUDIT=ROOT/'deployment/age-quality-final/common-audit-20261124'
REFERENCE=ROOT/'deployment/training-runs/highres-texture-20261124-104731'


def prepare(submit):
    sys.path.insert(0,str(ROOT/'integration'))
    from deploy_yicloud import client,IMAGE
    audit=json.loads((AUDIT/'audit.json').read_text())
    for name,digest in audit['files'].items():
        if sha256(AUDIT/name)!=digest:raise ValueError('Audit input changed')
    rows=[json.loads(x) for x in (AUDIT/'train.jsonl').read_text().splitlines()]
    eligible=[r for r in rows if r.get('reference_path') and r.get('source_face_short_edge',r.get('native_face_short_edge',0))>=512]
    rng=random.Random(20261125);selected=[];groups=set()
    for low,count in [(70,20),(50,20),(20,8),(30,6),(40,6)]:
        pool=[r for r in eligible if r['age_interval'][0]==low];rng.shuffle(pool)
        pool.sort(key=lambda r:r.get('published_demographic_label')!='Asian')
        current=[]
        for row in pool:
            if row['identity_group'] in groups:continue
            current.append(row);groups.add(row['identity_group'])
            if len(current)==count:break
        if len(current)!=count:raise ValueError('Insufficient distinct audit groups')
        selected.extend(current)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    run=ROOT/'deployment/training-runs'/('matched-latent-'+stamp)
    source=run/'source';source.mkdir(parents=True,exist_ok=False);(run/'data').mkdir()
    for name in ['matched_latent_study.py','run_matched_latent.sh','train_highres_texture.py','evaluate_highres_texture.py','deterministic_quality_ops.py','probe_inversion.py']:
        shutil.copy2(Path(__file__).with_name(name),source/name)
    for name in ['latent_quality.py','source_color.py']:shutil.copy2(ROOT/'integration'/name,source/name)
    (run/'data/train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    for name in ['eval.jsonl','external.jsonl']:shutil.copy2(AUDIT/name,run/'data'/name)
    shutil.copy2(AUDIT/'audit.json',run/'common-audit.json')
    shutil.copy2(REFERENCE/'selection.json',run/'selection.json')
    evaluation=json.loads((run/'selection.json').read_text())
    if groups&{r['identity_group'] for r in evaluation}:raise ValueError('Evaluation group overlap')
    frozen={}
    for item in json.loads((ROOT/'integration/quality/baselines/sam-wplus-v1.json').read_text())['weights']:
        path=ROOT/item['path']
        if sha256(path)!=item['sha256']:raise ValueError('Baseline model changed')
        frozen[str(path)]=item['sha256']
    files=list(source.iterdir())+list((run/'data').iterdir())+[run/'selection.json',run/'common-audit.json']
    files.extend(ROOT/'integration/ascend/src'/name for name in ['sam_npu.py','sam_losses.py','faceattr.py'])
    files.append(ROOT/'deployment/gan-research/face_recognition_sface_2021dec.onnx')
    for row in selected:
        for path,key in [('image_path','sha256'),('reference_path','reference_sha256')]:
            if sha256(row[path])!=row[key]:raise ValueError('Training portrait changed')
            frozen[row[path]]=row[key]
    for row in evaluation:files.append(REFERENCE/'evaluation'/row['name']/'fixed_codes.pt')
    files.append(REFERENCE/'evaluation/report.json')
    frozen.update({str(path):sha256(path) for path in files})
    experiment=dict(profile='matched_training_latent_distribution',steps=1000,min_resolution=64,generator_lr=5e-5,adversarial_weight=0,
        generator_routes={'direct_latent':'direct_latent','matched_wplus':'matched_wplus'},fixed_code_run=str(REFERENCE),
        frozen_inputs=frozen,train=len(selected),selection_count=len(evaluation),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        criteria=dict(identity_drop_max=.02,age_span_min_ratio=.90,missing_faces_increase_max=0,native_review_required=True,no_automatic_promotion=True),
        resources=dict(gpu=1,cpu=4,memory='48Gi',max_running='48m'),
        hypothesis='Change only generator-training latent source: direct Release encoder vs frozen baseline W+ fitting. Both reconstruction-only; no input flip; identical crops, sampling, steps and losses.',
        limitations=['60 training photos: a mechanism pilot, not population validation.','Previously inspected regression set, not a blind test.','Both arms disable horizontal input flip because cached W+ is view-specific.','W+ fitting uses the previous batch-four evaluation recipe; serving may fit requested ages separately.'])
    write_json(run/'experiment.json',experiment);print('PREPARED',run,flush=True)
    if submit:submit_run(run)


def submit_run(run):
    sys.path.insert(0,str(ROOT/'integration'))
    from deploy_yicloud import client,IMAGE
    if (run/'receipt.json').exists():raise ValueError('Run already submitted')
    for filename,digest in json.loads((run/'experiment.json').read_text())['frozen_inputs'].items():
        if sha256(filename)!=digest:raise ValueError('Prepared input changed: '+filename)
    job,ctx=client()
    active=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames='face-age',Self=True,Limit=100,Phases='Pending,Running,Inqueue,Starting,Stopping'))
    if active.Items:raise RuntimeError('One GPU limit: another face-age task is active')
    name='face-age-'+run.name
    payload=dict(Name=name,Project='fdj-infra',ZoneId='cn-shanghai-a',QuotaGroup='fdj_gpu_taskpool',Priority='3',UsePrivateMachine=False,
        AutoDeleteDuration='168h',TaskSpecs={'pilot':dict(Command=['bash',str(run/'source/run_matched_latent.sh'),str(ROOT),str(run)],Env={},Image=IMAGE,
        Replicas=1,CPU='4',Memory='48Gi',GPU='1',Internet=False,MaxRunningDuration='48m',MaxWaitDuration='10m',RestartPolicy='Never',MountGPFS='gpfs://gpfs2/haobinzhang:/root')})
    write_json(run/'request.json',payload)
    specs={k:job.models.CreateJobReqTaskSpec(**v) for k,v in payload['TaskSpecs'].items()}
    try:result=job.create_job(ctx,job.models.CreateJobReq(**(payload|{'TaskSpecs':specs})))
    except Exception:
        found=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames=name,Self=True,Limit=100))
        exact=[x for x in found.Items or [] if x.Name==name]
        if not exact:raise
        result=exact[0]
    receipt=dict(job_id=result.JobId,name=name,run_dir=str(run),gpus=1)
    write_json(run/'receipt.json',receipt);print(json.dumps(receipt),flush=True)


def fit(run):
    if os.environ.get('FACE_AGE_EXPERIMENT_ALLOCATION')!='1':raise RuntimeError('Dedicated GPU allocation required')
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from torch.nn import functional as F
    from latent_quality import LatentQuality
    from deterministic_quality_ops import replace_pools,resize,verify
    base=Path(os.environ['FACE_AGE_BASE']);sys.path.insert(0,str(base/'src'))
    from sam_npu import load_sam
    from sam_losses import AgeEst
    if torch.cuda.device_count()!=1:raise RuntimeError('Exactly one GPU required')
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(20261124)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False;verify()
    net=load_sam('cuda:0').eval().requires_grad_(False)
    release=deepcopy(net.encoder).eval().requires_grad_(False)
    release.load_state_dict(torch.load(base/'out/ft_l1/encoder_step1000.pt',map_location='cpu',weights_only=True))
    age=AgeEst().cuda().eval().requires_grad_(False);replace_pools(age)
    def tensor(rgb):return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1)[None].cuda().float()/127.5-1
    helper=LatentQuality(SimpleNamespace(torch=torch,dev='cuda:0',net=net,identity_encoder=release,age_net=age,tensor=tensor),'constrained',resize=resize)
    replace_pools(helper.identity);replace_pools(helper.perceptual)
    def decode(code):return net(code,input_code=True,input_is_full=True,randomize_noise=False,resize=False)
    folder=run/'train-codes';folder.mkdir(exist_ok=False)
    index=dict(recipe='Frozen baseline: 200-step W+ source anchor + 180-step batch-four constrained edits; same as prior evaluator; source-equal targets use anchor.',items={})
    rows=[json.loads(x) for x in (run/'data/train.jsonl').read_text().splitlines()]
    started=time.monotonic()
    for row in rows:
        if sha256(row['reference_path'])!=row['reference_sha256']:raise ValueError('Training input changed')
        rgb=np.asarray(Image.open(row['reference_path']).convert('RGB'));small=cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA)
        x=tensor(small);source=row['source_age'];item=SimpleNamespace(aligned=small,reference_rgb=rgb,reference_age=source,quality_state=None)
        t=time.monotonic();helper.run(item,source,lambda:None);state=item.quality_state;targets=[20,40,60,75]
        cond=lambda n:torch.cat([x,torch.full_like(x[:,:1],n/100)],1)
        with torch.no_grad():
            delta=torch.cat([release(cond(n)) for n in targets])-release(cond(source))
            pivot=state['anchor']+delta;prior=resize(decode(pivot),(256,256),kind='area')
            teacher=resize(decode(state['source']+delta),(256,256),kind='area')
        def logits(image):
            value=resize((image+1)/2,(224,224),kind='bilinear');return age.net.age(age.net.b((value-age.m)/age.sd)).float()
        with torch.no_grad():prob=logits(teacher).softmax(1);color_ref=F.avg_pool2d(x,32).expand(4,-1,-1,-1)
        correction=torch.zeros_like(pivot[:,:8]).requires_grad_(True);opt=torch.optim.Adam([correction],lr=.012)
        for step in range(180):
            opt.param_groups[0]['lr']=.012*(.2+.8*(1-step/180))
            code=pivot+torch.cat([correction,torch.zeros_like(pivot[:,8:])],1)
            edited=resize(decode(code),(256,256),kind='area')
            ident=(1-(helper.identity.feats(edited)*state['identity']).sum(1)).mean()
            loss=.8*ident+.5*F.kl_div(logits(edited).log_softmax(1),prob,reduction='batchmean')+.6*helper.perceptual(edited,prior).mean()+.05*correction.square().mean()+.1*F.l1_loss(F.avg_pool2d(edited,32),color_ref)
            opt.zero_grad(set_to_none=True);loss.backward()
            if not torch.isfinite(loss) or not torch.isfinite(correction.grad).all():raise RuntimeError('Nonfinite training-side W+ fit')
            opt.step()
        codes=torch.cat([state['anchor'],(pivot+torch.cat([correction,torch.zeros_like(pivot[:,8:])],1)).detach()])
        for i,target in enumerate(targets,1):
            if target==source:codes[i].copy_(codes[0])
        if not torch.isfinite(codes).all():raise ValueError('Invalid cached W+')
        path=folder/(row['name']+'.pt');torch.save(dict(codes=codes.cpu(),conditions=['same','20','40','60','75']),path)
        index['items'][row['name']]=dict(file=path.name,sha256=sha256(path),input_sha256=row['reference_sha256'],source_age=source,seconds=time.monotonic()-t)
        write_json(folder/'index.json',index)
        print('TRAIN_WPLUS_CACHED',len(index['items']),'/',len(rows),row['name'],round(time.monotonic()-t,2),flush=True)
        del loss,edited,correction,opt,codes,item,state,teacher,prior,pivot,delta
    index.update(total_seconds=time.monotonic()-started,count=len(rows),cuda_peak_bytes=torch.cuda.max_memory_allocated())
    write_json(folder/'index.json',index);(folder/'DONE').write_text('Training-only W+ cache complete\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('--submit',action='store_true')
    p=sub.add_parser('submit');p.add_argument('run',type=Path)
    p=sub.add_parser('fit');p.add_argument('run',type=Path)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.submit)
    elif args.action=='submit':submit_run(args.run.resolve())
    else:fit(args.run.resolve())

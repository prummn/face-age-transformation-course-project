"""A paired low-rank W+ residual editor with the entire SAM generator frozen.

Uses existing training-only anchors/codes and 381 available older native training
references. The sole paired difference is real age/region patch GAN supervision.
This is a mechanism study, not a new-data or population-adaptation claim.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
from functools import lru_cache
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from probe_inversion import sha256,write_json

ROOT=Path(os.environ.get('FACE_AGE_REPO',Path(__file__).resolve().parents[2]))
CACHE=ROOT/'deployment/training-runs/original-delta-match-20261125-050503'
PARENT=ROOT/'deployment/training-runs/original-delta-20261124-160055'
NATIVE=ROOT/'deployment/training-runs/native-full-20261124-141846'
REGIONS=((.34,.40),(.66,.40),(.32,.57),(.68,.57),(.50,.69))
ROUTES=['wplus_base','latent_editor_control','latent_editor_gan']


def lines(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def validate(run):
    for filename,digest in json.loads((run/'experiment.json').read_text())['frozen_inputs'].items():
        if sha256(filename)!=digest:raise ValueError('Frozen input changed: '+filename)


def prepare():
    sources=lines(CACHE/'data/train.jsonl')
    references=[r for r in lines(NATIVE/'data/train-resolved.jsonl') if r['age_interval'][0] in (50,70)]
    protected=lines(NATIVE/'data/eval.jsonl')+lines(NATIVE/'data/external.jsonl')
    evaluation=json.loads((PARENT/'selection.json').read_text())
    if len(sources)!=60 or len(evaluation)!=35 or len(references)!=381:raise ValueError('Expected frozen60/35/381 design')
    held_groups={r['identity_group'] for r in protected+evaluation};held_hashes={r['sha256'] for r in protected+evaluation}
    if {r['identity_group'] for r in sources+references}&held_groups:raise ValueError('Protected common group overlap')
    if {r['sha256'] for r in sources+references}&held_hashes:raise ValueError('Protected file overlap')
    if any(r.get('source_face_short_edge',0)<512 for r in sources+references):raise ValueError('Native reference threshold violated')
    run=ROOT/'deployment/training-runs'/('latent-age-editor-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
    (run/'source').mkdir(parents=True,exist_ok=False);(run/'data').mkdir()
    for name in ['latent_age_editor_study.py','run_latent_age_editor.sh','deterministic_quality_ops.py','probe_inversion.py']:
        shutil.copy2(Path(__file__).with_name(name),run/'source'/name)
    shutil.copy2(ROOT/'integration/source_color.py',run/'source/source_color.py')
    for name,data in [('train',sources),('references',references)]:
        (run/'data'/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data))
    write_json(run/'selection.json',evaluation)
    index=json.loads((CACHE/'train-codes-original/index.json').read_text())['items'];frozen={}
    for row in sources+references:
        for key,digest_key in [('image_path','sha256'),('reference_path','reference_sha256')]:
            if sha256(row[key])!=row[digest_key]:raise ValueError('Source image changed')
            frozen[row[key]]=row[digest_key]
    for row in sources:
        spec=index[row['name']]
        if spec['source_age']!=row['source_age'] or spec['input_sha256']!=row['reference_sha256']:raise ValueError('Cache source mismatch')
        frozen[str(CACHE/'train-codes-original'/spec['file'])]=spec['sha256']
    for spec in json.loads((ROOT/'integration/quality/baselines/sam-wplus-v1.json').read_text())['weights']:
        frozen[str(ROOT/spec['path'])]=spec['sha256']
    files=list((run/'source').iterdir())+list((run/'data').iterdir())+[run/'selection.json',CACHE/'train-codes-original/index.json',PARENT/'evaluation/report.json']
    for row in evaluation:files += [PARENT/'evaluation'/row['name']/f for f in ['reference.png','original_delta_constrained_codes.pt']]
    files += [ROOT/'integration/ascend/src'/f for f in ['sam_npu.py','sam_losses.py','faceattr.py']]
    files += [ROOT/'deployment/gan-research/face_recognition_sface_2021dec.onnx']
    frozen.update({str(p):sha256(p) for p in files})
    exp=dict(profile='frozen_generator_low_rank_latent_editor_pair',cases=35,train=60,real_references=381,steps=2000,
        seed=20261126,cache_run=str(CACHE),parent_run=str(PARENT),routes=ROUTES,frozen_inputs=frozen,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_age_counts=dict(Counter(str(r['age_interval']) for r in sources)),reference_age_counts=dict(Counter(str(r['age_interval']) for r in references)),
        common_recipe=dict(frozen='All SAM encoders, generator, identity/perceptual/age networks',
            input='Same frozen original-SAM-delta plus180step constrained W+; saved source anchor',
            module='Shared per-layer1035->128->64 MLP with8dim layer embedding; fixed64dim PCA basis of4096 StyleGAN mapped random W samples; zero output initialization',
            correction='Add bounded low-rank residual; no target-age or base-displacement rescaling',
            coefficient_scale=.2*math.sqrt(512/64),active='Only target>=55 and target!=source; same/20/40 exactly unchanged',
            learning_rate=2e-4,identity=.8,lpips256=.4,age_kl_three_views=1,coarse_weighted_correction_trust=.1,
            train_targets=[60,75],patch_gan_weight=.1,patch_edge=256,discriminator_lr=2e-4,lazy_r1_interval=16),
        sole_paired_difference='Additional conditional age-bin/region native grayscale/high-pass patch GAN against381 real older training references',
        criteria=dict(identity_drop_max=.02,age_span_min_ratio=.9,missing_faces_increase_max=0,native_review_required=True,no_automatic_promotion=True),
        resources=dict(gpus=1,cpu=4,memory='48Gi',max_running='45m'),
        limitations=['Previously inspected35 development cases; no fresh reserve consumed.',
            'Coarse publisher apparent ages and FairFace pseudo source ages; no new chronological supervision.',
            'PCA limits residual directions, not a guarantee of natural or on-manifold W+.',
            'Only the paired GAN effect is isolated; this editor differs from historical generator adapters.',
            'Offline research only; default API route and weights unchanged.'])
    write_json(run/'experiment.json',exp);print('PREPARED',run,flush=True)


def submit(run):
    validate(run)
    if (run/'receipt.json').exists():raise ValueError('Already submitted')
    sys.path.insert(0,str(ROOT/'integration'));from deploy_yicloud import client,IMAGE
    job,ctx=client();active=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames='face-age',Self=True,Limit=100,Phases='Pending,Running,Inqueue,Starting,Stopping'))
    if active.Items:raise RuntimeError('One-card limit: another project job is active')
    name='face-age-'+run.name
    payload=dict(Name=name,Project='fdj-infra',ZoneId='cn-shanghai-a',QuotaGroup='fdj_gpu_taskpool',Priority='3',UsePrivateMachine=False,AutoDeleteDuration='168h',
        TaskSpecs={'study':dict(Command=['bash',str(run/'source/run_latent_age_editor.sh'),str(ROOT),str(run)],Env={},Image=IMAGE,Replicas=1,CPU='4',Memory='48Gi',GPU='1',Internet=False,MaxRunningDuration='45m',MaxWaitDuration='10m',RestartPolicy='Never',MountGPFS='gpfs://gpfs2/haobinzhang:/root')})
    write_json(run/'request.json',payload);specs={k:job.models.CreateJobReqTaskSpec(**v) for k,v in payload['TaskSpecs'].items()}
    try:r=job.create_job(ctx,job.models.CreateJobReq(**(payload|{'TaskSpecs':specs})))
    except Exception:
        found=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames=name,Self=True,Limit=100))
        exact=[x for x in found.Items or [] if x.Name==name]
        if not exact:raise
        r=exact[0]
    receipt=dict(job_id=r.JobId,name=name,run_dir=str(run),gpus=1);write_json(run/'receipt.json',receipt);print(json.dumps(receipt),flush=True)


def make_editor(torch,basis,mean,scale):
    nn=torch.nn
    class Editor(nn.Module):
        def __init__(self):
            super().__init__();self.register_buffer('basis',basis);self.register_buffer('mean',mean);self.register_buffer('scale',scale)
            self.layer=nn.Embedding(18,8)
            self.body=nn.Sequential(nn.Linear(1035,128),nn.LeakyReLU(.2),nn.Linear(128,64))
            nn.init.zeros_(self.body[-1].weight);nn.init.zeros_(self.body[-1].bias)
        def forward(self,anchor,code,source,target):
            b=code.shape[0];source=torch.as_tensor(source,device=code.device,dtype=code.dtype).reshape(-1).expand(b)
            target=torch.as_tensor(target,device=code.device,dtype=code.dtype).reshape(-1).expand(b)
            age=torch.stack([source/100,target/100,(target-source)/100],-1)[:,None,:].expand(-1,18,-1)
            layer=self.layer(torch.arange(18,device=code.device))[None].expand(b,-1,-1)
            x=torch.cat([(anchor-self.mean)/self.scale,(code-anchor)/self.scale,layer,age],-1)
            coefficients=self.body(x).tanh()*(.2*math.sqrt(512/64))
            active=((target>=55)&(target!=source)).to(code.dtype)[:,None,None]
            correction=coefficients@self.basis.T*active
            return code+correction,correction
    return Editor()


def worker(run):
    validate(run)
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from torch import nn
    from torch.nn import functional as F
    import lpips
    from deterministic_quality_ops import replace_pools,resize,verify
    from source_color import preserve_source_chroma
    if os.environ.get('FACE_AGE_EXPERIMENT_ALLOCATION')!='1' or torch.cuda.device_count()!=1:raise ValueError('Exactly one dedicated GPU required')
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(20261126)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False;verify()
    write_json(run/'device.json',dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(0),device_count=torch.cuda.device_count(),
        total_memory_bytes=torch.cuda.get_device_properties(0).total_memory))
    exp=json.loads((run/'experiment.json').read_text());parent=Path(exp['parent_run']);cache=Path(exp['cache_run'])
    base=Path(os.environ['FACE_AGE_BASE']);sys.path.insert(0,str(base/'src'))
    from sam_npu import load_sam
    from sam_losses import IDLoss,AgeEst
    net=load_sam('cuda:0').eval().requires_grad_(False)
    identity=IDLoss().cuda().eval().requires_grad_(False);age_net=AgeEst().cuda().eval().requires_grad_(False)
    lp=lpips.LPIPS(net='alex',verbose=False).cuda().eval().requires_grad_(False)
    for model in [identity,age_net,lp]:replace_pools(model)
    def fingerprint():
        h=hashlib.sha256()
        for name,t in net.state_dict().items():h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    frozen=fingerprint()
    def decode(code):return net(code,input_code=True,input_is_full=True,randomize_noise=False,resize=False)
    def tensor(rgb):return torch.from_numpy(np.array(rgb,copy=True,order='C')).permute(2,0,1)[None].cuda().float()/127.5-1
    def logits(y):
        x=resize((y+1)/2,(224,224),kind='bilinear')
        return age_net.net.age(age_net.net.b((x-age_net.m)/age_net.sd)).float()
    def views(y):return [y,F.avg_pool2d(y,3,1,1),y[:,:,8:-8,8:-8]]
    evaluation=json.loads((run/'selection.json').read_text())
    prior_report=json.loads((parent/'evaluation/report.json').read_text())
    old={(r['name'],r['condition']):r['sha256'] for r in prior_report['outputs'] if r['route']=='original_delta_constrained'}
    eval_codes={};preflight=[]
    for row in evaluation:
        data=torch.load(parent/'evaluation'/row['name']/'original_delta_constrained_codes.pt',map_location='cuda',weights_only=True)
        codes=data['codes'];eval_codes[row['name']]=codes
        if tuple(codes.shape)!=(5,18,512):raise ValueError('Unexpected eval code shape')
        with torch.no_grad():images=decode(codes)
        for im,label in zip(images,['same','20','40','60','75']):
            rgb=((im.clamp(-1,1)+1)*127.5).permute(1,2,0).byte().cpu().numpy();buffer=BytesIO();Image.fromarray(rgb).save(buffer,format='PNG')
            if hashlib.sha256(buffer.getvalue()).hexdigest()!=old[row['name'],label]:raise ValueError('Baseline PNG differs before training: '+row['name']+'/'+label)
        preflight.append(row['name'])
    write_json(run/'baseline-preflight.json',dict(cases=preflight,pngs=175,exact=True,training_started=False))
    print('EDITOR_BASELINE_PREFLIGHT_OK',flush=True)
    with torch.no_grad():
        mapped=torch.cat([net.decoder.style(torch.randn(256,512,device='cuda')) for _ in range(16)])
        mean=mapped.mean(0);center=mapped-mean;cov=(center.T@center/(len(mapped)-1)).cpu().double()
        eigenvalues,eigenvectors=torch.linalg.eigh(cov);basis=eigenvectors[:,-64:].float().cuda();scale=mapped.std(0).clamp_min(.05)
    write_json(run/'mapped-w-basis.json',dict(samples=4096,rank=64,explained_variance=float(eigenvalues[-64:].sum()/eigenvalues.sum()),basis_sha256=hashlib.sha256(basis.cpu().numpy().tobytes()).hexdigest()))
    del mapped,center,cov,eigenvalues,eigenvectors
    editor=make_editor(torch,basis,mean,scale).cuda();initial=deepcopy(editor.state_dict())
    source_rows=lines(run/'data/train.jsonl');reference_rows=lines(run/'data/references.jsonl')
    index=json.loads((cache/'train-codes-original/index.json').read_text())['items'];cached={};teachers={}
    @lru_cache(maxsize=48)
    def image(path):
        with Image.open(path) as im:im.load();return np.array(im.convert('RGB'))
    for i,row in enumerate(source_rows):
        spec=index[row['name']];data=torch.load(cache/'train-codes-original'/spec['file'],map_location='cuda',weights_only=True)
        if data['conditions']!=['same','60','75'] or tuple(data['codes'].shape)!=(3,18,512):raise ValueError('Invalid training cache')
        cached[row['name']]=data['codes'];x=tensor(cv2.resize(image(row['reference_path']),(256,256),interpolation=cv2.INTER_AREA))
        for j,target in [(1,60),(2,75)]:
            with torch.no_grad():
                raw=resize(decode(data['codes'][j:j+1]),(256,256))
                teachers[row['name'],target]=dict(image=raw,probabilities=[logits(v).softmax(1) for v in views(raw)],identity=identity.feats(x))
        if (i+1)%10==0:print('EDITOR_TEACHERS',i+1,'/60',flush=True)
    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__();layers=[];cin=2
            for cout in (32,64,128,128):layers.extend([nn.utils.spectral_norm(nn.Conv2d(cin,cout,3,2,1)),nn.LeakyReLU(.2)]);cin=cout
            self.body=nn.Sequential(*layers);self.out=nn.Linear(cin,1);self.age=nn.Embedding(2,cin);self.region=nn.Embedding(5,cin)
        def forward(self,x,ages,regions):
            f=self.body(x).mean((2,3));return self.out(f).flatten()+(f*(self.age(ages)+self.region(regions))).sum(1)/math.sqrt(f.shape[1])
    d=Discriminator().cuda();initial_d=deepcopy(d.state_dict())
    def patches(x,jitter,flip,contrast):
        result=[]
        for (cx,cy),(dx,dy) in zip(REGIONS,jitter):
            left=round(cx*1024)-128+dx;top=round(cy*1024)-128+dy;result.append(x[:,:,top:top+256,left:left+256])
        rgb=torch.cat(result);gray=(rgb*rgb.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
        gray=(gray-gray.mean((2,3),keepdim=True))*contrast
        value=torch.cat([gray,gray-F.avg_pool2d(gray,7,1,3)],1)
        return value.flip(-1) if flip else value
    bins={k:[r for r in source_rows if r['age_interval'][0]==k] for k in [20,30,40,50,70]}
    real_bins={k:[r for r in reference_rows if r['age_interval'][0]==k] for k in [50,70]}
    def arm(route,steps,out):
        out.mkdir(exist_ok=False);editor.load_state_dict(initial);d.load_state_dict(initial_d);editor.train()
        torch.cuda.reset_peak_memory_stats()
        gopt=torch.optim.Adam(editor.parameters(),lr=2e-4,betas=(0.0,.99));dopt=torch.optim.Adam(d.parameters(),lr=2e-4,betas=(0.0,.99))
        rng=random.Random(20261126);used=set();used_real=set();started=time.monotonic()
        for step in range(steps):
            row=rng.choice(bins[rng.choice(list(bins))]);name=row['name'];target=rng.choice([60,75]);real=rng.choice(real_bins[50 if target==60 else 70])
            jitter=[(rng.randint(-12,12),rng.randint(-12,12)) for _ in REGIONS];flip=rng.random()<.5;contrast=rng.uniform(.85,1.15)
            codes=cached[name];anchor=codes[:1];code=codes[1:2] if target==60 else codes[2:3]
            edited,correction=editor(anchor,code,row['source_age'],target);raw=decode(edited);small=resize(raw,(256,256));teacher=teachers[name,target]
            fake=patches(raw,jitter,flip,contrast);ages=torch.full((5,),int(target==75),device='cuda',dtype=torch.long);regions=torch.arange(5,device='cuda')
            loss_d=raw.new_zeros(());r1=raw.new_zeros(());adv=raw.new_zeros(())
            if route=='latent_editor_gan':
                real_patch=patches(tensor(image(real['reference_path'])),jitter,flip,contrast).detach().requires_grad_(step%16==0)
                d.requires_grad_(True).train();dopt.zero_grad(set_to_none=True)
                logits_real=d(real_patch,ages,regions);logits_fake=d(fake.detach(),ages,regions)
                loss_d=F.softplus(-logits_real).mean()+F.softplus(logits_fake).mean()
                if step%16==0:
                    grad=torch.autograd.grad(logits_real.sum(),real_patch,create_graph=True)[0];r1=grad.square().sum((1,2,3)).mean();loss_d=loss_d+16*r1
                if not torch.isfinite(loss_d):raise ValueError('Invalid discriminator loss')
                loss_d.backward();torch.nn.utils.clip_grad_norm_(d.parameters(),5);dopt.step();d.requires_grad_(False).eval()
                adv=F.softplus(-d(fake,ages,regions)).mean()
            ident=(1-(identity.feats(small)*teacher['identity']).sum(1)).mean()
            perceptual=lp(small,teacher['image']).mean()
            age_loss=sum(F.kl_div(logits(v).log_softmax(1),q,reduction='batchmean') for v,q in zip(views(small),teacher['probabilities']))/3
            trust=4*correction[:,:8].square().mean()+correction[:,8:].square().mean()
            loss=.8*ident+.4*perceptual+age_loss+.1*trust+.1*min(1,(step+1)/100)*adv
            gopt.zero_grad(set_to_none=True);loss.backward()
            if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in editor.parameters()):raise ValueError('Invalid editor gradient')
            norm=torch.nn.utils.clip_grad_norm_(editor.parameters(),1);gopt.step();used.add(name);used_real.add(real['name'])
            record=dict(step=step+1,name=name,target=target,real=real['name'],jitter=jitter,flip=flip,contrast=contrast,loss=float(loss.detach()),identity=float(ident.detach()),lpips=float(perceptual.detach()),age_kl=float(age_loss.detach()),trust=float(trust.detach()),adversarial=float(adv.detach()),discriminator=float(loss_d.detach()),r1=float(r1.detach()),correction_rms=float(correction.detach().square().mean().sqrt()),gradient_norm=float(norm))
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if step==0 or (step+1)%100==0:print('LATENT_EDITOR_TRAIN',route,json.dumps(record),flush=True)
            if (step+1)%1000==0:torch.save(editor.state_dict(),out/f'editor_step{step+1}.pt')
        path=out/'editor.pt';torch.save(editor.state_dict(),path)
        if route=='latent_editor_gan':torch.save(d.state_dict(),out/'discriminator.pt')
        update=max(float((v-initial[k]).abs().max()) for k,v in editor.state_dict().items())
        if update==0:raise ValueError('Editor did not update')
        write_json(out/'completed.json',dict(steps=steps,seconds=time.monotonic()-started,source_images=len(used),sampled_real_references=len(used_real),
            real_references_used_for_gradients=len(used_real) if route=='latent_editor_gan' else 0,
            parameter_count=sum(p.numel() for p in editor.parameters()),maximum_state_update=update,editor_sha256=sha256(path),cuda_peak_bytes=torch.cuda.max_memory_allocated()))
        return deepcopy(editor.state_dict())
    arm('latent_editor_control',2,run/'probe_control');arm('latent_editor_gan',2,run/'probe_gan')
    states={route:arm(route,exp['steps'],run/route) for route in ROUTES[1:]}
    logs=[lines(run/route/'train.jsonl') for route in ROUTES[1:]]
    paired_fields=['step','name','target','real','jitter','flip','contrast']
    if any(len(log)!=exp['steps'] for log in logs) or any(any(a[k]!=b[k] for k in paired_fields) for a,b in zip(*logs)):
        raise ValueError('Paired training sampling differs')
    if any(len({r['name'] for r in log})!=len(source_rows) for log in logs):
        raise ValueError('Training source coverage incomplete')
    if fingerprint()!=frozen:raise ValueError('Frozen SAM weights changed')
    write_json(run/'frozen-verification.json',dict(before=frozen,after=frozen,all_sam_parameters_unchanged=True))
    out=run/'evaluation';out.mkdir(exist_ok=False);detector=cv2.FaceDetectorYN.create(str(base/'models/yunet/face_detection_yunet_2023mar.onnx'),'',(256,256),.6,.3,100)
    sf=cv2.FaceRecognizerSF.create(str(ROOT/'deployment/gan-research/face_recognition_sface_2021dec.onnx'),'')
    def feature(rgb):
        bgr=cv2.cvtColor(cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2BGR);_,faces=detector.detect(bgr)
        if faces is None:return None
        face=max(faces,key=lambda f:f[2]*f[3]);v=sf.feature(sf.alignCrop(bgr,face)).ravel();return v/(np.linalg.norm(v)+1e-12)
    report=dict(records=evaluation,routes=ROUTES,outputs=[],timings=[],baseline_route='original_delta_constrained',same_anchor=True,comparison=exp['sole_paired_difference'])
    checked=[]
    for row in evaluation:
        folder=out/row['name'];folder.mkdir();ref=np.asarray(Image.open(parent/'evaluation'/row['name']/'reference.png').convert('RGB'));Image.fromarray(ref).save(folder/'reference.png');own=feature(ref)
        codes=eval_codes[row['name']];targets=torch.tensor([row['source_age'],20,40,60,75],device='cuda')
        for route in ROUTES:
            if route!='wplus_base':editor.load_state_dict(states[route]);editor.eval()
            started=time.monotonic()
            with torch.no_grad():
                final_codes=codes if route=='wplus_base' else editor(codes[:1].expand(5,-1,-1),codes,row['source_age'],targets)[0]
                if route!='wplus_base' and not torch.equal(final_codes[:3],codes[:3]):raise ValueError('Same/20/40 latent codes changed')
                images=decode(final_codes)
            for im,label in zip(images,['same','20','40','60','75']):
                rgb=((im.clamp(-1,1)+1)*127.5).permute(1,2,0).byte().cpu().numpy();path=folder/(route+'_'+label+'.png');Image.fromarray(rgb).save(path);digest=sha256(path)
                if route=='wplus_base' or label in ['same','20','40'] or (label!='same' and int(label)==row['source_age']):
                    if digest!=old[row['name'],label]:raise ValueError('Protected baseline output differs: '+row['name']+'/'+route+'/'+label)
                corrected,_=preserve_source_chroma(rgb,ref);color=path.with_name(path.stem+'_color.png');Image.fromarray(corrected).save(color)
                feat=feature(rgb);target=row['source_age'] if label=='same' else int(label)
                report['outputs'].append(dict(name=row['name'],cohort=row['cohort'],route=route,condition=label,target=target,native=str(path.relative_to(out)),preserved=str(color.relative_to(out)),sha256=digest,detected=feat is not None,source_cosine=float(own@feat) if own is not None and feat is not None else None))
            report['timings'].append(dict(name=row['name'],route=route,decode_save_feature_seconds=time.monotonic()-started))
        checked.append(row['name']);write_json(out/'report.json',report);print('LATENT_EDITOR_EVALUATED',len(checked),'/35',row['name'],flush=True)
    write_json(run/'paired-verification.json',dict(seed=20261126,steps=2000,source_sampling_identical=True,real_sampling_identical=True,same_initial_editor=True,all_sam_frozen=True,baseline_preflight_exact=True,young_same_outputs_bit_exact=True))
    write_json(run/'automatic-completion.json',dict(status='ready_for_independent_CPU_scoring_and_native_review',default_model_changed=False,native_review_pending=True))
    (run/'DONE').write_text('Paired latent editor training and native evaluation complete\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','submit','worker','verify']);p.add_argument('run',nargs='?',type=Path);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.run is None:p.error('run directory required')
    elif a.action=='submit':submit(a.run.resolve())
    elif a.action=='verify':validate(a.run.resolve())
    else:worker(a.run.resolve())

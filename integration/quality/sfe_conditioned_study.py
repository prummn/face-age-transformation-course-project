"""One-GPU paired source/target-conditioned SFE adaptation, no live promotion.

Both arms have identical trainable modules, initialization, samples and age/ID
losses. Only the protected arm adds eye/surround and race/gender consistency.
These are preservation proxies, not a supervised eyeglass detector.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from probe_inversion import sha256, write_json
from sfe_study import NAMES, PARENT, ROOT, prepare as prepare_runtime

NATIVE = ROOT/'deployment/training-runs/native-full-20261124-141846'
RUNTIME = ROOT/'deployment/training-runs/sfe-cpu-20261127-112816/upstream'
ROUTES = ['fixed_mapping', 'age_id_editor', 'protected_editor']
TARGETS = [20, 30, 40, 50, 60, 75]


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def prepare(steps, mode='full'):
    if steps<1:raise ValueError('Training steps must be positive')
    all_train = lines(NATIVE/'data/train-resolved.jsonl')
    protected = lines(NATIVE/'data/eval.jsonl')+lines(NATIVE/'data/external.jsonl')
    evaluation = json.loads((PARENT/'selection.json').read_text())
    groups = {r['identity_group'] for r in protected+evaluation}
    digests = {r[k] for r in protected+evaluation for k in ['sha256', 'reference_sha256'] if k in r}
    eligible = [r for r in all_train if r.get('training_authorized') and r.get('source_face_short_edge',0)>=512
                and r['identity_group'] not in groups and r['reference_sha256'] not in digests and r['sha256'] not in digests]
    # Equal bin counts, unique conservative identity components, deterministic.
    rng = random.Random(20261128); rng.shuffle(eligible); chosen=[]; used=set()
    for age in [20,30,40,50,70]:
        count=0
        for row in eligible:
            if row['age_interval'][0]!=age or row['identity_group'] in used: continue
            chosen.append(row);used.add(row['identity_group']);count+=1
            if count==24:break
        if count!=24:raise ValueError('Insufficient training identities in bin '+str(age))
    run=ROOT/'deployment/training-runs'/('sfe-'+mode+'-conditioned-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
    (run/'source').mkdir(parents=True,exist_ok=False);(run/'data').mkdir()
    for name in ['sfe_conditioned_study.py','sfe_study.py','run_sfe_conditioned.sh','probe_inversion.py']:
        shutil.copy2(Path(__file__).with_name(name),run/'source'/name)
    shutil.copy2(ROOT/'integration/source_color.py',run/'source/source_color.py')
    # New machines can rebuild the pinned inference runtime from the official
    # checkouts; previous experiment directories are an optional local cache.
    runtime=RUNTIME if RUNTIME.exists() else prepare_runtime(cpu_pilot=True)/'upstream'
    shutil.copytree(runtime,run/'upstream',ignore=shutil.ignore_patterns('__pycache__'))
    write_json(run/'data/train.json',chosen)
    rows={r['name']:r for r in evaluation};write_json(run/'selection.json',[rows[n] for n in NAMES])
    specs=[json.loads(p.read_text()) for p in sorted((ROOT/'integration/models').glob('sfe-*.json'))]
    write_json(run/'model-specs.json',specs)
    frozen={}
    for r in chosen:
        if sha256(r['reference_path'])!=r['reference_sha256']:raise ValueError('Training image changed')
        frozen[r['reference_path']]=r['reference_sha256']
    for n in NAMES:
        p=PARENT/'evaluation'/n/'reference.png';frozen[str(p)]=sha256(p)
    for spec in specs:
        p=ROOT/'deployment/gan-models/stylefeatureeditor'/spec['name'];frozen[str(p)]=spec['sha256']
    for rel in ['deployment/model-files/face-age/models/irse50/model_ir_se50.pth',
                'deployment/model-files/face-age/models/faceattr/convnext_t_fairface.pt',
                'deployment/gan-research/face_recognition_sface_2021dec.onnx']:
        p=ROOT/rel;frozen[str(p)]=sha256(p)
    for folder in ['source','data','upstream']:
        for p in (run/folder).rglob('*'):
            if p.is_file():frozen[str(p)]=sha256(p)
    for p in [run/'selection.json',run/'model-specs.json']:frozen[str(p)]=sha256(p)
    write_json(run/'experiment.json',dict(profile='source_target_sfe_paired_protection',adapter_mode=mode,steps_per_arm=steps,seed=20261128,
        train=len(chosen),train_age_bins=dict(Counter(str(r['age_interval']) for r in chosen)),
        train_published_demographics=dict(Counter(str(r.get('published_demographic_label')) for r in chosen)),
        evaluation=NAMES,targets=TARGETS,routes=ROUTES,frozen_inputs=frozen,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_age='Frozen FairFace expected apparent age, not true age',
        architecture=('Frozen SFE inversion, e4e, feature editor and StyleGAN2; trained conditional strength plus bounded W+ and 16-channel spatial residuals; exact same-age gate' if mode=='full' else
            'Residual-ablation: identical module initialization, sampling and losses; W+ and spatial residuals forced to zero; learn only conditional scalar strength, with the same [-5,10] bounds and exact same-age gate'),
        common_losses=dict(age_kl=1,age_expectation=.5,arcface=1,lpips=.5,residual=.05),
        paired_difference=dict(eye_blurred_l1=2,surround_blurred_l1=1,gender_kl=.5,race_kl=.25),
        criteria=dict(identity_drop_max=.02,age_span_ratio_min=.9,age_proxy_mae_gain_min=1,missing_faces_increase_max=0,
            native_visual_review=True,no_automatic_promotion=True),
        resources=dict(gpu=1,max_running='45m'),
        limitations=['120 existing training identities; no new high quality cross-age same-person pairs.',
            '8 previously inspected development images, possible FFHQ pretraining overlap; reserved unseen images untouched.',
            'FairFace age supervision ceiling75; no85/90 training or chronological-age claim.',
            'Eye/surround consistency may also suppress valid aging; independent age/identity and visual inspection required.',
            'This is an initial bounded mechanism study, not full convergence or population-level evaluation.']))
    print('PREPARED',run,flush=True)


def validate(run):
    exp=json.loads((run/'experiment.json').read_text())
    for p,digest in exp['frozen_inputs'].items():
        if sha256(p)!=digest:raise ValueError('Frozen input changed: '+p)
    return exp


def submit(run):
    validate(run)
    if (run/'receipt.json').exists():raise ValueError('Already submitted')
    sys.path.insert(0,str(ROOT/'integration'));from deploy_yicloud import client,IMAGE
    job,ctx=client()
    active=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames='face-age',Self=True,
        Limit=100,Phases='Pending,Running,Inqueue,Starting,Stopping'))
    # Conservatively wait for all current project tasks, including CPU studies.
    if active.Items:raise RuntimeError('Wait for existing project tasks before allocating GPU')
    name='face-age-'+run.name
    payload=dict(Name=name,Project='fdj-infra',ZoneId='cn-shanghai-a',QuotaGroup='fdj_gpu_taskpool',Priority='3',
        UsePrivateMachine=False,AutoDeleteDuration='168h',TaskSpecs={'study':dict(
        Command=['bash',str(run/'source/run_sfe_conditioned.sh'),str(ROOT),str(run)],Env={},Image=IMAGE,Replicas=1,
        CPU='4',Memory='48Gi',GPU='1',Internet=False,MaxRunningDuration='45m',MaxWaitDuration='10m',
        RestartPolicy='Never',MountGPFS='gpfs://gpfs2/haobinzhang:/root')})
    write_json(run/'request.json',payload)
    req=payload|{'TaskSpecs':{k:job.models.CreateJobReqTaskSpec(**v) for k,v in payload['TaskSpecs'].items()}}
    try:r=job.create_job(ctx,job.models.CreateJobReq(**req))
    except Exception:
        found=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames=name,Self=True,Limit=100))
        exact=[x for x in found.Items or [] if x.Name==name]
        if not exact:raise
        r=exact[0]
    receipt=dict(job_id=r.JobId,name=name,run_dir=str(run),gpus=1)
    write_json(run/'receipt.json',receipt);print(json.dumps(receipt),flush=True)


def worker(run):
    exp=validate(run)
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from torch import nn
    from torch.nn import functional as F
    import lpips
    import timm
    from source_color import preserve_source_chroma
    if os.environ.get('FACE_AGE_EXPERIMENT_ALLOCATION')!='1' or torch.cuda.device_count()!=1:
        raise ValueError('Exactly one allocated GPU required')
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(exp['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    sys.path.insert(0,str(run/'upstream'))
    from inference_only import InferenceOnly,EditingOnly
    from models.psp.encoders.model_irse import Backbone
    paths=ROOT/'deployment/gan-models/stylefeatureeditor'
    tick=time.monotonic()
    model=InferenceOnly(paths/'sfe_editor_light.pt',paths/'e4e_ffhq_encode.pt',paths/'stylegan2-ffhq-config-f.pt')
    direction=torch.load(run/'upstream/age.pt',map_location='cuda',weights_only=False).reshape(1,1,512)
    runner=EditingOnly(model,direction)
    identity=Backbone(input_size=112,num_layers=50,drop_ratio=.6,mode='ir_se').cuda()
    identity.load_state_dict(torch.load(ROOT/'deployment/model-files/face-age/models/irse50/model_ir_se50.pth',map_location='cpu',weights_only=True),strict=True)
    class Attributes(nn.Module):
        def __init__(self):
            super().__init__();self.b=timm.create_model('convnext_tiny.fb_in22k_ft_in1k',pretrained=False,num_classes=0)
            self.race=nn.Linear(self.b.num_features,7);self.gender=nn.Linear(self.b.num_features,2);self.age=nn.Linear(self.b.num_features,9)
        def forward(self,x):
            mean=x.new_tensor([.485,.456,.406])[None,:,None,None];std=x.new_tensor([.229,.224,.225])[None,:,None,None]
            f=self.b((F.interpolate((x+1)/2,(224,224),mode='bilinear',align_corners=False)-mean)/std)
            return self.race(f),self.gender(f),self.age(f)
    attributes=Attributes().cuda()
    attributes.load_state_dict(torch.load(ROOT/'deployment/model-files/face-age/models/faceattr/convnext_t_fairface.pt',map_location='cpu',weights_only=True),strict=True)
    perceptual=lpips.LPIPS(net='alex',verbose=False).cuda()
    for net in [identity,attributes,perceptual]:net.eval().requires_grad_(False)
    centers=torch.tensor([1,6,15,25,35,45,55,65,75.],device='cuda')
    def idfeat(x):return identity(F.adaptive_avg_pool2d(x[:,:,35:223,32:220],(112,112)))
    def fingerprint():
        h=hashlib.sha256()
        for prefix,net in [('sfe',model),('id',identity),('attributes',attributes),('lpips',perceptual)]:
            for k,v in net.state_dict().items():h.update((prefix+k).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        h.update(model.latent_avg.cpu().numpy().tobytes());h.update(direction.cpu().numpy().tobytes())
        return h.hexdigest()
    frozen=fingerprint()
    def tensor(path):
        rgb=np.array(Image.open(path).convert('RGB'),copy=True)
        if rgb.shape!=(1024,1024,3):raise ValueError('Expected native1024 source')
        return torch.from_numpy(rgb).permute(2,0,1)[None].cuda().float()/127.5-1
    def pixels(y):
        if tuple(y.shape)!=(1,3,1024,1024) or not torch.isfinite(y).all():raise ValueError('Invalid output')
        return ((y[0].clamp(-1,1)+1)*127.5).permute(1,2,0).byte().cpu().numpy()
    @torch.no_grad()
    def invert(path):
        source=tensor(path);_,inv=runner._run_on_batch(source)
        x=F.interpolate(source,(256,256),mode='bilinear',align_corners=False)
        r,g,a=attributes(x);source_age=float(a.softmax(1)@centers)
        _,feats=model.decoder([inv['w_e4e']],input_is_latent=True,randomize_noise=False,return_features=True,early_stop=64)
        return dict(w=inv['latents'].detach(),e4e=inv['w_e4e'].detach(),fused=inv['fused_feat'].to('cuda').detach(),
            fs0=feats[9].detach(),x=x.detach(),source_age=source_age,id=idfeat(x).detach(),race=r.softmax(1).detach(),gender=g.softmax(1).detach())
    class Adapter(nn.Module):
        def __init__(self):
            super().__init__();self.body=nn.Sequential(nn.Linear(515,128),nn.LeakyReLU(.2),nn.Linear(128,128),nn.LeakyReLU(.2))
            self.out=nn.Linear(128,1+18*512)
            self.spatial=nn.Sequential(nn.Conv2d(1024,16,1),nn.LeakyReLU(.2),nn.Conv2d(16,512,1))
            nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
            nn.init.zeros_(self.spatial[-1].weight);nn.init.zeros_(self.spatial[-1].bias)
        def control(self,inv,target):
            w=inv['w'];source=inv['source_age'];d=float(target)-source;gate=min(abs(d)/40,1.)
            cond=w.new_tensor([[source/75,float(target)/75,d/55]])
            out=self.out(self.body(torch.cat([w.mean(1)/3,cond],1)))
            strength=((d/10)+2*torch.tanh(out[:,:1])*gate).clamp(-5,10)
            residual=.15*torch.tanh(out[:,1:].reshape(1,18,512))*gate
            # Preserve the coarsest identity/pose and finest stochastic styles.
            mask=residual.new_zeros(1,18,1);mask[:,4:14]=1
            return strength,residual*mask*(0 if exp.get('adapter_mode')=='strength' else 1),gate
    adapter=Adapter().cuda();initial=deepcopy(adapter.state_dict())
    def decode(inv,target,adapt=True):
        strength,residual,gate=adapter.control(inv,target)
        if not adapt:
            strength=strength.detach()*0+max(-5,min(10,(float(target)-inv['source_age'])/10));residual=residual*0
        w=inv['w']+strength[:,:,None]/2*direction+residual
        edited_e4e=inv['e4e']+strength[:,:,None]/2*direction
        _,features=model.decoder([edited_e4e],input_is_latent=True,randomize_noise=False,return_features=True,early_stop=64)
        delta=inv['fs0']-features[9]
        edited=model.encoder(torch.cat([inv['fused'],delta],1))
        spatial=.1*torch.tanh(adapter.spatial(torch.cat([inv['fused'],delta],1)))*gate
        if not adapt or exp.get('adapter_mode')=='strength':spatial=spatial*0
        edited=edited+spatial
        y,_=model.decoder([w],input_is_latent=True,new_features=[None]*9+[edited]+[None]*8,
            feature_scale=1.,randomize_noise=False)
        return y,dict(strength=strength,residual=residual,spatial=spatial)
    training=json.loads((run/'data/train.json').read_text())
    cache=[]
    for i,row in enumerate(training):
        inv=invert(row['reference_path']);cache.append({k:v.cpu() if torch.is_tensor(v) else v for k,v in inv.items()})
        if (i+1)%20==0:print('CACHE',i+1,'/',len(training),flush=True)
    def cuda(inv):return {k:v.cuda() if torch.is_tensor(v) else v for k,v in inv.items()}
    write_json(run/'cache-summary.json',dict(items=len(cache),source_ages=[x['source_age'] for x in cache],
        seconds=time.monotonic()-tick,cache='CPU memory, recreated from hashed sources; no foreign W+ caches'))
    eye=torch.zeros((1,1,256,256),device='cuda');eye[:,:,84:118,52:204]=1
    surround=torch.ones_like(eye);surround[:,:,45:220,38:218]=0
    @torch.no_grad()
    def same_check(inv):
        adapter.load_state_dict(initial)
        source=inv['source_age'];a,_=decode(inv,source,False);b,c=decode(inv,source,True)
        if not torch.equal(a,b):raise ValueError('Zero-init/same-age identity gate differs')
        return dict(raw_pixels_equal=True,zero_strength=float(c['strength'])==0,shape=list(a.shape))
    write_json(run/'same-age-preflight.json',same_check(cuda(cache[0])))
    trained={};logs={}
    for route in ROUTES[1:]:
        adapter.load_state_dict(initial);adapter.train();optimizer=torch.optim.Adam(adapter.parameters(),lr=2e-4,betas=(.9,.99))
        rng=random.Random(exp['seed']);started=time.monotonic();arm_logs=[];used=set()
        for step in range(exp['steps_per_arm']):
            index=rng.randrange(len(cache));inv=cuda(cache[index]);used.add(index)
            target=TARGETS[rng.randrange(len(TARGETS))]
            optimizer.zero_grad(set_to_none=True);y,controls=decode(inv,target)
            small=F.interpolate(y,(256,256),mode='area');r,g,a=attributes(small)
            probability=torch.softmax(-.5*((centers-target)/8)**2,0)[None]
            age=F.kl_div(a.log_softmax(1),probability,reduction='batchmean')
            age_expectation=F.smooth_l1_loss((a.softmax(1)@centers)/30,small.new_tensor([target/30]))
            ident=1-(idfeat(small)*inv['id']).sum(1).mean()
            lp=perceptual(small,inv['x']).mean()
            trust=controls['residual'].square().mean()+controls['spatial'].square().mean()
            blurred=F.avg_pool2d(small,7,1,3);source_blurred=F.avg_pool2d(inv['x'],7,1,3)
            difference=(blurred-source_blurred).abs()
            eyes=(difference*eye).sum()/(eye.sum()*3)
            outer=(difference*surround).sum()/(surround.sum()*3)
            gender=F.kl_div(g.log_softmax(1),inv['gender'],reduction='batchmean')
            race=F.kl_div(r.log_softmax(1),inv['race'],reduction='batchmean')
            loss=age+.5*age_expectation+ident+.5*lp+.05*trust
            if route=='protected_editor':loss=loss+2*eyes+outer+.5*gender+.25*race
            if not torch.isfinite(loss):raise ValueError('Non-finite loss')
            loss.backward()
            grads=[p.grad for p in adapter.parameters() if p.grad is not None]
            if not grads or any(not torch.isfinite(v).all() for v in grads):raise ValueError('Invalid gradients')
            if step==0:
                magnitude=sum(float(v.abs().sum()) for v in grads)
                if magnitude<=0:raise ValueError('No learning signal')
                if any(p.grad is not None for p in model.parameters()):raise ValueError('Frozen model acquired gradients')
                write_json(run/(route+'-backward.json'),dict(finite=True,gradient_abs_sum=magnitude,frozen_model_gradients=False))
            torch.nn.utils.clip_grad_norm_(adapter.parameters(),5);optimizer.step()
            if step==0 or (step+1)%20==0:
                row=dict(step=step+1,index=index,target=target,source_age=inv['source_age'],
                    **{k:float(v.detach()) for k,v in dict(loss=loss,age=age,id=ident,lpips=lp,eye=eyes,surround=outer,gender=gender,race=race,strength=controls['strength']).items()},
                    seconds=time.monotonic()-started)
                arm_logs.append(row);write_json(run/(route+'-log.json'),arm_logs);print('TRAIN',route,json.dumps(row),flush=True)
        trained[route]=deepcopy(adapter.cpu().state_dict());adapter.cuda()
        torch.save(dict(state_dict=trained[route],experiment=exp,route=route),run/(route+'.pt'))
        logs[route]=dict(steps=exp['steps_per_arm'],seconds=time.monotonic()-started,unique_sources=len(used))
    adapter.eval();out=run/'evaluation';out.mkdir(exist_ok=False)
    det=cv2.FaceDetectorYN.create(str(ROOT/'deployment/model-files/face-age/models/yunet/face_detection_yunet_2023mar.onnx'),'',(256,256),.6,.3,100)
    rec=cv2.FaceRecognizerSF.create(str(ROOT/'deployment/gan-research/face_recognition_sface_2021dec.onnx'),'')
    def feature(rgb):
        bgr=cv2.cvtColor(cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2BGR);_,faces=det.detect(bgr)
        if faces is None:return None
        f=rec.feature(rec.alignCrop(bgr,max(faces,key=lambda v:v[2]*v[3]))).ravel();return f/(np.linalg.norm(f)+1e-12)
    report=dict(records=json.loads((run/'selection.json').read_text()),routes=ROUTES,targets=TARGETS,outputs=[],training=logs)
    with torch.no_grad():
        for record in report['records']:
            name=record['name'];folder=out/name;folder.mkdir();ref=PARENT/'evaluation'/name/'reference.png'
            shutil.copy2(ref,folder/'reference.png');source=np.array(Image.open(ref).convert('RGB'));source_id=feature(source);inv=invert(ref)
            for route in ROUTES:
                adapter.load_state_dict(initial if route=='fixed_mapping' else trained[route])
                for condition in ['same']+[str(t) for t in TARGETS]:
                    target=inv['source_age'] if condition=='same' else int(condition)
                    torch.cuda.synchronize();start=time.monotonic();y,c=decode(inv,target,route!='fixed_mapping');torch.cuda.synchronize()
                    seconds=time.monotonic()-start;rgb=pixels(y);f=feature(rgb)
                    path=folder/(route+'_'+condition+'.png');Image.fromarray(rgb).save(path)
                    color=path.with_name(path.stem+'_color.png');color_rgb,_=preserve_source_chroma(source,rgb);Image.fromarray(color_rgb).save(color)
                    report['outputs'].append(dict(name=name,cohort=record['cohort'],route=route,condition=condition,target=target,
                        source_age=inv['source_age'],scale=float(c['strength']),native=str(path.relative_to(out)),preserved=str(color.relative_to(out)),
                        sha256=sha256(path),source_cosine=float(source_id@f) if source_id is not None and f is not None else None,
                        detected=f is not None,seconds=seconds,reconstruction_l1=float(np.abs(rgb.astype(float)-source).mean()) if condition=='same' else None))
            write_json(out/'report.partial.json',report);print('EVAL',name,flush=True)
    after=fingerprint()
    if frozen!=after:raise ValueError('Frozen model tensors changed')
    write_json(run/'frozen-verification.json',dict(before=frozen,after=after,equal=True))
    write_json(run/'device.json',dict(torch=torch.__version__,python=sys.version,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),device_count=1,peak_bytes=torch.cuda.max_memory_allocated()))
    report['total_seconds']=time.monotonic()-tick;write_json(out/'report.json',report)
    (run/'DONE').write_text('success\n');print('CONDITIONED_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','submit','worker'])
    p.add_argument('run',type=Path,nargs='?');p.add_argument('--steps',type=int,default=400);p.add_argument('--mode',choices=['full','strength'],default='full');a=p.parse_args()
    if a.action=='prepare':prepare(a.steps,a.mode)
    elif a.run is None:p.error('run required')
    else:globals()[a.action](a.run.resolve())

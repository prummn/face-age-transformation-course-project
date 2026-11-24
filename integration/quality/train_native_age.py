"""Full native-face dataset adaptation with matched cached codes and RGB GAN.

Cache fitting uses training faces only. Both arms see every training row before
weighted replay; the sole paired difference is age/region-conditioned RGB GAN.
Default serving models and original checkpoint files are never written.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from probe_inversion import sha256,write_json

REGIONS=((.34,.40),(.66,.40),(.32,.57),(.68,.57),(.50,.69))
def rows(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def setup():
    import cv2
    import torch
    if os.environ.get('FACE_AGE_EXPERIMENT_ALLOCATION')!='1' or torch.cuda.device_count()!=1:raise ValueError('Exactly one dedicated GPU required')
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(20261124)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    sys.path.insert(0,str(Path(os.environ['FACE_AGE_BASE'])/'src'))
    from sam_npu import load_sam
    from sam_losses import IDLoss,AgeEst
    import lpips
    net=load_sam('cuda:0').eval().requires_grad_(False)
    net.encoder.load_state_dict(torch.load(Path(os.environ['FACE_AGE_BASE'])/'out/ft_l1/encoder_step1000.pt',map_location='cpu',weights_only=True))
    return net,IDLoss().cuda().eval().requires_grad_(False),AgeEst().cuda().eval().requires_grad_(False),lpips.LPIPS(net='alex',verbose=False).cuda().eval().requires_grad_(False)


def cache(run,batch_size=8):
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from torch.nn import functional as F
    net,identity,age,lp=setup();data=rows(run/'data/train.jsonl');folder=run/'train-codes';folder.mkdir(exist_ok=True)
    index=json.loads((folder/'index.json').read_text()) if (folder/'index.json').exists() else dict(items={},source_steps=80,target_steps=60,batch_size=batch_size,
        recipe='Training-only W+ matching:80step source anchor,60step front8layer target correction at60/75; independently fitted to baseline, no candidate-specific fitting. Approximation to200/180serving recipe, disclosed distribution difference.')
    def image(row):
        if sha256(row['reference_path'])!=row['reference_sha256']:raise ValueError('Original changed')
        return np.asarray(Image.open(row['reference_path']).convert('RGB'))
    def tensor(rgb):return torch.from_numpy(np.ascontiguousarray(rgb)).permute(0,3,1,2).contiguous().cuda().float()/127.5-1
    def decode(code):return net(code,input_code=True,input_is_full=True,randomize_noise=False,resize=False)
    def logits(y):
        v=F.interpolate((y+1)/2,(224,224),mode='bilinear',align_corners=False)
        return age.net.age(age.net.b((v-age.m)/age.sd)).float()
    pending=[]
    for row in data:
        saved=index['items'].get(row['name'])
        if saved:
            if sha256(folder/saved['file'])!=saved['sha256'] or saved['input_sha256']!=row['reference_sha256']:raise ValueError('Existing code changed')
        else:pending.append(row)
    start=time.monotonic();grid=torch.linspace(0,1,512,device='cuda');yy,xx=torch.meshgrid(grid,grid,indexing='ij');mask=(.2+.8*((1-((xx-.5)/.31)**2-((yy-.53)/.36)**2)/.25).clamp(0,1))[None,None]
    for first in range(0,len(pending),batch_size):
        batch=pending[first:first+batch_size];t=time.monotonic();rgb=[image(r) for r in batch];x=tensor(np.stack([cv2.resize(v,(256,256),interpolation=cv2.INTER_AREA) for v in rgb]));ref=tensor(np.stack([cv2.resize(v,(512,512),interpolation=cv2.INTER_AREA) for v in rgb]));n=len(batch)
        with torch.no_grad():
            source_age=((age(x)[0]*100).clamp(0,100).round().long()).flatten()
            # Existing references keep the frozen published SAM conditioning;
            # new rows obtain the same frozen FairFace pseudo source age.
            source_age=torch.tensor([r['source_age'] if r.get('source_age') is not None else int(source_age[i]) for i,r in enumerate(batch)],device='cuda')
            cond=lambda ages:torch.cat([x,ages[:,None,None,None].expand(-1,1,256,256)/100],1)
            base=net.pretrained_encoder(x)+net.latent_avg;source=base+net.encoder(cond(source_age));idref=identity.feats(x)
        anchor=source.detach().clone().requires_grad_(True);opt=torch.optim.Adam([anchor],lr=.02)
        for step in range(80):
            opt.param_groups[0]['lr']=.02*(.2+.8*(1-step/80));y=decode(anchor);s=F.interpolate(y,(256,256),mode='area');large=F.interpolate(y,(512,512),mode='area')
            pixel=((large-ref).abs()*mask).sum()/(mask.sum()*3*n);ident=(1-(identity.feats(s)*idref).sum(1)).mean()
            loss=.5*pixel+.8*lp(s,x).mean()+.8*ident+.002*(anchor-source).square().mean();opt.zero_grad(set_to_none=True);loss.backward()
            if not torch.isfinite(loss) or not torch.isfinite(anchor.grad).all():raise ValueError('Nonfinite source fit')
            opt.step()
        anchor=anchor.detach()
        with torch.no_grad():
            targetcodes=[]
            for target in (60,75):targetcodes.append(net.encoder(cond(torch.full_like(source_age,target)))-net.encoder(cond(source_age)))
            delta=torch.stack(targetcodes,dim=1).flatten(0,1);pivot=anchor[:,None].expand(-1,2,-1,-1).flatten(0,1)+delta
            teacher=F.interpolate(decode(source[:,None].expand(-1,2,-1,-1).flatten(0,1)+delta),(256,256),mode='area');prob=logits(teacher).softmax(1)
            prior=F.interpolate(decode(pivot),(256,256),mode='area');ids=idref[:,None].expand(-1,2,-1).flatten(0,1);color=F.avg_pool2d(x,32)[:,None].expand(-1,2,-1,-1,-1).flatten(0,1)
        correction=torch.zeros_like(pivot[:,:8]).requires_grad_(True);opt=torch.optim.Adam([correction],lr=.012)
        for step in range(60):
            opt.param_groups[0]['lr']=.012*(.2+.8*(1-step/60));edited=F.interpolate(decode(pivot+torch.cat([correction,torch.zeros_like(pivot[:,8:])],1)),(256,256),mode='area')
            loss=.8*(1-(identity.feats(edited)*ids).sum(1)).mean()+.5*F.kl_div(logits(edited).log_softmax(1),prob,reduction='batchmean')+.6*lp(edited,prior).mean()+.05*correction.square().mean()+.1*F.l1_loss(F.avg_pool2d(edited,32),color)
            opt.zero_grad(set_to_none=True);loss.backward()
            if not torch.isfinite(loss) or not torch.isfinite(correction.grad).all():raise ValueError('Nonfinite target fit')
            opt.step()
        target=(pivot+torch.cat([correction,torch.zeros_like(pivot[:,8:])],1)).detach().reshape(n,2,18,512)
        codes=torch.cat([anchor[:,None],target],1)
        for i,row in enumerate(batch):
            own=int(source_age[i])
            for j,targ in enumerate((60,75),1):
                if own==targ:codes[i,j].copy_(codes[i,0])
            path=folder/(row['name']+'.pt');torch.save(dict(codes=codes[i].cpu(),conditions=['same','60','75']),path)
            index['items'][row['name']]=dict(file=path.name,sha256=sha256(path),input_sha256=row['reference_sha256'],source_age=own)
        index.update(count=len(index['items']),elapsed_seconds=time.monotonic()-start,cuda_peak_bytes=torch.cuda.max_memory_allocated());write_json(folder/'index.json',index)
        print('NATIVE_CODES',len(index['items']),'/',len(data),'batch_seconds',round(time.monotonic()-t,1),'eta_minutes',round((time.monotonic()-start)/(first+len(batch))*(len(pending)-first-len(batch))/60,1),flush=True)
        del loss,y,s,large,opt,correction,codes,target,pivot,prior,teacher,prob,anchor,source,delta
    data=[dict(r,source_age=index['items'][r['name']]['source_age']) for r in data]
    (run/'data/train-resolved.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data));(folder/'DONE').write_text('All native training source/target codes cached\n')


def train(run,arm,steps=4000,batch_size=2):
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from torch import nn
    from torch.nn import functional as F
    net,identity,age,lp=setup();data=rows(run/'data/train-resolved.jsonl');held=rows(run/'data/eval.jsonl')+rows(run/'data/external.jsonl')
    if {r['identity_group'] for r in data}&{r['identity_group'] for r in held}:raise ValueError('Protected groups overlap')
    for i,module in enumerate(net.decoder.convs):
        if 2**(3+i//2)>=64:module.conv.requires_grad_(True);module.activate.requires_grad_(True)
    named={n:p for n,p in net.decoder.named_parameters() if p.requires_grad};initial={n:p.detach().clone() for n,p in named.items()}
    def frozen_digest():
        h=hashlib.sha256()
        for n,v in net.state_dict().items():
            if n.startswith('decoder.') and n[8:] in named:continue
            h.update(n.encode());h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    before=frozen_digest();out=run/arm;out.mkdir(exist_ok=False);index=json.loads((run/'train-codes/index.json').read_text());cached={}
    native={}
    for r in data:
        spec=index['items'][r['name']];path=run/'train-codes'/spec['file']
        if sha256(path)!=spec['sha256'] or sha256(r['reference_path'])!=r['reference_sha256']:raise ValueError('Training input changed')
        cached[r['name']]=torch.load(path,map_location='cuda',weights_only=True)['codes']
        native[r['name']]=np.asarray(Image.open(r['reference_path']).convert('RGB'))
    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__();layers=[];cin=4
            for cout in (32,64,128,192,192):layers.extend([nn.utils.spectral_norm(nn.Conv2d(cin,cout,3,2,1)),nn.LeakyReLU(.2)]);cin=cout
            self.body=nn.Sequential(*layers);self.out=nn.Linear(192,1);self.age=nn.Embedding(5,192);self.region=nn.Embedding(6,192)
        def forward(self,x,ages,regions):
            f=self.body(x).mean((2,3));return self.out(f).flatten()+(f*(self.age(ages)+self.region(regions))).sum(1)/math.sqrt(192)
    d=Discriminator().cuda();gopt=torch.optim.Adam(list(named.values()),lr=1e-4,betas=(0.,.99));dopt=torch.optim.Adam(d.parameters(),lr=1e-4,betas=(0.,.99))
    bins={lo:[r for r in data if r['age_interval'][0]==lo] for lo in (20,30,40,50,70)};rng=random.Random(20261124);order=list(data);rng.shuffle(order);used=set();used_real=set();start=time.monotonic()
    def tensor(names):return torch.from_numpy(np.stack([native[n] for n in names])).permute(0,3,1,2).contiguous().cuda().float()/127.5-1
    def decode(code):return net(code,input_code=True,input_is_full=True,randomize_noise=False,resize=False)
    def logits(y):
        # Age protection on blurred views avoids explicitly preserving sharp
        # wrinkle patterns from the original generator.
        y=F.avg_pool2d(y,9,1,4);v=F.interpolate((y+1)/2,(224,224),mode='bilinear',align_corners=False)
        return age.net.age(age.net.b((v-age.m)/age.sd)).float()
    def patches(images,jitters,flip,contrast):
        output=[]
        for i,image in enumerate(images):
            for (cx,cy),(dx,dy) in zip(REGIONS,jitters[i]):
                x=round(cx*1024)-128+dx;y=round(cy*1024)-128+dy;output.append(image[:,y:y+256,x:x+256])
            output.append(F.interpolate(image[None,:,180:880,180:844],(256,256),mode='bilinear',align_corners=False)[0])
        z=torch.stack(output)
        if flip:z=z.flip(-1)
        z=(z-z.mean((2,3),keepdim=True))*contrast
        gray=(z*z.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True);hp=gray-F.avg_pool2d(gray,7,1,3)
        return torch.cat([z,hp],1)
    plan=dict(arm=arm,train=len(data),steps=steps,batch=batch_size,seed=20261124,all_rows_first=True,min_resolution=64,generator_lr=1e-4,
        latent_recipe=index['recipe'],age_conditioned=False,teacher_resolution=32,teacher_weight=.05,age_KL_weight=.2,
        objective='Native512 same-age reconstruction + LPIPS/identity + target identity + low-frequency32 teacher + blurred age KL; experimental arm adds RGB/highpass256 region and central-face GAN with balanced real age bins.',
        weights=dict(pixel=1.2,lpips=.8,same_identity=1.,target_identity=.8,local_same=.25,teacher32=.05,age_KL=.2,parameter_trust=.01,GAN=.5),
        tensor_scope=list(named),parameter_count=sum(p.numel() for p in named.values()),notes=['All ages share generator patch;20/40 may change and require regression.','No target-age rescaling, no image blend, no original photo pasted into generated output.','RGB plus regional context adds genuine color/detail signal; no claim it guarantees realism.'])
    write_json(out/'plan.json',plan)
    for step in range(steps):
        batch=[]
        for j in range(batch_size):
            position=step*batch_size+j
            if position<len(order):r=order[position]
            else:
                lo=rng.choices((20,30,40,50,70),weights=(.12,.12,.12,.28,.36))[0];pool=bins[lo];asian=[r for r in pool if r.get('published_demographic_label')=='Asian'];r=rng.choice(asian if asian and rng.random()<.5 else pool)
            batch.append(r)
        target=rng.choice((60,75));realpool=bins[50 if target==60 else 70];asian_real=[r for r in realpool if r.get('published_demographic_label')=='Asian']
        # Oversample only published labels, with a minimum reference count;
        # never estimate ethnicity from a photograph or treat all data as Asian.
        real=[rng.choice(asian_real if len(asian_real)>=8 and rng.random()<.4 else realpool) for _ in batch]
        jitter=[[(rng.randint(-12,12),rng.randint(-12,12)) for _ in REGIONS] for _ in batch];flip=rng.random()<.5;contrast=rng.uniform(.8,1.2)
        names=[r['name'] for r in batch];realnames=[r['name'] for r in real];used.update(names);used_real.update(realnames)
        ref=tensor(names);realimage=tensor(realnames);x=F.interpolate(ref,(256,256),mode='area');samecode=torch.stack([cached[n][0] for n in names]);targetcode=torch.stack([cached[n][1 if target==60 else 2] for n in names])
        with torch.no_grad():
            saved={n:p.detach().clone() for n,p in named.items()}
            for n,p in named.items():p.copy_(initial[n])
            teachernative=decode(targetcode);teacher32=F.interpolate(teachernative,(32,32),mode='area');teacherprob=logits(F.interpolate(teachernative,(256,256),mode='area')).softmax(1)
            for n,p in named.items():p.copy_(saved[n])
        same=decode(samecode);edit=decode(targetcode);same256=F.interpolate(same,(256,256),mode='area');edit256=F.interpolate(edit,(256,256),mode='area')
        fake=patches(edit,jitter,flip,contrast);realpatch=patches(realimage,jitter,flip,contrast);ages=torch.full((batch_size*6,),3 if target==60 else 4,device='cuda',dtype=torch.long);regions=torch.arange(6,device='cuda').repeat(batch_size)
        dloss=edit.new_zeros(());r1=edit.new_zeros(());adversarial=edit.new_zeros(())
        if arm=='native_gan':
            d.requires_grad_(True).train();dopt.zero_grad(set_to_none=True);realpatch=realpatch.detach().requires_grad_(step%16==0)
            scores=d(realpatch,ages,regions);dloss=F.softplus(-scores).mean()+F.softplus(d(fake.detach(),ages,regions)).mean()
            if step%16==0:
                grad=torch.autograd.grad(scores.sum(),realpatch,create_graph=True)[0];r1=grad.square().sum((1,2,3)).mean();dloss=dloss+8*r1
            if not torch.isfinite(dloss):raise ValueError('Nonfinite discriminator')
            dloss.backward();torch.nn.utils.clip_grad_norm_(d.parameters(),5);dopt.step();d.requires_grad_(False).eval();adversarial=F.softplus(-d(fake,ages,regions)).mean()
        pixel=F.l1_loss(F.interpolate(same,(512,512),mode='area'),F.interpolate(ref,(512,512),mode='area'));perceptual=lp(same256,x).mean();sameid=(1-identity.cos(same256,x)).mean();editid=(1-identity.cos(edit256,x)).mean()
        local=F.l1_loss(patches(same,jitter,False,1),patches(ref,jitter,False,1));teacher=F.l1_loss(F.interpolate(edit,(32,32),mode='area'),teacher32)
        age_loss=F.kl_div(logits(edit256).log_softmax(1),teacherprob,reduction='batchmean');trust=sum((p-initial[n]).square().mean() for n,p in named.items())
        ramp=min(1,(step+1)/500);loss=1.2*pixel+.8*perceptual+sameid+.8*editid+.25*local+.05*teacher+.2*age_loss+.01*trust+.5*ramp*adversarial
        gopt.zero_grad(set_to_none=True);loss.backward()
        if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in named.values()):raise ValueError('Nonfinite or missing generator gradient')
        norm=torch.nn.utils.clip_grad_norm_(list(named.values()),2);gopt.step()
        point=dict(step=step+1,names=names,real=realnames,target=target,jitter=jitter,flip=flip,contrast=contrast,loss=float(loss.detach()),pixel=float(pixel.detach()),lpips=float(perceptual.detach()),same_id=float(sameid.detach()),target_id=float(editid.detach()),local=float(local.detach()),teacher=float(teacher.detach()),age_KL=float(age_loss.detach()),GAN=float(adversarial.detach()),D=float(dloss.detach()),r1=float(r1.detach()),norm=float(norm))
        with (out/'train.jsonl').open('a') as f:f.write(json.dumps(point)+'\n')
        if step==0 or (step+1)%100==0:print('NATIVE_TRAIN',arm,step+1,'/',steps,'seconds',round(time.monotonic()-start,1),'used',len(used),'loss',round(point['loss'],4),flush=True)
        if (step+1)%1000==0:
            torch.save({n:p.detach().cpu() for n,p in named.items()},out/f'generator_step{step+1}.pt');write_json(out/'progress.json',dict(step=step+1,seconds=time.monotonic()-start,unique_images=len(used),real_references=len(used_real)))
    if frozen_digest()!=before:raise ValueError('Frozen model tensors changed')
    checkpoint=out/'generator_patch.pt';torch.save({n:p.detach().cpu() for n,p in named.items()},checkpoint)
    if arm=='native_gan':torch.save(d.state_dict(),out/'discriminator.pt')
    write_json(out/'completed.json',dict(steps=steps,batch=batch_size,seconds=time.monotonic()-start,unique_images=len(used),unique_real_references=len(used_real),
        patch_sha256=sha256(checkpoint),frozen_tensors_sha256=before,frozen_verified=True,cuda_peak_bytes=torch.cuda.max_memory_allocated(),generator_max_update=max(float((p.detach()-initial[n]).abs().max()) for n,p in named.items())))
    print('NATIVE_ARM_DONE',arm,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['cache','train']);p.add_argument('run',type=Path);p.add_argument('--arm',choices=['native_control','native_gan']);p.add_argument('--steps',type=int,default=4000);p.add_argument('--batch',type=int)
    a=p.parse_args()
    if a.action=='cache':cache(a.run.resolve(),a.batch or 8)
    else:
        if not a.arm or not 1<=a.steps<=8000:raise ValueError('Specify arm and1..8000steps')
        train(a.run.resolve(),a.arm,a.steps,a.batch or 2)

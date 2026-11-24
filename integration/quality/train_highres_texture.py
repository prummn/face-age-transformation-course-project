"""Paired, bounded high-resolution generator control vs age-matched patch GAN.

This is an offline research adapter. The SAM age encoder, pSp, coarse generator,
ToRGB and default serving weights remain frozen. No automatic promotion.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from probe_inversion import sha256,write_json

REGIONS=((.34,.40),(.66,.40),(.32,.57),(.68,.57),(.50,.69))


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('--arm',choices=['control','patch_gan'],required=True)
    p.add_argument('--steps',type=int,default=500);p.add_argument('--execute',action='store_true')
    p.add_argument('--min-resolution',type=int,choices=(64,128,256),default=256)
    p.add_argument('--generator-lr',type=float,default=2e-5)
    p.add_argument('--adversarial-weight',type=float,default=.02)
    a=p.parse_args();run=a.run.resolve();out=run/a.arm
    if not 1<=a.steps<=3000:raise ValueError('Bounded study requires 1..3000 steps')
    if not 0<a.generator_lr<=1e-4 or not 0<=a.adversarial_weight<=.2:raise ValueError('Learning rate or adversarial weight exceeds bounded study')
    records=read(run/'data/train.jsonl');evaluation=read(run/'data/eval.jsonl')+read(run/'data/external.jsonl')
    if {r['identity_group'] for r in records}&{r['identity_group'] for r in evaluation}:
        raise ValueError('Common audit groups overlap')
    if {r['sha256'] for r in records}&{r['sha256'] for r in evaluation}:raise ValueError('Pixel overlap')
    eligible=[r for r in records if r.get('reference_path') and r.get('source_face_short_edge',r.get('native_face_short_edge',0))>=512]
    high={k:[r for r in eligible if r['age_interval'][0]==k] for k in (50,70)}
    if min(map(len,high.values()))<20:raise ValueError('Need >=20 eligible real references per older age bin')
    plan=dict(arm=a.arm,steps=a.steps,seed=20261124,eligible=len(eligible),
        real_high_age_counts={k:len(v) for k,v in high.items()},train_by_age=dict(Counter(str(r['age_interval']) for r in eligible)),
        shared=f'frozen Release encoder + pSp; generator conv/activation resolution >={a.min_resolution}; ToRGB frozen',
        min_resolution=a.min_resolution,generator_lr=a.generator_lr,adversarial_weight=a.adversarial_weight,
        control='same reconstruction512, LPIPS256, identity, native local high-pass reconstruction, teacher128 and parameter trust',
        patch_gan='conditional age-bin and region discriminator; real native crops; common jitter/flip/contrast; patch-space lazy R1',
        target_ages=[60,75],weights=dict(pixel512=.5,lpips=.8,identity=.8,local_reconstruction=.15,
            teacher128=.5,target_identity=.5,generator_trust=.1,adversarial=a.adversarial_weight),
        caution='Exploratory generator pilot. FFHQ pretraining source; coarse apparent-age bins are not exact chronological labels. Direct encoder training must also be checked on fixed W+ latents.')
    print(json.dumps(plan,ensure_ascii=False),flush=True)
    if not a.execute:return
    if os.environ.get('FACE_AGE_EXPERIMENT_ALLOCATION')!='1':raise RuntimeError('Dedicated allocation required')
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from torch import nn
    from torch.nn import functional as F
    import lpips
    if torch.cuda.device_count()!=1:raise RuntimeError('Exactly one allocated GPU required')
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(20261124)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    base=Path(os.environ['FACE_AGE_BASE']);sys.path.insert(0,str(base/'src'))
    from sam_npu import load_sam
    from sam_losses import IDLoss
    net=load_sam('cuda:0').eval().requires_grad_(False)
    net.encoder.load_state_dict(torch.load(base/'out/ft_l1/encoder_step1000.pt',map_location='cpu',weights_only=True))
    def digest_tensors(items):
        h=hashlib.sha256()
        for name,t in items:
            h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    for index,module in enumerate(net.decoder.convs):
        if 2**(3+index//2)>=a.min_resolution:
            module.conv.requires_grad_(True);module.activate.requires_grad_(True)
    named={n:q for n,q in net.decoder.named_parameters() if q.requires_grad}
    initial={n:q.detach().clone() for n,q in named.items()}
    frozen_before=digest_tensors((n,t) for n,t in net.state_dict().items() if n not in {'decoder.'+k for k in named})
    identity=IDLoss().cuda().eval().requires_grad_(False)
    perceptual=lpips.LPIPS(net='alex',verbose=False).cuda().eval().requires_grad_(False)
    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__();layers=[];cin=2
            for cout in (32,64,128,128):
                layers.extend([nn.utils.spectral_norm(nn.Conv2d(cin,cout,3,2,1)),nn.LeakyReLU(.2)]);cin=cout
            self.body=nn.Sequential(*layers);self.out=nn.Linear(cin,1)
            self.age=nn.Embedding(2,cin);self.region=nn.Embedding(len(REGIONS),cin)
        def forward(self,x,age,region):
            f=self.body(x).mean((2,3));condition=self.age(age)+self.region(region)
            return self.out(f).flatten()+(f*condition).sum(1)/(f.shape[1]**.5)
    d=Discriminator().cuda();gopt=torch.optim.Adam(list(named.values()),lr=a.generator_lr,betas=(0.0,.99))
    dopt=torch.optim.Adam(d.parameters(),lr=2e-4,betas=(0.0,.99))
    bins={low:[r for r in eligible if r['age_interval'][0]==low] for low in sorted({r['age_interval'][0] for r in eligible})}
    rng=random.Random(20261124);used=set();used_real=set();started=time.monotonic()
    out.mkdir(exist_ok=False);plan.update(generator_parameters=list(named),parameter_count=sum(q.numel() for q in named.values()),gpu=torch.cuda.get_device_name())
    write_json(out/'plan.json',plan)
    def image(r,key):
        path=r[key];expected=r.get('reference_sha256' if key=='reference_path' else 'sha256')
        if sha256(path)!=expected:raise ValueError('Image digest changed: '+r['name'])
        rgb=np.asarray(Image.open(path).convert('RGB'))
        if key=='image_path' and rgb.shape[:2]!=(256,256):raise ValueError('Expected aligned256')
        if key=='reference_path' and rgb.shape[:2]!=(1024,1024):raise ValueError('Native1024 reference required')
        return torch.from_numpy(np.array(rgb,copy=True,order='C')).permute(2,0,1)[None].cuda().float()/127.5-1
    def cond(x,age):return torch.cat([x,torch.full_like(x[:,:1],age/100)],1)
    def decode(code):return net(code,input_code=True,input_is_full=True,randomize_noise=False,resize=False)
    def patches(x,jitter,flip,contrast):
        result=[]
        for (cx,cy),(dx,dy) in zip(REGIONS,jitter):
            left=round(cx*1024)-64+dx;top=round(cy*1024)-64+dy
            result.append(x[:,:,top:top+128,left:left+128])
        rgb=torch.cat(result);gray=(rgb*rgb.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
        gray=(gray-gray.mean((2,3),keepdim=True))*contrast
        local=gray-F.avg_pool2d(gray,7,1,3)
        z=torch.cat([gray,local],1)
        return z.flip(-1) if flip else z
    for step in range(a.steps):
        low=rng.choice(list(bins));r=rng.choice(bins[low]);target=rng.choice((60,75));real=rng.choice(high[50 if target==60 else 70])
        flip=rng.random()<.5;jitter=[(rng.randint(-12,12),rng.randint(-12,12)) for _ in REGIONS]
        patch_flip=rng.random()<.5;contrast=rng.uniform(.85,1.15)
        x=image(r,'image_path');ref=image(r,'reference_path');real_native=image(real,'reference_path')
        if flip:x=x.flip(-1);ref=ref.flip(-1);real_native=real_native.flip(-1)
        used.add(r['name']);used_real.add(real['name'])
        with torch.no_grad():
            b=net.pretrained_encoder(x)+net.latent_avg
            same_code=b+net.encoder(cond(x,r['source_age']));target_code=b+net.encoder(cond(x,target))
            # Evaluate the unmodified decoder at the same codes. Restore only the
            # explicitly trainable tensors, then put this arm's tensors back.
            saved={n:q.detach().clone() for n,q in named.items()}
            for n,q in named.items():q.copy_(initial[n])
            teacher=F.interpolate(decode(target_code),(128,128),mode='area')
            for n,q in named.items():q.copy_(saved[n])
        same_native=decode(same_code);edited_native=decode(target_code)
        same=F.interpolate(same_native,(256,256),mode='area');edited=F.interpolate(edited_native,(256,256),mode='area')
        fake=patches(edited_native,jitter,patch_flip,contrast)
        real_patch=patches(real_native,jitter,patch_flip,contrast)
        age=torch.full((len(REGIONS),),int(target==75),device='cuda',dtype=torch.long)
        regions=torch.arange(len(REGIONS),device='cuda')
        loss_d=edited.new_zeros(());r1=edited.new_zeros(())
        if a.arm=='patch_gan':
            d.requires_grad_(True).train();dopt.zero_grad(set_to_none=True)
            real_patch=real_patch.detach().requires_grad_(step%16==0)
            real_logits=d(real_patch,age,regions);fake_logits=d(fake.detach(),age,regions)
            loss_d=F.softplus(-real_logits).mean()+F.softplus(fake_logits).mean()
            if step%16==0:
                grad=torch.autograd.grad(real_logits.sum(),real_patch,create_graph=True)[0]
                r1=grad.square().sum((1,2,3)).mean();loss_d=loss_d+16*r1
            if not torch.isfinite(loss_d):raise RuntimeError('Nonfinite discriminator loss')
            loss_d.backward();torch.nn.utils.clip_grad_norm_(d.parameters(),5);dopt.step();d.requires_grad_(False).eval()
        local=(patches(same_native,jitter,False,1)-patches(ref,jitter,False,1)).abs().mean()
        pixel=(F.interpolate(same_native,(512,512),mode='area')-F.interpolate(ref,(512,512),mode='area')).abs().mean()
        lp=perceptual(same,x).mean();ids=(1-identity.cos(same,x)).mean()
        target_id=(1-identity.cos(edited,x)).mean();teacher_loss=F.l1_loss(F.interpolate(edited_native,(128,128),mode='area'),teacher)
        trust=sum((q-initial[n]).square().mean() for n,q in named.items())
        adversarial=F.softplus(-d(fake,age,regions)).mean() if a.arm=='patch_gan' else edited.new_zeros(())
        ramp=min(1,(step+1)/100)
        loss=.5*pixel+.8*lp+.8*ids+.15*local+.5*teacher_loss+.5*target_id+.1*trust+a.adversarial_weight*ramp*adversarial
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite generator loss')
        gopt.zero_grad(set_to_none=True);loss.backward()
        if not all(q.grad is not None and torch.isfinite(q.grad).all() for q in named.values()):raise RuntimeError('Missing or nonfinite generator gradient')
        norm=torch.nn.utils.clip_grad_norm_(list(named.values()),1);gopt.step()
        point=dict(step=step+1,name=r['name'],real=real['name'],target=target,flip=flip,jitter=jitter,patch_flip=patch_flip,contrast=contrast,
            loss=float(loss.detach()),pixel=float(pixel.detach()),lpips=float(lp.detach()),same_id=float(ids.detach()),
            target_id=float(target_id.detach()),local=float(local.detach()),teacher=float(teacher_loss.detach()),
            adversarial=float(adversarial.detach()),discriminator=float(loss_d.detach()),r1=float(r1.detach()),norm=float(norm))
        with (out/'train.jsonl').open('a') as f:f.write(json.dumps(point)+'\n')
        if step==0 or (step+1)%25==0:print(json.dumps(point),flush=True)
        if (step+1)%250==0:
            torch.save({n:q.detach().cpu() for n,q in named.items()},out/f'generator_step{step+1}.pt')
    frozen_after=digest_tensors((n,t) for n,t in net.state_dict().items() if n not in {'decoder.'+k for k in named})
    if frozen_after!=frozen_before:raise RuntimeError('Frozen SAM tensors changed')
    update=max(float((q.detach()-initial[n]).abs().max()) for n,q in named.items())
    if update==0:raise RuntimeError('Generator did not update')
    path=out/'generator_patch.pt';torch.save({n:q.detach().cpu() for n,q in named.items()},path)
    if a.arm=='patch_gan':torch.save(d.state_dict(),out/'discriminator.pt')
    write_json(out/'completed.json',dict(steps=a.steps,seconds=time.monotonic()-started,unique_images=len(used),
        unique_real_references=len(used_real),generator_max_update=update,patch_sha256=sha256(path),
        frozen_tensors_sha256=frozen_after,frozen_verified=True,cuda_peak_bytes=torch.cuda.max_memory_allocated()))
    print('HIGHRES_ARM_DONE',a.arm,flush=True)

if __name__=='__main__':main()

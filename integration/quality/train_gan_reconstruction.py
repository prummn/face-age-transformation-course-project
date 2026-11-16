"""Dry-run by default: paired encoder vs encoder+limited-generator ablation.

Requires explicit, hash-verified, conservatively grouped train/eval manifests. The
user portrait and diagnostic benchmark fixtures cannot enter this training.
Use an independently allocated CUDA task, not the running demo's device.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from probe_inversion import sha256,write_json


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate(args):
    train,evaluation=rows(args.train_manifest),rows(args.eval_manifest)
    if not train or not evaluation:raise ValueError('Train/eval manifests must be nonempty')
    forbidden=set()
    diagnostic=Path(os.environ.get('FACE_AGE_REPO',str(Path(__file__).resolve().parents[2])))/'deployment/gan-research/diagnostic-manifest.json'
    if diagnostic.exists():forbidden={r['sha256'] for r in json.loads(diagnostic.read_text())['samples']}
    for split in [train,evaluation]:
        if len({r['sha256'] for r in split})!=len(split):raise ValueError('Duplicate image content within split')
        for r in split:
            if not r.get('identity_group'):raise ValueError('Explicit identity_group candidate is required')
            if r.get('group_kind') not in ['verified_subject','conservative_source_similarity_component']:
                raise ValueError('Explicit verified or conservative group_kind required; no invented subject IDs')
            if not r.get('source') or not r.get('training_authorized',False):raise ValueError('Source and training authorization required')
            if r['sha256'] in forbidden:raise ValueError('Diagnostic portrait cannot enter train/eval training manifests')
            if sha256(r['image_path'])!=r['sha256']:raise ValueError('Input hash mismatch')
            if 'reference512_path' in r and sha256(r['reference512_path'])!=r['reference512_sha256']:
                raise ValueError('High-resolution reference hash mismatch')
            age=float(r['source_age'])
            if not 0<=age<=100 or r.get('age_kind') not in ['verified','pseudo']:
                raise ValueError('Explicit source_age and verified/pseudo age_kind required')
    if {r['identity_group'] for r in train}&{r['identity_group'] for r in evaluation}:
        raise ValueError('Train/eval identity overlap')
    if {r['sha256'] for r in train}&{r['sha256'] for r in evaluation}:raise ValueError('Train/eval image overlap')
    if args.output.exists():raise ValueError('Choose a new experiment output directory')
    if not 1<=args.steps<=1000:raise ValueError('Pilot steps must be1..1000')
    if not 1<=args.batch_size<=8:raise ValueError('Pilot batch size must be1..8')
    if not math.isfinite(getattr(args,'target_id_weight',.1)) or getattr(args,'target_id_weight',.1)<0:
        raise ValueError('Target identity weight must be finite and nonnegative')
    return train,evaluation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-manifest',type=Path,required=True)
    p.add_argument('--eval-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=['encoder','encoder_generator'],required=True)
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--target-id-weight',type=float,default=.1,
                   help='Cross-age identity loss coefficient; original configuration is0.1')
    p.add_argument('--same-id-weight',type=float,default=.1)
    p.add_argument('--generator-scope',choices=['affines','mid_convs'],default='affines')
    p.add_argument('--generator-lr',type=float,help='Explicit generator LR; default is encoder LR times0.1')
    p.add_argument('--seed',type=int,default=20261112)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args();training,evaluation=validate(args)
    plan=dict(arm=args.arm,steps=args.steps,batch_size=args.batch_size,lr=args.lr,seed=args.seed,
              train_manifest_sha256=sha256(args.train_manifest),eval_manifest_sha256=sha256(args.eval_manifest),
              train_count=len(training),eval_count=len(evaluation),targets=[20,75],
              target_scope='Explicit adult pilot range20..75, not a change to online target ages or a90-year accuracy claim',
              group_kinds=sorted({r['group_kind'] for r in training+evaluation}),
              grouping_limit='Conservative account/similarity groups do not prove subject-disjoint splits',
              common_changes='same-age L2/LPIPS/ID256 + true-reference face-weighted L1_512; paired stratified sampling; frozen BN; teacher-latent trust; direct fine route',
              loss_weights=dict(mse256=1,lpips256=.5,same_id=args.same_id_weight,l1_512=.2,target_age=1,target_id=args.target_id_weight,latent_trust=.01,generator_trust=.1),
              generator_arm=args.generator_scope,
              caveats='Pilot configuration; no training until --execute. Validation sample manifests required separately.')
    print(json.dumps(plan,ensure_ascii=False,indent=2))
    if not args.execute:return
    if os.environ.get('FACE_AGE_EXPERIMENT_ALLOCATION')!='1':
        raise RuntimeError('Execute only inside the dedicated experiment allocation')
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    import lpips
    from copy import deepcopy
    from torch.nn.modules.batchnorm import _BatchNorm
    base=Path(os.environ['FACE_AGE_BASE']);sys.path.insert(0,str(base/'src'))
    from sam_npu import load_sam
    from sam_losses import AgeEst,IDLoss
    torch.cuda.set_device('cuda:0');torch.manual_seed(args.seed);random.seed(args.seed)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    net=load_sam('cuda:0').eval().requires_grad_(False)
    assert net.opts.start_from_encoded_w_plus and not net.opts.start_from_latent_avg,'Unexpected SAM latent formulation'
    plan.update(gpu=torch.cuda.get_device_name(),torch=torch.__version__,precision='float32',generator_base='original SAM frozen except explicitly listed parameters')
    net.encoder.load_state_dict(torch.load(base/'out/ft_l1/encoder_step1000.pt',map_location='cpu',weights_only=True))
    teacher=deepcopy(net.encoder).eval().requires_grad_(False)
    net.encoder.requires_grad_(True).train()
    for module in net.encoder.modules():
        if isinstance(module,_BatchNorm):module.eval()
    generator_names=[];initial_generator={}
    if args.arm=='encoder_generator':
        for index,module in enumerate(net.decoder.convs):
            resolution=2**(3+index//2)
            if args.generator_scope=='mid_convs' and resolution>=64:
                module.conv.requires_grad_(True)
                module.activate.requires_grad_(True)
            elif args.generator_scope=='affines' and resolution>=128:
                module.conv.modulation.requires_grad_(True)
        for name,param in net.decoder.named_parameters():
            if param.requires_grad:
                generator_names.append(name);initial_generator[name]=param.detach().clone()
    age_net=AgeEst().cuda().eval().requires_grad_(False)
    identity=IDLoss().cuda().eval().requires_grad_(False)
    perception=lpips.LPIPS(net='alex',verbose=False).cuda().eval().requires_grad_(False)
    generator_params=[q for q in net.decoder.parameters() if q.requires_grad]
    groups=[dict(params=list(net.encoder.parameters()),lr=args.lr)]
    if generator_params:groups.append(dict(params=generator_params,lr=args.generator_lr if args.generator_lr is not None else args.lr*.1))
    opt=torch.optim.Adam(groups)
    args.output.mkdir(parents=True,exist_ok=False)
    plan['generator_parameters']=generator_names;plan['generator_parameter_count']=sum(q.numel() for q in generator_params)
    write_json(args.output/'plan.json',plan)
    rng=random.Random(args.seed);logs=[];started=time.monotonic()
    bins={}
    for i,r in enumerate(training):bins.setdefault(tuple(r['age_interval']),[]).append(i)
    bin_keys=sorted(bins)
    yy,xx=torch.meshgrid(torch.linspace(0,1,512,device='cuda'),torch.linspace(0,1,512,device='cuda'),indexing='ij')
    weight=(.2+.8*((1-((xx-.5)/.31)**2-((yy-.53)/.36)**2)/.25).clamp(0,1))[None,None]
    used=set()
    def conditioned(x,a):return torch.cat([x,a[:,None,None,None].expand(-1,1,256,256)],1)
    def decode(code):return net(code,input_code=True,input_is_full=True,randomize_noise=False,resize=False)
    for step in range(args.steps):
        indices=[rng.choice(bins[bin_keys[rng.randrange(len(bin_keys))]]) for _ in range(args.batch_size)]
        used.update(indices)
        flips=[rng.random()<.5 for _ in indices]
        ages=[rng.uniform(.2,.75) for _ in indices]
        images=[];references=[]
        for i,flip in zip(indices,flips):
            rgb=np.asarray(Image.open(training[i]['image_path']).convert('RGB'))
            if rgb.shape[:2]!=(256,256):raise ValueError('Training data must be aligned256, no hidden resize')
            images.append(rgb[:,::-1].copy() if flip else rgb)
            ref=np.asarray(Image.open(training[i]['reference512_path']).convert('RGB'))
            if ref.shape[:2]!=(512,512):raise ValueError('True reference512 required')
            references.append(ref[:,::-1].copy() if flip else ref)
        x=torch.from_numpy(np.stack(images)).permute(0,3,1,2).contiguous().cuda().float()/127.5-1
        reference=torch.from_numpy(np.stack(references)).permute(0,3,1,2).contiguous().cuda().float()/127.5-1
        source=torch.tensor([float(training[i]['source_age'])/100 for i in indices],device='cuda')
        target=torch.tensor(ages,device='cuda')
        with torch.no_grad():
            b=net.pretrained_encoder(x)+net.latent_avg
            teacher_target=teacher(conditioned(x,target))+b
        code_same=net.encoder(conditioned(x,source))+b
        code_target=net.encoder(conditioned(x,target))+b
        same_native=decode(code_same);edited_native=decode(code_target)
        same=F.interpolate(same_native,(256,256),mode='area');edited=F.interpolate(edited_native,(256,256),mode='area')
        highres=((F.interpolate(same_native,(512,512),mode='area')-reference).abs()*weight).sum()/(weight.sum()*3*args.batch_size)
        loss_same=.2*highres+F.mse_loss(same,x)+.5*perception(same,x).mean()+args.same_id_weight*(1-identity.cos(same,x)).mean()
        loss_age=F.mse_loss(age_net(edited)[0],target)
        loss_id=args.target_id_weight*(1-identity.cos(edited,x)).mean()
        loss_latent=.01*(code_target-teacher_target).square().mean()
        trust=sum((q-initial_generator[n]).square().mean() for n,q in net.decoder.named_parameters() if q.requires_grad)
        loss=loss_same+loss_age+loss_id+loss_latent+.1*trust
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        opt.zero_grad(set_to_none=True);loss.backward()
        parameters=[q for group in groups for q in group['params']]
        if not all(q.grad is None or torch.isfinite(q.grad).all() for q in parameters):raise RuntimeError('Nonfinite gradient')
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
        opt.step()
        point=dict(step=step+1,indices=indices,flips=flips,source=source.tolist(),target=target.tolist(),
                   loss=float(loss.detach()),same=float(loss_same.detach()),age=float(loss_age.detach()),highres_l1=float(highres.detach()),target_identity=float(loss_id.detach()),latent_trust=float(loss_latent.detach()),grad_norm=float(norm))
        logs.append(point)
        with (args.output/'train.jsonl').open('a') as stream:stream.write(json.dumps(point)+'\n')
        if step==0 or (step+1)%20==0:print(json.dumps(point),flush=True)
    encoder_delta=max(float((p.detach()-dict(teacher.named_parameters())[n]).abs().max()) for n,p in net.encoder.named_parameters())
    if encoder_delta==0:raise RuntimeError('Encoder did not update')
    generator_delta=max([float((q.detach()-initial_generator[n]).abs().max()) for n,q in net.decoder.named_parameters() if q.requires_grad] or [0.])
    if generator_params and generator_delta==0:raise RuntimeError('Generator parameters did not update')
    for n,buf in net.encoder.named_buffers():
        if 'running_' in n or 'num_batches_tracked' in n:
            assert torch.equal(buf,dict(teacher.named_buffers())[n]),'BN state changed'
    torch.save({k:v.cpu() for k,v in net.encoder.state_dict().items()},args.output/'encoder.pt')
    if generator_params:
        patch_name='generator_affines.pt' if args.generator_scope=='affines' else 'generator_patch.pt'
        torch.save({n:q.detach().cpu() for n,q in net.decoder.named_parameters() if q.requires_grad},args.output/patch_name)
    write_json(args.output/'completed.json',dict(steps=args.steps,seconds=time.monotonic()-started,unique_training_images=len(used),total_training_candidates=len(training),
               encoder_max_abs_update=encoder_delta,generator_max_abs_update=generator_delta,
               batchnorm_frozen_verified=True,cuda_peak_bytes=torch.cuda.max_memory_allocated(),encoder_sha256=sha256(args.output/'encoder.pt'),
               note='Training completed; candidate is not evaluated/promoted by this entrypoint'))


if __name__=='__main__':main()

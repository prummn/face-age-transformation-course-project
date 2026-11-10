#!/usr/bin/env python3
"""Opt-in small SAM encoder ablation based on the existing train_sam_ft.py.

Default action only validates manifests and prints a plan. Training requires
--execute. Never writes in the original face-age project. Use a free physical NPU outside 0/2/3; pass --device explicitly.
ASCEND_RT_VISIBLE_DEVICES is resolved before checking the physical device. No new pretrained models.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
import random
import sys


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def read_manifest(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def physical_device(logical_device):
    match=re.fullmatch(r'npu:(\d+)',logical_device)
    if not match:
        raise ValueError('--device must be a logical NPU such as npu:0 or npu:1.')
    logical=int(match.group(1))
    visible=os.environ.get('ASCEND_RT_VISIBLE_DEVICES')
    if visible is not None:
        tokens=[token.strip() for token in visible.split(',')]
        if not tokens or any(not token.isdigit() for token in tokens) or len(set(tokens))!=len(tokens):
            raise ValueError('Unrecognized ASCEND_RT_VISIBLE_DEVICES mapping; use a unique comma-separated physical device list.')
        mapping=[int(token) for token in tokens]
        if len(set(mapping))!=len(mapping):
            raise ValueError('ASCEND_RT_VISIBLE_DEVICES contains duplicate physical devices.')
        if logical>=len(mapping):
            raise ValueError('Logical device is outside ASCEND_RT_VISIBLE_DEVICES.')
        physical=mapping[logical]
    else:
        physical=logical
    if physical in {0,2,3}:
        raise ValueError(f'Physical NPU {physical} is reserved (0 serves the demo; 2/3 are reserved for other services).')
    return physical,visible


def ensure_device_idle(physical):
    # Only called in --execute; a dry run never imports torch or probes hardware.
    result=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=15,check=False)
    if result.returncode!=0:
        raise RuntimeError('npu-smi failed; refusing to start training without an idle-device check.')
    if not re.search(r'No running processes found in NPU\s+'+str(physical)+r'(?:\s|\|)',result.stdout):
        raise RuntimeError(f'Physical NPU {physical} is busy or its status is unknown; training was not started.')


def validate(args):
    base=Path(args.base).resolve(); output=Path(args.out).resolve()
    if output==base or base in output.parents:
        raise ValueError('Output must be outside the original face-age project.')
    training=read_manifest(args.train_manifest); evaluation=read_manifest(args.eval_manifest)
    if not training or not evaluation:
        raise ValueError('Both manifests must be nonempty.')
    train_keys={r['key'] for r in training}; eval_keys={r['key'] for r in evaluation}
    if train_keys & eval_keys:
        raise ValueError('Train/eval keys overlap.')
    train_groups={r['duplicate_candidate_group'] for r in training}
    eval_groups={r['duplicate_candidate_group'] for r in evaluation}
    if train_groups & eval_groups:
        raise ValueError('Train/eval near-duplicate groups overlap.')
    for row in training+evaluation:
        if not Path(row['image_path']).is_file():
            raise ValueError('Missing image: '+row['image_path'])
    for row in evaluation:
        if sha256(row['image_path'])!=row['sha256']:
            raise ValueError('Frozen evaluation image changed: '+row['key'])
    if not Path(args.initial_encoder).is_file():
        raise ValueError('Missing initial encoder: '+args.initial_encoder)
    physical,visible=physical_device(args.device)
    arms={'control':(0.,0.),'colour_only':(args.colour_weight,0.),'same_age_only':(0.,args.same_age_weight),'colour_same_age':(args.colour_weight,args.same_age_weight)}
    colour_weight,same_weight=arms[args.arm]
    return training,evaluation,{'arm':args.arm,'base':str(base),'out':str(output),'initial_encoder':args.initial_encoder,'train_manifest_sha256':sha256(args.train_manifest),'eval_manifest_sha256':sha256(args.eval_manifest),'train_count':len(training),'eval_count':len(evaluation),'steps':args.steps,'validation_count':args.val_count,'batch_size':args.bs,'learning_rate':args.lr,'seed':args.seed,'device':args.device,'physical_device':physical,'ascend_rt_visible_devices':visible,'common_changes_vs_historical_training':['quality-filtered and near-duplicate-quarantined manifest','capped pseudo-age weighted sampling','frozen encoder BatchNorm running statistics','target ages restricted to 20-75 for this ablation','single-device batch2 and lr1e-5 from ft_l1_step1000'],'control_definition':'The control shares all these changes; it isolates added colour/same-age terms and is NOT a reproduction of historical train_sam_ft.py.','colour_weight':colour_weight,'same_age_weight':same_weight,'target_range':[20,75],'decoder_frozen':True,'pretrained_encoder_frozen':True,'encoder_batchnorm_running_stats_frozen':True,'disclosure':'Training plan, not evidence of improved quality. Eval is a previously used frozen regression set, not a new blind test set.'}


def train(args,rows,evaluation,plan):
    # Deferred imports mean the default validation path cannot allocate an NPU.
    ensure_device_idle(plan['physical_device'])
    import os,time,math
    import numpy as np
    import cv2
    import torch
    import torch_npu  # noqa: F401
    import torch.nn.functional as F
    from torch.nn.modules.batchnorm import _BatchNorm
    sys.path.insert(0,str(Path(args.base)/'src'))
    from sam_npu import load_sam
    from sam_losses import AgeEst,IDLoss
    import lpips
    torch.npu.set_device(args.device)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    output=Path(args.out)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'train_log.jsonl').exists() or list(output.glob('encoder_step*.pt')):
        raise ValueError('Refusing to overwrite an earlier run; choose a new --out directory.')
    (output/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    net=load_sam(args.device)
    net.encoder.load_state_dict(torch.load(args.initial_encoder,map_location='cpu'))
    net.train()
    for module in (net.decoder,net.pretrained_encoder):
        module.eval()
        for p in module.parameters(): p.requires_grad_(False)
    for module in net.encoder.modules():
        if isinstance(module,_BatchNorm): module.eval()
    identity=IDLoss().to(args.device)
    ages=AgeEst().to(args.device)
    perceptual=lpips.LPIPS(net='alex',verbose=False).to(args.device).eval()
    for p in perceptual.parameters(): p.requires_grad_(False)
    parameters=[p for p in net.encoder.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(parameters,lr=args.lr)
    weights=[row['sampling_weight'] for row in rows]
    # A geometric face mask excludes most hair/background. This is explicitly
    # NOT skin segmentation; always inspect cheeks, hair, eyes and background.
    yy,xx=torch.meshgrid(torch.linspace(-1,1,256,device=args.device),torch.linspace(-1,1,256,device=args.device),indexing='ij')
    radial=(xx/.72)**2+((yy-.04)/.84)**2
    mask=(1-radial).clamp(0,1)[None,None]
    mask=F.avg_pool2d(mask,16,16)
    colour_coeff=torch.tensor([[-.168736,-.331264,.5],[.5,-.418688,-.081312]],device=args.device)
    def chroma(image):
        rgb=(image+1)/2
        low=F.avg_pool2d(rgb,16,16)
        return torch.einsum('kc,bchw->bkhw',colour_coeff,low)
    def colour_loss(generated,original):
        # Retain luma/texture freedom; no reference ethnicity, global skin-tone
        # target, background histogram matching, or texture synthesis is used.
        delta=(chroma(generated)-chroma(original.detach())).abs()
        return (delta*mask).sum()/(mask.sum()*delta.shape[0]*delta.shape[1])
    def forward(image,target):
        condition=target.view(-1,1,1,1).expand(-1,1,256,256)
        return net(torch.cat([image,condition],1),randomize_noise=False,resize=True)
    def validate_checkpoint(step):
        print(f'VALIDATION_START step={step}',flush=True)
        # Round-robin by EXISTING pseudo-age decade; never infer demographic
        # attributes from images. Save the chosen keys for cross-arm pairing.
        bins={}
        for row in evaluation:
            bins.setdefault(int(row['pseudo_age']//10),[]).append(row)
        ordered=[]
        for position in range(max(map(len,bins.values()))):
            for decade in sorted(bins):
                if position<len(bins[decade]): ordered.append(bins[decade][position])
        chosen=ordered[:args.val_count]
        folder=output/f'validation_step{step}'
        folder.mkdir(exist_ok=True)
        net.eval(); records=[]
        def uint8_rgb(value):
            return ((value.clamp(-1,1)+1)*127.5).permute(1,2,0).byte().cpu().numpy()
        def cheek_lab(rgb):
            lab=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
            return np.median(np.concatenate([lab[123:159,65:105].reshape(-1,3),lab[123:159,151:191].reshape(-1,3)]),axis=0)
        with torch.no_grad():
            for index,row in enumerate(chosen):
                original=cv2.cvtColor(cv2.imread(row['image_path']),cv2.COLOR_BGR2RGB)
                inp=torch.from_numpy(original).permute(2,0,1).float().div(127.5).sub(1)[None].to(args.device)
                source,_=ages(inp)
                input_lab=cheek_lab(original)
                sheet=[original]
                for label,target in [('same_age',source.item()),('a20',.20),('a40',.40),('a60',.60),('a75',.75)]:
                    condition=torch.full((1,1,256,256),target,device=args.device)
                    native=net(torch.cat([inp,condition],1),randomize_noise=False,resize=False)
                    pooled=net.face_pool(native)
                    rgb=uint8_rgb(pooled[0]);sheet.append(rgb)
                    difference=cheek_lab(rgb)-input_lab
                    age_est,_=ages(pooled)
                    # Age and identity models overlap training losses: explicitly
                    # secondary diagnostics, never the sole acceptance criterion.
                    records.append({'key':row['key'],'tag':label,'cheek_delta_L':float(difference[0]),'cheek_delta_a':float(difference[1]),'cheek_delta_b':float(difference[2]),'cheek_chroma_shift':float(np.linalg.norm(difference[1:])),'same_age_mae_rgb':float(np.abs(rgb.astype(float)-original).mean()) if label=='same_age' else None,'diagnostic_id_cos_shared_training_model':float(identity.cos(pooled,inp).item()),'diagnostic_age_shared_training_model':float(age_est.item()*100)})
                    if index<4:
                        cv2.imwrite(str(folder/f'{row["key"]}_{label}_native1024.png'),uint8_rgb(native[0])[:,:,::-1])
                cv2.imwrite(str(folder/f'{row["key"]}_input_sameage_20_40_60_75.png'),np.concatenate(sheet,axis=1)[:,:,::-1])
        (folder/'metrics.json').write_text(json.dumps({'step':step,'keys':[r['key'] for r in chosen],'metrics':records,'disclosure':'Fixed cheek ROIs are a proxy, not skin segmentation. Inspect paired contact sheets and native1024 outputs; do not promote a checkpoint from FairFace age or training-ID scores alone.'},ensure_ascii=False,indent=2))
        (folder/'blind_review_template.csv').write_text('key,blinded_arm_code,same_person_1to5,color_consistency_1to5,age_change_1to5,texture_artifacts_1to5,notes\n'+''.join(r['key']+',,,,,,\n' for r in chosen))
        print(f'VALIDATION_DONE step={step} images={len(chosen)}',flush=True)
        net.train()
        for module in (net.decoder,net.pretrained_encoder): module.eval()
        for module in net.encoder.modules():
            if isinstance(module,_BatchNorm): module.eval()
    validate_checkpoint(0)
    log=(output/'train_log.jsonl').open('x')
    skipped=0
    try:
        for step in range(1,args.steps+1):
            selected=random.choices(rows,weights=weights,k=args.bs)
            rgb=[cv2.cvtColor(cv2.imread(row['image_path']),cv2.COLOR_BGR2RGB) for row in selected]
            x=torch.from_numpy(np.stack(rgb)).permute(0,3,1,2).float().div(127.5).sub(1).to(args.device)
            flips=torch.rand(args.bs,1,1,1,device=args.device)<.5
            x=torch.where(flips,x,x.flip(3))
            with torch.no_grad(): source,_=ages(x)
            target=torch.empty(args.bs,device=args.device).uniform_(.20,.75)
            target=torch.where(torch.rand(args.bs,device=args.device)<1/3,source,target)
            started=time.monotonic()
            aged=forward(x,target); reconstructed=forward(aged,source)
            estimated,_=ages(aged)
            same_age=forward(x,source) if plan['same_age_weight'] else None
            terms={'aging':F.mse_loss(estimated,target),'id':.1*((1-(target-source).abs())*(1-identity.cos(aged,x))).mean(),'cycle_l2':.25*F.mse_loss(reconstructed,x),'cycle_lpips':.1*perceptual(reconstructed,x).mean(),'cycle_id':.1*(1-identity.cos(reconstructed,x)).mean()}
            colour=colour_loss(aged,x)
            direct=(.25*F.mse_loss(same_age,x)+.1*perceptual(same_age,x).mean()+.1*(1-identity.cos(same_age,x)).mean()) if same_age is not None else x.new_zeros(())
            loss=sum(terms.values())+plan['colour_weight']*colour+plan['same_age_weight']*direct
            optimizer.zero_grad(set_to_none=True)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f'Non-finite loss at step {step}; aborting this experiment without changing any deployed model.')
            if torch.isfinite(loss):
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
            else:
                norm=x.new_tensor(float('nan'))
            if not bool(torch.isfinite(norm)):
                raise RuntimeError(f'Non-finite gradient at step {step}; aborting this experiment.')
            skip=norm.item()>50
            if skip: skipped+=1
            else: optimizer.step()
            torch.npu.synchronize()
            record={'step':step,'loss':float(loss.detach()) if torch.isfinite(loss) else None,'chroma_raw':float(colour.detach()),'same_age_raw':float(direct.detach()),'gradient_norm':float(norm) if torch.isfinite(norm) else None,'skipped':skip,'seconds':round(time.monotonic()-started,3),**{key:float(value.detach()) for key,value in terms.items()}}
            log.write(json.dumps(record)+'\n');log.flush()
            if step==1 or step%10==0: print(json.dumps(record),flush=True)
            if step%args.save_every==0 or step==args.steps:
                destination=output/f'encoder_step{step}.pt'
                torch.save({key:value.detach().cpu() for key,value in net.encoder.state_dict().items()},str(destination)+'.tmp')
                Path(str(destination)+'.tmp').replace(destination)
                validate_checkpoint(step)
    finally:
        log.close()
    print(f'EXPERIMENT_FINISHED skipped={skipped}; evaluate independent of training losses before promoting any checkpoint.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',default='/data/home/1120231322/scow/face-age')
    parser.add_argument('--train-manifest',required=True)
    parser.add_argument('--eval-manifest',required=True)
    parser.add_argument('--initial-encoder',default='/data/home/1120231322/scow/face-age/out/ft_l1/encoder_step1000.pt')
    parser.add_argument('--out',required=True)
    parser.add_argument('--device',required=True,help='Logical NPU id; resolves ASCEND_RT_VISIBLE_DEVICES. Physical NPU 0/2/3 are forbidden.')
    parser.add_argument('--arm',choices=['control','colour_only','same_age_only','colour_same_age'],default='colour_same_age')
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--bs',type=int,default=2)
    parser.add_argument('--lr',type=float,default=1e-5)
    parser.add_argument('--seed',type=int,default=20261110)
    parser.add_argument('--colour-weight',type=float,default=1.)
    parser.add_argument('--same-age-weight',type=float,default=1.)
    parser.add_argument('--save-every',type=int,default=50)
    parser.add_argument('--val-count',type=int,default=24,help='Frozen regression subset, round-robin by existing pseudo-age bins')
    parser.add_argument('--execute',action='store_true',help='Actually train; omitted by default')
    args=parser.parse_args()
    if args.steps<1 or args.bs<1 or args.save_every<1 or args.val_count<1 or args.lr<=0 or args.colour_weight<0 or args.same_age_weight<0:
        parser.error('steps, bs, save-every and lr must be positive')
    rows,evaluation,plan=validate(args)
    print(json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
    if args.execute:
        train(args,rows,evaluation,plan)
    else:
        print('DRY_RUN_ONLY: no torch import, no NPU allocation, no training. Add --execute only after reviewing the plan and confirming the selected NPU is free.')


if __name__=='__main__':
    main()

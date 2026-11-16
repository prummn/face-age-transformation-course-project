"""Follow up observed low-resolution fitting blur; keep the first trial intact.

Uses the same fixed input, noise and initial W+, with full reference512, local
SSIM, and the existing frozen pSp trunk as a perceptual reconstruction loss.
The pSp feature loss is NOT an independent face recognition/identity metric.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=120)
    args = parser.parse_args()
    if not 1 <= args.steps <= 300:
        parser.error('steps must be 1..300')
    args.output.mkdir(parents=True, exist_ok=False)
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import torch
    import torch.nn.functional as F
    from probe_inversion import sha256, write_json

    base = Path(os.environ['FACE_AGE_BASE'])
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(base / 'src'))
    sys.path.insert(0, str(repo / 'integration'))
    from sam_npu import load_sam
    from sam_losses import AgeEst
    from source_color import preserve_source_chroma
    torch.manual_seed(20261112)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = 'cuda:0'
    torch.cuda.set_device(device)
    net = load_sam(device)
    age_net = AgeEst().to(device)
    for model in (net, age_net):
        model.eval().requires_grad_(False)
    if not net.opts.start_from_encoded_w_plus:
        raise RuntimeError('Expected release checkpoint with pretrained pSp encoder')
    report = deepcopy(json.loads((args.previous_results / 'report.json').read_text()))
    for item in report['weights']:
        if sha256(item['path']) != item['sha256']:
            raise RuntimeError('Weight changed since first trial')
    report['follow_up'] = dict(script_sha256=sha256(__file__), previous_report_sha256=sha256(args.previous_results/'report.json'),
                            steps=args.steps, lr=.03, feature_model='existing frozen pSp trunk at body6,20,23',
                            objective='.4 multi-resolution L1 + .3 normalized pSp feature L1 + '
                                      '.25 local SSIM loss512 + .05 relative gradient L1_512 + .01 W+ trust MSE',
                            supervision='Full1024 reference downsampled512 in addition to original input256',
                            caveat='Exploratory follow-up prompted by visible smoothing in first trial, not a blind test')
    started = time.monotonic()
    def tensor(rgb):
        return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1)[None].to(device).float()/127.5-1
    def decode(w):
        return net(w, input_code=True, input_is_full=True, randomize_noise=False, resize=False)
    def features(x):
        value = net.pretrained_encoder.input_layer(x)
        result = []
        for i, layer in enumerate(net.pretrained_encoder.body):
            value = layer(value)
            if i in (6,20,23):
                result.append(value)
        return result
    def wl1(a,b,mask):
        return ((a-b).abs()*mask).sum()/(mask.sum()*a.shape[1])
    def gradients(a):
        return a[:,:,:,1:]-a[:,:,:,:-1], a[:,:,1:]-a[:,:,:-1]
    def ssim_loss(a,b,mask):
        # Standard local SSIM on RGB in [0,1], window11; only used for fitting.
        a,b=(a+1)/2,(b+1)/2
        mu_a=F.avg_pool2d(a,11,1,5);mu_b=F.avg_pool2d(b,11,1,5)
        var_a=F.avg_pool2d(a*a,11,1,5)-mu_a*mu_a
        var_b=F.avg_pool2d(b*b,11,1,5)-mu_b*mu_b
        cov=F.avg_pool2d(a*b,11,1,5)-mu_a*mu_b
        score=((2*mu_a*mu_b+.01**2)*(2*cov+.03**2))/((mu_a.square()+mu_b.square()+.01**2)*(var_a+var_b+.03**2))
        return ((1-score)*mask).sum()/(mask.sum()*3)
    for sample in report['samples']:
        name=sample['name']; old=args.previous_results/name; folder=args.output/name; folder.mkdir()
        # Hard links duplicate no private photo contents and leave previous run unchanged.
        for file in old.iterdir():
            if file.suffix in ('.png','.jpg'):
                os.link(file,folder/file.name)
        latent=torch.load(old/'latents.pt',map_location=device,weights_only=True)
        anchor=latent['anchor']; deltas=latent['original_deltas']; source_age=sample['source_condition']
        reference=np.asarray(Image.open(old/'reference.png').convert('RGB'))
        rgb=np.asarray(Image.open(old/'input256.png').convert('RGB'));x=tensor(rgb)
        target512=tensor(cv2.resize(reference,(512,512),interpolation=cv2.INTER_AREA))
        with torch.no_grad():
            source_features=features(x)
            psp=net.pretrained_encoder(x)+net.latent_avg

        def save(route,age,code):
            tick=time.monotonic()
            with torch.no_grad():
                torch.cuda.synchronize(); t=time.monotonic(); image=decode(code); torch.cuda.synchronize()
                decode_ms=(time.monotonic()-t)*1000
                estimated=float(age_net(F.interpolate(image,(256,256),mode='area'))[0])*100
                native=((image[0].clamp(-1,1)+1)*127.5).permute(1,2,0).byte().cpu().numpy()
            stem=f'{route}_a{age:02d}'
            Image.fromarray(native).save(folder/f'{stem}_raw.png')
            corrected,metadata=preserve_source_chroma(rgb,native)
            Image.fromarray(corrected).save(folder/f'{stem}_color.png')
            diff=np.abs(cv2.resize(native,(256,256),interpolation=cv2.INTER_AREA).astype('float32')-rgb.astype('float32'))
            sample['outputs'].append(dict(route=route,condition=age,raw=f'{name}/{stem}_raw.png',color=f'{name}/{stem}_color.png',
                reference_age_diagnostic=estimated,decode_ms=decode_ms,generation_and_save_seconds=time.monotonic()-tick,
                rgb_mae_0_255=float(diff.mean()),central_face_mae_0_255=float(diff[70:205,60:196].mean()),
                reconstruction_metric_valid=age==source_age,color_metadata=metadata))
        for age in sample['targets']:
            save('psp_delta',age,psp+deltas[age])
        refined=anchor.detach().clone().requires_grad_(True)
        grid=torch.linspace(0,1,512,device=device);yy,xx=torch.meshgrid(grid,grid,indexing='ij')
        face=((1-(((xx-.5)/.31)**2+((yy-.53)/.36)**2))/.25).clamp(0,1)[None,None]
        weight=.2+.8*face
        gx,gy=gradients(target512)
        edge_scale=(wl1(gx,torch.zeros_like(gx),face[:,:,:,1:])+wl1(gy,torch.zeros_like(gy),face[:,:,1:])).clamp_min(.02)
        optimizer=torch.optim.Adam([refined],lr=.03)
        points=[];fit_start=time.monotonic()
        for step in range(args.steps):
            optimizer.param_groups[0]['lr']=.03*(.15+.85*(1-step/args.steps))
            optimizer.zero_grad(set_to_none=True)
            image=decode(refined);y512=F.interpolate(image,(512,512),mode='area');y256=F.interpolate(image,(256,256),mode='area')
            pixel=wl1(y512,target512,weight)+wl1(y256,x,F.interpolate(weight,(256,256),mode='area'))
            feature=sum((a-b).abs().mean()/b.abs().mean().clamp_min(.05) for a,b in zip(features(y256),source_features))/3
            dx,dy=gradients(y512)
            edge=(wl1(dx,gx,face[:,:,:,1:])+wl1(dy,gy,face[:,:,1:]))/edge_scale
            structural=ssim_loss(y512,target512,weight)
            loss=.4*pixel+.3*feature+.25*structural+.05*edge+.01*(refined-anchor).square().mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite detail loss')
            loss.backward()
            if not torch.isfinite(refined.grad).all():raise RuntimeError('Nonfinite detail gradient')
            optimizer.step()
            if step==0 or (step+1)%20==0:
                torch.cuda.synchronize()
                point=dict(step=step+1,loss_before_update=float(loss.detach()),pixel=float(pixel.detach()),
                           feature=float(feature.detach()),ssim_loss=float(structural.detach()),relative_edge=float(edge.detach()),
                           elapsed_seconds=time.monotonic()-fit_start)
                points.append(point);print('DETAIL_FIT',name,json.dumps(point),flush=True)
        sample['detail_fitting_seconds']=time.monotonic()-fit_start
        sample['detail_fitting']=points
        assert all(p.grad is None and not p.requires_grad for model in (net,age_net) for p in model.parameters())
        for age in sample['targets']:
            save('refined_detail',age,refined.detach()+deltas[age])
        torch.save(dict(anchor=anchor.cpu(),refined_detail=refined.detach().cpu()),folder/'detail_latents.pt')
        # Small original-vs-pixel-vs-detail sheet; no hidden method selection.
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
        routes=['mixed','psp_delta','refined','refined_detail'];ages=[source_age,20,60,75]
        board=Image.new('RGB',(160+5*240,4*270),'#efefef');draw=ImageDraw.Draw(board)
        for row,route in enumerate(routes):
            y=row*270;draw.text((6,y+100),route,font=font,fill='black')
            for col,age in enumerate([None]+ages):
                path=folder/'reference.png' if age is None else folder/f'{route}_a{age:02d}_raw.png'
                board.paste(Image.open(path).resize((240,240)),(160+col*240,y+30))
                draw.text((164+col*240,y+4),'source' if age is None else f'condition {age}',font=font,fill='black')
        board.save(folder/'detail-overview.jpg',quality=93)
        print('DETAIL_SAMPLE_DONE',name,flush=True)
    report['follow_up']['total_seconds']=time.monotonic()-started
    write_json(args.output/'report.json',report)
    (args.output/'DONE').write_text('Frozen-weight detail follow-up complete\n')


if __name__=='__main__':
    main()

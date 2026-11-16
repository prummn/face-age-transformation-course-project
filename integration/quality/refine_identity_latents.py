"""Identity/perceptual test-time W+ optimization; network weights stay frozen."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import time
from probe_inversion import sha256,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('--anchor-steps',type=int,default=200)
    p.add_argument('--edit-steps',type=int,default=120)
    p.add_argument('--cached-run',type=Path,help='Reuse prior fitted anchors; constrain edits and retain prior outputs')
    a=p.parse_args();run=a.run;constrained=a.cached_run is not None
    if constrained and a.edit_steps!=180:p.error('--cached-run requires --edit-steps180')
    if constrained:a.anchor_steps=0
    out=run/('latent-constrained' if constrained else 'latent-refinement');out.mkdir(exist_ok=False)
    import cv2
    import lpips
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    base=Path(os.environ['FACE_AGE_BASE']);root=Path(os.environ['FACE_AGE_REPO'])
    sys.path.insert(0,str(base/'src'))
    from sam_npu import load_sam
    from sam_losses import IDLoss,AgeEst
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(20261116)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    net=load_sam('cuda:0').eval().requires_grad_(False)
    fine=deepcopy(net.encoder).eval().requires_grad_(False)
    fine.load_state_dict(torch.load(base/'out/ft_l1/encoder_step1000.pt',map_location='cpu',weights_only=True))
    identity=IDLoss().cuda().eval().requires_grad_(False)
    perceptual=lpips.LPIPS(net='alex',verbose=False).cuda().eval().requires_grad_(False)
    age=AgeEst().cuda().eval().requires_grad_(False)
    detector=cv2.FaceDetectorYN.create(str(base/'models/yunet/face_detection_yunet_2023mar.onnx'),'',(256,256),.6,.3,100)
    recognizer=cv2.FaceRecognizerSF.create(str(root/'deployment/gan-research/face_recognition_sface_2021dec.onnx'),'')
    records=json.loads((run/'latent-inputs.json').read_text())
    routes=['fine_direct','latent_anchor','latent_refined'] if not constrained else ['fine_direct','latent_anchor','latent_refined','latent_constrained60','latent_constrained180']
    report=dict(records=records,routes=routes,outputs=[],
        protocol=dict(anchor_steps=a.anchor_steps,edit_steps=a.edit_steps,seed=20261116,
            optimized='Only this input W+; every encoder/generator/loss-network weight is frozen.',
            age_constraint='KL to original per-target GAN age-head distribution; not chronological target accuracy.',
            inference='Each new input requires fresh optimization; not universal training or generalization.',
            no_compositing=True,constrained_first_styles=8 if constrained else None,
            constrained_objective='.8 identity + .5 ageKL + .6 LPIPS to anchored age edit + .05 latent trust + .1 low-frequency source color' if constrained else None),timings=[])
    previous=json.loads((a.cached_run/'latent-refinement/report.json').read_text()) if constrained else None
    def tensor(rgb):return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1)[None].contiguous().cuda().float()/127.5-1
    def decode(code):return net(code,input_code=True,input_is_full=True,randomize_noise=False,resize=False)
    def logits(y):
        value=F.interpolate((y+1)/2,(224,224),mode='bilinear',align_corners=False)
        return age.net((value-age.m)/age.sd)[2].float()
    def feature(rgb,face):
        if face is None:return None
        v=recognizer.feature(recognizer.alignCrop(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),face)).ravel()
        return v/(np.linalg.norm(v)+1e-12)
    grid=torch.linspace(0,1,512,device='cuda');yy,xx=torch.meshgrid(grid,grid,indexing='ij')
    weight=(.2+.8*((1-((xx-.5)/.31)**2-((yy-.53)/.36)**2)/.25).clamp(0,1))[None,None]
    started=time.monotonic()
    for row in records:
        filename=row.get('reference_path',row['image_path'])
        assert sha256(filename)==row.get('reference_sha256',row['sha256'])
        rgb=np.asarray(Image.open(filename).convert('RGB'))
        small=cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA);x=tensor(small)
        ref=tensor(cv2.resize(rgb,(512,512),interpolation=cv2.INTER_AREA))
        folder=out/row['name'];folder.mkdir();Image.fromarray(rgb).save(folder/'reference.png')
        _,faces=detector.detect(cv2.cvtColor(small,cv2.COLOR_RGB2BGR))
        face=max(faces,key=lambda f:f[2]*f[3]) if faces is not None else None
        source_feature=feature(small,face)
        source=row['source_age'];targets=[('same',source)]+[(str(t),t) for t in [20,40,60,75]]
        with torch.no_grad():
            b=net.pretrained_encoder(x)+net.latent_avg
            codes=torch.cat([b+fine(torch.cat([x,torch.full_like(x[:,:1],t/100)],1)) for _,t in targets])
            teacher=decode(codes)
            target_prob=logits(F.interpolate(teacher[1:],(256,256),mode='area')).softmax(1)
            id_reference=identity.feats(x).detach()
        def record(route,images):
            with torch.no_grad():
                for (condition,target),image in zip(targets,images):
                    image=image[None];assert tuple(image.shape)==(1,3,1024,1024) and torch.isfinite(image).all()
                    native=((image[0].clamp(-1,1)+1)*127.5).permute(1,2,0).byte().cpu().numpy()
                    reduced=cv2.resize(native,(256,256),interpolation=cv2.INTER_AREA)
                    y=F.interpolate(image,(256,256),mode='area')
                    predicted=feature(reduced,face)
                    stem=route+'_'+condition
                    Image.fromarray(native).save(folder/(stem+'.png'));Image.fromarray(reduced).save(folder/(stem+'_preview.png'))
                    with Image.open(folder/(stem+'.png')) as check:check.load();assert check.size==(1024,1024)
                    report['outputs'].append(dict(name=row['name'],cohort=row['cohort'],route=route,condition=condition,
                        target=target,source_age=source,shared_output_age=float(age(y)[0])*100,
                        sface_cosine=float(source_feature@predicted) if source_feature is not None else None,
                        lpips_shared=float(perceptual(y,x).mean()),native_size=[1024,1024],
                        native=row['name']+'/'+stem+'.png',preview=row['name']+'/'+stem+'_preview.png'))
        record('fine_direct',teacher);del teacher
        initial_pivot=codes[:1]
        if constrained:
            initial_pivot=torch.load(a.cached_run/'latent-refinement'/row['name']/'codes.pt',map_location='cuda',weights_only=True)['anchor']
            for old in previous['outputs']:
                if old['name']==row['name'] and old['route']=='latent_refined':
                    for field in ['native','preview']:
                        shutil.copy2(a.cached_run/'latent-refinement'/old[field],out/old[field])
                    report['outputs'].append(old)
        pivot=initial_pivot.detach().clone().requires_grad_(True)
        optimizer=torch.optim.Adam([pivot],lr=.02);tick=time.monotonic();points=[]
        for step in range(a.anchor_steps):
            optimizer.param_groups[0]['lr']=.02*(.2+.8*(1-step/a.anchor_steps))
            y=decode(pivot);y256=F.interpolate(y,(256,256),mode='area');y512=F.interpolate(y,(512,512),mode='area')
            pixel=((y512-ref).abs()*weight).sum()/(weight.sum()*3)
            ident=(1-(identity.feats(y256)*id_reference).sum(1)).mean()
            loss=.5*pixel+.8*perceptual(y256,x).mean()+.8*ident+.002*(pivot-codes[:1]).square().mean()
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if not torch.isfinite(loss) or not torch.isfinite(pivot.grad).all():raise RuntimeError('Invalid anchor optimization')
            optimizer.step()
            if (step+1)%50==0:
                points.append(dict(step=step+1,loss=float(loss.detach()),identity_loss=float(ident.detach())))
                print('ANCHOR',row['name'],points[-1],flush=True)
        torch.cuda.synchronize();anchor_seconds=time.monotonic()-tick
        anchor=pivot.detach();target_anchor=anchor+codes[1:]-codes[:1]
        with torch.no_grad():record('latent_anchor',decode(torch.cat([anchor,target_anchor])))
        parameter=(torch.zeros_like(target_anchor[:,:8]) if constrained else target_anchor.detach().clone()).requires_grad_(True)
        optimizer=torch.optim.Adam([parameter],lr=.012);tick=time.monotonic();edit_points=[]
        with torch.no_grad():
            prior_images=F.interpolate(decode(target_anchor),(256,256),mode='area') if constrained else None
        for step in range(a.edit_steps):
            optimizer.param_groups[0]['lr']=.012*(.2+.8*(1-step/a.edit_steps))
            refined=target_anchor+torch.cat([parameter,torch.zeros_like(target_anchor[:,8:])],1) if constrained else parameter
            y=decode(refined);small_y=F.interpolate(y,(256,256),mode='area')
            ident=(1-(identity.feats(small_y)*id_reference).sum(1)).mean()
            distribution=F.kl_div(logits(small_y).log_softmax(1),target_prob,reduction='batchmean')
            if constrained:
                color=F.l1_loss(F.avg_pool2d(small_y,32),F.avg_pool2d(x,32).expand(4,-1,-1,-1))
                loss=.8*ident+.5*distribution+.6*perceptual(small_y,prior_images).mean()+.05*parameter.square().mean()+.1*color
            else:
                loss=ident+2*distribution+.05*perceptual(small_y,x.expand(4,-1,-1,-1)).mean()+.02*(refined-target_anchor).square().mean()
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if not torch.isfinite(loss) or not torch.isfinite(parameter.grad).all():raise RuntimeError('Invalid edit optimization')
            optimizer.step()
            if (step+1)%40==0:
                edit_points.append(dict(step=step+1,identity_loss=float(ident.detach()),age_kl=float(distribution.detach())))
                print('EDIT_REFINE',row['name'],edit_points[-1],flush=True)
            if constrained and step+1 in [60,180]:
                with torch.no_grad():
                    snapshot=target_anchor+torch.cat([parameter,torch.zeros_like(target_anchor[:,8:])],1)
                    record('latent_constrained'+str(step+1),decode(torch.cat([anchor,snapshot])))
        torch.cuda.synchronize();edit_seconds=time.monotonic()-tick
        refined=target_anchor+torch.cat([parameter,torch.zeros_like(target_anchor[:,8:])],1) if constrained else parameter
        if not constrained:
            with torch.no_grad():record('latent_refined',decode(torch.cat([anchor,refined.detach()])))
        assert all(not q.requires_grad and q.grad is None for model in [net,fine,identity,perceptual,age] for q in model.parameters())
        torch.save(dict(anchor=anchor.cpu(),target_codes=refined.detach().cpu(),source=source),folder/'codes.pt')
        report['timings'].append(dict(name=row['name'],anchor_seconds=anchor_seconds,edit_seconds=edit_seconds,anchor_log=points,edit_log=edit_points))
        write_json(out/'report.partial.json',report);print('REFINEMENT_SAMPLE_DONE',row['name'],anchor_seconds,edit_seconds,flush=True)
    report['total_seconds']=time.monotonic()-started;write_json(out/'report.json',report)
    (out/'DONE').write_text('Frozen-weight per-input latent refinement complete\n')


if __name__=='__main__':main()

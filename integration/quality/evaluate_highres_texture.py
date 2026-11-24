"""Compare generator adapters at exactly the same fixed W+ codes."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from probe_inversion import sha256,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();run=a.run.resolve()
    experiment=json.loads((run/'experiment.json').read_text())
    reuse=Path(experiment['fixed_code_run']) if experiment.get('fixed_code_run') else None
    old_report=json.loads((reuse/'evaluation/report.json').read_text()) if reuse else None
    old_hashes={(o['name'],o['condition']):o['sha256'] for o in old_report['outputs'] if o['route']=='wplus_base'} if old_report else {}
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from torch.nn import functional as F
    from latent_quality import LatentQuality
    from deterministic_quality_ops import replace_pools,resize,verify
    from source_color import preserve_source_chroma
    base=Path(os.environ['FACE_AGE_BASE']);repo=Path(os.environ['FACE_AGE_REPO']);sys.path.insert(0,str(base/'src'))
    from sam_npu import load_sam
    from sam_losses import AgeEst
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(20261124)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False;verify()
    if torch.cuda.device_count()!=1:raise RuntimeError('Single GPU required')
    net=load_sam('cuda:0').eval().requires_grad_(False)
    release=deepcopy(net.encoder).eval().requires_grad_(False)
    release.load_state_dict(torch.load(base/'out/ft_l1/encoder_step1000.pt',map_location='cpu',weights_only=True))
    age=AgeEst().cuda().eval().requires_grad_(False);replace_pools(age)
    def tensor(rgb):return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1)[None].cuda().float()/127.5-1
    helper=LatentQuality(SimpleNamespace(torch=torch,dev='cuda:0',net=net,identity_encoder=release,age_net=age,tensor=tensor),'constrained',resize=resize)
    replace_pools(helper.identity);replace_pools(helper.perceptual)
    detector=cv2.FaceDetectorYN.create(str(base/'models/yunet/face_detection_yunet_2023mar.onnx'),'',(256,256),.6,.3,100)
    sf=cv2.FaceRecognizerSF.create(str(repo/'deployment/gan-research/face_recognition_sface_2021dec.onnx'),'')
    named=dict(net.decoder.named_parameters());patches={}
    generator_routes=experiment.get('generator_routes',{'highres_control':'control','highres_patch_gan':'patch_gan'})
    for label,arm in generator_routes.items():
        patches[label]=torch.load(run/arm/'generator_patch.pt',map_location='cuda:0',weights_only=True)
        if not set(patches[label])<=set(named):raise ValueError('Unknown generator parameter')
    scopes=list(patches.values())
    if len(scopes)!=2 or set(scopes[0])!=set(scopes[1]):raise ValueError('Paired generator scopes differ')
    original={n:named[n].detach().clone() for n in scopes[0]}
    def apply(state):
        with torch.no_grad():
            for n,value in state.items():named[n].copy_(value)
    def decode(code):return net(code,input_code=True,input_is_full=True,randomize_noise=False,resize=False)
    def small(rgb):return cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA)
    def feature(rgb):
        bgr=cv2.cvtColor(small(rgb),cv2.COLOR_RGB2BGR);_,faces=detector.detect(bgr)
        if faces is None:return None
        face=max(faces,key=lambda f:f[2]*f[3]);v=sf.feature(sf.alignCrop(bgr,face)).ravel();return v/(np.linalg.norm(v)+1e-12)
    out=run/'evaluation';out.mkdir(exist_ok=False)
    records=json.loads((run/'selection.json').read_text());routes=['wplus_base',*patches]
    report=dict(records=records,routes=routes,outputs=[],timings=[],comparison='Identical frozen W+ source/target codes for all generator routes; no candidate-specific fitting',
        caveats=['Exploratory FFHQ regression and broad publisher Asian APPA annotation; no new ethnicity inference.','MiVOLO age is an appearance estimate. Local statistics do not establish realistic texture.'])
    started=time.monotonic()
    for row in records:
        apply(original);path=row.get('reference_path',row['image_path']);digest=row.get('reference_sha256',row['sha256'])
        if sha256(path)!=digest:raise ValueError('Input digest changed')
        rgb=np.asarray(Image.open(path).convert('RGB'));x=tensor(small(rgb));source=row['source_age'];own=feature(rgb)
        folder=out/row['name'];folder.mkdir();Image.fromarray(rgb).save(folder/'reference.png')
        item=SimpleNamespace(aligned=small(rgb),reference_rgb=rgb,reference_age=source,quality_state=None)
        if reuse:
            saved=torch.load(reuse/'evaluation'/row['name']/'fixed_codes.pt',map_location='cpu',weights_only=True)
            if saved['conditions']!=['same','20','40','60','75'] or tuple(saved['codes'].shape)!=(5,18,512):raise ValueError('Unexpected fixed codes')
            codes=saved['codes'].cuda();fit_seconds=0;targets=[20,40,60,75]
        else:
            t=time.monotonic();helper.run(item,source,lambda:None);state=item.quality_state
            targets=[20,40,60,75];cond=lambda n:torch.cat([x,torch.full_like(x[:,:1],n/100)],1)
            with torch.no_grad():
                delta=torch.cat([release(cond(t)) for t in targets])-release(cond(source))
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
                if not torch.isfinite(loss) or not torch.isfinite(correction.grad).all():raise RuntimeError('Nonfinite W+ fit')
                opt.step()
            codes=torch.cat([state['anchor'],(pivot+torch.cat([correction,torch.zeros_like(pivot[:,8:])],1)).detach()])
            torch.cuda.synchronize();fit_seconds=time.monotonic()-t
        # Source-equal targets must use the same reconstruction anchor.
        for index,target in enumerate(targets,1):
            if target==source:codes[index].copy_(codes[0])
        torch.save(dict(codes=codes.cpu(),conditions=['same','20','40','60','75']),folder/'fixed_codes.pt')
        for route in routes:
            apply(original if route=='wplus_base' else patches[route]);t=time.monotonic()
            with torch.no_grad():
                if experiment.get('age_conditioned') and route!='wplus_base':
                    generated=[]
                    for code,target in zip(codes,[source,*targets]):
                        gate=max(0,min(1,(target-40)/35))
                        state={n:original[n]+gate*(value-original[n]) for n,value in patches[route].items()}
                        apply(state);generated.append(decode(code[None]))
                    images=torch.cat(generated)
                else:images=decode(codes)
            torch.cuda.synchronize();seconds=time.monotonic()-t
            for image,label,target in zip(images,['same','20','40','60','75'],[source,*targets]):
                if tuple(image.shape)!=(3,1024,1024) or not torch.isfinite(image).all():raise RuntimeError('Invalid1024 output')
                native=((image.clamp(-1,1)+1)*127.5).permute(1,2,0).byte().cpu().numpy();f=feature(native)
                file=folder/f'{route}_{label}.png';Image.fromarray(native).save(file)
                with Image.open(file) as check:check.load();assert check.size==(1024,1024) and check.format=='PNG'
                if reuse and route=='wplus_base' and sha256(file)!=old_hashes[row['name'],'same' if target==source else label]:raise RuntimeError('Reused baseline PNG differs from original fixed-code comparison')
                color,_=preserve_source_chroma(rgb,native);preserved=file.with_name(file.stem+'_color.png');Image.fromarray(color).save(preserved)
                report['outputs'].append(dict(name=row['name'],cohort=row['cohort'],route=route,condition=label,target=target,
                    native=str(file.relative_to(out)),preserved=str(preserved.relative_to(out)),sha256=sha256(file),
                    source_cosine=float(own@f) if own is not None and f is not None else None,detected=f is not None))
            report['timings'].append(dict(name=row['name'],route=route,wplus_fit_seconds=fit_seconds,native_decode_seconds=seconds))
        write_json(out/'report.partial.json',report);print('TEXTURE_EVAL',row['name'],round(fit_seconds,2),flush=True)
        del codes,images
        if not reuse:del loss,edited,correction,opt
    report['fixed_code_run']=str(reuse) if reuse else None;report['reused_baseline_exact']=bool(reuse);report['source_equal_target_uses_anchor']=True
    report['total_seconds']=time.monotonic()-started;write_json(out/'report.json',report);(out/'DONE').write_text('Paired native generator comparison complete\n')
    print('TEXTURE_EVALUATION_DONE',len(report['outputs']),flush=True)

if __name__=='__main__':main()

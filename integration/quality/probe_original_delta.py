"""Compare original SAM age displacement at the exact accepted W+ anchors."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from probe_inversion import sha256,write_json

ROOT=Path(os.environ['FACE_AGE_REPO']).resolve()


def main(run):
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from torch.nn import functional as F
    import lpips
    from deterministic_quality_ops import replace_pools,resize,verify
    from source_color import preserve_source_chroma
    if os.environ.get('FACE_AGE_EXPERIMENT_ALLOCATION')!='1' or torch.cuda.device_count()!=1:raise ValueError('Exactly one allocated GPU required')
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(20261124);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False;verify()
    exp=json.loads((run/'experiment.json').read_text());parent=Path(exp['parent_run'])
    for path,digest in exp['frozen_inputs'].items():
        if sha256(path)!=digest:raise ValueError('Frozen original-delta input changed')
    base=Path(os.environ['FACE_AGE_BASE']);sys.path.insert(0,str(base/'src'))
    from sam_npu import load_sam
    from sam_losses import IDLoss,AgeEst
    net=load_sam('cuda:0').eval().requires_grad_(False);fine=deepcopy(net.encoder).eval().requires_grad_(False);fine.load_state_dict(torch.load(base/'out/ft_l1/encoder_step1000.pt',map_location='cpu',weights_only=True))
    identity=IDLoss().cuda().eval().requires_grad_(False);age=AgeEst().cuda().eval().requires_grad_(False);lp=lpips.LPIPS(net='alex',verbose=False).cuda().eval().requires_grad_(False)
    for model in (identity,age,lp):replace_pools(model)
    def fingerprint():
        h=hashlib.sha256()
        for prefix,model in [('net',net),('fine',fine)]:
            for name,value in model.state_dict().items():h.update((prefix+name).encode());h.update(value.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    frozen=fingerprint()
    def tensor(rgb):return torch.from_numpy(np.array(rgb,copy=True,order='C')).permute(2,0,1).contiguous()[None].cuda().float()/127.5-1
    def decode(code):return net(code,input_code=True,input_is_full=True,randomize_noise=False,resize=False)
    def logits(image):
        value=resize((image+1)/2,(224,224),kind='bilinear');return age.net.age(age.net.b((value-age.m)/age.sd)).float()
    detector=cv2.FaceDetectorYN.create(str(base/'models/yunet/face_detection_yunet_2023mar.onnx'),'',(256,256),.6,.3,100);sf=cv2.FaceRecognizerSF.create(str(ROOT/'deployment/gan-research/face_recognition_sface_2021dec.onnx'),'')
    def feature(rgb):
        bgr=cv2.cvtColor(cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2BGR);_,faces=detector.detect(bgr)
        if faces is None:return None
        face=max(faces,key=lambda f:f[2]*f[3]);v=sf.feature(sf.alignCrop(bgr,face)).ravel();return v/(np.linalg.norm(v)+1e-12)
    parent_report=json.loads((parent/'evaluation/report.json').read_text());old={(o['name'],o['condition']):o['sha256'] for o in parent_report['outputs'] if o['route']=='wplus_base'}
    records=json.loads((run/'selection.json').read_text());out=run/'evaluation';out.mkdir(exist_ok=False);start=time.monotonic()
    report=dict(records=records,routes=['wplus_base','original_delta','original_delta_constrained'],outputs=[],timings=[],comparison='Same original generator, source reference age and frozen W+ reconstruction anchor. Compare ft_l1 age displacement with original SAM displacement; optional matched180step front8 correction. No target rescaling or image blend.',caveats=['Four previously inspected development inputs, not a blinded population test.','Teacher uses original SAM delta at the unrefined ft_l1 source anchor, matching the historical hybrid formula.','Raw original-delta route lacks the baseline target refinement; constrained original-delta route restores the same loss weights and steps.'])
    for row in records:
        folder=out/row['name'];folder.mkdir();rgb=np.asarray(Image.open(parent/'evaluation'/row['name']/'reference.png').convert('RGB'));Image.fromarray(rgb).save(folder/'reference.png');own=feature(rgb);x=tensor(cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA));source=row['source_age'];saved=torch.load(parent/'evaluation'/row['name']/'fixed_codes.pt',map_location='cuda',weights_only=True);baseline=saved['codes'];t=time.monotonic()
        condition=lambda n:torch.cat([x,torch.full_like(x[:,:1],n/100)],1)
        with torch.no_grad():
            delta=torch.cat([net.encoder(condition(n)) for n in (20,40,60,75)])-net.encoder(condition(source));pivot=baseline[:1]+delta
            unrefined=net.pretrained_encoder(x)+net.latent_avg+fine(condition(source));teacher=resize(decode(unrefined+delta),(256,256),kind='area');prob=logits(teacher).softmax(1);prior=resize(decode(pivot),(256,256),kind='area');ids=identity.feats(x).expand(4,-1);color=F.avg_pool2d(x,32).expand(4,-1,-1,-1)
        active=torch.tensor([n!=source for n in (20,40,60,75)],device='cuda')[:,None,None];correction=torch.zeros_like(pivot[:,:8]).requires_grad_(True);opt=torch.optim.Adam([correction],lr=.012)
        for step in range(180):
            opt.param_groups[0]['lr']=.012*(.2+.8*(1-step/180));edited=resize(decode(pivot+torch.cat([correction*active,torch.zeros_like(pivot[:,8:])],1)),(256,256),kind='area')
            ident=(1-(identity.feats(edited)*ids).sum(1)).mean();age_loss=F.kl_div(logits(edited).log_softmax(1),prob,reduction='batchmean');loss=.8*ident+.5*age_loss+.6*lp(edited,prior).mean()+.05*(correction*active).square().mean()+.1*F.l1_loss(F.avg_pool2d(edited,32),color)
            opt.zero_grad(set_to_none=True);loss.backward()
            if not torch.isfinite(loss) or not torch.isfinite(correction.grad).all():raise ValueError('Nonfinite original-delta correction')
            opt.step()
        raw=torch.cat([baseline[:1],pivot]);constrained=torch.cat([baseline[:1],pivot+torch.cat([correction.detach()*active,torch.zeros_like(pivot[:,8:])],1)])
        for code in (raw,constrained):
            for i,n in enumerate((20,40,60,75),1):
                if n==source:code[i].copy_(baseline[0])
        fit_seconds=time.monotonic()-t
        for route,code in [('wplus_base',baseline),('original_delta',raw),('original_delta_constrained',constrained)]:
            with torch.no_grad():images=decode(code)
            for image,label,target in zip(images,['same','20','40','60','75'],[source,20,40,60,75]):
                native=((image.clamp(-1,1)+1)*127.5).permute(1,2,0).byte().cpu().numpy();path=folder/f'{route}_{label}.png';Image.fromarray(native).save(path);digest=sha256(path)
                if (route=='wplus_base' or label=='same') and digest!=old[row['name'],label]:raise ValueError('Identical baseline/source anchor PNG changed')
                colorimage,_=preserve_source_chroma(rgb,native);preserved=path.with_name(path.stem+'_color.png');Image.fromarray(colorimage).save(preserved);f=feature(native)
                report['outputs'].append(dict(name=row['name'],cohort=row['cohort'],route=route,condition=label,target=target,native=str(path.relative_to(out)),preserved=str(preserved.relative_to(out)),sha256=digest,source_cosine=float(own@f) if own is not None and f is not None else None,detected=f is not None))
            torch.save(dict(codes=code.detach().cpu(),conditions=saved['conditions']),folder/(route+'_codes.pt'))
        report['timings'].append(dict(name=row['name'],original_delta_and180step_seconds=fit_seconds));write_json(out/'report.partial.json',report);print('ORIGINAL_DELTA_CASE_DONE',row['name'],round(fit_seconds,2),flush=True)
        del loss,edited,correction,opt,images,raw,constrained
    if fingerprint()!=frozen:raise ValueError('Frozen SAM weights changed')
    report.update(total_seconds=time.monotonic()-start,frozen_model_sha256=frozen,baseline_and_source_png_exact=True,cuda_peak_bytes=torch.cuda.max_memory_allocated(),default_model_changed=False);write_json(out/'report.json',report);(run/'DONE').write_text('Original SAM delta ablation complete\n');print('ORIGINAL_DELTA_DONE',flush=True)


if __name__=='__main__':main(Path(sys.argv[1]).resolve())

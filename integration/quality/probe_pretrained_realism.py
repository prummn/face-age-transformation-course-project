"""Paired W+ refinement with a frozen official FFHQ discriminator prior.

This is a four-case development diagnostic, not an independent realism metric.
All SAM/StyleGAN and discriminator weights remain frozen. Only fine W+ layers
are optimized; target ages and source/target coarse codes remain unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from probe_inversion import sha256,write_json

ROOT=Path(os.environ.get('FACE_AGE_REPO',Path(__file__).resolve().parents[2])).resolve()
PRIOR_SHA='a205a346e86a9ddaae702e118097d014b7b8bd719491396a162cca438f2f524c'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();run=a.run.resolve()
    exp=json.loads((run/'experiment.json').read_text());parent=Path(exp['parent_run'])
    if not (parent/'DONE').exists():raise ValueError('Parent GPU study must finish first')
    for path,digest in exp['frozen_inputs'].items():
        if sha256(path)!=digest:raise ValueError('Frozen diagnostic input changed')
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from torch.nn import functional as F
    from train_native_age import setup
    from deterministic_quality_ops import replace_pools,resize,verify
    from source_color import preserve_source_chroma
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    net,identity,age,lp=setup();replace_pools(identity);replace_pools(age);replace_pools(lp);verify()
    sys.path.insert(0,str(ROOT/'deployment/CUSP'))
    from torch_utils.ops import bias_act,upfirdn2d,conv2d_gradfix
    # Standard reference operators avoid compiling old CUDA extensions.
    bias_act._init=lambda:False;upfirdn2d._init=lambda:False;conv2d_gradfix.enabled=False
    import legacy
    prior=ROOT/'deployment/gan-research/stylegan2ada_ffhq_1024.pkl'
    if sha256(prior)!=PRIOR_SHA:raise ValueError('Official prior bytes changed')
    with prior.open('rb') as f:loaded=legacy.load_network_pkl(f)
    d=loaded['D'].float().cuda().eval().requires_grad_(False);del loaded
    if d.c_dim!=0 or d.img_resolution!=1024:raise ValueError('Expected unconditional1024 discriminator')
    # Remove interactions between different people/target ages in this loss.
    # At inference a one-image batch uses this same minibatch-std convention.
    if d.b4.mbstd is not None:d.b4.mbstd.group_size=1
    def state_digest(module):
        h=hashlib.sha256()
        for name,tensor in module.state_dict().items():h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    frozen=dict(generator=state_digest(net.decoder),discriminator=state_digest(d))
    def tensor(rgb):return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1).contiguous()[None].cuda().float()/127.5-1
    def decode(code):return net(code,input_code=True,input_is_full=True,randomize_noise=False,resize=False)
    def score(image):return d(image,None,force_fp32=True).flatten()
    def logits(image):
        image=F.avg_pool2d(image,9,1,4);v=resize((image+1)/2,(224,224),kind='bilinear')
        return age.net.age(age.net.b((v-age.m)/age.sd)).float()
    base=Path(os.environ['FACE_AGE_BASE']);detector=cv2.FaceDetectorYN.create(str(base/'models/yunet/face_detection_yunet_2023mar.onnx'),'',(256,256),.6,.3,100)
    sf=cv2.FaceRecognizerSF.create(str(ROOT/'deployment/gan-research/face_recognition_sface_2021dec.onnx'),'')
    def feature(rgb):
        bgr=cv2.cvtColor(cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2BGR);_,faces=detector.detect(bgr)
        if faces is None:return None
        face=max(faces,key=lambda f:f[2]*f[3]);v=sf.feature(sf.alignCrop(bgr,face)).ravel();return v/(np.linalg.norm(v)+1e-12)
    parent_report=json.loads((parent/'evaluation/report.json').read_text());old={(o['name'],o['condition']):o['sha256'] for o in parent_report['outputs'] if o['route']=='wplus_base'}
    records=json.loads((run/'selection.json').read_text());out=run/'evaluation';out.mkdir(exist_ok=False)
    report=dict(records=records,routes=['wplus_base','prior_control','pretrained_D'],outputs=[],timings=[],discriminator_minibatch_std_group_size=1,comparison='Identical baseline W+; paired150step fine-layer optimization, sole added term frozen official FFHQ discriminator. All model parameters frozen.',caveats=['Four previously examined development inputs, no blinded population claim.','FFHQ discriminator was trained on FFHQ; its scores are optimization diagnostics, not independent naturalness evidence.','Same/20/40 use identical baseline codes; only60/75 fine layers8:18 may change.'])
    start=time.monotonic();log=run/'fit.jsonl'
    for row in records:
        folder=out/row['name'];folder.mkdir();reference=parent/'evaluation'/row['name']/'reference.png';rgb=np.asarray(Image.open(reference).convert('RGB'));Image.fromarray(rgb).save(folder/'reference.png');own=feature(rgb)
        source=tensor(cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA));saved=torch.load(parent/'evaluation'/row['name']/'fixed_codes.pt',map_location='cuda',weights_only=True);codes=saved['codes']
        if saved['conditions']!=['same','20','40','60','75'] or tuple(codes.shape)!=(5,18,512):raise ValueError('Unexpected frozen W+')
        with torch.no_grad():
            native=decode(codes);baseline=resize(native[3:5],(256,256),kind='area');prob=logits(baseline).softmax(1);ids=identity.feats(source).expand(2,-1);color=F.avg_pool2d(source,32).expand(2,-1,-1,-1)
            source_score=float(score(tensor(rgb)).mean());base_scores=score(native[3:5]).cpu().tolist()
        routes={'wplus_base':codes};scores={};elapsed={}
        for arm in ('prior_control','pretrained_D'):
            t=time.monotonic();delta=torch.zeros_like(codes[3:5,8:]).requires_grad_(True);opt=torch.optim.Adam([delta],lr=.01)
            active=torch.tensor([row['source_age']!=target for target in (60,75)],device='cuda')[:,None,None]
            for step in range(150):
                opt.param_groups[0]['lr']=.01*(.2+.8*(1-step/150));fine=delta*active;edited=decode(codes[3:5]+torch.cat([torch.zeros_like(codes[3:5,:8]),fine],1));small=resize(edited,(256,256),kind='area')
                ident=(1-(identity.feats(small)*ids).sum(1)).mean();perceptual=lp(small,baseline).mean();age_loss=F.kl_div(logits(small).log_softmax(1),prob,reduction='batchmean');chroma=F.l1_loss(F.avg_pool2d(small,32),color);trust=fine.square().mean()
                realism=F.softplus(-score(edited)).mean() if arm=='pretrained_D' else edited.new_zeros(())
                loss=ident+perceptual+.5*age_loss+.1*chroma+.1*trust+.05*realism
                opt.zero_grad(set_to_none=True);loss.backward()
                if not torch.isfinite(loss) or not torch.isfinite(delta.grad).all():raise ValueError('Nonfinite prior refinement')
                torch.nn.utils.clip_grad_norm_([delta],1);opt.step()
                if step==0 or (step+1)%25==0:
                    point=dict(name=row['name'],arm=arm,step=step+1,loss=float(loss.detach()),identity=float(ident.detach()),LPIPS=float(perceptual.detach()),age_KL=float(age_loss.detach()),realism=float(realism.detach()),fine_delta_rms=float(fine.detach().square().mean().sqrt()))
                    with log.open('a') as f:f.write(json.dumps(point)+'\n')
            candidate=codes.detach().clone();candidate[3:5,8:]+=delta.detach()*active;routes[arm]=candidate;elapsed[arm]=time.monotonic()-t
            with torch.no_grad():scores[arm]=score(decode(candidate)[3:5]).cpu().tolist()
            del loss,edited,small,opt,delta
        for route,code in routes.items():
            with torch.no_grad():images=decode(code)
            for image,label,target in zip(images,['same','20','40','60','75'],[row['source_age'],20,40,60,75]):
                native=((image.clamp(-1,1)+1)*127.5).permute(1,2,0).byte().cpu().numpy();path=folder/f'{route}_{label}.png';Image.fromarray(native).save(path);digest=sha256(path)
                if (route=='wplus_base' or label in ('same','20','40')) and digest!=old[row['name'],label]:raise ValueError('Unmodified baseline PNG differs')
                preserved,_=preserve_source_chroma(rgb,native);colorpath=path.with_name(path.stem+'_color.png');Image.fromarray(preserved).save(colorpath);f=feature(native)
                report['outputs'].append(dict(name=row['name'],cohort=row['cohort'],route=route,condition=label,target=target,native=str(path.relative_to(out)),preserved=str(colorpath.relative_to(out)),sha256=digest,source_cosine=float(own@f) if own is not None and f is not None else None,detected=f is not None))
            torch.save(dict(codes=code.cpu(),conditions=saved['conditions']),folder/(route+'_codes.pt'))
        report['timings'].append(dict(name=row['name'],source_D_score=source_score,baseline_D_scores=base_scores,refined_D_scores=scores,refinement_seconds=elapsed))
        write_json(out/'report.partial.json',report);print('PRETRAINED_PRIOR_CASE_DONE',row['name'],elapsed,flush=True)
    if state_digest(net.decoder)!=frozen['generator'] or state_digest(d)!=frozen['discriminator']:raise ValueError('Frozen generator/discriminator changed')
    report.update(total_seconds=time.monotonic()-start,default_model_changed=False,frozen_hashes=frozen,baseline_and_young_png_exact=True,cuda_peak_bytes=torch.cuda.max_memory_allocated())
    write_json(out/'report.json',report);(run/'DONE').write_text('Paired pretrained-discriminator W+ diagnostic complete\n');print('PRETRAINED_PRIOR_DONE',flush=True)


if __name__=='__main__':main()

"""One frozen-weight W+ ablation: preserve low-frequency source chroma."""
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    run=p.parse_args().run;protocol=json.loads((run/'experiment.json').read_text())
    previous=Path(protocol['previous_run']);old=previous/'external-results'
    old_report=json.loads((old/'report.json').read_text())
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    base=Path(os.environ['FACE_AGE_BASE']);root=Path(os.environ['FACE_AGE_REPO']);sys.path.insert(0,str(base/'src'))
    from sam_npu import load_sam
    from sam_losses import AgeEst
    from latent_quality import LatentQuality
    assert torch.cuda.device_count()==1
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(20261116)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    net=load_sam('cuda:0').eval().requires_grad_(False);fine=deepcopy(net.encoder).eval().requires_grad_(False)
    fine.load_state_dict(torch.load(base/'out/ft_l1/encoder_step1000.pt',map_location='cpu',weights_only=True))
    age=AgeEst().cuda().eval().requires_grad_(False)
    def tensor(rgb):return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1)[None].contiguous().cuda().float()/127.5-1
    engine=SimpleNamespace(torch=torch,dev='cuda:0',net=net,identity_encoder=fine,age_net=age,tensor=tensor)
    refiner=LatentQuality(engine,'texture')
    detector=cv2.FaceDetectorYN.create(str(base/'models/yunet/face_detection_yunet_2023mar.onnx'),'',(256,256),.6,.3,100)
    recognizer=cv2.FaceRecognizerSF.create(str(root/'deployment/gan-research/face_recognition_sface_2021dec.onnx'),'')
    def decode(code):return net(code,input_code=True,input_is_full=True,randomize_noise=False,resize=False)
    def logits(y):
        y=F.interpolate((y+1)/2,(224,224),mode='bilinear',align_corners=False)
        return age.net((y-age.m)/age.sd)[2].float()
    def chroma(image):
        pooled=F.adaptive_avg_pool2d(image,(16,16))
        return pooled-pooled.mean(1,keepdim=True)
    def feature(rgb,face):
        if face is None:return None
        f=recognizer.feature(recognizer.alignCrop(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),face)).ravel()
        return f/(np.linalg.norm(f)+1e-12)
    out=run/'chroma-repair';out.mkdir(exist_ok=False)
    records=old_report['records'];records.sort(key=lambda r:(r['review_split']!='diagnostic',r['cohort'],r['name']))
    report=dict(records=records,routes=['baseline_v1','texture_half','texture_chroma'],outputs=deepcopy(old_report['outputs']),timings=[],protocol=protocol,
        title='新增人像：低频颜色潜变量约束',note='原始GAN输出对照；仅增加16×16色度损失，不粘贴源像素、不做输出保色。')
    started=time.monotonic()
    for row in records:
        name=row['name'];folder=out/name;folder.mkdir();source=old/name
        # Existing controls are immutable symlinks, not rerun or relabeled candidates.
        for path in source.iterdir():
            if path.suffix=='.png':(folder/path.name).symlink_to(path.resolve())
        assert sha256(row['image_path'])==row['sha256']
        rgb=np.asarray(Image.open(source/'reference.png').convert('RGB'));small=cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA);x=tensor(small)
        _,faces=detector.detect(cv2.cvtColor(small,cv2.COLOR_RGB2BGR));face=max(faces,key=lambda f:f[2]*f[3]) if faces is not None else None
        src_feature=feature(small,face)
        cache=source/'texture_half_codes.pt';assert sha256(cache)==protocol['cache_sha256'][name]
        saved=torch.load(cache,map_location='cuda:0',weights_only=True);anchor=saved['anchor'];s=saved['source'];targets=[20,40,60,75]
        with torch.no_grad():
            identity=refiner.identity.feats(x).detach();basecode=net.pretrained_encoder(x)+net.latent_avg
            def code(t):return basecode+fine(torch.cat([x,torch.full_like(x[:,:1],t/100)],1))
            source_code=code(s);original=torch.cat([code(t) for t in targets]);direction=original-source_code
            target=anchor+torch.cat([direction[:,:8],.5*direction[:,8:]],1)
            prior=F.interpolate(decode(target),(256,256),mode='area')
            prob=logits(F.interpolate(decode(original),(256,256),mode='area')).softmax(1)
            source_chroma=chroma(x).expand(4,-1,-1,-1)
        tick=time.monotonic()
        for chroma_weight in ([0.,protocol['chroma_weight']] if name==records[0]['name'] else [protocol['chroma_weight']]):
            delta=torch.zeros_like(target[:,:8]).requires_grad_(True);optimizer=torch.optim.Adam([delta],lr=.012)
            for step in range(180):
                optimizer.param_groups[0]['lr']=.012*(.2+.8*(1-step/180))
                y=F.interpolate(decode(target+torch.cat([delta,torch.zeros_like(target[:,8:])],1)),(256,256),mode='area')
                ident=(1-(refiner.identity.feats(y)*identity).sum(1)).mean()
                kl=F.kl_div(logits(y).log_softmax(1),prob,reduction='batchmean')
                color=F.l1_loss(F.avg_pool2d(y,32),F.avg_pool2d(x,32).expand(4,-1,-1,-1))
                loss=.8*ident+.5*kl+.6*refiner.perceptual(y,prior).mean()+.05*delta.square().mean()+.1*color
                loss=loss+chroma_weight*F.l1_loss(chroma(y),source_chroma)
                optimizer.zero_grad(set_to_none=True);loss.backward();assert torch.isfinite(loss) and torch.isfinite(delta.grad).all();optimizer.step()
            if chroma_weight==0.:
                with torch.no_grad():
                    control=target+torch.cat([delta,torch.zeros_like(target[:,8:])],1)
                    code_error=float((control-saved['target_codes']).abs().max())
                    assert code_error<1e-4,('Unchanged control failed reproduction',code_error)
                write_json(run/'control-reproduction.json',dict(name=name,max_latent_error=code_error,zero_chroma_replayed=True))
        torch.cuda.synchronize();seconds=time.monotonic()-tick
        with torch.no_grad():
            codes=target+torch.cat([delta,torch.zeros_like(target[:,8:])],1)
            native_images=decode(codes)
            same=deepcopy(next(o for o in old_report['outputs'] if o['name']==name and o['route']=='texture_half' and o['condition']=='same'))
            same.update(route='texture_chroma',native=name+'/texture_chroma_same.png',preview=name+'/texture_chroma_same_preview.png')
            (folder/'texture_chroma_same.png').symlink_to((source/'texture_half_same.png').resolve())
            (folder/'texture_chroma_same_preview.png').symlink_to((source/'texture_half_same_preview.png').resolve());report['outputs'].append(same)
            for t,image in zip(targets,native_images):
                assert tuple(image.shape)==(3,1024,1024) and torch.isfinite(image).all()
                native=((image.clamp(-1,1)+1)*127.5).permute(1,2,0).byte().cpu().numpy();reduced=cv2.resize(native,(256,256),interpolation=cv2.INTER_AREA)
                y256=F.interpolate(image[None],(256,256),mode='area');f=feature(reduced,face)
                ages=[float(logits(v).softmax(1)@age.mid)*100 for v in [y256,F.avg_pool2d(y256,3,1,1),y256[:,:,8:-8,8:-8]]]
                a=cv2.resize(native,(64,64)).astype(float);b=cv2.resize(rgb,(64,64)).astype(float)
                chroma_mae=float(np.abs((a-a.mean(2,keepdims=True))-(b-b.mean(2,keepdims=True))).mean())
                gray=cv2.cvtColor(cv2.resize(native,(512,512)),cv2.COLOR_RGB2GRAY).astype(np.float32)/255
                energy=np.abs(gray-cv2.GaussianBlur(gray,(0,0),1))[220:375,130:382].mean()
                stem='texture_chroma_'+str(t);Image.fromarray(native).save(folder/(stem+'.png'));Image.fromarray(reduced).save(folder/(stem+'_preview.png'))
                with Image.open(folder/(stem+'.png')) as check:check.load();assert check.size==(1024,1024)
                report['outputs'].append(dict(name=name,cohort=row['cohort'],review_split=row['review_split'],route='texture_chroma',condition=str(t),source_age=s,target=t,
                    sface_cosine=float(src_feature@f) if f is not None and src_feature is not None else None,lpips_shared=float(refiner.perceptual(y256,x).mean()),
                    shared_output_age=ages[0],age_views=ages,age_view_spread=max(ages)-min(ages),chroma_mae=chroma_mae,rgb_mae=float(np.abs(a-b).mean()),
                    face_high_frequency_energy=float(energy),native_size=[1024,1024],native=name+'/'+stem+'.png',preview=name+'/'+stem+'_preview.png'))
        torch.save(dict(anchor=anchor.cpu(),target_codes=codes.detach().cpu(),source=s),folder/'texture_chroma_codes.pt')
        assert all(not q.requires_grad and q.grad is None for model in [net,fine,age,refiner.identity,refiner.perceptual] for q in model.parameters())
        report['timings'].append(dict(name=name,route='texture_chroma',edit_seconds=seconds,anchor_reused=True))
        write_json(out/'report.partial.json',report);print('CHROMA_SAMPLE_DONE',name,row['review_split'],round(seconds,2),flush=True)
    report['total_seconds']=time.monotonic()-started;write_json(out/'report.json',report)
    assert len(report['outputs'])==len(records)*15
    (out/'DONE').write_text('One chroma W+ ablation complete; no network-weight training\n');(run/'DONE').write_text('Chroma W+ evaluation done; GPU exits\n')


if __name__=='__main__':main()

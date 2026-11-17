"""Frozen-weight batch-age evaluation of newly collected external portraits."""
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
    a=p.parse_args();run=a.run;protocol=json.loads((run/'experiment.json').read_text())
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    base=Path(os.environ['FACE_AGE_BASE']);root=Path(os.environ['FACE_AGE_REPO'])
    sys.path.insert(0,str(base/'src'))
    from sam_npu import load_sam
    from sam_losses import AgeEst
    from latent_quality import LatentQuality
    assert torch.cuda.device_count()==1
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(20261116)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    net=load_sam('cuda:0').eval().requires_grad_(False)
    fine=deepcopy(net.encoder).eval().requires_grad_(False)
    fine.load_state_dict(torch.load(base/'out/ft_l1/encoder_step1000.pt',map_location='cpu',weights_only=True))
    age=AgeEst().cuda().eval().requires_grad_(False)
    def tensor(rgb):return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1)[None].contiguous().cuda().float()/127.5-1
    engine=SimpleNamespace(torch=torch,dev='cuda:0',net=net,identity_encoder=fine,age_net=age,tensor=tensor)
    refiner=LatentQuality(engine,'constrained')
    detector=cv2.FaceDetectorYN.create(str(base/'models/yunet/face_detection_yunet_2023mar.onnx'),'',(256,256),.6,.3,100)
    recognizer=cv2.FaceRecognizerSF.create(str(root/'deployment/gan-research/face_recognition_sface_2021dec.onnx'),'')
    out=run/'external-results';out.mkdir(exist_ok=False)
    records=json.loads((run/'inputs.pre-age.json').read_text())
    # Development cases first, confirmation not inspected until repair is fixed.
    records.sort(key=lambda r:(r['review_split']!='diagnostic',r['cohort'],r['name']))
    report=dict(records=records,routes=['baseline_v1','texture_half'],outputs=[],timings=[],protocol=protocol,
        title='新增外部人像：基线与纹理候选',note='官方APPA测试图；同龄重建后四目标联合优化，全部网络权重冻结。')
    def decode(code):return net(code,input_code=True,input_is_full=True,randomize_noise=False,resize=False)
    def logits(y):
        y=F.interpolate((y+1)/2,(224,224),mode='bilinear',align_corners=False)
        return age.net((y-age.m)/age.sd)[2].float()
    def feature(rgb,face):
        if face is None:return None
        f=recognizer.feature(recognizer.alignCrop(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),face)).ravel()
        return f/(np.linalg.norm(f)+1e-12)
    started=time.monotonic()
    for row in records:
        path=row['image_path'];assert sha256(path)==row['sha256']
        rgb=np.asarray(Image.open(path).convert('RGB'));small=cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA);x=tensor(small)
        folder=out/row['name'];folder.mkdir();Image.fromarray(rgb).save(folder/'reference.png')
        _,faces=detector.detect(cv2.cvtColor(small,cv2.COLOR_RGB2BGR));face=max(faces,key=lambda f:f[2]*f[3]) if faces is not None else None
        source_feature=feature(small,face)
        with torch.no_grad():estimate=float(age(x)[0])*100
        source=round(estimate);row.update(source_age=source,shared_reference_age=estimate,age_kind='pseudo',
            source_age_method='Frozen FairFace expectation; independent publisher ages kept separately')
        item=SimpleNamespace(aligned=small,reference_rgb=rgb,reference_age=source,quality_state=None)
        tick=time.monotonic();same=refiner.run(item,source,lambda:None);torch.cuda.synchronize();anchor_seconds=time.monotonic()-tick
        state=item.quality_state;anchor=state['anchor'];targets=[('same',source)]+[(str(t),t) for t in [20,40,60,75]]
        with torch.no_grad():
            original=torch.cat([state['base']+fine(torch.cat([x,torch.full_like(x[:,:1],t/100)],1)) for t in [20,40,60,75]])
            direction=original-state['source'];teacher=F.interpolate(decode(original),(256,256),mode='area');prob=logits(teacher).softmax(1)
        def record(route,images):
            with torch.no_grad():
                for (condition,target),image in zip(targets,images):
                    y=image[None];assert tuple(y.shape)==(1,3,1024,1024) and torch.isfinite(y).all()
                    native=((y[0].clamp(-1,1)+1)*127.5).permute(1,2,0).byte().cpu().numpy()
                    reduced=cv2.resize(native,(256,256),interpolation=cv2.INTER_AREA);y256=F.interpolate(y,(256,256),mode='area');f=feature(reduced,face)
                    ages=[float(logits(v).softmax(1)@age.mid)*100 for v in [y256,F.avg_pool2d(y256,3,1,1),y256[:,:,8:-8,8:-8]]]
                    uv=lambda im:im-im.mean(2,keepdims=True)
                    rgb_delta=np.mean(np.abs(cv2.resize(native,(64,64)).astype(float)-cv2.resize(rgb,(64,64)).astype(float)))
                    chroma_delta=np.mean(np.abs(uv(cv2.resize(native,(64,64)).astype(float))-uv(cv2.resize(rgb,(64,64)).astype(float))))
                    gray=cv2.cvtColor(cv2.resize(native,(512,512)),cv2.COLOR_RGB2GRAY).astype(np.float32)/255
                    energy=np.abs(gray-cv2.GaussianBlur(gray,(0,0),1))[220:375,130:382].mean()
                    stem=route+'_'+condition;Image.fromarray(native).save(folder/(stem+'.png'));Image.fromarray(reduced).save(folder/(stem+'_preview.png'))
                    with Image.open(folder/(stem+'.png')) as check:check.load();assert check.size==(1024,1024)
                    report['outputs'].append(dict(name=row['name'],cohort=row['cohort'],review_split=row['review_split'],route=route,condition=condition,
                        source_age=source,target=target,sface_cosine=float(source_feature@f) if source_feature is not None and f is not None else None,
                        lpips_shared=float(refiner.perceptual(y256,x).mean()),shared_output_age=ages[0],age_views=ages,age_view_spread=max(ages)-min(ages),
                        chroma_mae=chroma_delta,rgb_mae=rgb_delta,face_high_frequency_energy=float(energy),native_size=[1024,1024],
                        native=row['name']+'/'+stem+'.png',preview=row['name']+'/'+stem+'_preview.png'))
        for route in report['routes']:
            scale=torch.ones_like(direction)
            if route=='texture_half':scale[:,8:]=.5
            target=anchor+direction*scale
            with torch.no_grad():prior=F.interpolate(decode(target),(256,256),mode='area')
            delta=torch.zeros_like(target[:,:8]).requires_grad_(True);optimizer=torch.optim.Adam([delta],lr=.012);tick=time.monotonic()
            for step in range(180):
                optimizer.param_groups[0]['lr']=.012*(.2+.8*(1-step/180))
                y=F.interpolate(decode(target+torch.cat([delta,torch.zeros_like(target[:,8:])],1)),(256,256),mode='area')
                ident=(1-(refiner.identity.feats(y)*state['identity']).sum(1)).mean()
                kl=F.kl_div(logits(y).log_softmax(1),prob,reduction='batchmean')
                color=F.l1_loss(F.avg_pool2d(y,32),F.avg_pool2d(x,32).expand(4,-1,-1,-1))
                loss=.8*ident+.5*kl+.6*refiner.perceptual(y,prior).mean()+.05*delta.square().mean()+.1*color
                optimizer.zero_grad(set_to_none=True);loss.backward()
                assert torch.isfinite(loss) and torch.isfinite(delta.grad).all();optimizer.step()
            torch.cuda.synchronize();seconds=time.monotonic()-tick
            with torch.no_grad():
                codes=target+torch.cat([delta,torch.zeros_like(target[:,8:])],1)
                record(route,torch.cat([same,decode(codes)]))
            torch.save(dict(anchor=anchor.cpu(),target_codes=codes.detach().cpu(),source=source),folder/(route+'_codes.pt'))
            report['timings'].append(dict(name=row['name'],route=route,anchor_seconds=anchor_seconds,edit_seconds=seconds))
        assert all(not q.requires_grad and q.grad is None for model in [net,fine,age,refiner.identity,refiner.perceptual] for q in model.parameters())
        write_json(out/'report.partial.json',report);write_json(run/'annotated-inputs.json',records)
        print('EXTERNAL_SAMPLE_DONE',row['name'],row['review_split'],source,round(anchor_seconds,2),flush=True)
    report['total_seconds']=time.monotonic()-started;write_json(out/'report.json',report)
    (out/'DONE').write_text('External baseline/candidate evaluation complete; no training\n')


if __name__=='__main__':main()

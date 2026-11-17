"""Evaluate a fixed texture candidate on additional cases using serving code."""
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
    import cv2
    import lpips
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    base=Path(os.environ['FACE_AGE_BASE']);root=Path(os.environ['FACE_AGE_REPO'])
    sys.path.insert(0,str(base/'src'))
    from sam_npu import load_sam
    from sam_losses import AgeEst
    from latent_quality import LatentQuality
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(20261116)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.device_count()==1
    net=load_sam('cuda:0').eval().requires_grad_(False)
    fine=deepcopy(net.encoder).eval().requires_grad_(False)
    fine.load_state_dict(torch.load(base/'out/ft_l1/encoder_step1000.pt',map_location='cpu',weights_only=True))
    age=AgeEst().cuda().eval().requires_grad_(False)
    def tensor(rgb):return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1)[None].contiguous().cuda().float()/127.5-1
    engine=SimpleNamespace(torch=torch,dev='cuda:0',net=net,identity_encoder=fine,age_net=age,tensor=tensor)
    refiner=LatentQuality(engine,'constrained')
    detector=cv2.FaceDetectorYN.create(str(base/'models/yunet/face_detection_yunet_2023mar.onnx'),'',(256,256),.6,.3,100)
    recognizer=cv2.FaceRecognizerSF.create(str(root/'deployment/gan-research/face_recognition_sface_2021dec.onnx'),'')
    records=json.loads((run/'validation-inputs.json').read_text())
    route=protocol['candidate_route'];mode=protocol['candidate_mode']
    out=run/'age-texture';out.mkdir(exist_ok=False)
    report=dict(records=records,routes=['baseline_v1',route],outputs=[],timings=[],protocol=protocol,
        title='追加样例：基线v1与纹理候选',note='使用实际后端优化代码；11张追加样例，不用于本轮选参数。')
    def logits(y):
        y=F.interpolate((y+1)/2,(224,224),mode='bilinear',align_corners=False)
        return age.net((y-age.m)/age.sd)[2].float()
    def feature(rgb,face):
        if face is None:return None
        v=recognizer.feature(recognizer.alignCrop(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),face)).ravel()
        return v/(np.linalg.norm(v)+1e-12)
    started=time.monotonic()
    for row in records:
        path=row.get('reference_path',row['image_path']);assert sha256(path)==row.get('reference_sha256',row['sha256'])
        rgb=np.asarray(Image.open(path).convert('RGB'));small=cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA);x=tensor(small)
        folder=out/row['name'];folder.mkdir();Image.fromarray(rgb).save(folder/'reference.png')
        _,faces=detector.detect(cv2.cvtColor(small,cv2.COLOR_RGB2BGR))
        face=max(faces,key=lambda f:f[2]*f[3]) if faces is not None else None;source_feature=feature(small,face)
        item=SimpleNamespace(aligned=small,reference_rgb=rgb,reference_age=row['source_age'],quality_state=None)
        targets=[('same',row['source_age'])]+[(str(t),t) for t in [20,40,60,75]]
        for method,quality in [('baseline_v1','constrained'),(route,mode)]:
            refiner.mode=quality
            for condition,target in targets:
                tick=time.monotonic();y=refiner.run(item,target,lambda:None)
                torch.cuda.synchronize();elapsed=time.monotonic()-tick
                with torch.no_grad():
                    native=((y[0].clamp(-1,1)+1)*127.5).permute(1,2,0).byte().cpu().numpy()
                    assert native.shape==(1024,1024,3) and torch.isfinite(y).all()
                    reduced=cv2.resize(native,(256,256),interpolation=cv2.INTER_AREA);y256=F.interpolate(y,(256,256),mode='area')
                    estimates=[float(logits(v).softmax(1)@age.mid)*100 for v in [y256,F.avg_pool2d(y256,3,1,1),y256[:,:,8:-8,8:-8]]]
                    f=feature(reduced,face);gray=cv2.cvtColor(cv2.resize(native,(512,512)),cv2.COLOR_RGB2GRAY).astype(np.float32)/255
                    energy=np.abs(gray-cv2.GaussianBlur(gray,(0,0),1))[220:375,130:382].mean()
                    stem=method+'_'+condition;Image.fromarray(native).save(folder/(stem+'.png'));Image.fromarray(reduced).save(folder/(stem+'_preview.png'))
                    with Image.open(folder/(stem+'.png')) as check:check.load();assert check.size==(1024,1024)
                    report['outputs'].append(dict(name=row['name'],cohort=row['cohort'],route=method,condition=condition,target=target,
                        source_age=row['source_age'],sface_cosine=float(source_feature@f) if f is not None else None,
                        lpips_shared=float(refiner.perceptual(y256,x).mean()),shared_output_age=estimates[0],age_views=estimates,
                        age_view_spread=max(estimates)-min(estimates),face_high_frequency_energy=float(energy),native_size=[1024,1024],
                        native=row['name']+'/'+stem+'.png',preview=row['name']+'/'+stem+'_preview.png'))
                report['timings'].append(dict(name=row['name'],route=method,condition=condition,edit_seconds=elapsed))
            print('TEXTURE_VALIDATION',row['name'],method,flush=True)
        assert all(not q.requires_grad and q.grad is None for model in [net,fine,age,refiner.identity,refiner.perceptual] for q in model.parameters())
        write_json(out/'report.partial.json',report)
    report['total_seconds']=time.monotonic()-started;write_json(out/'report.json',report)
    (out/'DONE').write_text('Additional11 cases checked with serving code\n')


if __name__=='__main__':main()

"""Paired frozen-weight age/texture experiments against accepted W+ baseline."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
from probe_inversion import sha256, write_json


ROUTES = ['baseline_v1', 'texture_half', 'robust_age', 'texture_robust']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    a = p.parse_args(); run = a.run
    protocol = json.loads((run/'experiment.json').read_text())
    previous = Path(protocol['previous_run'])
    out = run/'age-texture'; out.mkdir(exist_ok=False)
    import cv2
    import lpips
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    base = Path(os.environ['FACE_AGE_BASE']); root = Path(os.environ['FACE_AGE_REPO'])
    sys.path.insert(0, str(base/'src'))
    from sam_npu import load_sam
    from sam_losses import IDLoss, AgeEst
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4); cv2.setNumThreads(2); torch.manual_seed(20261116)
    torch.backends.cudnn.benchmark = False; torch.backends.cuda.matmul.allow_tf32 = False
    net = load_sam('cuda:0').eval().requires_grad_(False)
    fine = deepcopy(net.encoder).eval().requires_grad_(False)
    fine.load_state_dict(torch.load(base/'out/ft_l1/encoder_step1000.pt',map_location='cpu',weights_only=True))
    identity = IDLoss().cuda().eval().requires_grad_(False)
    perceptual = lpips.LPIPS(net='alex',verbose=False).cuda().eval().requires_grad_(False)
    age = AgeEst().cuda().eval().requires_grad_(False)
    detector = cv2.FaceDetectorYN.create(str(base/'models/yunet/face_detection_yunet_2023mar.onnx'),'',(256,256),.6,.3,100)
    recognizer = cv2.FaceRecognizerSF.create(str(root/'deployment/gan-research/face_recognition_sface_2021dec.onnx'),'')
    records = json.loads((run/'latent-inputs.json').read_text())
    report = dict(records=records,routes=ROUTES,outputs=[],timings=[],protocol=protocol)
    def tensor(rgb):
        return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1)[None].contiguous().cuda().float()/127.5-1
    def decode(code):
        return net(code,input_code=True,input_is_full=True,randomize_noise=False,resize=False)
    def views(y):
        return [y,F.avg_pool2d(y,3,1,1),y[:,:,8:-8,8:-8]]
    def logits(y):
        y = F.interpolate((y+1)/2,(224,224),mode='bilinear',align_corners=False)
        return age.net((y-age.m)/age.sd)[2].float()
    def feature(rgb,face):
        if face is None:return None
        v = recognizer.feature(recognizer.alignCrop(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),face)).ravel()
        return v/(np.linalg.norm(v)+1e-12)
    started = time.monotonic()
    for row in records:
        image_path = row.get('reference_path',row['image_path'])
        assert sha256(image_path) == row.get('reference_sha256',row['sha256'])
        rgb = np.asarray(Image.open(image_path).convert('RGB'))
        small = cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA); x = tensor(small)
        folder = out/row['name']; folder.mkdir(); Image.fromarray(rgb).save(folder/'reference.png')
        _,faces = detector.detect(cv2.cvtColor(small,cv2.COLOR_RGB2BGR))
        face = max(faces,key=lambda f:f[2]*f[3]) if faces is not None else None
        source_feature = feature(small,face)
        source = row['source_age']; targets = [('same',source)]+[(str(t),t) for t in [20,40,60,75]]
        cached_path = previous/'latent-constrained'/row['name']/'codes.pt'
        assert sha256(cached_path) == protocol['cached_codes_sha256'][row['name']]
        cached = torch.load(cached_path,map_location='cuda',weights_only=True)
        assert cached['source'] == source
        anchor = cached['anchor']; baseline_codes = torch.cat([anchor,cached['target_codes']])
        with torch.no_grad():
            b = net.pretrained_encoder(x)+net.latent_avg
            codes = torch.cat([b+fine(torch.cat([x,torch.full_like(x[:,:1],t/100)],1)) for _,t in targets])
            teacher = F.interpolate(decode(codes[1:]),(256,256),mode='area')
            probabilities = [logits(v).softmax(1) for v in views(teacher)]
            source_identity = identity.feats(x).detach()
        def record(route,images):
            with torch.no_grad():
                for (condition,target),image in zip(targets,images):
                    y = image[None]
                    assert tuple(y.shape) == (1,3,1024,1024) and torch.isfinite(y).all()
                    native = ((y[0].clamp(-1,1)+1)*127.5).permute(1,2,0).byte().cpu().numpy()
                    reduced = cv2.resize(native,(256,256),interpolation=cv2.INTER_AREA)
                    y256 = F.interpolate(y,(256,256),mode='area')
                    f = feature(reduced,face)
                    estimates = [float(logits(v).softmax(1)@age.mid)*100 for v in views(y256)]
                    gray = cv2.cvtColor(cv2.resize(native,(512,512)),cv2.COLOR_RGB2GRAY).astype(np.float32)/255
                    band = np.abs(gray-cv2.GaussianBlur(gray,(0,0),1.0))[220:375,130:382]
                    stem = route+'_'+condition
                    Image.fromarray(native).save(folder/(stem+'.png'))
                    Image.fromarray(reduced).save(folder/(stem+'_preview.png'))
                    with Image.open(folder/(stem+'.png')) as check:check.load();assert check.size==(1024,1024)
                    result = dict(name=row['name'],cohort=row['cohort'],route=route,condition=condition,
                        source_age=source,target=target,sface_cosine=float(source_feature@f) if f is not None else None,
                        lpips_shared=float(perceptual(y256,x).mean()),shared_output_age=estimates[0],
                        age_views=estimates,age_view_spread=max(estimates)-min(estimates),
                        face_high_frequency_energy=float(band.mean()),native_size=[1024,1024],
                        native=row['name']+'/'+stem+'.png',preview=row['name']+'/'+stem+'_preview.png')
                    if route=='baseline_v1':
                        prior = np.asarray(Image.open(previous/'latent-constrained'/row['name']/('latent_constrained180_'+condition+'.png')).convert('RGB'))
                        result['baseline_png_max_difference'] = int(np.abs(native.astype(np.int16)-prior.astype(np.int16)).max())
                        assert result['baseline_png_max_difference'] <= 1, result
                    report['outputs'].append(result)
        with torch.no_grad():record('baseline_v1',decode(baseline_codes))
        for route in ROUTES[1:]:
            half_texture = route in ['texture_half','texture_robust']
            robust = route in ['robust_age','texture_robust']
            direction = codes[1:]-codes[:1]
            scale = torch.ones_like(direction)
            if half_texture:scale[:,8:] = .5
            target = anchor+direction*scale
            with torch.no_grad():prior = F.interpolate(decode(target),(256,256),mode='area')
            delta = torch.zeros_like(target[:,:8]).requires_grad_(True)
            optimizer = torch.optim.Adam([delta],lr=.012);tick = time.monotonic()
            points = []
            for step in range(180):
                optimizer.param_groups[0]['lr'] = .012*(.2+.8*(1-step/180))
                code = target+torch.cat([delta,torch.zeros_like(target[:,8:])],1)
                y = F.interpolate(decode(code),(256,256),mode='area')
                ident = (1-(identity.feats(y)*source_identity).sum(1)).mean()
                transformed = views(y) if robust else [y]
                age_kl = sum(F.kl_div(logits(v).log_softmax(1),q,reduction='batchmean')
                             for v,q in zip(transformed,probabilities))/len(transformed)
                color = F.l1_loss(F.avg_pool2d(y,32),F.avg_pool2d(x,32).expand(4,-1,-1,-1))
                loss = .8*ident+.5*age_kl+.6*perceptual(y,prior).mean()+.05*delta.square().mean()+.1*color
                optimizer.zero_grad(set_to_none=True);loss.backward()
                if not torch.isfinite(loss) or not torch.isfinite(delta.grad).all():raise RuntimeError('Nonfinite latent gradient')
                optimizer.step()
                if (step+1)%60==0:
                    points.append(dict(step=step+1,loss=float(loss.detach()),identity=float(ident.detach()),age_kl=float(age_kl.detach())))
                    print('AGE_TEXTURE',row['name'],route,points[-1],flush=True)
            torch.cuda.synchronize();elapsed = time.monotonic()-tick
            with torch.no_grad():
                edited = target+torch.cat([delta,torch.zeros_like(target[:,8:])],1)
                record(route,decode(torch.cat([anchor,edited])))
            torch.save(dict(anchor=anchor.cpu(),target_codes=edited.detach().cpu(),source=source),folder/(route+'_codes.pt'))
            report['timings'].append(dict(name=row['name'],route=route,edit_seconds=elapsed,log=points))
        assert all(not q.requires_grad and q.grad is None for model in [net,fine,identity,perceptual,age] for q in model.parameters())
        write_json(out/'report.partial.json',report)
        print('AGE_TEXTURE_SAMPLE_DONE',row['name'],flush=True)
    report['total_seconds'] = time.monotonic()-started
    write_json(out/'report.json',report)
    (out/'DONE').write_text('Paired age/texture ablation completed; weights frozen\n')


if __name__=='__main__':main()

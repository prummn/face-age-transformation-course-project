"""Frozen paired regression/Asian test/diagnostic evaluation. No promotion."""
import argparse,json,os,sys,time
from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import lpips
from probe_inversion import sha256,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--routes',nargs='+',default=['mixed','fine_direct','encoder','encoder_generator'])
    a=p.parse_args();run=a.run
    if len(set(a.routes))!=len(a.routes) or any(not name.replace('_','').isalnum() or name.startswith('.') for name in a.routes):
        raise ValueError('Routes must be unique simple directory names')
    out=run/'evaluation';out.mkdir(exist_ok=False)
    root=Path(os.environ['FACE_AGE_REPO']);base=Path(os.environ['FACE_AGE_BASE']);sys.path.insert(0,str(base/'src'))
    from sam_npu import load_sam
    from sam_losses import AgeEst
    torch.set_num_threads(4);cv2.setNumThreads(4);torch.manual_seed(20261113)
    net=load_sam('cuda:0').eval().requires_grad_(False)
    fine=deepcopy(net.encoder).eval().requires_grad_(False)
    fine_initial=torch.load(base/'out/ft_l1/encoder_step1000.pt',map_location='cpu',weights_only=True)
    decoder_initial={n:p.detach().clone() for n,p in net.decoder.named_parameters() if '.modulation.' in n}
    age_net=AgeEst().cuda().eval().requires_grad_(False)
    perceptual=lpips.LPIPS(net='alex',verbose=False).cuda().eval().requires_grad_(False)
    detector=cv2.FaceDetectorYN.create(str(base/'models/yunet/face_detection_yunet_2023mar.onnx'),'',(256,256),.6,.3,100)
    recognizer=cv2.FaceRecognizerSF.create(str(root/'deployment/gan-research/face_recognition_sface_2021dec.onnx'),'')
    records=[]
    for split in ['eval','external']:
        records += [json.loads(x) for x in (run/'manifests'/(split+'.jsonl')).read_text().splitlines()]
    # Native outputs for all12 APPA, user,3 author examples, and10 stratified FFHQ; all other outputs have256 previews and metrics.
    chosen=set();counts={}
    for r in records:
        if r['cohort']=='appa_asian_test' or r['name'] in ['uploaded','cusp_00000','cusp_00001','cusp_00002']:chosen.add(r['name'])
        if r['cohort']=='ffhq_regression':
            key=tuple(r['age_interval']);counts[key]=counts.get(key,0)+1
            if counts[key]<=2:chosen.add(r['name'])
    def tensor(rgb):return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1)[None].contiguous().cuda().float()/127.5-1
    def feature(rgb,face):
        if face is None:return None
        v=recognizer.feature(recognizer.alignCrop(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),face)).ravel()
        return v/(np.linalg.norm(v)+1e-12)
    report=dict(records=records,outputs=[],native_samples=sorted(chosen),
        external_alignment_failures=json.loads((run/'data-audit.json').read_text())['external_alignment_failures'],
        color_metric='Norm of mean central-face OpenCV8bit Lab ab difference; not DeltaE00',routes=a.routes,
        caveats=['FFHQ is a base-generator training source: regression only, not independent pretraining generalization.',
                 'APPA Asian annotation is broad, not verified East Asian; cases and ages are publisher labels.',
                 'SFace only evaluated here; conservative dedup used it, but it never enters training loss.',
                 'LPIPS and FairFace are shared training networks; their scores are diagnostic, not independent evidence.',
                 'Cross-age source pixel/LPIPS errors confound intended aging: reconstruction conclusions use same condition only.',
                 'No paired future appearance truth, verified subject split, blind population test or automatic model promotion.',
                 'model_seconds times target encoder + native decoder only, after source pSp/anchor setup; excludes metrics/PNG/I/O.'])
    tick=time.monotonic()
    for route in report['routes']:
        fine.load_state_dict(fine_initial if route in ['mixed','fine_direct'] else torch.load(run/route/'encoder.pt',map_location='cpu',weights_only=True))
        with torch.no_grad():
            for name,param in net.decoder.named_parameters():
                if name in decoder_initial:param.copy_(decoder_initial[name])
            if route=='encoder_generator':
                state=torch.load(run/route/'generator_affines.pt',map_location='cpu',weights_only=True)
                named=dict(net.decoder.named_parameters())
                for name,value in state.items():named[name].copy_(value)
        for index,r in enumerate(records):
            reference=r.get('reference_path',r['image_path'])
            assert sha256(reference)==r.get('reference_sha256',r['sha256'])
            rgb=np.asarray(Image.open(reference).convert('RGB'));small=cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA)
            x=tensor(small);source=r['source_age'];targets=[('same',source)]+[(str(t),t) for t in [20,40,60,75]]
            folder=out/r['name'];folder.mkdir(exist_ok=True)
            if not (folder/'reference.png').exists():Image.fromarray(rgb).save(folder/'reference.png')
            _,faces=detector.detect(cv2.cvtColor(small,cv2.COLOR_RGB2BGR))
            face=max(faces,key=lambda f:f[2]*f[3]) if faces is not None else None
            source_feature=feature(small,face)
            lab_source=cv2.cvtColor(small,cv2.COLOR_RGB2LAB).astype(float)
            with torch.no_grad():
                b=net.pretrained_encoder(x)+net.latent_avg
                def cond(age):return torch.cat([x,torch.full_like(x[:,:1],age/100)],1)
                if route=='mixed':
                    anchor=b+fine(cond(source));zero=net.encoder(cond(source))
                for label,target in targets:
                    torch.cuda.synchronize();t=time.monotonic()
                    code=anchor+net.encoder(cond(target))-zero if route=='mixed' else b+fine(cond(target))
                    y=net(code,input_code=True,input_is_full=True,randomize_noise=False,resize=False)
                    torch.cuda.synchronize();seconds=time.monotonic()-t
                    assert tuple(y.shape)==(1,3,1024,1024) and torch.isfinite(y).all()
                    native=((y[0].clamp(-1,1)+1)*127.5).permute(1,2,0).byte().cpu().numpy()
                    reduced=cv2.resize(native,(256,256),interpolation=cv2.INTER_AREA)
                    y256=F.interpolate(y,(256,256),mode='area')
                    estimate=float(age_net(y256)[0])*100
                    lp=float(perceptual(y256,x).mean())
                    pred_feature=feature(reduced,face)
                    cosine=float(source_feature@pred_feature) if source_feature is not None else None
                    lab=cv2.cvtColor(reduced,cv2.COLOR_RGB2LAB).astype(float)
                    roi=(slice(70,205),slice(60,196))
                    chroma=float(np.linalg.norm((lab-lab_source)[roi][:,:,1:3].mean((0,1))))
                    file=f'{route}_{label}'
                    Image.fromarray(reduced).save(folder/(file+'_preview.png'))
                    native_path=None
                    if r['name'] in chosen:
                        path=folder/(file+'.png');Image.fromarray(native).save(path)
                        with Image.open(path) as check:check.load();assert check.size==(1024,1024) and check.format=='PNG'
                        native_path=str(path.relative_to(out))
                    row=dict(name=r['name'],cohort=r['cohort'],route=route,condition=label,target=target,
                        source_age=source,shared_output_age=estimate,shared_age_error=abs(estimate-target),
                        sface_cosine=cosine,lpips_shared=lp,face_chroma_shift=chroma,
                        rgb_mae=float(np.abs(reduced.astype(float)-small).mean()),
                        model_seconds=seconds,native_size=[1024,1024],native=native_path,
                        preview=str((folder/(file+'_preview.png')).relative_to(out)))
                    report['outputs'].append(row)
            if (index+1)%10==0:print('EVAL',route,index+1,len(records),flush=True)
            write_json(out/'report.partial.json',report)
        print('ROUTE_DONE',route,flush=True)
    report['total_seconds']=time.monotonic()-tick
    write_json(out/'report.json',report);(out/'DONE').write_text('Complete\n')
    print('EVALUATION_DONE',len(report['outputs']),flush=True)
if __name__=='__main__':main()

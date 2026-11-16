"""GAN-only diagnostic baseline and perceptual, identity-aware W+ inversion.

No network weights are trained. Independent SFace evaluation is never used in
the fitting objective. All target ages remain unchanged in the SAM routes.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=120)
    parser.add_argument('--cusp',action='store_true',help='Run separate CUSP worker instead of SAM')
    args=parser.parse_args()
    if not 1<=args.steps<=300:parser.error('steps outside1..300')
    args.output.mkdir(parents=True,exist_ok=False)
    import cv2
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    import torch
    import torch.nn.functional as F
    from probe_inversion import sha256,write_json
    root=Path(os.environ.get('FACE_AGE_REPO',str(Path(__file__).resolve().parents[2])))
    base=Path(os.environ['FACE_AGE_BASE'])
    manifest=json.loads(args.manifest.read_text())
    device='cuda:0';torch.cuda.set_device(device);torch.set_num_threads(4)
    torch.manual_seed(20261112);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    sys.path.insert(0,str(base/'src'))
    from sam_losses import AgeEst
    age_net=AgeEst().to(device).eval().requires_grad_(False)
    detector=cv2.FaceDetectorYN.create(str(base/'models/yunet/face_detection_yunet_2023mar.onnx'),'',(256,256),.6,.3,100)
    recognizer=cv2.FaceRecognizerSF.create(str(root/'deployment/gan-research/face_recognition_sface_2021dec.onnx'),'')
    expected='0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79'
    assert sha256(root/'deployment/gan-research/face_recognition_sface_2021dec.onnx')==expected
    def tensor(a):return torch.from_numpy(np.ascontiguousarray(a)).permute(2,0,1)[None].to(device).float()/127.5-1
    def as_rgb(t):return ((t[0].detach().clamp(-1,1)+1)*127.5).permute(1,2,0).byte().cpu().numpy()
    def down(t,size=256):return F.interpolate(t,(size,size),mode='area')
    def age(t):return age_net(down(t))[0]
    def feature(rgb,landmarks):
        if landmarks is None:return None
        crop=recognizer.alignCrop(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),landmarks)
        v=recognizer.feature(crop).ravel();return v/(np.linalg.norm(v)+1e-12)
    tick=time.monotonic()
    if args.cusp:
        sys.path.insert(0,str(root/'deployment/CUSP'))
        import legacy
        from torch_utils import persistence
        from torch_utils.ops import bias_act,upfirdn2d
        # Use upstream reference implementations; do not compile old CUDA extensions.
        original_bias=bias_act.bias_act;original_filter=upfirdn2d.upfirdn2d
        def ref_bias(*a,**kw):kw['impl']='ref';return original_bias(*a,**kw)
        def ref_filter(*a,**kw):kw['impl']='ref';return original_filter(*a,**kw)
        bias_act.bias_act=ref_bias;upfirdn2d.upfirdn2d=ref_filter
        ckpt=root/'deployment/gan-research/cusp_ffhq_ls_005780.pkl'
        metadata=json.loads((root/'deployment/gan-research/cusp-checkpoint.json').read_text())
        assert sha256(ckpt)==metadata['sha256']
        classifier_states=[]
        @persistence.import_hook
        def inspect_classifier_state(meta):
            if meta.class_name=='DEXAgeClassifier':
                classifier_states.append(dict(meta.state))
                print('CUSP_CLASSIFIER_STATE',list(meta.state),flush=True)
            return meta
        with ckpt.open('rb') as stream:payload=legacy.load_network_pkl(stream)
        net=payload['G_ema'].to(device).eval().requires_grad_(False);del payload
        print('CUSP_INIT',net.init_kwargs,flush=True)
        wrapper=net.skip_grad_blur
        classifier=wrapper.model
        print('CUSP_CLASSIFIER_RESTORED',vars(classifier).keys(),
              'type',type(classifier),'states',len(classifier_states),flush=True)
        print('CUSP_CLASSIFIER_METHODS',[(name,getattr(getattr(classifier,name),'__self__',None) is classifier)
              for name in ['get_classifier','zero_grad','to']],flush=True)
        # The legacy pickle contains an empty classifier instance referenced by
        # its bound methods as well as the full G_ema state. Recover that exact
        # serialized state, not a new classifier or substituted age network.
        repaired_classifier=False
        if not hasattr(classifier,'classifier'):
            full=[state for state in classifier_states if 'classifier' in state]
            if len(full)!=2:raise RuntimeError('Unexpected legacy G/G_ema classifier states')
            classifier.__dict__.update(full[-1])
            repaired_classifier=True
        classifier.to(device)
        classifier.get_classifier().eval().requires_grad_(False)
        # Pickled hook methods may refer to the duplicate empty wrapper too.
        # Install the original hook functions bound to the restored live object.
        target_layer=classifier.get_hook()
        target_layer._forward_hooks.clear();target_layer._backward_hooks.clear()
        wrapper.register_hooks()
        print('CUSP_CLASSIFIER_READY','restored_serialized_state',repaired_classifier,
              'parameters',sum(p.numel() for p in classifier.get_classifier().parameters()),flush=True)
        # Upstream CAM-based skip preservation differentiates the input even in eval.
        # Keep autograd enabled. Stored synthesis noise is fixed for comparisons.
        for module in net.modules():
            if hasattr(module,'noise_strength') and hasattr(module,'noise_const'):
                original=module.forward
                def fixed_noise(*a,_original=original,**kw):kw['noise_mode']='const';return _original(*a,**kw)
                module.forward=fixed_noise
        bins=[(0,2),(3,6),(7,9),(10,14),(15,19),(20,29),(30,39),(40,49),(50,69),(70,120)]
        def generate_cusp(x,target):
            index=next(i for i,(lo,hi) in enumerate(bins) if lo<=target<=hi)
            classes=net.attr_map.z_dim
            if classes!=len(bins):raise RuntimeError(f'Expected FFHQ-LS10 classes, got {classes}')
            condition=torch.zeros(1,classes,device=device);condition[0,index]=1
            # Reset random blur sampling for every age, so target is the variable.
            np.random.seed(20261112);torch.manual_seed(20261112)
            out=net(x,condition,return_msk=False)
            return out,dict(index=index,range=list(bins[index]))
        model_info=dict(kind='CUSP FFHQ-LS',checkpoint=metadata,commit=manifest['cusp_commit'],
                        input_size=256,native_size=256,condition='10 age bins',
                        restored_legacy_classifier_state=repaired_classifier,
                        adaptation='upstream reference bias_act/upfirdn2d; stored noise; fixed blur seed')
    else:
        from sam_npu import load_sam
        from sam_losses import IDLoss
        import lpips
        net=load_sam(device).eval().requires_grad_(False)
        fine=deepcopy(net.encoder)
        fine.load_state_dict(torch.load(base/'out/ft_l1/encoder_step1000.pt',map_location='cpu',weights_only=True))
        fine.eval().requires_grad_(False)
        identity=IDLoss().to(device).eval().requires_grad_(False)
        perceptual=lpips.LPIPS(net='alex',verbose=False).to(device).eval().requires_grad_(False)
        def decode(code):return net(code,input_code=True,input_is_full=True,randomize_noise=False,resize=False)
        model_info=dict(kind='SAM',input_size=256,native_size=1024,condition='numeric age/100',
                        inversion=dict(steps=args.steps,lr=.03,
                        objective='.8 LPIPS256 + .2 face-weighted L1_512 + .15 shared IR-SE50 ID '
                                  '+ 2 source-age MSE_0_1 + .01 W+ trust MSE',
                        note='Network weights frozen; inversion objective changed before this benchmark; not blind tuning'))
    torch.cuda.synchronize()
    report=dict(manifest_sha256=sha256(args.manifest),model=model_info,
                script_sha256=sha256(__file__),setup_seconds=time.monotonic()-tick,
                gpu=torch.cuda.get_device_name(),torch=torch.__version__,weights_trained=False,
                identity_evaluation='Independent SFace cosine; fixed source landmarks for source/output crop; uncalibrated threshold',
                age_evaluation='Shared training FairFace diagnostic only; no true ages',samples=[])
    for item in manifest['samples']:
        assert sha256(item['path'])==item['sha256']
        folder=args.output/item['name'];folder.mkdir()
        ref=np.asarray(Image.open(item['path']).convert('RGB'))
        Image.fromarray(ref).save(folder/'reference.png')
        rgb=cv2.resize(ref,(256,256),interpolation=cv2.INTER_AREA);x=tensor(rgb)
        _,faces=detector.detect(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        landmarks=max(faces,key=lambda f:f[2]*f[3]) if faces is not None and len(faces) else None
        source_feature=feature(rgb,landmarks)
        with torch.no_grad():source_float=float(age(x))*100
        source=round(source_float);targets=list(dict.fromkeys([source]+manifest['targets']))
        sample=dict(name=item['name'],source_condition=source,reference_age=source_float,
                    original_size=item['size'],sface_source_detected=landmarks is not None,outputs=[])
        def save(route,target,result,seconds,**extra):
            native=as_rgb(result);path=folder/f'{route}_a{target:02d}.png';Image.fromarray(native).save(path)
            small=cv2.resize(native,(256,256),interpolation=cv2.INTER_AREA)
            pred_feature=feature(small,landmarks)
            cosine=float(source_feature@pred_feature) if source_feature is not None else None
            with torch.no_grad():estimated=float(age(result))*100
            difference=np.abs(small.astype('float32')-rgb.astype('float32'))
            output=dict(route=route,target=target,path=str(path.relative_to(args.output)),native_size=list(native.shape[:2]),
                        sface_cosine=cosine,shared_age_estimate=estimated,inference_seconds=seconds,
                        same_source_condition=target==source,rgb_mae=float(difference.mean()),
                        face_mae=float(difference[70:205,60:196].mean()),**extra)
            sample['outputs'].append(output)
        if args.cusp:
            for target in targets:
                t=time.monotonic();result,condition=generate_cusp(x,target);torch.cuda.synchronize()
                save('cusp',target,result,time.monotonic()-t,age_bin=condition)
        else:
            with torch.no_grad():
                def conditioned(a):return torch.cat([x,torch.full_like(x[:,:1],a/100)],1)
                e0={a:net.encoder(conditioned(a)) for a in targets}
                eft={a:fine(conditioned(a)) for a in targets}
                b=net.pretrained_encoder(x)+net.latent_avg
                anchor=eft[source]+b;deltas={a:e0[a]-e0[source] for a in targets}
                for target in targets:
                    for route,code in [('original',e0[target]+b),('fine_direct',eft[target]+b),('mixed',anchor+deltas[target])]:
                        t=time.monotonic();result=decode(code);torch.cuda.synchronize()
                        save(route,target,result,time.monotonic()-t)
            w=anchor.detach().clone().requires_grad_(True)
            opt=torch.optim.Adam([w],lr=.03)
            # Only true reference pixels enter reconstruction;256 inputs are not upscaled for512 targets.
            target_size=min(512,ref.shape[0]);target=tensor(cv2.resize(ref,(target_size,target_size),interpolation=cv2.INTER_AREA))
            grid=torch.linspace(0,1,target_size,device=device);yy,xx=torch.meshgrid(grid,grid,indexing='ij')
            weight=.2+.8*((1-((xx-.5)/.31)**2-((yy-.53)/.36)**2)/.25).clamp(0,1)[None,None]
            with torch.no_grad():id_source=identity.feats(x)
            logs=[];t=time.monotonic()
            for step in range(args.steps):
                opt.param_groups[0]['lr']=.03*(.15+.85*(1-step/args.steps))
                opt.zero_grad(set_to_none=True);image=decode(w);y=down(image)
                lp=perceptual(y,x).mean()
                pixel=((down(image,target_size)-target).abs()*weight).sum()/(weight.sum()*3)
                idloss=(1-(identity.feats(y)*id_source).sum(1)).mean()
                age_loss=(age(y)-source_float/100).square().mean()
                loss=.8*lp+.2*pixel+.15*idloss+2*age_loss+.01*(w-anchor).square().mean()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite inversion loss')
                loss.backward()
                if not torch.isfinite(w.grad).all():raise RuntimeError('Nonfinite inversion gradient')
                opt.step()
                if step==0 or (step+1)%40==0:
                    log=dict(step=step+1,loss=float(loss.detach()),lpips=float(lp.detach()),
                             id_loss=float(idloss.detach()),age_mse=float(age_loss.detach()))
                    logs.append(log)
            torch.cuda.synchronize();sample['fit_seconds']=time.monotonic()-t;sample['fit_log']=logs
            assert all(p.grad is None and not p.requires_grad for m in [net,fine,identity,perceptual,age_net] for p in m.parameters())
            with torch.no_grad():
                for target_age in targets:
                    t=time.monotonic();result=decode(w+deltas[target_age]);torch.cuda.synchronize()
                    save('perceptual_inversion',target_age,result,time.monotonic()-t)
            torch.save(dict(w=w.detach().cpu(),source=source),folder/'inversion.pt')
        report['samples'].append(sample);write_json(args.output/'report.partial.json',report)
        print('SAMPLE_DONE',item['name'],'source',source,'outputs',len(sample['outputs']),flush=True)
    report['total_seconds']=time.monotonic()-tick
    write_json(args.output/'report.json',report);(args.output/'DONE').write_text('Complete\n')
    print('DONE',args.output,flush=True)


if __name__=='__main__':main()

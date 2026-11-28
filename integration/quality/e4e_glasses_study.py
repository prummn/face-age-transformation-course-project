"""Study source-channel preservation and fixed eyeglass corrections in e4e edits.

The source method retains four official StyleSpace channels at two fixed levels.
The sweep method compares three W+ offsets and three StyleSpace offsets at two
high aging strengths. Every known development portrait sees every fixed setting,
including an existing-glasses control. No training, per-image search or blending;
unconditional negative offsets are research controls, not an automatic policy.
"""
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from probe_inversion import sha256,write_json
from sfe_study import ROOT,NAMES,REV,validate,submit

PARENT=ROOT/'deployment/training-runs/sfe-20261128-013310'
SCALES=[0.,5.,7.5,10.]
ROUTES=['projected_e4e','source_channels_50','source_channels_100']


def function(source,name):
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    return '\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])+'\n'


def prepare(method='source'):
    upstream=ROOT/'deployment/StyleFeatureEditor'
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=upstream,text=True).strip()!=REV:
        raise ValueError('Wrong upstream revision')
    if subprocess.check_output(['git','status','--porcelain'],cwd=upstream,text=True):
        raise ValueError('Upstream checkout must remain clean')
    run=ROOT/'deployment/training-runs'/('e4e-glasses-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
    (run/'source').mkdir(parents=True,exist_ok=False)
    for name in ['e4e_glasses_study.py','sfe_study.py','probe_inversion.py']:
        shutil.copy2(Path(__file__).with_name(name),run/'source'/name)
    shutil.copy2(ROOT/'integration/source_color.py',run/'source/source_color.py')
    shutil.copytree(PARENT/'upstream',run/'upstream',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(upstream/'utils/model_utils.py',run/'upstream/official_model_utils.txt')
    shutil.copy2(upstream/'editings/latent_editor.py',run/'upstream/official_latent_editor.txt')
    (run/'upstream/stylespace.py').write_text(function((upstream/'utils/model_utils.py').read_text(),'get_stylespace_from_w'))
    latent=ast.parse((upstream/'editings/latent_editor.py').read_text())
    mapping=ast.literal_eval(next(n.value for n in latent.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='STYLESPACE_IDX' for t in n.targets)))
    human=next(n for n in ast.walk(latent) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='stylespace_directions' for t in n.targets))
    glasses=ast.literal_eval(human.value)['glasses']
    channels=sorted(set((mapping[layer],channel) for layer,channel in glasses))
    if channels!=[(1,97),(1,175),(2,120),(2,288)]:raise ValueError('Official glasses channels changed')
    for name in ['selection.json','model-specs.json']:shutil.copy2(PARENT/name,run/name)
    records=json.loads((run/'selection.json').read_text())
    if [r['name'] for r in records]!=NAMES:raise ValueError('Wrong fixed development images')
    frozen={str(p):sha256(p) for folder in ['source','upstream'] for p in (run/folder).rglob('*') if p.is_file()}
    for p in [run/'selection.json',run/'model-specs.json',PARENT/'evaluation/report.json',PARENT/'independent-age.json']:
        frozen[str(p)]=sha256(p)
    parent=json.loads((PARENT/'evaluation/report.json').read_text())
    for r in records:
        p=PARENT/'evaluation'/r['name']/'reference.png';frozen[str(p)]=sha256(p)
    for o in parent['outputs']:
        if o['route']=='e4e_age' and o['scale'] in SCALES:
            frozen[str(PARENT/'evaluation'/o['native'])]=o['sha256']
    write_json(run/'experiment.json',dict(profile='e4e_source_eyeglass_channels',parent_run=str(PARENT),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        upstream_commit=REV,device='cuda:0',gpu=1,training=False,cases=8,scales=SCALES,routes=ROUTES,
        channels=channels,frozen_inputs=frozen,
        recipe='Use the previous norm-preserved eyeglass-orthogonal W+ age direction. After W+ to StyleSpace affine conversion, replace only four official eyeglass channels by 50%/100% of their source e4e values; all other convolution and ToRGB styles stay edited.',
        controls='Fixed8 inspected development portraits; ffhq_65817 is a visually verified existing-glasses preservation control. No per-person selection or image-space patching.',
        criteria=dict(mean_identity_drop_max=.02,mean_age_change_abs_max=2,age_span_ratio_min=.9,missing_increase_max=0,
            visual_glasses_required=True,existing_glasses_preserved=True,no_automatic_promotion=True),
        limitations=['Channels are published semantic handles, not a guarantee of pure eyeglass control.',
            'Retaining source channels may reduce valid aging or alter eyes; raw native inspection required.',
            '8 previously inspected development images, only1 existing-glasses control, not independent population evaluation.',
            'No target numerical age calibration, unseen41 reserved portraits untouched.']))
    if method=='sweep':
        exp=json.loads((run/'experiment.json').read_text())
        exp.update(profile='e4e_eyeglass_correction_sweep',scales=[7.5,10.],method=method,
            routes=['projected_e4e','boundary_minus4','boundary_minus8','boundary_minus12','style_minus1','style_minus2','style_minus3'],
            corrections=[dict(route='boundary_minus'+str(s),kind='wplus_boundary',amount=-s) for s in [4,8,12]]+
                [dict(route='style_minus'+str(s),kind='stylespace_channels',amount=-3*s) for s in [1,2,3]],
            recipe='Fixed high-age7.5/10 scan. Compare the norm-preserved eyeglass-orthogonal e4e baseline with negative W+ eyeglass-boundary offsets, or negative shifts of four official glasses StyleSpace channels. Every portrait sees every fixed correction. No automated route, no per-image tuning, no pixel blending.',
            controls='The existing-glasses source ffhq_65817 is an explicit side-effect control; these unconditional corrections may remove its original glasses and must not be promoted on that basis.',
            limitations=exp['limitations']+['Both signs/effect size come from published edit handles; negative direction is a hypothesis to verify visually, not a supervised detector.'])
        write_json(run/'experiment.json',exp)
    (run/'source/run_sfe.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
repo="$1"
run="$2"
cd "$repo"
exec >> "$run/worker.log" 2>&1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export FACE_AGE_REPO="$repo" FACE_AGE_EXPERIMENT_ALLOCATION=1 HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -m venv --system-site-packages /tmp/face-age-e4e-glasses
py=/tmp/face-age-e4e-glasses/bin/python
"$py" -m pip install --no-index --no-deps "$repo"/deployment/wheels/opencv_python_headless-4.11.0.86-*.whl
"$py" "$run/source/e4e_glasses_study.py" worker "$run"
''')
    exp=json.loads((run/'experiment.json').read_text());exp['frozen_inputs'][str(run/'source/run_sfe.sh')]=sha256(run/'source/run_sfe.sh')
    write_json(run/'experiment.json',exp)
    print('PREPARED',run,flush=True)


def worker(run):
    exp,weights=validate(run)
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from torch.nn import functional as F
    from source_color import preserve_source_chroma
    if os.environ.get('FACE_AGE_EXPERIMENT_ALLOCATION')!='1' or torch.cuda.device_count()!=1:
        raise ValueError('One allocated GPU required')
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(20261127)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    sys.path.insert(0,str(run/'upstream'))
    from inference_only import InferenceOnly
    from stylespace import get_stylespace_from_w
    tick=time.monotonic()
    model=InferenceOnly(weights['sfe_editor_light.pt'],weights['e4e_ffhq_encode.pt'],weights['stylegan2-ffhq-config-f.pt'])
    age=torch.load(run/'upstream/age.pt',map_location='cuda',weights_only=False).reshape(1,1,512).expand(1,18,512).clone()
    glasses=torch.from_numpy(np.load(run/'upstream/eyeglasses.npy',allow_pickle=False)).reshape(1,18,512).cuda()
    direction=age-(age*glasses).sum()/glasses.square().sum()*glasses
    direction*=age.norm()/direction.norm()
    def fingerprint():
        h=hashlib.sha256()
        for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        h.update(model.latent_avg.cpu().numpy().tobytes());h.update(direction.cpu().numpy().tobytes())
        return h.hexdigest()
    frozen=fingerprint()
    def pixels(x):
        if tuple(x.shape)!=(1,3,1024,1024) or not torch.isfinite(x).all():raise ValueError('Invalid native output')
        return ((x[0].clamp(-1,1)+1)*127.5).permute(1,2,0).byte().cpu().numpy()
    @torch.inference_mode()
    def invert(path):
        rgb=np.array(Image.open(path).convert('RGB'),copy=True)
        if rgb.shape!=(1024,1024,3):raise ValueError('Expected1024 reference')
        x=torch.from_numpy(rgb).permute(2,0,1)[None].cuda().float()/127.5-1
        x=F.interpolate(x,(256,256),mode='bilinear',align_corners=False)
        return model.e4e_encoder(x)+model.latent_avg
    @torch.inference_mode()
    def infer(w,source,scale,alpha):
        edited=w+scale/2*direction
        styles,rgb=get_stylespace_from_w(edited,model.decoder)
        for layer,channel in exp['channels']:
            styles[layer][:,channel]=styles[layer][:,channel]*(1-alpha)+source[0][layer][:,channel]*alpha
        y,_=model.decoder((styles,rgb),is_stylespace=True,input_is_latent=True,randomize_noise=False)
        return pixels(y)
    @torch.inference_mode()
    def correction(w,scale,spec):
        edited=w+scale/2*direction
        if spec['kind']=='wplus_boundary':
            y,_=model.decoder([edited+spec['amount']*glasses],input_is_latent=True,randomize_noise=False)
        else:
            styles,rgb=get_stylespace_from_w(edited,model.decoder)
            for layer,channel in exp['channels']:styles[layer][:,channel]+=spec['amount']
            y,_=model.decoder((styles,rgb),is_stylespace=True,input_is_latent=True,randomize_noise=False)
        return pixels(y)
    det=cv2.FaceDetectorYN.create(str(ROOT/'deployment/model-files/face-age/models/yunet/face_detection_yunet_2023mar.onnx'),'',(256,256),.6,.3,100)
    rec=cv2.FaceRecognizerSF.create(str(ROOT/'deployment/gan-research/face_recognition_sface_2021dec.onnx'),'')
    def feature(rgb):
        bgr=cv2.cvtColor(cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2BGR);_,faces=det.detect(bgr)
        if faces is None:return None
        f=rec.feature(rec.alignCrop(bgr,max(faces,key=lambda x:x[2]*x[3]))).ravel();return f/(np.linalg.norm(f)+1e-12)
    parent=Path(exp['parent_run']);prior=json.loads((parent/'evaluation/report.json').read_text())
    lookup={(o['name'],o['scale']):o for o in prior['outputs'] if o['route']=='e4e_age'}
    out=run/'evaluation';out.mkdir(exist_ok=False)
    records=json.loads((run/'selection.json').read_text())
    report=dict(records=records,routes=exp['routes'],scales=exp['scales'],outputs=[],timings=[])
    parity=[];repeat=[]
    for record in records:
        name=record['name'];folder=out/name;folder.mkdir()
        reference=parent/'evaluation'/name/'reference.png';shutil.copy2(reference,folder/'reference.png')
        source=np.array(Image.open(reference).convert('RGB'));source_id=feature(source)
        w=invert(reference)
        with torch.inference_mode():ss=get_stylespace_from_w(w,model.decoder)
        for scale in exp['scales']:
            condition='scale_'+format(scale,'g');base=lookup[name,scale]
            baseline_path=parent/'evaluation'/base['native'];baseline=np.array(Image.open(baseline_path).convert('RGB'))
            # Verify affine conversion is mathematically the same unprotected decoder path.
            direct=infer(w,ss,scale,0.)
            difference=np.abs(direct.astype(int)-baseline.astype(int))
            parity.append(dict(name=name,scale=scale,max_pixel_error=int(difference.max()),different_pixels=int(np.count_nonzero(difference))))
            if difference.max()!=0:raise ValueError('StyleSpace conversion baseline PNG differs')
            interventions=([dict(route='projected_e4e',kind='baseline',amount=0)]+exp['corrections']) if exp.get('method')=='sweep' else [dict(route=r,kind='source',amount=a) for r,a in zip(ROUTES,[0.,.5,1.])]
            for spec in interventions:
                route=spec['route'];alpha=spec['amount']
                torch.cuda.synchronize();start=time.monotonic()
                y=baseline if alpha==0 else (correction(w,scale,spec) if exp.get('method')=='sweep' else infer(w,ss,scale,alpha))
                torch.cuda.synchronize();seconds=time.monotonic()-start
                if scale==0 and not np.array_equal(y,baseline):raise ValueError('Same-age channel preservation changed output')
                if name==records[0]['name'] and scale==10 and (alpha==1 or (exp.get('method')=='sweep' and route==exp['routes'][-1])):
                    repeated=correction(w,scale,spec) if exp.get('method')=='sweep' else infer(w,ss,scale,alpha)
                    identical=np.array_equal(y,repeated);repeat.append(dict(name=name,scale=scale,alpha=alpha,pixels_equal=identical))
                    if not identical:raise ValueError('Non-repeatable channel preservation')
                path=folder/(route+'_'+condition+'.png')
                if alpha==0:shutil.copy2(baseline_path,path)
                else:Image.fromarray(y).save(path)
                c,_=preserve_source_chroma(source,y);color=path.with_name(path.stem+'_color.png');Image.fromarray(c).save(color)
                f=feature(y)
                report['outputs'].append(dict(name=name,cohort=record['cohort'],route=route,condition=condition,scale=scale,alpha=alpha,
                    native=str(path.relative_to(out)),preserved=str(color.relative_to(out)),sha256=sha256(path),
                    source_cosine=float(source_id@f) if source_id is not None and f is not None else None,detected=f is not None,
                    reference_glasses=name=='ffhq_65817',baseline_sha256=base['sha256'],model_inference_seconds=seconds if alpha else None))
        write_json(out/'report.partial.json',report);print('CASE',name,len(report['outputs']),flush=True)
    after=fingerprint()
    if frozen!=after:raise ValueError('Frozen weights changed')
    write_json(run/'frozen-verification.json',dict(before=frozen,after=after,equal=True))
    write_json(run/'parity.json',dict(checks=parity,exact=True))
    write_json(run/'repeatability.json',dict(checks=repeat,equal=True))
    write_json(run/'device.json',dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
        device_count=1,peak_bytes=torch.cuda.max_memory_allocated()))
    report['total_seconds']=time.monotonic()-tick;write_json(out/'report.json',report)
    (run/'DONE').write_text('success\n');print('GLASSES_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','submit','worker']);p.add_argument('run',type=Path,nargs='?');p.add_argument('--method',choices=['source','sweep'],default='source');a=p.parse_args()
    if a.action=='prepare':prepare(a.method)
    elif a.run is None:p.error('run required')
    else:globals()[a.action](a.run.resolve())

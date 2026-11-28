"""Fixed -12 eyeglass control across e4e age trajectories, 29 known dev faces.

Compare the projected baseline, smooth age-only -12 correction, and the same
correction with a predeclared +1 aging-strength compensation. No fitting or
model training. Source-eyeglass labels are manual diagnostic groups only and
never choose a route or alter inference. Keep the unseen41 reserve untouched.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from probe_inversion import sha256, write_json
from sfe_study import ROOT, REV, NAMES, submit, validate

RUNTIME = ROOT/'deployment/training-runs/e4e-glasses-20261128-074728'
BASELINE = ROOT/'deployment/training-runs/sfe-20261128-013310'
SELECTION = ROOT/'deployment/training-runs/original-delta-20261124-160055/selection.json'
SCALES = [-10., -5., 0., 2.5, 5., 7.5, 10.]
ROUTES = ['projected_e4e', 'smooth_minus12', 'smooth_minus12_boost1']
GLASSES = ['ffhq_69847', 'ffhq_61680', 'ffhq_61690', 'ffhq_63721',
           'ffhq_63046', 'ffhq_69817', 'ffhq_60058',
           'ffhq_66577', 'ffhq_65817', 'ffhq_65714']


def schedule(scale, route):
    if route not in ROUTES:
        raise ValueError('Unknown route')
    t = min(1., max(0., scale / 7.5))
    gate = t * t * (3 - 2 * t)
    offset = 0. if route == 'projected_e4e' else -12 * gate
    strength = scale + (gate if route == 'smooth_minus12_boost1' else 0.)
    return strength, offset


def prepare(cpu=False):
    upstream = ROOT/'deployment/StyleFeatureEditor'
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=upstream, text=True).strip() != REV:
        raise ValueError('Wrong upstream revision')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=upstream, text=True):
        raise ValueError('Keep upstream clean')
    records = [r for r in json.loads(SELECTION.read_text()) if r['name'].startswith('ffhq_')]
    if len(records) != 29 or len({r['name'] for r in records}) != 29:
        raise ValueError('Expected 29 previously used FFHQ development portraits')
    if not set(GLASSES).issubset({r['name'] for r in records}):
        raise ValueError('Invalid manual diagnostic labels')
    run = ROOT/'deployment/training-runs'/(('e4e-schedule-cpu-' if cpu else 'e4e-schedule-')+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
    (run/'source').mkdir(parents=True, exist_ok=False)
    for f in ['e4e_no_glasses_schedule.py', 'sfe_study.py', 'probe_inversion.py']:
        shutil.copy2(Path(__file__).with_name(f), run/'source'/f)
    shutil.copy2(ROOT/'integration/source_color.py', run/'source/source_color.py')
    shutil.copytree(RUNTIME/'upstream', run/'upstream', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(RUNTIME/'model-specs.json', run/'model-specs.json')
    for r in records:
        if sha256(r['reference_path']) != r['reference_sha256']:
            raise ValueError('Reference changed')
        r['reference_glasses'] = r['name'] in GLASSES
        r['glasses_label_kind'] = 'Manual inspected source crop; diagnostic grouping, not automatic inference input'
    write_json(run/'selection.json', records)
    (run/'source/run_sfe.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
repo="$1"
run="$2"
cd "$repo"
exec >> "$run/worker.log" 2>&1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export FACE_AGE_REPO="$repo" FACE_AGE_EXPERIMENT_ALLOCATION=1 HF_HUB_OFFLINE=1
if [ "${FACE_AGE_EXPERIMENT_DEVICE:-cuda:0}" != "cpu" ]; then
  nvidia-smi --query-gpu=name,memory.total --format=csv
fi
python3 -m venv --system-site-packages /tmp/face-age-e4e-schedule
py=/tmp/face-age-e4e-schedule/bin/python
"$py" -m pip install --no-index --no-deps "$repo"/deployment/wheels/opencv_python_headless-4.11.0.86-*.whl
"$py" "$run/source/e4e_no_glasses_schedule.py" worker "$run"
''')
    frozen = {str(p): sha256(p) for folder in ['source', 'upstream'] for p in (run/folder).rglob('*') if p.is_file()}
    for p in [SELECTION, run/'selection.json', run/'model-specs.json', BASELINE/'evaluation/report.json', RUNTIME/'evaluation/report.json']:
        frozen[str(p)] = sha256(p)
    for r in records:
        frozen[r['reference_path']] = r['reference_sha256']
    prior = json.loads((BASELINE/'evaluation/report.json').read_text())
    for o in prior['outputs']:
        if o['route'] == 'e4e_age':
            frozen[str(BASELINE/'evaluation'/o['native'])] = o['sha256']
    prev = json.loads((RUNTIME/'evaluation/report.json').read_text())
    for o in prev['outputs']:
        if o['route'] == 'boundary_minus12':
            frozen[str(RUNTIME/'evaluation'/o['native'])] = o['sha256']
    write_json(run/'experiment.json', dict(profile='e4e_no_glasses_schedule', training=False, cases=29,
        gpu=0 if cpu else 1, device='cpu' if cpu else 'cuda:0', source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        upstream_commit=REV, baseline_run=str(BASELINE), previous_minus12_run=str(RUNTIME),
        scales=SCALES, routes=ROUTES, frozen_inputs=frozen,
        schedule=dict(gate='t=clip(strength/7.5,0,1); gate=t*t*(3-2*t)', eyeglasses_offset='-12*gate',
            boost='optional +1*gate applied to aging strength before decoder', negative_and_zero='exactly unchanged',
            fixed_recipe=True, per_person_selection=False),
        expected_outputs=29*len(SCALES)*len(ROUTES), manual_source_glasses=GLASSES,
        criteria=dict(identity_drop_limit=.02, apparent_age_mean_absolute_change_limit=2,
            keep_existing_glasses=True, no_added_glasses=True, zero_and_younger_exact=True, no_automatic_promotion=True),
        limitations=['Known development set, no independent blind evaluation; possible FFHQ pretraining overlap.',
            'Manual source-eyeglass labels are diagnostic only, not automatic detection or route selection.',
            'Strengths are not target ages. Boost1 at10 uses internal11 and is an explicit extrapolation check.',
            'Eyeglass suppression can alter eyes, identity, apparent age and original glasses.',
            'Unseen41 reserve untouched; no new data or training.']))
    print('PREPARED', run, flush=True)


def worker(run):
    exp, weights = validate(run)
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from torch.nn import functional as F
    from source_color import preserve_source_chroma
    device=exp['device'];cpu=device=='cpu'
    if os.environ.get('FACE_AGE_EXPERIMENT_ALLOCATION') != '1' or torch.cuda.device_count() != (0 if cpu else 1):
        raise ValueError('Device allocation differs from fixed protocol')
    def sync():
        if not cpu:torch.cuda.synchronize()
    torch.set_num_threads(8 if cpu else 4); cv2.setNumThreads(2); torch.manual_seed(20261127)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False
    sys.path.insert(0, str(run/'upstream'))
    from inference_only import InferenceOnly
    tick=time.monotonic()
    model=InferenceOnly(weights['sfe_editor_light.pt'],weights['e4e_ffhq_encode.pt'],weights['stylegan2-ffhq-config-f.pt'],device=device)
    age=torch.load(run/'upstream/age.pt',map_location=device,weights_only=False).reshape(1,1,512).expand(1,18,512).clone()
    glasses=torch.from_numpy(np.load(run/'upstream/eyeglasses.npy',allow_pickle=False)).reshape(1,18,512).to(device)
    direction=age-(age*glasses).sum()/glasses.square().sum()*glasses
    direction*=age.norm()/direction.norm()
    def fingerprint():
        h=hashlib.sha256()
        for k,v in model.state_dict().items():
            h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
        h.update(model.latent_avg.cpu().numpy().tobytes()); h.update(direction.cpu().numpy().tobytes())
        return h.hexdigest()
    before=fingerprint()
    @torch.inference_mode()
    def invert(rgb):
        x=torch.from_numpy(rgb.copy()).permute(2,0,1)[None].to(device).float()/127.5-1
        return model.e4e_encoder(F.interpolate(x,(256,256),mode='bilinear',align_corners=False))+model.latent_avg
    @torch.inference_mode()
    def infer(w,strength,offset):
        y,_=model.decoder([w+strength/2*direction+offset*glasses],input_is_latent=True,randomize_noise=False)
        if tuple(y.shape)!=(1,3,1024,1024) or not torch.isfinite(y).all():
            raise ValueError('Invalid native image')
        return ((y[0].clamp(-1,1)+1)*127.5).permute(1,2,0).byte().cpu().numpy()
    det=cv2.FaceDetectorYN.create(str(ROOT/'deployment/model-files/face-age/models/yunet/face_detection_yunet_2023mar.onnx'),'',(256,256),.6,.3,100)
    rec=cv2.FaceRecognizerSF.create(str(ROOT/'deployment/gan-research/face_recognition_sface_2021dec.onnx'),'')
    def feature(rgb):
        bgr=cv2.cvtColor(cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2BGR); _,faces=det.detect(bgr)
        if faces is None:return None
        f=rec.feature(rec.alignCrop(bgr,max(faces,key=lambda x:x[2]*x[3]))).ravel()
        return f/(np.linalg.norm(f)+1e-12)
    baseline=Path(exp['baseline_run']); previous=Path(exp['previous_minus12_run'])
    old={(o['name'],o['scale']):o for o in json.loads((baseline/'evaluation/report.json').read_text())['outputs'] if o['route']=='e4e_age'}
    old12={(o['name'],o['scale']):o for o in json.loads((previous/'evaluation/report.json').read_text())['outputs'] if o['route']=='boundary_minus12'}
    records=json.loads((run/'selection.json').read_text()); out=run/'evaluation';out.mkdir(exist_ok=False)
    report=dict(records=records,routes=ROUTES,scales=SCALES,outputs=[],timings=[])
    parity=[];protected=[];repeats=[]
    for record in records:
        name=record['name'];folder=out/name;folder.mkdir()
        shutil.copy2(record['reference_path'],folder/'reference.png')
        source=np.array(Image.open(folder/'reference.png').convert('RGB'),copy=True)
        if source.shape!=(1024,1024,3):raise ValueError('Expected native reference')
        source_id=feature(source)
        sync();start=time.monotonic();w=invert(source);sync()
        report['timings'].append(dict(name=name,inversion_seconds=time.monotonic()-start))
        for scale in SCALES:
            cache={}
            for route in ROUTES:
                strength,offset=schedule(scale,route);key=(strength,offset)
                path=folder/(route+'_scale_'+format(scale,'g')+'.png');color=path.with_name(path.stem+'_color.png')
                cached=key in cache;seconds=None
                if cached:
                    raw_path,color_path,y,identity,detected=cache[key]
                    os.link(raw_path,path);os.link(color_path,color)
                else:
                    sync();start=time.monotonic();y=infer(w,strength,offset);sync()
                    seconds=time.monotonic()-start
                    Image.fromarray(y).save(path)
                    c,_=preserve_source_chroma(source,y);Image.fromarray(c).save(color)
                    f=feature(y);identity=float(source_id@f) if source_id is not None and f is not None else None
                    detected=f is not None;cache[key]=(path,color,y,identity,detected)
                digest=sha256(path)
                if route=='projected_e4e' and (name,scale) in old:
                    exact=digest==old[name,scale]['sha256']
                    diff=np.abs(y.astype(int)-np.array(Image.open(baseline/'evaluation'/old[name,scale]['native'])).astype(int))
                    parity.append(dict(name=name,scale=scale,kind='parent_baseline',exact=exact,max_pixel_error=int(diff.max()),mean_pixel_error=float(diff.mean())))
                    if not exact and (not cpu or diff.max()>8 or diff.mean()>.1):raise ValueError('Projected baseline exceeds declared device tolerance')
                if route=='smooth_minus12' and (name,scale) in old12:
                    exact=digest==old12[name,scale]['sha256']
                    diff=np.abs(y.astype(int)-np.array(Image.open(previous/'evaluation'/old12[name,scale]['native'])).astype(int))
                    parity.append(dict(name=name,scale=scale,kind='prior_minus12',exact=exact,max_pixel_error=int(diff.max()),mean_pixel_error=float(diff.mean())))
                    if not exact and (not cpu or diff.max()>8 or diff.mean()>.1):raise ValueError('High aging -12 exceeds declared device tolerance')
                if scale<=0 and route!='projected_e4e':
                    exact=digest==sha256(folder/('projected_e4e_scale_'+format(scale,'g')+'.png'))
                    protected.append(dict(name=name,scale=scale,route=route,exact=exact))
                    if not exact:raise ValueError('Protected zero/younger output changed')
                if name==records[0]['name'] and scale==10 and route!='projected_e4e':
                    equal=np.array_equal(y,infer(w,strength,offset));repeats.append(dict(route=route,pixels_equal=equal))
                    if not equal:raise ValueError('Repeat failed')
                report['outputs'].append(dict(name=name,cohort=record['cohort'],route=route,condition='scale_'+format(scale,'g'),
                    scale=scale,effective_strength=strength,glasses_offset=offset,reference_glasses=record['reference_glasses'],
                    native=str(path.relative_to(out)),preserved=str(color.relative_to(out)),sha256=digest,source_cosine=identity,
                    detected=detected,model_inference_seconds=seconds,exact_duplicate=cached))
        write_json(out/'report.partial.json',report);print('CASE',name,len(report['outputs']),flush=True)
    after=fingerprint()
    if before!=after:raise ValueError('Frozen model changed')
    write_json(run/'frozen-verification.json',dict(before=before,after=after,equal=True))
    write_json(run/'parity.json',dict(checks=parity,exact=all(v['exact'] for v in parity),
        cross_device=cpu,note='CPU/GPU image errors reported, not asserted bitwise equal.' if cpu else 'Same-device byte-identical baseline checks.'))
    write_json(run/'protected.json',dict(checks=protected,exact=True))
    write_json(run/'repeatability.json',dict(checks=repeats,equal=True))
    write_json(run/'device.json',dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,gpu=None if cpu else torch.cuda.get_device_name(0),
        device_count=0 if cpu else 1,peak_bytes=None if cpu else torch.cuda.max_memory_allocated()))
    report['total_seconds']=time.monotonic()-tick;write_json(out/'report.json',report)
    (run/'DONE').write_text('success\n');print('SCHEDULE_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','submit','worker']);p.add_argument('run',type=Path,nargs='?');p.add_argument('--cpu',action='store_true');a=p.parse_args()
    if a.action=='prepare':prepare(a.cpu)
    elif a.run is None:p.error('run required')
    else:globals()[a.action](a.run.resolve())

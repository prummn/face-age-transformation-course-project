"""Bounded official StyleGANEX age-edit comparison; never a serving backend."""
import argparse
from argparse import Namespace
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

ROOT = Path(os.environ.get('FACE_AGE_REPO', Path(__file__).resolve().parents[2]))
PARENT = ROOT/'deployment/training-runs/original-delta-20261124-160055'
NAMES = ['ffhq_64376', 'ffhq_61908', 'ffhq_62644', 'ffhq_64919',
         'ffhq_65817', 'ffhq_67140', 'ffhq_60764', 'ffhq_68335']
SCALES = [-1., -.5, 0., .5, 1., 1.5, 2.]


def prepare():
    spec = json.loads((ROOT/'integration/models/styleganex-20261127.json').read_text())
    upstream = ROOT/'deployment/StyleGANEX'
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=upstream, text=True).strip()
    if commit != spec['code_commit']: raise ValueError('Unexpected upstream revision')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=upstream, text=True):
        raise ValueError('Preserve clean upstream checkout')
    run = ROOT/'deployment/training-runs'/('styleganex-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
    (run/'source').mkdir(parents=True, exist_ok=False)
    for filename in ['styleganex_study.py', 'run_styleganex.sh', 'probe_inversion.py']:
        shutil.copy2(Path(__file__).with_name(filename), run/'source'/filename)
    shutil.copy2(ROOT/'integration/source_color.py', run/'source/source_color.py')
    # Copy only the model package, config and license. Omit plotting imports
    # unused by inference; keep all model arithmetic and weights unchanged.
    runtime = run/'upstream'; runtime.mkdir()
    for folder in ['models', 'configs']:
        shutil.copytree(upstream/folder, runtime/folder, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(upstream/'LICENSE.md', runtime/'LICENSE.md')
    original = (runtime/'models/psp.py').read_text()
    adapted = original.replace("import matplotlib\nmatplotlib.use('Agg')\n", '')
    if adapted == original: raise ValueError('Plotting import adaptation not applied')
    (runtime/'models/psp.py').write_text(adapted)
    rows = {row['name']: row for row in json.loads((PARENT/'selection.json').read_text())}
    selected = [rows[name] for name in NAMES]
    write_json(run/'selection.json', selected)
    write_json(run/'model-spec.json', spec)
    frozen = {str(p): sha256(p) for folder in [run/'source', runtime]
              for p in folder.rglob('*') if p.is_file()}
    frozen[str(run/'selection.json')] = sha256(run/'selection.json')
    frozen[str(run/'model-spec.json')] = sha256(run/'model-spec.json')
    frozen[str(PARENT/'evaluation/report.json')] = sha256(PARENT/'evaluation/report.json')
    for row in selected:
        folder = PARENT/'evaluation'/row['name']
        for p in [folder/'reference.png', Path(row['image_path'])]: frozen[str(p)] = sha256(p)
    for path in ['deployment/gan-research/face_recognition_sface_2021dec.onnx',
                 'deployment/model-files/face-age/models/yunet/face_detection_yunet_2023mar.onnx']:
        frozen[str(ROOT/path)] = sha256(ROOT/path)
    write_json(run/'experiment.json', dict(profile='official_styleganex_age_feedforward',
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        upstream_commit=commit, cases=len(selected), scales=SCALES, targets_not_calibrated=True,
        frozen_inputs=frozen, gpu=1, max_running_duration='20m', training=False,
        preprocessing='Same saved FFHQ-aligned256 RGB as SAM; x1=x2, no second detector/alignment. Native1024 references retained.',
        inference='Official age checkpoint, pSp forward, checkpoint use_skip, zero_noise=True, resize=False; fixed strength sweep.',
        adaptations=['Remove unused matplotlib import in isolated runtime copy only. No upstream checkout mutation.'],
        selection='Eight previously inspected native-HD development portraits; not fresh holdout or population test.',
        criteria=['Inspect zero-strength reconstruction before evaluating editing.',
                  'Report all strengths and failures; no per-person best-strength cherry-picking.',
                  'Independent SFace and MiVOLO evaluation only, no optimization or selection.',
                  'No claim of target60/75 accuracy from uncalibrated strength.',
                  'Visible aging, source identity, eye/mouth integrity and native texture all required before expansion.',
                  'First-case repeat PNG must be exact; all outputs finite, native1024 PNG; model state frozen.',
                  'No automatic live deployment, no fresh41 holdout use.']))
    print('PREPARED', run, flush=True)


def validate(run):
    e = json.loads((run/'experiment.json').read_text())
    for path, digest in e['frozen_inputs'].items():
        if sha256(path) != digest: raise ValueError('Frozen source changed: '+path)
    spec = json.loads((run/'model-spec.json').read_text())
    weight = ROOT/'deployment/gan-models/styleganex'/spec['name']
    if weight.stat().st_size != spec['bytes'] or sha256(weight) != spec['sha256']:
        raise ValueError('Official model SHA256 mismatch')
    return e, spec, weight


def submit(run):
    validate(run)
    if (run/'receipt.json').exists(): raise ValueError('Already submitted')
    sys.path.insert(0, str(ROOT/'integration'))
    from deploy_yicloud import client, IMAGE
    job, ctx = client()
    active = job.list_jobs(ctx, job.models.ListJobsReq(Project='fdj-infra', ZoneId='cn-shanghai-a',
        JobNames='face-age', Self=True, Limit=100, Phases='Pending,Running,Inqueue,Starting,Stopping'))
    if active.Items: raise RuntimeError('One-card limit: wait for existing project job')
    name = 'face-age-'+run.name
    payload = dict(Name=name, Project='fdj-infra', ZoneId='cn-shanghai-a', QuotaGroup='fdj_gpu_taskpool',
        Priority='3', UsePrivateMachine=False, AutoDeleteDuration='168h', TaskSpecs={'compare':dict(
            Command=['bash', str(run/'source/run_styleganex.sh'), str(ROOT), str(run)], Env={}, Image=IMAGE,
            Replicas=1, CPU='4', Memory='32Gi', GPU='1', Internet=False,
            MaxRunningDuration='20m', MaxWaitDuration='10m', RestartPolicy='Never',
            MountGPFS='gpfs://gpfs2/haobinzhang:/root')})
    write_json(run/'request.json', payload)
    request = payload | {'TaskSpecs': {k:job.models.CreateJobReqTaskSpec(**v) for k,v in payload['TaskSpecs'].items()}}
    try: result = job.create_job(ctx, job.models.CreateJobReq(**request))
    except Exception:
        matches = job.list_jobs(ctx, job.models.ListJobsReq(Project='fdj-infra', ZoneId='cn-shanghai-a',
            JobNames=name, Self=True, Limit=100))
        exact = [x for x in matches.Items or [] if x.Name == name]
        if not exact: raise
        result = exact[0]
    receipt = dict(job_id=result.JobId, name=name, run_dir=str(run), gpus=1)
    write_json(run/'receipt.json', receipt); print(json.dumps(receipt), flush=True)


def worker(run):
    e, spec, weight = validate(run)
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    if os.environ.get('FACE_AGE_EXPERIMENT_ALLOCATION') != '1' or torch.cuda.device_count() != 1:
        raise ValueError('Exactly one allocated CUDA GPU required')
    torch.set_num_threads(4); cv2.setNumThreads(2); torch.manual_seed(20261127)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, str(run/'upstream'))
    from models.psp import pSp
    from source_color import preserve_source_chroma
    started = time.monotonic()
    checkpoint = torch.load(weight, map_location='cpu', weights_only=False)
    opts = dict(checkpoint['opts']); original_opts = dict(opts)
    if opts.get('toonify_weights'): raise ValueError('Unexpected toonify checkpoint dependency')
    opts.update(checkpoint_path=str(weight), device='cuda:0')
    model = pSp(Namespace(**opts), ckpt=checkpoint).cuda().eval().requires_grad_(False)
    model.latent_avg = model.latent_avg.cuda()
    incompatible = model.load_state_dict(checkpoint['state_dict'], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys: raise ValueError('Model state incomplete')
    direction = checkpoint['editing_w'][0:1].clone().cuda()
    if direction.shape != (1,18,512) or not torch.isfinite(direction).all():
        raise ValueError('Unexpected edit direction shape/values')
    write_json(run/'checkpoint-info.json', dict(opts=original_opts, keys=list(checkpoint),
        editing_w_shape=list(checkpoint['editing_w'].shape), strict_state_loading=True))
    del checkpoint
    def fingerprint():
        h = hashlib.sha256()
        for name, value in model.state_dict().items():
            h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
        h.update(model.latent_avg.detach().cpu().numpy().tobytes())
        h.update(direction.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    before = fingerprint()
    write_json(run/'device.json', dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), device_count=1, memory_bytes=torch.cuda.get_device_properties(0).total_memory))
    base = ROOT/'deployment/model-files/face-age'
    detector = cv2.FaceDetectorYN.create(str(base/'models/yunet/face_detection_yunet_2023mar.onnx'), '', (256,256), .6,.3,100)
    recognizer = cv2.FaceRecognizerSF.create(str(ROOT/'deployment/gan-research/face_recognition_sface_2021dec.onnx'), '')
    def feature(rgb):
        bgr = cv2.cvtColor(cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2BGR)
        _, faces = detector.detect(bgr)
        if faces is None: return None
        f = recognizer.feature(recognizer.alignCrop(bgr,max(faces,key=lambda v:v[2]*v[3]))).ravel()
        return f/(np.linalg.norm(f)+1e-12)
    def infer(x, scale):
        with torch.no_grad():
            y = model(x1=x, x2=x, use_skip=model.opts.use_skip, zero_noise=True,
                      resize=False, editing_w=direction*scale)
        if y.shape != (1,3,1024,1024) or not torch.isfinite(y).all():
            raise ValueError('Invalid native output')
        return ((y[0].clamp(-1,1)+1)*127.5).permute(1,2,0).byte().cpu().numpy()
    records = json.loads((run/'selection.json').read_text()); out = run/'evaluation'; out.mkdir(exist_ok=False)
    report = dict(records=records, routes=['sam_baseline','styleganex_age'], outputs=[], timings=[],
        scales=e['scales'], targets_not_calibrated=True, model_weights_frozen=True, default_model_changed=False)
    old = json.loads((PARENT/'evaluation/report.json').read_text())
    # Warmup and exact repetition, not counted as steady inference time.
    first = records[0]; small = np.asarray(Image.open(first['image_path']).convert('RGB'))
    x = torch.from_numpy(np.array(small,copy=True)).permute(2,0,1)[None].cuda().float()/127.5-1
    first_a, first_b = infer(x,1.), infer(x,1.)
    if not np.array_equal(first_a,first_b): raise ValueError('Repeated image differs')
    write_json(run/'repeatability.json', dict(case=first['name'],scale=1.,raw_pixels_equal=True))
    report['cold_setup_seconds'] = time.monotonic()-started
    for row in records:
        name = row['name']; folder = out/name; folder.mkdir()
        ref = np.asarray(Image.open(PARENT/'evaluation'/name/'reference.png').convert('RGB'))
        small = np.asarray(Image.open(row['image_path']).convert('RGB'))
        if not np.array_equal(small,cv2.resize(ref,(256,256),interpolation=cv2.INTER_AREA)):
            raise ValueError('SAM and StyleGANEX receive different aligned pixels')
        Image.fromarray(ref).save(folder/'reference.png'); source_f = feature(ref)
        def measure(path, condition, route, target=None, scale=None):
            pixels = np.asarray(Image.open(path).convert('RGB')); f = feature(pixels)
            if route == 'styleganex_age':
                preserved = path.with_name(path.stem+'_color.png')
                color, _ = preserve_source_chroma(ref,pixels); Image.fromarray(color).save(preserved)
            else: preserved = path.with_name(path.stem+'_color.png')
            report['outputs'].append(dict(name=name,cohort=row['cohort'],route=route,condition=condition,
                target=target,scale=scale,native=str(path.relative_to(out)),preserved=str(preserved.relative_to(out)),
                sha256=sha256(path),source_cosine=float(source_f@f) if source_f is not None and f is not None else None,
                detected=f is not None,reconstruction_l1=float(np.abs(pixels.astype(float)-ref.astype(float)).mean()) if condition in ['same','scale_0'] else None))
        for o in old['outputs']:
            if o['name'] != name or o['route'] != 'original_delta_constrained': continue
            path = folder/('sam_'+o['condition']+'.png')
            if sha256(PARENT/'evaluation'/o['native']) != o['sha256']: raise ValueError('SAM baseline changed')
            shutil.copy2(PARENT/'evaluation'/o['native'],path)
            shutil.copy2(PARENT/'evaluation'/o['preserved'],path.with_name(path.stem+'_color.png'))
            measure(path,o['condition'],'sam_baseline',target=o['target'])
        x = torch.from_numpy(np.array(small,copy=True)).permute(2,0,1)[None].cuda().float()/127.5-1
        for scale in e['scales']:
            torch.cuda.synchronize(); tick = time.monotonic(); pixels = infer(x,scale)
            torch.cuda.synchronize(); elapsed = time.monotonic()-tick
            condition = 'scale_'+format(scale,'g'); path = folder/('styleganex_'+condition+'.png')
            Image.fromarray(pixels).save(path); measure(path,condition,'styleganex_age',scale=scale)
            report['timings'].append(dict(name=name,scale=scale,seconds=elapsed))
        write_json(out/'report.partial.json',report)
        print('STYLEGANEX_CASE',name,len(report['outputs']),flush=True)
    after = fingerprint()
    write_json(run/'frozen-verification.json',dict(before=before,after=after,equal=before==after))
    if before != after: raise ValueError('Model state changed')
    report.update(total_seconds=time.monotonic()-started,cuda_peak_bytes=torch.cuda.max_memory_allocated())
    write_json(out/'report.json',report); (run/'DONE').write_text('StyleGANEX fixed-strength comparison complete\n')
    print('STYLEGANEX_DONE',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action',choices=['prepare','submit','worker'])
    p.add_argument('run',type=Path,nargs='?'); args = p.parse_args()
    if args.action == 'prepare': prepare()
    elif args.run is None: p.error('run required')
    elif args.action == 'submit': submit(args.run.resolve())
    else: worker(args.run.resolve())

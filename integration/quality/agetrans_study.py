"""Official AgeTransGAN ten-group checkpoint: fixed development-set comparison."""
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

from sfe_study import ROOT, PARENT, NAMES
from probe_inversion import sha256, write_json

REV='5d1f8354d903f8858ea9da639c0eac80180cfde2'


def prepare(cpu_pilot=False,cpu_full=False):
    cpu=cpu_pilot or cpu_full
    upstream=ROOT/'deployment/AgeTransGAN'
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=upstream,text=True).strip()!=REV:
        raise ValueError('Unexpected upstream revision')
    spec=json.loads((ROOT/'integration/models/agetrans-10group.json').read_text())
    run=ROOT/'deployment/training-runs'/(('agetrans-cpu-' if cpu else 'agetrans-')+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
    (run/'source').mkdir(parents=True,exist_ok=False)
    for name in ['agetrans_study.py','sfe_study.py','probe_inversion.py','run_agetrans.sh']:
        shutil.copy2(Path(__file__).with_name(name),run/'source'/name)
    shutil.copy2(ROOT/'integration/source_color.py',run/'source/source_color.py')
    runtime=run/'upstream';runtime.mkdir()
    shutil.copy2(upstream/'test/model.py',runtime/'model.py')
    shutil.copytree(upstream/'test/op',runtime/'op',ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['fused_act.py','upfirdn2d.py']:
        shutil.copy2(ROOT/'deployment/StyleGANEX/models/stylegan2/op'/name,runtime/'op'/name)
    shutil.copy2(upstream/'README.md',runtime/'UPSTREAM_README.md')
    shutil.copy2(ROOT/'deployment/StyleGANEX/LICENSE.md',runtime/'STYLEGANEX_LICENSE.md')
    rows={r['name']:r for r in json.loads((PARENT/'selection.json').read_text())}
    selected=['ffhq_64376','ffhq_62644'] if cpu_pilot else NAMES
    write_json(run/'selection.json',[rows[n] for n in selected]);write_json(run/'model-spec.json',spec)
    frozen={str(p):sha256(p) for d in [run/'source',runtime] for p in d.rglob('*') if p.is_file()}
    for p in [run/'selection.json',run/'model-spec.json',PARENT/'evaluation/report.json']:
        frozen[str(p)]=sha256(p)
    for name in selected:
        p=PARENT/'evaluation'/name/'reference.png';frozen[str(p)]=sha256(p)
    write_json(run/'experiment.json',dict(profile='official_agetrans_cpu_pilot' if cpu_pilot else 'official_agetrans_cpu_full' if cpu_full else 'official_agetrans_10group',upstream_commit=REV,
        groups=list(range(10)),cases=len(selected),gpu=0 if cpu else 1,device='cpu' if cpu else 'cuda:0',training=False,frozen_inputs=frozen,
        preprocessing='Same saved native1024 FFHQ reference, RGB [-1,1], no second alignment.',
        adaptations=['Use model directly, omit desktop/video/landmark and TensorFlow utilities.',
                     'Official default randomize_noise=True; reset RNG seed per forward for repeatable, shared noise across age groups.',
                     'PyTorch reference StyleGAN operators, no change to generator network or weights.'],
        limitations=['Age-group indices are not numerical ages. README table has ambiguous middle-group bounds; do not invent them.',
                     'Code license not separately stated; do not redistribute upstream source/checkpoint or assert CAF license covers model.',
                     'Publisher SHA256 unavailable; record local complete SHA256 and official Drive origin honestly.',
                     'Eight previously inspected development portraits only; possible FFHQ pretraining overlap.',
                     'No automatic replacement of upload model.']))
    print('PREPARED',run,flush=True)


def validate(run):
    exp=json.loads((run/'experiment.json').read_text())
    for path,digest in exp['frozen_inputs'].items():
        if sha256(path)!=digest:raise ValueError('Frozen input changed: '+path)
    spec=json.loads((run/'model-spec.json').read_text());path=ROOT/spec['local_path']
    if path.stat().st_size!=spec['bytes'] or sha256(path)!=spec['sha256']:raise ValueError('Checkpoint hash/size differs')
    return exp,path


def submit(run):
    exp,_=validate(run)
    if (run/'receipt.json').exists():raise ValueError('Already submitted')
    sys.path.insert(0,str(ROOT/'integration'))
    from deploy_yicloud import client, IMAGE
    job,ctx=client()
    active=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames='face-age',Self=True,Limit=100,Phases='Pending,Running,Inqueue,Starting,Stopping'))
    if active.Items and exp['gpu']:raise ValueError('One-card limit: existing project job')
    name='face-age-'+run.name
    payload=dict(Name=name,Project='fdj-infra',ZoneId='cn-shanghai-a',QuotaGroup='fdj_gpu_taskpool',Priority='3',UsePrivateMachine=False,AutoDeleteDuration='168h',
        TaskSpecs={'compare':dict(Command=['bash',str(run/'source/run_agetrans.sh'),str(ROOT),str(run)],Env={'FACE_AGE_EXPERIMENT_DEVICE':exp.get('device','cuda:0')},Image=IMAGE,Replicas=1,CPU='8' if not exp['gpu'] else '4',Memory='32Gi',GPU=str(exp['gpu']),Internet=False,MaxRunningDuration='20m',MaxWaitDuration='10m',RestartPolicy='Never',MountGPFS='gpfs://gpfs2/haobinzhang:/root')})
    write_json(run/'request.json',payload)
    req=payload|{'TaskSpecs':{k:job.models.CreateJobReqTaskSpec(**v) for k,v in payload['TaskSpecs'].items()}}
    try:result=job.create_job(ctx,job.models.CreateJobReq(**req))
    except Exception:
        r=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames=name,Self=True,Limit=100))
        exact=[x for x in r.Items or [] if x.Name==name]
        if not exact:raise
        result=exact[0]
    receipt=dict(job_id=result.JobId,name=name,run_dir=str(run),gpus=exp['gpu'])
    write_json(run/'receipt.json',receipt);print(json.dumps(receipt),flush=True)


def worker(run):
    exp,weight=validate(run)
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    device=exp.get('device','cuda:0');cpu=device=='cpu'
    if os.environ.get('FACE_AGE_EXPERIMENT_ALLOCATION')!='1' or torch.cuda.device_count()!=(0 if cpu else 1):
        raise ValueError('Device allocation differs from experiment')
    def sync():
        if not cpu:torch.cuda.synchronize()
    torch.set_num_threads(8 if cpu else 4);cv2.setNumThreads(2);torch.manual_seed(20261127)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    sys.path.insert(0,str(run/'upstream'))
    from model import Generator
    from source_color import preserve_source_chroma
    start=time.monotonic();model=Generator(1024,512,8,10,channel_multiplier=2)
    ckpt=torch.load(weight,map_location='cpu',weights_only=False)
    model.load_state_dict(ckpt['g_ema'],strict=True)
    write_json(run/'checkpoint-info.json',dict(keys=list(ckpt),strict_loading=True));del ckpt
    model=model.to(device).eval().requires_grad_(False)
    write_json(run/'device.json',dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,gpu=None if cpu else torch.cuda.get_device_name(0),device_count=0 if cpu else 1,memory_bytes=0 if cpu else torch.cuda.get_device_properties(0).total_memory))
    def fingerprint():
        h=hashlib.sha256()
        for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    before=fingerprint()
    det=cv2.FaceDetectorYN.create(str(ROOT/'deployment/model-files/face-age/models/yunet/face_detection_yunet_2023mar.onnx'),'',(256,256),.6,.3,100)
    rec=cv2.FaceRecognizerSF.create(str(ROOT/'deployment/gan-research/face_recognition_sface_2021dec.onnx'),'')
    def feature(rgb):
        bgr=cv2.cvtColor(cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2BGR)
        _,faces=det.detect(bgr)
        if faces is None:return None
        v=rec.feature(rec.alignCrop(bgr,max(faces,key=lambda f:f[2]*f[3]))).ravel();return v/(np.linalg.norm(v)+1e-12)
    def tensor(rgb):return torch.from_numpy(np.array(rgb,copy=True)).permute(2,0,1)[None].to(device).float()/127.5-1
    @torch.inference_mode()
    def infer(x,group):
        label=torch.eye(10,device=device)[group:group+1]
        # The upstream fixed buffers start at4x4, while this conditional encoder
        # starts synthesis at16x16. Use its actual official demo path instead.
        with torch.random.fork_rng(devices=[] if cpu else [0]):
            torch.manual_seed(20261127)
            y,_=model(x,label,randomize_noise=True)
        if y.shape!=(1,3,1024,1024) or not torch.isfinite(y).all():raise ValueError('Invalid native1024 output')
        return ((y[0].clamp(-1,1)+1)*127.5).permute(1,2,0).byte().cpu().numpy()
    rows=json.loads((run/'selection.json').read_text());ref=np.asarray(Image.open(PARENT/'evaluation'/rows[0]['name']/'reference.png').convert('RGB'))
    x=tensor(ref);a,b=infer(x,9),infer(x,9)
    if not np.array_equal(a,b):raise ValueError('Nonrepeatable output')
    write_json(run/'repeatability.json',dict(case=rows[0]['name'],group=9,raw_pixels_equal=True))
    out=run/'evaluation';out.mkdir(exist_ok=False)
    report=dict(records=rows,routes=['sam_baseline','agetrans'],groups=list(range(10)),outputs=[],timings=[],cold_setup_seconds=time.monotonic()-start,targets_not_calibrated=True)
    old=json.loads((PARENT/'evaluation/report.json').read_text())
    for row in rows:
        name=row['name'];folder=out/name;folder.mkdir()
        ref=np.asarray(Image.open(PARENT/'evaluation'/name/'reference.png').convert('RGB'));Image.fromarray(ref).save(folder/'reference.png');sf=feature(ref)
        def measure(path,route,condition,target=None,group=None):
            rgb=np.asarray(Image.open(path).convert('RGB'));f=feature(rgb);color=path.with_name(path.stem+'_color.png')
            if route=='agetrans':c,_=preserve_source_chroma(ref,rgb);Image.fromarray(c).save(color)
            report['outputs'].append(dict(name=name,cohort=row['cohort'],route=route,condition=condition,target=target,scale=None,group=group,
                native=str(path.relative_to(out)),preserved=str(color.relative_to(out)),sha256=sha256(path),source_cosine=float(sf@f) if sf is not None and f is not None else None,detected=f is not None,reconstruction_l1=None))
        for item in old['outputs']:
            if item['name']!=name or item['route']!='original_delta_constrained':continue
            source=PARENT/'evaluation'/item['native']
            if sha256(source)!=item['sha256']:raise ValueError('Baseline changed')
            path=folder/('sam_'+item['condition']+'.png');shutil.copy2(source,path);shutil.copy2(PARENT/'evaluation'/item['preserved'],path.with_name(path.stem+'_color.png'))
            measure(path,'sam_baseline',item['condition'],target=item['target'])
        x=tensor(ref)
        for group in range(10):
            sync();tick=time.monotonic();rgb=infer(x,group);sync();elapsed=time.monotonic()-tick
            path=folder/('agetrans_group_'+str(group)+'.png');Image.fromarray(rgb).save(path);measure(path,'agetrans','group_'+str(group),group=group)
            report['timings'].append(dict(name=name,group=group,seconds=elapsed))
        write_json(out/'report.partial.json',report);print('AGETRANS_CASE',name,len(report['outputs']),flush=True)
    after=fingerprint();write_json(run/'frozen-verification.json',dict(before=before,after=after,equal=before==after))
    if before!=after:raise ValueError('Model state mutated')
    report.update(total_seconds=time.monotonic()-start,cuda_peak_bytes=0 if cpu else torch.cuda.max_memory_allocated())
    write_json(out/'report.json',report);(run/'DONE').write_text('success\n');print('AGETRANS_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','submit','worker']);p.add_argument('run',type=Path,nargs='?')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--cpu-pilot',action='store_true');mode.add_argument('--cpu',action='store_true');a=p.parse_args()
    if a.action=='prepare':prepare(a.cpu_pilot,a.cpu)
    elif a.run is None:p.error('run required')
    else:globals()[a.action](a.run.resolve())

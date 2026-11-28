"""Bounded official StyleFeatureEditor pilot; inference only, one allocated GPU."""
import argparse
import ast
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
SCALES = [-10., -5., 0., 2.5, 5., 7.5, 10.]
REV = '00996c309ba5f04eec9ad7456f521e1501f7cc67'


def method(path, cls, name):
    source = path.read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == cls)
    member = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return '\n'.join(source.splitlines()[member.lineno-1:member.end_lineno])+'\n'


def prepare(cpu_pilot=False, dense_younger=False, glasses_projection=False):
    if glasses_projection and (cpu_pilot or dense_younger):
        raise ValueError('The fixed projection protocol uses8 cases/7 strengths/1GPU; omit CPU and dense flags')
    upstream = ROOT/'deployment/StyleFeatureEditor'
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=upstream,text=True).strip() != REV:
        raise ValueError('Wrong upstream revision')
    if subprocess.check_output(['git','status','--porcelain'],cwd=upstream,text=True):
        raise ValueError('Keep upstream checkout clean')
    run = ROOT/'deployment/training-runs'/(('sfe-cpu-' if cpu_pilot else 'sfe-')+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
    (run/'source').mkdir(parents=True,exist_ok=False)
    for name in ['sfe_study.py','run_sfe.sh','probe_inversion.py']:
        shutil.copy2(Path(__file__).with_name(name),run/'source'/name)
    shutil.copy2(ROOT/'integration/source_color.py',run/'source/source_color.py')
    runtime = run/'upstream'; runtime.mkdir()
    shutil.copytree(upstream/'models/psp',runtime/'models/psp',ignore=shutil.ignore_patterns('__pycache__'))
    (runtime/'models/__init__.py').touch()
    shutil.copy2(upstream/'LICENSE',runtime/'LICENSE')
    shutil.copy2(ROOT/'deployment/StyleGANEX/LICENSE.md',runtime/'STYLEGANEX_LICENSE.md')
    shutil.copy2(upstream/'editings/interfacegan_directions/age.pt',runtime/'age.pt')
    if glasses_projection:
        shutil.copy2(upstream/'editings/bound/Eyeglasses_boundary.npy',runtime/'eyeglasses.npy')
    enc = runtime/'models/psp/encoders/psp_encoders.py'
    s = enc.read_text(); line = '        resnet50.load_state_dict(torch.load(opts.arcface_model_path))\n'
    if s.count(line) != 1: raise ValueError('ArcFace initialization adaptation mismatch')
    enc.write_text(s.replace(line,'        # All used backbone tensors are loaded strictly from the trained inverter below.\n'))
    # PyTorch reference operators, already used by the official StyleGANEX tree.
    # Avoid compiling legacy CUDA extensions; preserve kernel, padding and activation math.
    ops = ROOT/'deployment/StyleGANEX/models/stylegan2/op'
    for name in ['fused_act.py','upfirdn2d.py']:
        shutil.copy2(ops/name,runtime/'models/psp/stylegan2/op'/name)
    header = '''import torch
import torch.nn.functional as F
from torch import nn
from argparse import Namespace
from models.psp.encoders import psp_encoders
from models.psp.stylegan2.model import Generator

def keys(ckpt, prefix):
    return {k[len(prefix)+1:]:v for k,v in ckpt['state_dict'].items() if k.startswith(prefix+'.')}

class InferenceOnly(nn.Module):
    def __init__(self, editor, e4e, generator, device='cuda:0'):
        super().__init__()
        self.opts = Namespace(stylegan_size=1024)
        self.device = device
        self.encoder = self.set_encoder()
        self.decoder = Generator(1024,512,8)
        ckpt = torch.load(editor,map_location='cpu',weights_only=False)
        self.encoder.load_state_dict(keys(ckpt,'encoder'),strict=True)
        self.inverter.load_state_dict(keys(ckpt,'inverter'),strict=True)
        self.checkpoint_keys = list(ckpt)
        del ckpt
        ckpt = torch.load(generator,map_location='cpu',weights_only=False)
        self.decoder.load_state_dict(ckpt['g_ema'],strict=True)
        self.latent_avg = ckpt['latent_avg'].to(device)
        del ckpt
        ckpt = torch.load(e4e,map_location='cpu',weights_only=False)
        self.e4e_encoder.load_state_dict(keys(ckpt,'encoder'),strict=True)
        self.to(device).eval().requires_grad_(False)

'''
    body = header+method(upstream/'models/methods.py','FSEFull','set_encoder')+'\n'
    body += method(upstream/'models/methods.py','FSEFull','forward')
    body += '''
class EditingOnly:
    def __init__(self, model, direction):
        self.method = model
        self.device = model.device
        self.direction = direction

    def get_edited_latent(self, latent, editing_name, factors, inputs):
        assert editing_name == 'age'
        return [latent + factor / 2 * self.direction for factor in factors]

'''
    body += method(upstream/'runners/inference_runners.py','FSEInferenceRunner','_run_on_batch')+'\n'
    body += method(upstream/'runners/inference_runners.py','FSEInferenceRunner','_run_editing_on_batch')
    (runtime/'inference_only.py').write_text(body)
    reference = 'import torch\nimport torch.nn.functional as F\n'+method_as_function(upstream/'models/psp/stylegan2/op/upfirdn2d.py','upfirdn2d_native')
    (runtime/'reference_filter.py').write_text(reference)
    specs = [json.loads(p.read_text()) for p in sorted((ROOT/'integration/models').glob('sfe-*.json'))]
    if len(specs) != 3: raise ValueError('Expected exactly three inference checkpoints')
    write_json(run/'model-specs.json',specs)
    rows = {r['name']:r for r in json.loads((PARENT/'selection.json').read_text())}
    selected = ['ffhq_67140','ffhq_60764','ffhq_68335'] if dense_younger else (['ffhq_64376','ffhq_62644'] if cpu_pilot else NAMES)
    scales = [i/2 for i in range(-20,1)] if dense_younger else ([0.,5.,10.] if cpu_pilot else SCALES)
    write_json(run/'selection.json',[rows[name] for name in selected])
    frozen = {str(p):sha256(p) for folder in [run/'source',runtime] for p in folder.rglob('*') if p.is_file()}
    for p in [run/'model-specs.json',run/'selection.json',PARENT/'evaluation/report.json']:
        frozen[str(p)] = sha256(p)
    for name in selected:
        p = PARENT/'evaluation'/name/'reference.png'; frozen[str(p)] = sha256(p)
    write_json(run/'experiment.json',dict(profile='official_sfe_glasses_projected' if glasses_projection else ('official_sfe_dense_younger' if dense_younger else ('official_sfe_cpu_pilot' if cpu_pilot else 'official_sfe_age_pilot')),upstream_commit=REV,
        glasses_projection=glasses_projection,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        scales=scales,cases=len(selected),gpu=0 if cpu_pilot else 1,device='cpu' if cpu_pilot else 'cuda:0',training=False,frozen_inputs=frozen,targets_not_calibrated=True,
        preprocessing='Same saved native1024 FFHQ reference. Official forward bilinear-downsamples to256, unlike SAM area-aligned256 input.',
        adaptations=['Inference-only module omits unused discriminator, trainers and unrelated edit models.',
                     'All used components strictly load official weights; omit redundant ArcFace preinitialization.',
                     'Reference PyTorch StyleGAN operators from official StyleGANEX; no legacy CUDA compilation.',
                     'Official forward and edit functions copied verbatim; no mask, repair, superresolution or blending.'],
        criteria=['Fixed strengths, no per-image fitting; e4e same-direction control at every strength.',
                  'Native1024 finite output, deterministic repeat, frozen tensors, raw and color retained.',
                  'Previously inspected development cases; not fresh holdout and possible FFHQ pretraining overlap.',
                  'Inspect achieved aging with identity/skin; never infer chronological-age accuracy from strength.',
                  'No automatic live-model replacement.']))
    print('PREPARED',run,flush=True)
    return run


def validate(run):
    exp = json.loads((run/'experiment.json').read_text())
    for path,digest in exp['frozen_inputs'].items():
        if sha256(path) != digest: raise ValueError('Frozen source changed: '+path)
    weights = {}
    for spec in json.loads((run/'model-specs.json').read_text()):
        path = ROOT/'deployment/gan-models/stylefeatureeditor'/spec['name']
        if path.stat().st_size != spec['bytes'] or sha256(path) != spec['sha256']:
            raise ValueError('Model size/SHA256 mismatch: '+str(path))
        weights[spec['name']] = path
    return exp,weights


def method_as_function(path,name):
    source=path.read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return '\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])+'\n'


def submit(run):
    exp,_ = validate(run)
    if (run/'receipt.json').exists(): raise ValueError('Already submitted')
    sys.path.insert(0,str(ROOT/'integration'))
    from deploy_yicloud import client, IMAGE
    job,ctx = client()
    active = job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',
        JobNames='face-age',Self=True,Limit=100,Phases='Pending,Running,Inqueue,Starting,Stopping'))
    if active.Items and exp['gpu']: raise RuntimeError('One-card limit: existing project job')
    name = 'face-age-'+run.name
    payload = dict(Name=name,Project='fdj-infra',ZoneId='cn-shanghai-a',QuotaGroup='fdj_gpu_taskpool',
        Priority='3',UsePrivateMachine=False,AutoDeleteDuration='168h',TaskSpecs={'compare':dict(
            Command=['bash',str(run/'source/run_sfe.sh'),str(ROOT),str(run)],Env={'FACE_AGE_EXPERIMENT_DEVICE':exp.get('device','cuda:0')},Image=IMAGE,
            Replicas=1,CPU='8' if not exp['gpu'] else '4',Memory='32Gi',GPU=str(exp['gpu']),Internet=False,MaxRunningDuration='20m',
            MaxWaitDuration='10m',RestartPolicy='Never',MountGPFS='gpfs://gpfs2/haobinzhang:/root')})
    write_json(run/'request.json',payload)
    req = payload | {'TaskSpecs':{k:job.models.CreateJobReqTaskSpec(**v) for k,v in payload['TaskSpecs'].items()}}
    try: result = job.create_job(ctx,job.models.CreateJobReq(**req))
    except Exception:
        found = job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames=name,Self=True,Limit=100))
        exact = [x for x in found.Items or [] if x.Name == name]
        if not exact: raise
        result = exact[0]
    receipt = dict(job_id=result.JobId,name=name,run_dir=str(run),gpus=exp['gpu'])
    write_json(run/'receipt.json',receipt); print(json.dumps(receipt),flush=True)


def worker(run):
    exp,weights = validate(run)
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    device=exp.get('device','cuda:0');cpu=device=='cpu'
    if os.environ.get('FACE_AGE_EXPERIMENT_ALLOCATION') != '1' or torch.cuda.device_count() != (0 if cpu else 1):
        raise ValueError('Device allocation does not match experiment')
    def sync():
        if not cpu:torch.cuda.synchronize()
    torch.set_num_threads(8 if cpu else 4); cv2.setNumThreads(2); torch.manual_seed(20261127)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False
    sys.path.insert(0,str(run/'upstream'))
    from inference_only import InferenceOnly, EditingOnly
    from reference_filter import upfirdn2d_native
    from models.psp.stylegan2.op.upfirdn2d import upfirdn2d
    from source_color import preserve_source_chroma
    operator_checks=[]
    for up,down,pad in [(1,1,(1,1)),(2,1,(2,1)),(1,2,(1,2)),(2,1,(-1,2))]:
        x=torch.randn(2,3,9,9);kernel=torch.tensor([[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]])/16
        expected=upfirdn2d_native(x.reshape(-1,9,9,1),kernel,up,up,down,down,*pad,*pad)
        actual=upfirdn2d(x,kernel,up=up,down=down,pad=pad)
        expected=expected.reshape_as(actual)
        error=float((expected-actual).abs().max())
        if error!=0:raise ValueError('Reference filter parity failed')
        operator_checks.append(dict(up=up,down=down,pad=pad,max_abs_error=error))
    write_json(run/'operator-verification.json',dict(reference='Pinned SFE native filter',checks=operator_checks))
    tick = time.monotonic()
    model = InferenceOnly(weights['sfe_editor_light.pt'],weights['e4e_ffhq_encode.pt'],weights['stylegan2-ffhq-config-f.pt'],device=device)
    direction = torch.load(run/'upstream/age.pt',map_location='cpu',weights_only=False).to(device)
    if exp.get('glasses_projection'):
        age=direction.reshape(1,1,512).expand(1,18,512).clone()
        glasses=torch.from_numpy(np.load(run/'upstream/eyeglasses.npy',allow_pickle=False)).reshape(1,18,512).to(device)
        before_cos=float((age*glasses).sum()/(age.norm()*glasses.norm()))
        projected=age-(age*glasses).sum()/glasses.square().sum()*glasses
        direction=projected*(age.norm()/projected.norm())
        write_json(run/'direction-projection.json',dict(original_cosine=before_cos,
            projected_cosine=float((direction*glasses).sum()/(direction.norm()*glasses.norm())),
            original_norm=float(age.norm()),projected_norm=float(direction.norm()),norm_restored=True,
            source='Pinned official SFE editings/bound/Eyeglasses_boundary.npy; W+ linear projection, no per-image fitting',
            limitation='Linear orthogonality does not guarantee semantic independence or matched achieved aging.'))
    if direction.shape not in [(1,512),(512,),(1,18,512)] or not torch.isfinite(direction).all():
        raise ValueError('Unexpected interfaceGAN direction')
    runner = EditingOnly(model,direction)
    write_json(run/'checkpoint-info.json',dict(keys=model.checkpoint_keys,strict_loading=True,direction_shape=list(direction.shape)))
    write_json(run/'device.json',dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=None if cpu else torch.cuda.get_device_name(0),device_count=0 if cpu else 1,memory_bytes=0 if cpu else torch.cuda.get_device_properties(0).total_memory))
    def fingerprint():
        h=hashlib.sha256()
        for k,v in model.state_dict().items(): h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
        h.update(model.latent_avg.cpu().numpy().tobytes()); h.update(direction.cpu().numpy().tobytes())
        return h.hexdigest()
    before=fingerprint()
    det=cv2.FaceDetectorYN.create(str(ROOT/'deployment/model-files/face-age/models/yunet/face_detection_yunet_2023mar.onnx'),'',(256,256),.6,.3,100)
    rec=cv2.FaceRecognizerSF.create(str(ROOT/'deployment/gan-research/face_recognition_sface_2021dec.onnx'),'')
    def feature(rgb):
        bgr=cv2.cvtColor(cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2BGR)
        _,faces=det.detect(bgr)
        if faces is None:return None
        v=rec.feature(rec.alignCrop(bgr,max(faces,key=lambda f:f[2]*f[3]))).ravel()
        return v/(np.linalg.norm(v)+1e-12)
    def tensor(path):
        rgb=np.asarray(Image.open(path).convert('RGB'))
        if rgb.shape!=(1024,1024,3):raise ValueError('Expected native1024 input')
        return torch.from_numpy(np.array(rgb,copy=True)).permute(2,0,1)[None].to(device).float()/127.5-1
    def pixels(y):
        if y.shape!=(1,3,1024,1024) or not torch.isfinite(y).all():raise ValueError('Invalid native output')
        return ((y[0].clamp(-1,1)+1)*127.5).permute(1,2,0).byte().cpu().numpy()
    @torch.inference_mode()
    def infer(inv,scale):
        y,_,e4e=runner._run_editing_on_batch(inv,'age',[scale],return_e4e=True)
        return pixels(y[0]),pixels(e4e)
    records=json.loads((run/'selection.json').read_text())
    with torch.inference_mode(): _,inv=runner._run_on_batch(tensor(PARENT/'evaluation'/records[0]['name']/'reference.png'))
    a,b=infer(inv,5.),infer(inv,5.)
    if any(not np.array_equal(x,y) for x,y in zip(a,b)):raise ValueError('Nonrepeatable output')
    write_json(run/'repeatability.json',dict(case=records[0]['name'],scale=5.,raw_pixels_equal=True))
    cold=time.monotonic()-tick
    out=run/'evaluation'; out.mkdir(exist_ok=False)
    report=dict(records=records,routes=['sam_baseline','sfe_age','e4e_age'],scales=exp['scales'],outputs=[],timings=[],cold_setup_seconds=cold,targets_not_calibrated=True)
    baseline=json.loads((PARENT/'evaluation/report.json').read_text())
    for row in records:
        name=row['name'];folder=out/name;folder.mkdir()
        ref=np.asarray(Image.open(PARENT/'evaluation'/name/'reference.png').convert('RGB'))
        Image.fromarray(ref).save(folder/'reference.png');sf=feature(ref)
        def measure(path,route,condition,scale=None,target=None):
            rgb=np.asarray(Image.open(path).convert('RGB'));f=feature(rgb)
            color=path.with_name(path.stem+'_color.png')
            if route!='sam_baseline':
                c,_=preserve_source_chroma(ref,rgb);Image.fromarray(c).save(color)
            report['outputs'].append(dict(name=name,cohort=row['cohort'],route=route,condition=condition,scale=scale,target=target,
                native=str(path.relative_to(out)),preserved=str(color.relative_to(out)),sha256=sha256(path),
                source_cosine=float(sf@f) if sf is not None and f is not None else None,detected=f is not None,
                reconstruction_l1=float(np.abs(rgb.astype(float)-ref.astype(float)).mean()) if condition in ['same','scale_0'] else None))
        for item in baseline['outputs']:
            if item['name']!=name or item['route']!='original_delta_constrained':continue
            source=PARENT/'evaluation'/item['native']
            if sha256(source)!=item['sha256']:raise ValueError('Baseline changed')
            path=folder/('sam_'+item['condition']+'.png');shutil.copy2(source,path)
            shutil.copy2(PARENT/'evaluation'/item['preserved'],path.with_name(path.stem+'_color.png'))
            measure(path,'sam_baseline',item['condition'],target=item['target'])
        sync();start=time.monotonic()
        with torch.inference_mode(): _,inv=runner._run_on_batch(tensor(folder/'reference.png'))
        sync();inversion_seconds=time.monotonic()-start
        for scale in exp['scales']:
            sync();start=time.monotonic();ys=infer(inv,scale)
            sync();seconds=time.monotonic()-start
            condition='scale_'+format(scale,'g')
            for route,y in zip(['sfe_age','e4e_age'],ys):
                path=folder/(route+'_'+condition+'.png');Image.fromarray(y).save(path);measure(path,route,condition,scale)
            report['timings'].append(dict(name=name,scale=scale,edit_pair_seconds=seconds,inversion_seconds=inversion_seconds))
        write_json(out/'report.partial.json',report);print('SFE_CASE',name,len(report['outputs']),flush=True)
    after=fingerprint();write_json(run/'frozen-verification.json',dict(before=before,after=after,equal=before==after))
    if before!=after:raise ValueError('Inference model mutated')
    report.update(total_seconds=time.monotonic()-tick,cuda_peak_bytes=0 if cpu else torch.cuda.max_memory_allocated())
    write_json(out/'report.json',report);(run/'DONE').write_text('success\n');print('SFE_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','submit','worker']);p.add_argument('run',type=Path,nargs='?');p.add_argument('--cpu-pilot',action='store_true');p.add_argument('--dense-younger',action='store_true');p.add_argument('--glasses-projection',action='store_true');a=p.parse_args()
    if a.action=='prepare':prepare(a.cpu_pilot,a.dense_younger,a.glasses_projection)
    elif a.run is None:p.error('run required')
    else:globals()[a.action](a.run.resolve())

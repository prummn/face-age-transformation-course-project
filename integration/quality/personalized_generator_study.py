"""Bounded per-photo generator fitting, with a paired locality-preservation arm."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from probe_inversion import sha256,write_json

ROOT=Path(os.environ.get('FACE_AGE_REPO',Path(__file__).resolve().parents[2]))
PARENT=ROOT/'deployment/training-runs/original-delta-20261124-160055'
NAMES=['ffhq_64376','ffhq_64919','ffhq_67140','ffhq_65817']


def prepare(confirm_hd=False):
    rows={r['name']:r for r in json.loads((PARENT/'selection.json').read_text())};selected=[rows[n] for n in NAMES]
    if confirm_hd:
        from PIL import Image
        selected=[]
        for row in rows.values():
            if row['name'] in NAMES or row.get('source_face_short_edge',row.get('native_face_short_edge',0))<512:continue
            with Image.open(PARENT/'evaluation'/row['name']/'reference.png') as image:
                if image.size!=(1024,1024):continue
            selected.append(row)
        if len(selected)!=25:raise ValueError('Expected the25 remaining native HD development inputs')
    run=ROOT/'deployment/training-runs'/(('personalized-generator-confirm-' if confirm_hd else 'personalized-generator-')+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'));(run/'source').mkdir(parents=True,exist_ok=False)
    for name in ['personalized_generator_study.py','run_personalized_generator.sh','probe_inversion.py','deterministic_quality_ops.py']:
        shutil.copy2(Path(__file__).with_name(name),run/'source'/name)
    shutil.copy2(ROOT/'integration/source_color.py',run/'source/source_color.py')
    write_json(run/'selection.json',selected);frozen={}
    for spec in json.loads((ROOT/'integration/quality/baselines/sam-wplus-v1.json').read_text())['weights']:
        p=ROOT/spec['path']
        if sha256(p)!=spec['sha256']:raise ValueError('Baseline checkpoint changed')
        frozen[str(p)]=spec['sha256']
    files=list((run/'source').iterdir())+[run/'selection.json',PARENT/'evaluation/report.json']
    for row in selected:files += [PARENT/'evaluation'/row['name']/f for f in ['reference.png','original_delta_constrained_codes.pt']]
    files += [ROOT/'integration/ascend/src'/name for name in ['sam_npu.py','sam_losses.py','faceattr.py']]
    files.append(ROOT/'deployment/gan-research/face_recognition_sface_2021dec.onnx');frozen.update({str(p):sha256(p) for p in files})
    write_json(run/'experiment.json',dict(profile='per_photo_generator_locality_pair',parent_run=str(PARENT),cases=len(selected),steps=200,
        selection_stage='Remaining25 native-HD development inputs, four diagnostic inputs excluded; same frozen recipe, no retuning.' if confirm_hd else 'Four diagnostic development inputs',
        criteria=dict(identity_drop_max=.02,age_span_min_ratio=.9,missing_faces_increase_max=0,native_review_required=True,no_automatic_promotion=True),frozen_inputs=frozen,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        common_recipe=dict(source_anchor='Same saved W+ pivot',generator_min_resolution=64,learning_rate=1e-4,source_steps=200,pixel512=.5,lpips=.8,identity=.8,native_local=.15,parameter_trust=.1),
        sole_paired_difference='Additional LPIPS/local low-frequency regularization to original-generator images near source-to-edit latent interpolations. This is a PTI-inspired bounded diagnostic, not an exact paper reproduction.',
        locality_recipe=dict(weight=1,lpips256=.8,pixel64=.2,interpolation_range=[.25,1],noise_std=.02),
        limitations=['Per-photo generator patches, not universal trained weights. No automatic deployment.', 'Previously inspected development inputs; confirmation excludes the four diagnostic inputs but is not a blinded or independent population test.', 'No source pixels blended into output, no age rescaling, no candidate-specific latent refitting.', 'Locality can preserve existing artifacts; improved same-age reconstruction alone is insufficient.']))
    print('PREPARED',run,flush=True)


def submit(run):
    e=json.loads((run/'experiment.json').read_text())
    for filename,digest in e['frozen_inputs'].items():
        if sha256(filename)!=digest:raise ValueError('Frozen input changed')
    if (run/'receipt.json').exists():raise ValueError('Already submitted')
    sys.path.insert(0,str(ROOT/'integration'));from deploy_yicloud import client,IMAGE
    job,ctx=client();active=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames='face-age',Self=True,Limit=100,Phases='Pending,Running,Inqueue,Starting,Stopping'))
    if active.Items:raise RuntimeError('One-card limit: another project job is active')
    name='face-age-'+run.name;payload=dict(Name=name,Project='fdj-infra',ZoneId='cn-shanghai-a',QuotaGroup='fdj_gpu_taskpool',Priority='3',UsePrivateMachine=False,AutoDeleteDuration='168h',
        TaskSpecs={'personalize':dict(Command=['bash',str(run/'source/run_personalized_generator.sh'),str(ROOT),str(run)],Env={},Image=IMAGE,Replicas=1,CPU='4',Memory='48Gi',GPU='1',Internet=False,MaxRunningDuration='20m',MaxWaitDuration='10m',RestartPolicy='Never',MountGPFS='gpfs://gpfs2/haobinzhang:/root')})
    write_json(run/'request.json',payload);specs={k:job.models.CreateJobReqTaskSpec(**v) for k,v in payload['TaskSpecs'].items()}
    try:r=job.create_job(ctx,job.models.CreateJobReq(**(payload|{'TaskSpecs':specs})))
    except Exception:
        found=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames=name,Self=True,Limit=100));exact=[x for x in found.Items or [] if x.Name==name]
        if not exact:raise
        r=exact[0]
    receipt=dict(job_id=r.JobId,name=name,run_dir=str(run),gpus=1);write_json(run/'receipt.json',receipt);print(json.dumps(receipt),flush=True)


def fit(run):
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from torch.nn import functional as F
    import lpips
    from source_color import preserve_source_chroma
    from deterministic_quality_ops import replace_pools,resize,verify
    e=json.loads((run/'experiment.json').read_text());parent=Path(e['parent_run'])
    for filename,digest in e['frozen_inputs'].items():
        if sha256(filename)!=digest:raise ValueError('Frozen input changed')
    if os.environ.get('FACE_AGE_EXPERIMENT_ALLOCATION')!='1' or torch.cuda.device_count()!=1:raise ValueError('One dedicated GPU required')
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False;verify()
    base=Path(os.environ['FACE_AGE_BASE']);sys.path.insert(0,str(base/'src'));from sam_npu import load_sam;from sam_losses import IDLoss
    torch.manual_seed(20261125);net=load_sam('cuda:0').eval().requires_grad_(False)
    for index,module in enumerate(net.decoder.convs):
        if 2**(3+index//2)>=64:module.conv.requires_grad_(True);module.activate.requires_grad_(True)
    named={n:q for n,q in net.decoder.named_parameters() if q.requires_grad};initial={n:q.detach().clone() for n,q in named.items()}
    identity=IDLoss().cuda().eval().requires_grad_(False);lp=lpips.LPIPS(net='alex',verbose=False).cuda().eval().requires_grad_(False)
    replace_pools(identity);replace_pools(lp)
    def tensor(rgb):return torch.from_numpy(np.array(rgb,copy=True,order='C')).permute(2,0,1).contiguous()[None].cuda().float()/127.5-1
    def apply(state):
        with torch.no_grad():
            for n,q in named.items():q.copy_(state[n])
    def decode(code):return net(code,input_code=True,input_is_full=True,randomize_noise=False,resize=False)
    def fingerprint():
        h=hashlib.sha256()
        for n,t in net.state_dict().items():
            if n not in {'decoder.'+key for key in named}:h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    frozen=fingerprint();det=cv2.FaceDetectorYN.create(str(base/'models/yunet/face_detection_yunet_2023mar.onnx'),'',(256,256),.6,.3,100);sf=cv2.FaceRecognizerSF.create(str(ROOT/'deployment/gan-research/face_recognition_sface_2021dec.onnx'),'')
    def feature(rgb):
        bgr=cv2.cvtColor(cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2BGR);_,faces=det.detect(bgr)
        if faces is None:return None
        face=max(faces,key=lambda f:f[2]*f[3]);v=sf.feature(sf.alignCrop(bgr,face)).ravel();return v/(np.linalg.norm(v)+1e-12)
    old={(r['name'],r['condition']):r['sha256'] for r in json.loads((parent/'evaluation/report.json').read_text())['outputs'] if r['route']=='original_delta_constrained'}
    # Check decoder/batch/determinism compatibility before spending training steps.
    from io import BytesIO
    preflight=[]
    for row in json.loads((run/'selection.json').read_text()):
        saved=torch.load(parent/'evaluation'/row['name']/'original_delta_constrained_codes.pt',map_location='cuda',weights_only=True)
        with torch.no_grad():baseline_images=decode(saved['codes'])
        for image,label in zip(baseline_images,['same','20','40','60','75']):
            rgb=((image.clamp(-1,1)+1)*127.5).permute(1,2,0).byte().cpu().numpy();buffer=BytesIO();Image.fromarray(rgb).save(buffer,format='PNG')
            if hashlib.sha256(buffer.getvalue()).hexdigest()!=old[row['name'],label]:raise ValueError('Baseline preflight PNG differs: '+row['name']+'/'+label)
        preflight.append(row['name'])
    del baseline_images,saved
    write_json(run/'baseline-preflight.json',dict(cases=preflight,pngs=5*len(preflight),exact=True,training_started=False))
    records=json.loads((run/'selection.json').read_text());out=run/'evaluation';out.mkdir(exist_ok=False);started=time.monotonic()
    report=dict(records=records,routes=['wplus_base','personalized_reconstruction','personalized_locality'],outputs=[],timings=[],baseline_route='original_delta_constrained',reused_baseline_exact=True,comparison='Same frozen original-delta W+ codes, baseline decoder versus two per-photo generator patches. Each source is fitted independently; patch is not shared across subjects.')
    def patches(y):return torch.cat([y[:,:,310:490,220:520],y[:,:,490:670,220:520],y[:,:,310:490,504:804],y[:,:,490:670,504:804]],0)
    for row in records:
        folder=out/row['name'];folder.mkdir();rgb=np.asarray(Image.open(parent/'evaluation'/row['name']/'reference.png').convert('RGB'));Image.fromarray(rgb).save(folder/'reference.png')
        ref=tensor(rgb);x=resize(ref,(256,256));own=feature(rgb)
        saved=torch.load(parent/'evaluation'/row['name']/'original_delta_constrained_codes.pt',map_location='cuda',weights_only=True);codes=saved['codes'];anchor=codes[:1];refs=patches(ref)
        state={};times={}
        for route in ['personalized_reconstruction','personalized_locality']:
            apply(initial);opt=torch.optim.Adam(list(named.values()),lr=1e-4,betas=(0.0,.99));rng=random.Random(20261125);noise_rng=torch.Generator(device='cuda').manual_seed(20261125);tick=time.monotonic();logs=[]
            for step in range(200):
                target=rng.choice([1,2,3,4]);alpha=rng.uniform(.25,1);near=anchor+alpha*(codes[target:target+1]-anchor)+.02*torch.randn(anchor.shape,device='cuda',generator=noise_rng)
                if route=='personalized_locality':
                    current={n:q.detach().clone() for n,q in named.items()};apply(initial)
                    with torch.no_grad():teacher=resize(decode(near),(256,256))
                    apply(current)
                y=decode(anchor);small=resize(y,(256,256))
                pixel=(resize(y,(512,512))-resize(ref,(512,512))).abs().mean()
                perceptual=lp(small,x).mean();ident=(1-identity.cos(small,x)).mean();local=(patches(y)-refs).abs().mean();trust=sum((q-initial[n]).square().mean() for n,q in named.items())
                locality=y.new_zeros(())
                if route=='personalized_locality':
                    near_output=resize(decode(near),(256,256));locality=.8*lp(near_output,teacher).mean()+.2*F.l1_loss(F.avg_pool2d(near_output,4),F.avg_pool2d(teacher,4))
                loss=.5*pixel+.8*perceptual+.8*ident+.15*local+.1*trust+locality
                opt.zero_grad(set_to_none=True);loss.backward()
                if not torch.isfinite(loss) or not all(q.grad is not None and torch.isfinite(q.grad).all() for q in named.values()):raise ValueError('Invalid personalized gradient')
                torch.nn.utils.clip_grad_norm_(list(named.values()),1);opt.step()
                point=dict(step=step+1,target_index=target,alpha=alpha,loss=float(loss.detach()),pixel=float(pixel.detach()),lpips=float(perceptual.detach()),identity=float(ident.detach()),local=float(local.detach()),locality=float(locality.detach()));logs.append(point)
            state[route]={n:q.detach().clone() for n,q in named.items()};times[route]=time.monotonic()-tick
            torch.save({n:q.cpu() for n,q in state[route].items()},folder/(route+'_patch.pt'));write_json(folder/(route+'_train.json'),logs)
            print('PERSONALIZED_TRAINED',row['name'],route,round(times[route],2),flush=True)
        for route in report['routes']:
            apply(initial if route=='wplus_base' else state[route])
            with torch.no_grad():images=decode(codes)
            for image,label,target in zip(images,['same','20','40','60','75'],[row['source_age'],20,40,60,75]):
                if tuple(image.shape)!=(3,1024,1024) or not torch.isfinite(image).all():raise ValueError('Invalid native1024 output')
                native=((image.clamp(-1,1)+1)*127.5).permute(1,2,0).byte().cpu().numpy();path=folder/f'{route}_{label}.png';Image.fromarray(native).save(path);digest=sha256(path)
                if route=='wplus_base' and digest!=old[row['name'],label]:raise ValueError('Baseline PNG differs')
                corrected,_=preserve_source_chroma(rgb,native);color=path.with_name(path.stem+'_color.png');Image.fromarray(corrected).save(color);v=feature(native)
                report['outputs'].append(dict(name=row['name'],cohort=row['cohort'],route=route,condition=label,target=target,native=str(path.relative_to(out)),preserved=str(color.relative_to(out)),sha256=digest,detected=v is not None,source_cosine=float(own@v) if own is not None and v is not None else None))
        report['timings'].append(dict(name=row['name'],**times));write_json(out/'report.partial.json',report)
        apply(initial);del state,images,loss,y,opt
    if fingerprint()!=frozen:raise ValueError('Frozen generator/encoder tensors changed')
    report.update(total_seconds=time.monotonic()-started,cuda_peak_bytes=torch.cuda.max_memory_allocated(),frozen_tensors_sha256=frozen)
    write_json(out/'report.json',report);(run/'DONE').write_text('Per-photo generator pair and native outputs complete\n');print('PERSONALIZED_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True);prep=sub.add_parser('prepare');prep.add_argument('--confirm-hd',action='store_true')
    for action in ['submit','fit']:sub.add_parser(action).add_argument('run',type=Path)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.confirm_hd)
    elif a.action=='submit':submit(a.run.resolve())
    else:fit(a.run.resolve())

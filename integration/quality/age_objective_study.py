"""Frozen-SAM high-age objective ablation; one GPU, no live promotion.

Six inspected development cases isolate four objectives; fixed final recipe and
teacher control continue unchanged on the remaining29 development cases.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from probe_inversion import sha256, write_json

ROOT=Path(os.environ.get('FACE_AGE_REPO',Path(__file__).resolve().parents[2]))
PARENT=ROOT/'deployment/training-runs/original-delta-20261124-160055'
DEV=['ffhq_64376','ffhq_62644','ffhq_64919','ffhq_65817','ffhq_67140','ffhq_60764']
ARMS=['teacher_control','target_age','target_age_order','target_age_order_floor']


def validate(run):
    for file,digest in json.loads((run/'experiment.json').read_text())['frozen_inputs'].items():
        if sha256(file)!=digest:raise ValueError('Frozen input changed: '+file)


def prepare():
    selection=json.loads((PARENT/'selection.json').read_text())
    if len(selection)!=35 or not set(DEV)<={r['name'] for r in selection}:raise ValueError('Expected35 fixed cases')
    run=ROOT/'deployment/training-runs'/('age-objective-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
    (run/'source').mkdir(parents=True,exist_ok=False)
    for name in ['age_objective_study.py','run_age_objective.sh','deterministic_quality_ops.py','probe_inversion.py']:
        shutil.copy2(Path(__file__).with_name(name),run/'source'/name)
    shutil.copy2(ROOT/'integration/source_color.py',run/'source/source_color.py')
    write_json(run/'selection.json',selection)
    files=list((run/'source').iterdir())+[run/'selection.json',PARENT/'evaluation/report.json']
    for row in selection:
        files += [PARENT/'evaluation'/row['name']/name for name in ['reference.png','original_delta_codes.pt','original_delta_constrained_codes.pt']]
    files += [ROOT/'integration/ascend/src'/f for f in ['sam_npu.py','sam_losses.py','faceattr.py']]
    files += [ROOT/'deployment/gan-research/face_recognition_sface_2021dec.onnx']
    frozen={str(p):sha256(p) for p in files}
    for spec in json.loads((ROOT/'integration/quality/baselines/sam-wplus-v1.json').read_text())['weights']:
        path=ROOT/spec['path']
        if sha256(path)!=spec['sha256']:raise ValueError('Original model hash mismatch')
        frozen[str(path)]=spec['sha256']
    write_json(run/'experiment.json',dict(profile='explicit_age_objective_frozen_sam',parent_run=str(PARENT),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),cases=35,
        diagnostic_cases=DEV,diagnostic_arms=ARMS,confirmation_arms=['teacher_control','target_age_order_floor'],
        frozen_inputs=frozen,steps=180,batch=2,targets=[60,75],seed=20261126,
        common_recipe=dict(all_networks_frozen=True,anchor='Fixed accepted source W+',layers='W+0:8',
            learning_rate=.012,lr_final_ratio=.2,identity=.8,age_KL=.5,lpips=.6,delta_L2=.05,color=.1,
            numerics='Deterministic resize/pool backward, CUDA deterministic, TF32 convolution only'),
        objective_recipe=dict(explicit_target='Barycentric distribution at unchanged60/75 over FairFace representatives;75 is top75 bin',
            order='Additional.25*relu((.075-(expected_age75-expected_age60))/.1)',
            floor='Identity hinge at same-target teacher-control ArcFace cosine minus.01, coefficient.8',
            inactive='Target==source fixed at source anchor; same/20/40 PNG protected exactly'),
        stability=dict(source_anchor_steps=200,repeats=2,source='ffhq_64376',require_equal_codes_and_png=True),
        criteria=dict(identity_drop_max=.02,age_span_min_ratio=.9,missing_faces_increase_max=0,
            high_age_error_must_improve=True,high_age60to75_must_not_shrink=True,protected_same_young_exact=True,
            native_review_required=True,no_automatic_promotion=True),resources=dict(gpu=1,cpu=4,memory='48Gi',max_running='45m'),
        caveats=['Previously inspected35 development cases, not a new blinded test.',
            'Independent MiVOLO is evaluation only; fixed recipe, no tuning against its gradients.',
            'Six diagnostic cases plus remaining29 confirmation, final recipe chosen before results.',
            'Two-target fitting differs from historical batch4 and live sequential fitting; compare same-budget teacher control.',
            'Pseudo-age head ending at75 provides no80/90 calibration.',
            'Classifier objective alone does not guarantee natural elderly skin.']))
    print('PREPARED',run,flush=True)


def submit(run):
    validate(run)
    if (run/'receipt.json').exists():raise ValueError('Already submitted')
    sys.path.insert(0,str(ROOT/'integration'));from deploy_yicloud import client,IMAGE
    job,ctx=client()
    active=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames='face-age',Self=True,Limit=100,Phases='Pending,Running,Inqueue,Starting,Stopping'))
    if active.Items:raise RuntimeError('One-card limit: another face-age job is active')
    name='face-age-'+run.name
    payload=dict(Name=name,Project='fdj-infra',ZoneId='cn-shanghai-a',QuotaGroup='fdj_gpu_taskpool',Priority='3',UsePrivateMachine=False,AutoDeleteDuration='168h',
        TaskSpecs={'study':dict(Command=['bash',str(run/'source/run_age_objective.sh'),str(ROOT),str(run)],Env={},Image=IMAGE,Replicas=1,CPU='4',Memory='48Gi',GPU='1',Internet=False,MaxRunningDuration='45m',MaxWaitDuration='10m',RestartPolicy='Never',MountGPFS='gpfs://gpfs2/haobinzhang:/root')})
    write_json(run/'request.json',payload)
    try:result=job.create_job(ctx,job.models.CreateJobReq(**(payload|{'TaskSpecs':{k:job.models.CreateJobReqTaskSpec(**v) for k,v in payload['TaskSpecs'].items()}})))
    except Exception:
        found=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames=name,Self=True,Limit=100))
        exact=[x for x in found.Items or [] if x.Name==name]
        if not exact:raise
        result=exact[0]
    receipt=dict(job_id=result.JobId,name=name,run_dir=str(run),gpus=1)
    write_json(run/'receipt.json',receipt);print(json.dumps(receipt),flush=True)


def worker(run):
    validate(run)
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from torch.nn import functional as F
    import lpips
    from deterministic_quality_ops import resize,replace_pools,verify
    from source_color import preserve_source_chroma
    if os.environ.get('FACE_AGE_EXPERIMENT_ALLOCATION')!='1' or torch.cuda.device_count()!=1:raise ValueError('Exactly one allocated GPU required')
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(20261126)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    write_json(run/'numeric-gradient-check.json',verify())
    write_json(run/'device.json',dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),device_count=torch.cuda.device_count(),memory_bytes=torch.cuda.get_device_properties(0).total_memory))
    base=Path(os.environ['FACE_AGE_BASE']);sys.path.insert(0,str(base/'src'))
    from sam_npu import load_sam
    from sam_losses import IDLoss,AgeEst
    net=load_sam('cuda:0').eval().requires_grad_(False)
    fine=deepcopy(net.encoder).eval().requires_grad_(False)
    fine.load_state_dict(torch.load(base/'out/ft_l1/encoder_step1000.pt',map_location='cpu',weights_only=True))
    identity=IDLoss().cuda().eval().requires_grad_(False);age=AgeEst().cuda().eval().requires_grad_(False)
    perceptual=lpips.LPIPS(net='alex',verbose=False).cuda().eval().requires_grad_(False)
    for model in [identity,age,perceptual]:replace_pools(model)
    def fingerprint():
        h=hashlib.sha256()
        for prefix,model in [('net',net),('fine',fine),('identity',identity),('age',age),('lp',perceptual)]:
            for name,value in model.state_dict().items():
                h.update((prefix+name).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    frozen=fingerprint()
    def tensor(rgb):return torch.from_numpy(np.array(rgb,copy=True,order='C')).permute(2,0,1)[None].cuda().float()/127.5-1
    def decode(c):return net(c,input_code=True,input_is_full=True,randomize_noise=False,resize=False)
    def logits(y):
        x=resize((y+1)/2,(224,224),kind='bilinear')
        return age.net.age(age.net.b((x-age.m)/age.sd)).float()
    def native(y):return ((y.clamp(-1,1)+1)*127.5).permute(1,2,0).byte().cpu().numpy()
    def digest(rgb):
        buf=BytesIO();Image.fromarray(rgb).save(buf,format='PNG');return hashlib.sha256(buf.getvalue()).hexdigest()
    def load(row):
        folder=PARENT/'evaluation'/row['name'];rgb=np.asarray(Image.open(folder/'reference.png').convert('RGB'))
        x=tensor(cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA))
        raw=torch.load(folder/'original_delta_codes.pt',map_location='cuda',weights_only=True)['codes']
        accepted=torch.load(folder/'original_delta_constrained_codes.pt',map_location='cuda',weights_only=True)['codes']
        if tuple(raw.shape)!=(5,18,512) or tuple(accepted.shape)!=(5,18,512):raise ValueError('Code shape mismatch')
        return rgb,x,raw,accepted
    def fit_anchor(rgb,x,source):
        condition=torch.cat([x,torch.full_like(x[:,:1],source/100)],1)
        with torch.no_grad():origin=net.pretrained_encoder(x)+net.latent_avg+fine(condition);own=identity.feats(x).detach()
        ref=tensor(cv2.resize(rgb,(512,512),interpolation=cv2.INTER_AREA))
        grid=torch.linspace(0,1,512,device='cuda');yy,xx=torch.meshgrid(grid,grid,indexing='ij')
        mask=(.2+.8*((1-((xx-.5)/.31)**2-((yy-.53)/.36)**2)/.25).clamp(0,1))[None,None]
        pivot=origin.clone().requires_grad_(True);opt=torch.optim.Adam([pivot],lr=.02)
        for step in range(200):
            opt.param_groups[0]['lr']=.02*(.2+.8*(1-step/200))
            y=decode(pivot);small=resize(y,(256,256));large=resize(y,(512,512))
            pixel=((large-ref).abs()*mask).sum()/(mask.sum()*3);ident=(1-(identity.feats(small)*own).sum(1)).mean()
            loss=.5*pixel+.8*perceptual(small,x).mean()+.8*ident+.002*(pivot-origin).square().mean()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            if not torch.isfinite(loss):raise ValueError('Nonfinite anchor fit')
        return pivot.detach()
    mid=age.mid.flatten();targets=torch.tensor([.60,.75],device='cuda');distribution=torch.zeros(2,9,device='cuda')
    for i,t in enumerate(targets):
        hi=int(torch.searchsorted(mid,t).item());lo=max(0,hi-1);hi=min(hi,len(mid)-1)
        if lo==hi:distribution[i,lo]=1
        else:
            ratio=(t-mid[lo])/(mid[hi]-mid[lo]);distribution[i,lo]=1-ratio;distribution[i,hi]=ratio
    if not torch.allclose(distribution@mid,targets,atol=1e-7):raise ValueError('Target expectation differs')
    write_json(run/'target-distribution.json',dict(targets=[60,75],representatives=(mid*100).tolist(),probabilities=distribution.tolist()))

    def refine(row,x,raw,arm,floor=None,diagnostic=False):
        started=time.monotonic();source=row['source_age']
        with torch.no_grad():
            own=identity.feats(x).expand(2,-1).detach();cond=torch.cat([x,torch.full_like(x[:,:1],source/100)],1)
            origin=net.pretrained_encoder(x)+net.latent_avg+fine(cond);direction=raw[3:]-raw[:1]
            prior=resize(decode(raw[3:]),(256,256));teacher=resize(decode(origin+direction),(256,256))
            teacher_prob=logits(teacher).softmax(1);color=F.avg_pool2d(x,32).expand(2,-1,-1,-1)
        active=torch.tensor([60!=source,75!=source],device='cuda')[:,None,None]
        delta=torch.zeros_like(raw[3:,:8]).requires_grad_(True);optimizer=torch.optim.Adam([delta],lr=.012)
        trace=[];gradient_report=None
        for step in range(180):
            optimizer.param_groups[0]['lr']=.012*(.2+.8*(1-step/180))
            correction=torch.cat([delta*active,torch.zeros_like(raw[3:,8:])],1);y=resize(decode(raw[3:]+correction),(256,256))
            sim=(identity.feats(y)*own).sum(1);ident=(1-sim).mean() if floor is None else (floor-sim).clamp_min(0).mean()
            logit=logits(y);prob=logit.softmax(1);pred=prob@mid
            age_loss=F.kl_div(logit.log_softmax(1),teacher_prob if arm=='teacher_control' else distribution,reduction='batchmean')
            order=.25*((.075-(pred[1]-pred[0]))/.1).clamp_min(0) if 'order' in arm else pred.sum()*0
            terms=dict(identity=.8*ident,age=.5*age_loss,perceptual=.6*perceptual(y,prior).mean(),
                trust=.05*(delta*active).square().mean(),color=.1*F.l1_loss(F.avg_pool2d(y,32),color),order=order)
            if diagnostic and step==0:
                gradients={k:torch.autograd.grad(v,delta,retain_graph=True)[0].flatten() for k,v in terms.items()}
                g_age=gradients['age'];direction_flat=direction[:,:8].flatten()
                def cosine(a,b):return float((a@b)/(a.norm()*b.norm()).clamp_min(1e-12))
                gradient_report={k:dict(norm=float(g.norm()),cosine_with_age_gradient=cosine(g,g_age),
                    descent_cosine_with_sam_age_displacement=cosine(-g,direction_flat)) for k,g in gradients.items()}
            loss=sum(terms.values());optimizer.zero_grad(set_to_none=True);loss.backward()
            if not torch.isfinite(loss) or not torch.isfinite(delta.grad).all():raise ValueError('Nonfinite objective gradient')
            optimizer.step()
            if step in [0,59,119,179]:trace.append(dict(step=step+1,loss=float(loss.detach()),
                expected_age=(pred.detach()*100).tolist(),identity_cosine=sim.detach().tolist(),
                **{'loss_'+k:float(v.detach()) for k,v in terms.items()}))
        code=raw[3:]+torch.cat([delta.detach()*active,torch.zeros_like(raw[3:,8:])],1)
        with torch.no_grad():y=resize(decode(code),(256,256));sim=(identity.feats(y)*own).sum(1);final_age=(logits(y).softmax(1)@mid*100).tolist()
        torch.cuda.synchronize()
        return code.detach(),dict(arm=arm,seconds=time.monotonic()-started,trace=trace,teacher_age=(teacher_prob@mid*100).tolist(),
            final_shared_age=final_age,final_identity=sim.tolist(),identity_floor=floor.tolist() if floor is not None else None,initial_gradient_terms=gradient_report)

    rows=json.loads((run/'selection.json').read_text());previous=json.loads((PARENT/'evaluation/report.json').read_text())
    old={(o['name'],o['condition']):o['sha256'] for o in previous['outputs'] if o['route']=='original_delta_constrained'}
    preflight=[]
    for row in rows:
        rgb,x,raw,accepted=load(row)
        with torch.no_grad():images=decode(accepted)
        for y,label in zip(images,['same','20','40','60','75']):
            if digest(native(y))!=old[row['name'],label]:raise ValueError('Baseline decode mismatch '+row['name']+'/'+label)
        preflight.append(row['name'])
    write_json(run/'baseline-preflight.json',dict(cases=preflight,pngs=175,exact=True));print('AGE_OBJECTIVE_BASELINE_OK',flush=True)
    row=next(r for r in rows if r['name']=='ffhq_64376');rgb,x,raw,accepted=load(row);repeats=[]
    for repeat in range(2):
        tick=time.monotonic();anchor=fit_anchor(rgb,x,row['source_age']);own_raw=raw.clone()
        own_raw[0]=anchor[0];own_raw[1:]=anchor+raw[1:]-raw[:1]
        codes,_=refine(row,x,own_raw,'teacher_control')
        with torch.no_grad():pixels=[native(y) for y in decode(torch.cat([anchor,codes]))]
        repeats.append(dict(anchor=anchor,codes=codes,pngs=[digest(im) for im in pixels],seconds=time.monotonic()-tick))
    stable=dict(repeats=2,anchor_codes_equal=torch.equal(repeats[0]['anchor'],repeats[1]['anchor']),
        target_codes_equal=torch.equal(repeats[0]['codes'],repeats[1]['codes']),pngs_equal=repeats[0]['pngs']==repeats[1]['pngs'],
        png_sha256=repeats[0]['pngs'],seconds=[r['seconds'] for r in repeats])
    write_json(run/'repeatability.json',stable)
    if not all(stable[k] for k in ['anchor_codes_equal','target_codes_equal','pngs_equal']):raise ValueError('Repeatability failed')
    del repeats;print('AGE_OBJECTIVE_REPEATABLE',flush=True)
    detector=cv2.FaceDetectorYN.create(str(base/'models/yunet/face_detection_yunet_2023mar.onnx'),'',(256,256),.6,.3,100)
    recognizer=cv2.FaceRecognizerSF.create(str(ROOT/'deployment/gan-research/face_recognition_sface_2021dec.onnx'),'')
    def feature(rgb):
        bgr=cv2.cvtColor(cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2BGR);_,faces=detector.detect(bgr)
        if faces is None:return None
        v=recognizer.feature(recognizer.alignCrop(bgr,max(faces,key=lambda f:f[2]*f[3]))).ravel();return v/(np.linalg.norm(v)+1e-12)
    out=run/'evaluation';out.mkdir(exist_ok=False)
    report=dict(records=rows,routes=['accepted_baseline']+ARMS,outputs=[],timings=[],model_weights_frozen=True,default_model_changed=False,
        selection_note='All35: accepted, teacher_control, target_age_order_floor; intermediate ablations only fixed six cases.')
    for row in sorted(rows,key=lambda r:(r['name'] not in DEV,DEV.index(r['name']) if r['name'] in DEV else r['name'])):
        rgb,x,raw,accepted=load(row);folder=out/row['name'];folder.mkdir();Image.fromarray(rgb).save(folder/'reference.png');own=feature(rgb)
        control,control_meta=refine(row,x,raw,'teacher_control',diagnostic=row['name'] in DEV)
        floor=torch.tensor(control_meta['final_identity'],device='cuda')-.01
        variants=[('accepted_baseline',accepted,None),('teacher_control',control,control_meta)]
        for arm in (ARMS[1:] if row['name'] in DEV else ['target_age_order_floor']):
            code,meta=refine(row,x,raw,arm,floor=floor if arm.endswith('_floor') else None,diagnostic=row['name'] in DEV)
            variants.append((arm,code,meta))
        for route,codes,meta in variants:
            code=accepted.clone()
            if route=='accepted_baseline':code=codes
            else:code[3:]=codes
            with torch.no_grad():images=decode(code)
            for image,label,target in zip(images,['same','20','40','60','75'],[row['source_age'],20,40,60,75]):
                pixels=native(image);path=folder/(route+'_'+label+'.png');Image.fromarray(pixels).save(path);value=sha256(path)
                if (route=='accepted_baseline' or label in ['same','20','40'] or target==row['source_age']) and value!=old[row['name'],label]:raise ValueError('Protected PNG changed')
                colored,_=preserve_source_chroma(rgb,pixels);preserved=path.with_name(path.stem+'_color.png');Image.fromarray(colored).save(preserved);f=feature(pixels)
                report['outputs'].append(dict(name=row['name'],cohort=row['cohort'],route=route,condition=label,target=target,native=str(path.relative_to(out)),
                    preserved=str(preserved.relative_to(out)),sha256=value,source_cosine=float(own@f) if own is not None and f is not None else None,detected=f is not None))
            torch.save(dict(codes=code.detach().cpu(),conditions=['same',20,40,60,75]),folder/(route+'_codes.pt'))
            if meta:report['timings'].append(dict(name=row['name'],**meta))
        write_json(out/'report.partial.json',report);print('AGE_OBJECTIVE_CASE',row['name'],len(report['outputs']),flush=True)
    after=fingerprint();write_json(run/'frozen-verification.json',dict(before=frozen,after=after,equal=frozen==after))
    if frozen!=after:raise ValueError('Frozen model state changed')
    report.update(total_seconds=sum(t['seconds'] for t in report['timings']),cuda_peak_bytes=torch.cuda.max_memory_allocated())
    write_json(out/'report.json',report);(run/'DONE').write_text('Frozen age objective study complete\n');print('AGE_OBJECTIVE_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','submit','worker']);p.add_argument('run',type=Path,nargs='?');args=p.parse_args()
    if args.action=='prepare':prepare()
    elif args.run is None:p.error('run required')
    elif args.action=='submit':submit(args.run.resolve())
    else:worker(args.run.resolve())

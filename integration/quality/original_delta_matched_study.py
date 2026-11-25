"""Pair generator training at old/new age-displacement codes; serve neither automatically."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from probe_inversion import sha256, write_json

ROOT = Path(os.environ.get('FACE_AGE_REPO', Path(__file__).resolve().parents[2]))
PARENT = ROOT / 'deployment/training-runs/original-delta-20261124-160055'
NATIVE = ROOT / 'deployment/training-runs/native-full-20261124-141846'
OLD = ROOT / 'deployment/training-runs/matched-latent-20261124-115525'


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def validate_frozen(run):
    for filename, expected in json.loads((run / 'experiment.json').read_text())['frozen_inputs'].items():
        if sha256(filename) != expected:
            raise ValueError('Frozen input changed: ' + filename)


def prepare():
    data = read(NATIVE / 'data/train-resolved.jsonl')
    if any(not isinstance(r.get('source_age'),int) or not 0<=r['source_age']<=100 for r in data):
        raise ValueError('Frozen FairFace source ages must be resolved before submission')
    old_index = json.loads((OLD / 'train-codes/index.json').read_text())
    rng = random.Random(20261125)
    selected, groups = [], set()
    for low, count in [(70, 20), (50, 20), (20, 8), (30, 6), (40, 6)]:
        pool = [r for r in data if r['age_interval'][0] == low]
        rng.shuffle(pool)
        pool.sort(key=lambda r: (r['name'] not in old_index['items'], r.get('published_demographic_label') != 'Asian'))
        chosen = []
        for row in pool:
            if row['identity_group'] in groups:
                continue
            chosen.append(row)
            groups.add(row['identity_group'])
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError('Insufficient audited groups')
        selected += chosen
    evaluation = json.loads((PARENT / 'selection.json').read_text())
    held = read(NATIVE / 'data/eval.jsonl') + read(NATIVE / 'data/external.jsonl')
    if groups & {r['identity_group'] for r in held + evaluation}:
        raise ValueError('Protected group overlap')
    if {r['sha256'] for r in selected} & {r['sha256'] for r in held + evaluation}:
        raise ValueError('Protected pixel overlap')
    run = ROOT / 'deployment/training-runs' / ('original-delta-match-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
    (run / 'source').mkdir(parents=True, exist_ok=False)
    (run / 'data').mkdir()
    for name in ['original_delta_matched_study.py', 'run_original_delta_match.sh', 'train_highres_texture.py', 'evaluate_highres_texture.py', 'deterministic_quality_ops.py', 'probe_inversion.py']:
        shutil.copy2(Path(__file__).with_name(name), run / 'source' / name)
    shutil.copy2(ROOT / 'integration/latent_quality.py', run / 'source/latent_quality.py')
    shutil.copy2(ROOT / 'integration/source_color.py', run / 'source/source_color.py')
    (run / 'data/train.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected))
    for split in ['eval', 'external']:
        shutil.copy2(NATIVE / 'data' / (split + '.jsonl'), run / 'data' / (split + '.jsonl'))
    write_json(run / 'selection.json', evaluation)
    reusable = {}
    frozen = {}
    for row in selected:
        for key, digest_key in [('image_path', 'sha256'), ('reference_path', 'reference_sha256')]:
            if sha256(row[key]) != row[digest_key]:
                raise ValueError('Training input changed')
            frozen[row[key]] = row[digest_key]
        spec = old_index['items'].get(row['name'])
        if spec and spec['source_age'] == row['source_age'] and spec['input_sha256'] == row['reference_sha256']:
            path = OLD / 'train-codes' / spec['file']
            if sha256(path) != spec['sha256']:
                raise ValueError('Anchor cache changed')
            reusable[row['name']] = dict(path=str(path), sha256=spec['sha256'])
            frozen[str(path)] = spec['sha256']
    write_json(run / 'reusable-anchors.json', reusable)
    for spec in json.loads((ROOT / 'integration/quality/baselines/sam-wplus-v1.json').read_text())['weights']:
        path = ROOT / spec['path']
        if sha256(path) != spec['sha256']:
            raise ValueError('Original checkpoint changed')
        frozen[str(path)] = spec['sha256']
    files = list((run / 'source').iterdir()) + list((run / 'data').iterdir()) + [run / 'selection.json', run / 'reusable-anchors.json', PARENT / 'evaluation/report.json', NATIVE / 'data/train-resolved.jsonl']
    for row in evaluation:
        files += [PARENT / 'evaluation' / row['name'] / 'original_delta_constrained_codes.pt', PARENT / 'evaluation' / row['name'] / 'reference.png']
    files += [ROOT / 'integration/ascend/src' / n for n in ['sam_npu.py', 'sam_losses.py', 'faceattr.py']]
    files += [ROOT / 'deployment/gan-research/face_recognition_sface_2021dec.onnx']
    frozen.update({str(path): sha256(path) for path in files})
    exp = dict(profile='original_delta_matched_generator_pair', train=len(selected), selection_count=len(evaluation), steps=2000,
        train_age_counts=dict(Counter(str(r['age_interval']) for r in selected)), reused_anchors=len(reusable),
        generator_routes={'ft_delta_training': 'control', 'original_delta_training': 'patch_gan'},
        fixed_code_run=str(PARENT), fixed_code_filename='original_delta_constrained_codes.pt', baseline_route='original_delta_constrained',
        frozen_inputs=frozen, code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        criteria=dict(identity_drop_max=.02, age_span_min_ratio=.9, missing_faces_increase_max=0, native_review_required=True, no_automatic_promotion=True),
        resources=dict(gpu=1, cpu=4, memory='48Gi', max_running='75m'),
        hypothesis='Only paired difference: generator training codes use ft_l1 versus original SAM age displacement. Same fitted source anchors, simultaneous four-code180step optimization, common RGB-independent grayscale/highpass patch GAN, identical sampling and losses. Evaluate both on frozen original_delta_constrained codes.',
        common_recipe=dict(source_anchor_steps=200, target_steps=180, training_targets=[60,75], fitting_batch=4, generator_min_resolution=64, lr=5e-5, patch_gan_weight=.1, patch_edge=256, teacher_size=32, teacher_weight=.05, age_preservation=.5, age_conditioned=False),
        limitations=['60-image mechanism study, not population adaptation or a new blind test.', 'Source ages remain FairFace pseudo labels; no replacement chronology or85+ supervision.', 'Reference35cases previously inspected; FFHQ pretraining overlap remains.', 'Both old/new target codes are fitted simultaneously in batch4; production requests fit targets sequentially.', 'Generator trained on fixed baseline codes; no candidate-specific latent refitting.'])
    write_json(run / 'experiment.json', exp)
    print('PREPARED', run, flush=True)
    return run


def submit(run):
    validate_frozen(run)
    if (run / 'receipt.json').exists():
        raise ValueError('Already submitted')
    sys.path.insert(0, str(ROOT / 'integration'))
    from deploy_yicloud import client, IMAGE
    job, ctx = client()
    active = job.list_jobs(ctx, job.models.ListJobsReq(Project='fdj-infra', ZoneId='cn-shanghai-a', JobNames='face-age', Self=True, Limit=100, Phases='Pending,Running,Inqueue,Starting,Stopping'))
    if active.Items:
        raise RuntimeError('One-card limit; active project job exists')
    name = 'face-age-' + run.name
    payload = dict(Name=name, Project='fdj-infra', ZoneId='cn-shanghai-a', QuotaGroup='fdj_gpu_taskpool', Priority='3', UsePrivateMachine=False, AutoDeleteDuration='168h',
        TaskSpecs={'pair': dict(Command=['bash', str(run / 'source/run_original_delta_match.sh'), str(ROOT), str(run)], Env={}, Image=IMAGE, Replicas=1, CPU='4', Memory='48Gi', GPU='1', Internet=False, MaxRunningDuration='75m', MaxWaitDuration='10m', RestartPolicy='Never', MountGPFS='gpfs://gpfs2/haobinzhang:/root')})
    write_json(run / 'request.json', payload)
    specs = {k: job.models.CreateJobReqTaskSpec(**v) for k,v in payload['TaskSpecs'].items()}
    try:
        result = job.create_job(ctx, job.models.CreateJobReq(**(payload | {'TaskSpecs': specs})))
    except Exception:
        found = job.list_jobs(ctx, job.models.ListJobsReq(Project='fdj-infra', ZoneId='cn-shanghai-a', JobNames=name, Self=True, Limit=100))
        exact = [x for x in found.Items or [] if x.Name == name]
        if not exact:
            raise
        result = exact[0]
    receipt = dict(job_id=result.JobId, name=name, run_dir=str(run), gpus=1)
    write_json(run / 'receipt.json', receipt)
    print(json.dumps(receipt), flush=True)


def fit(run):
    validate_frozen(run)
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from torch.nn import functional as F
    from deterministic_quality_ops import replace_pools, resize, verify
    from latent_quality import LatentQuality
    if os.environ.get('FACE_AGE_EXPERIMENT_ALLOCATION') != '1' or torch.cuda.device_count() != 1:
        raise ValueError('One dedicated GPU required')
    torch.set_num_threads(4); cv2.setNumThreads(2); torch.manual_seed(20261124)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=True; torch.backends.cuda.matmul.allow_tf32=False; verify()
    base = Path(os.environ['FACE_AGE_BASE']); sys.path.insert(0, str(base / 'src'))
    from sam_npu import load_sam
    from sam_losses import AgeEst
    net = load_sam('cuda:0').eval().requires_grad_(False)
    fine = deepcopy(net.encoder).eval().requires_grad_(False)
    fine.load_state_dict(torch.load(base / 'out/ft_l1/encoder_step1000.pt', map_location='cpu', weights_only=True))
    age = AgeEst().cuda().eval().requires_grad_(False); replace_pools(age)
    def tensor(rgb):
        return torch.from_numpy(np.array(rgb, copy=True, order='C')).permute(2,0,1).contiguous()[None].cuda().float()/127.5-1
    helper = LatentQuality(SimpleNamespace(torch=torch, dev='cuda:0', net=net, identity_encoder=fine, age_net=age, tensor=tensor), 'original_delta', resize=resize)
    replace_pools(helper.identity); replace_pools(helper.perceptual)
    def decode(code):
        return net(code, input_code=True, input_is_full=True, randomize_noise=False, resize=False)
    def logits(image):
        value=resize((image+1)/2,(224,224),kind='bilinear')
        return age.net.age(age.net.b((value-age.m)/age.sd)).float()
    def fingerprint():
        h=hashlib.sha256()
        for prefix, model in [('net',net),('fine',fine)]:
            for name, value in model.state_dict().items():
                h.update((prefix+name).encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    before=fingerprint(); reusable=json.loads((run/'reusable-anchors.json').read_text())
    folders={route:run/('train-codes-'+route) for route in ['ft','original']}
    indexes={route:dict(items={},conditions=['same','60','75'],recipe='Same200step anchor; both directions fitted simultaneously180steps, batch4=[ft60,ft75,original60,original75]') for route in folders}
    for folder in folders.values():folder.mkdir(exist_ok=False)
    rows=read(run/'data/train.jsonl');started=time.monotonic()
    for row in rows:
        t=time.monotonic(); rgb=np.asarray(Image.open(row['reference_path']).convert('RGB')); small=cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA)
        x=tensor(small); source=row['source_age']; cond=lambda n:torch.cat([x,torch.full_like(x[:,:1],n/100)],1)
        with torch.no_grad():
            raw_source=net.pretrained_encoder(x)+net.latent_avg+fine(cond(source)); ids=helper.identity.feats(x).expand(4,-1)
        if row['name'] in reusable:
            saved=torch.load(reusable[row['name']]['path'],map_location='cuda',weights_only=True)
            if saved['conditions']!=['same','20','40','60','75'] or tuple(saved['codes'].shape)!=(5,18,512):raise ValueError('Invalid cached anchor')
            anchor=saved['codes'][:1];del saved
        else:
            item=SimpleNamespace(aligned=small,reference_rgb=rgb,reference_age=source,quality_state=None)
            helper.run(item,source,lambda:None);anchor=item.quality_state['anchor'];del item
        with torch.no_grad():
            directions=[]
            for encoder in [fine,net.encoder]:
                own=encoder(cond(source))
                directions.extend(encoder(cond(target))-own for target in [60,75])
            direction=torch.cat(directions);pivot=anchor+direction
            prior=resize(decode(pivot),(256,256));prob=logits(resize(decode(raw_source+direction),(256,256))).softmax(1)
            color=F.avg_pool2d(x,32).expand(4,-1,-1,-1)
        active=torch.tensor([n!=source for n in [60,75,60,75]],device='cuda')[:,None,None]
        correction=torch.zeros_like(pivot[:,:8]).requires_grad_(True);opt=torch.optim.Adam([correction],lr=.012)
        for step in range(180):
            opt.param_groups[0]['lr']=.012*(.2+.8*(1-step/180))
            edited=resize(decode(pivot+torch.cat([correction*active,torch.zeros_like(pivot[:,8:])],1)),(256,256))
            ident=(1-(helper.identity.feats(edited)*ids).sum(1)).mean()
            loss=.8*ident+.5*F.kl_div(logits(edited).log_softmax(1),prob,reduction='batchmean')+.6*helper.perceptual(edited,prior).mean()+.05*(correction*active).square().mean()+.1*F.l1_loss(F.avg_pool2d(edited,32),color)
            opt.zero_grad(set_to_none=True);loss.backward()
            if not torch.isfinite(loss) or not torch.isfinite(correction.grad).all():raise ValueError('Nonfinite matched fit')
            opt.step()
        result=(pivot+torch.cat([correction.detach()*active,torch.zeros_like(pivot[:,8:])],1)).detach()
        for route, first in [('ft',0),('original',2)]:
            codes=torch.cat([anchor,result[first:first+2]])
            for i,target in enumerate([60,75],1):
                if target==source:codes[i].copy_(anchor[0])
            path=folders[route]/(row['name']+'.pt');torch.save(dict(codes=codes.cpu(),conditions=['same','60','75']),path)
            indexes[route]['items'][row['name']]=dict(file=path.name,sha256=sha256(path),input_sha256=row['reference_sha256'],source_age=source)
            write_json(folders[route]/'index.json',indexes[route])
        print('DIRECTION_CODES',len(indexes['ft']['items']),'/',len(rows),row['name'],round(time.monotonic()-t,2),flush=True)
        del loss,edited,correction,opt,result,pivot,prior,prob,direction,anchor
    after=fingerprint()
    if before!=after:raise ValueError('Frozen model changed during cache fitting')
    for route in folders:
        indexes[route].update(count=len(rows),seconds=time.monotonic()-started,frozen_model_sha256=after,cuda_peak_bytes=torch.cuda.max_memory_allocated())
        write_json(folders[route]/'index.json',indexes[route]);(folders[route]/'DONE').write_text('Training-only direction cache complete\n')
    print('DIRECTION_CACHE_DONE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    sub.add_parser('prepare')
    for name in ['submit','fit','verify']:
        sub.add_parser(name).add_argument('run',type=Path)
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='submit':submit(args.run.resolve())
    elif args.action=='fit':fit(args.run.resolve())
    else:validate_frozen(args.run.resolve());print('FROZEN_INPUTS_VERIFIED')

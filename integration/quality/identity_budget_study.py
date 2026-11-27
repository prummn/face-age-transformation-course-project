"""Frozen-SAM pilot: explicit aging with an enforced ArcFace budget.

Matched soft/hard arms start at the same accepted W+ codes. SFace and MiVOLO
are evaluation only. No recipe changes or automatic live promotion.
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

ROOT = Path(os.environ.get('FACE_AGE_REPO', Path(__file__).resolve().parents[2]))
PARENT = ROOT / 'deployment/training-runs/original-delta-20261124-160055'
CASES = ['ffhq_64376', 'ffhq_62644', 'ffhq_64919', 'ffhq_65817', 'ffhq_67140', 'ffhq_60764']
ROUTES = ['accepted_baseline', 'warm_soft_budget', 'hard_identity_budget']


def validate(run):
    for name, digest in json.loads((run / 'experiment.json').read_text())['frozen_inputs'].items():
        if sha256(name) != digest:
            raise ValueError('Frozen input changed: ' + name)


def prepare():
    selection = json.loads((PARENT / 'selection.json').read_text())
    rows = [next(r for r in selection if r['name'] == name) for name in CASES]
    run = ROOT / 'deployment/training-runs' / ('identity-budget-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
    (run / 'source').mkdir(parents=True, exist_ok=False)
    for name in ['identity_budget_study.py', 'run_identity_budget.sh', 'deterministic_quality_ops.py', 'probe_inversion.py']:
        shutil.copy2(Path(__file__).with_name(name), run / 'source' / name)
    shutil.copy2(ROOT / 'integration/source_color.py', run / 'source/source_color.py')
    write_json(run / 'selection.json', rows)
    files = list((run / 'source').iterdir()) + [run / 'selection.json', PARENT / 'evaluation/report.json']
    files += [ROOT / 'integration/ascend/src' / name for name in ['sam_npu.py', 'sam_losses.py', 'faceattr.py']]
    files += [ROOT / 'deployment/gan-research/face_recognition_sface_2021dec.onnx']
    for row in rows:
        files += [PARENT / 'evaluation' / row['name'] / name for name in
                  ['reference.png', 'original_delta_codes.pt', 'original_delta_constrained_codes.pt']]
    frozen = {str(path): sha256(path) for path in files}
    for spec in json.loads((ROOT / 'integration/quality/baselines/sam-wplus-v1.json').read_text())['weights']:
        path = ROOT / spec['path']
        if sha256(path) != spec['sha256']:
            raise ValueError('Model hash mismatch')
        frozen[str(path)] = spec['sha256']
    write_json(run / 'experiment.json', dict(
        profile='explicit_age_hard_identity_budget_pilot', parent_run=str(PARENT), cases=len(rows),
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        routes=ROUTES, diagnostic_cases=CASES, targets=[60, 75], steps=180, seed=20261127,
        frozen_inputs=frozen,
        recipe=dict(initialization='Accepted baseline high-age codes, same source anchor and original SAM direction',
                    variable='First8 W+ layers only', learning_rate=.012, lr_final_ratio=.2,
                    identity_hinge=.8, arcface_budget=.01, age_KL=.5, order=.25,
                    order_interval_years=7.5, lpips=.6, trust=.05, color=.1,
                    checkpoint='Every10 steps plus step0; lowest full batch objective; quantized native check',
                    hard_update='Project Adam proposal onto linearized per-target ArcFace budget, then nonlinear backtracking',
                    backtracking=[1, .5, .25, .125, .0625, .03125, .015625, .0078125],
                    hard_acceptance='Each target float ArcFace >= baseline_float-.01; otherwise retain previous code',
                    hard_output='Both float and quantized PNG ArcFace budgets must pass; fallback to baseline if necessary'),
        criteria=dict(high_age_identity_drop_max=.02, relative_age_span_min=.9,
                      mean_high_age_error_improvement_min=1., high_age60to75_must_not_shrink=True,
                      detection_must_not_drop=True, native_visual_review_required=True,
                      same_young_exact=True, no_automatic_promotion=True),
        resources=dict(gpu=1, cpu=4, memory='48Gi', max_running='30m'),
        caveats=['Six previously inspected development cases, not a blind test.',
                 'ArcFace budget is a proxy constraint, not a guarantee of human identity/naturalness.',
                 'SFace and MiVOLO are evaluation only; do not optimize/select steps with them.',
                 'Age targets unchanged; no80/90 calibration; no fresh reserve consumed.',
                 'A failed development gate stops larger confirmation.',
                 'Both arms change initialization relative to the previous raw-start study; compare matched arms.']))
    print('PREPARED', run, flush=True)


def submit(run):
    validate(run)
    if (run / 'receipt.json').exists():
        raise ValueError('Already submitted')
    sys.path.insert(0, str(ROOT / 'integration'))
    from deploy_yicloud import client, IMAGE
    job, ctx = client()
    active = job.list_jobs(ctx, job.models.ListJobsReq(Project='fdj-infra', ZoneId='cn-shanghai-a',
        JobNames='face-age', Self=True, Limit=100, Phases='Pending,Running,Inqueue,Starting,Stopping'))
    if active.Items:
        raise RuntimeError('One-card limit: another face-age job is active')
    name = 'face-age-' + run.name
    payload = dict(Name=name, Project='fdj-infra', ZoneId='cn-shanghai-a', QuotaGroup='fdj_gpu_taskpool',
        Priority='3', UsePrivateMachine=False, AutoDeleteDuration='168h',
        TaskSpecs={'study': dict(Command=['bash', str(run / 'source/run_identity_budget.sh'), str(ROOT), str(run)],
            Env={}, Image=IMAGE, Replicas=1, CPU='4', Memory='48Gi', GPU='1', Internet=False,
            MaxRunningDuration='30m', MaxWaitDuration='10m', RestartPolicy='Never',
            MountGPFS='gpfs://gpfs2/haobinzhang:/root')})
    write_json(run / 'request.json', payload)
    try:
        result = job.create_job(ctx, job.models.CreateJobReq(**(payload | {'TaskSpecs': {
            k: job.models.CreateJobReqTaskSpec(**v) for k, v in payload['TaskSpecs'].items()}})))
    except Exception:
        matches = job.list_jobs(ctx, job.models.ListJobsReq(Project='fdj-infra', ZoneId='cn-shanghai-a',
                                                          JobNames=name, Self=True, Limit=100))
        exact = [x for x in matches.Items or [] if x.Name == name]
        if not exact:
            raise
        result = exact[0]
    receipt = dict(job_id=result.JobId, name=name, run_dir=str(run), gpus=1)
    write_json(run / 'receipt.json', receipt)
    print(json.dumps(receipt), flush=True)


def worker(run):
    validate(run)
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from torch.nn import functional as F
    import lpips
    from deterministic_quality_ops import resize, replace_pools, verify
    from source_color import preserve_source_chroma
    if os.environ.get('FACE_AGE_EXPERIMENT_ALLOCATION') != '1' or torch.cuda.device_count() != 1:
        raise ValueError('Exactly one allocated GPU required')
    torch.set_num_threads(4); cv2.setNumThreads(2); torch.manual_seed(20261127)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    write_json(run / 'numeric-gradient-check.json', verify())
    write_json(run / 'device.json', dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), device_count=1, memory_bytes=torch.cuda.get_device_properties(0).total_memory))
    base = Path(os.environ['FACE_AGE_BASE']); sys.path.insert(0, str(base / 'src'))
    from sam_npu import load_sam
    from sam_losses import IDLoss, AgeEst
    net = load_sam('cuda:0').eval().requires_grad_(False)
    identity = IDLoss().cuda().eval().requires_grad_(False)
    age = AgeEst().cuda().eval().requires_grad_(False)
    perceptual = lpips.LPIPS(net='alex', verbose=False).cuda().eval().requires_grad_(False)
    for model in [identity, age, perceptual]: replace_pools(model)

    def fingerprint():
        digest = hashlib.sha256()
        for prefix, model in [('net', net), ('identity', identity), ('age', age), ('perceptual', perceptual)]:
            for name, value in model.state_dict().items():
                digest.update((prefix + name).encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    frozen = fingerprint()
    def tensor(rgb):
        return torch.from_numpy(np.array(rgb, copy=True, order='C')).permute(2, 0, 1)[None].cuda().float()/127.5-1
    def decode(codes):
        return net(codes, input_code=True, input_is_full=True, randomize_noise=False, resize=False)
    def native(image):
        return ((image.clamp(-1, 1)+1)*127.5).permute(1, 2, 0).byte().cpu().numpy()
    def digest_image(rgb):
        buffer = BytesIO(); Image.fromarray(rgb).save(buffer, format='PNG')
        return hashlib.sha256(buffer.getvalue()).hexdigest()
    def logits(image):
        value = resize((image+1)/2, (224, 224), kind='bilinear')
        return age.net.age(age.net.b((value-age.m)/age.sd)).float()
    def load(row):
        folder = PARENT / 'evaluation' / row['name']
        rgb = np.asarray(Image.open(folder / 'reference.png').convert('RGB'))
        x = tensor(cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA))
        raw = torch.load(folder / 'original_delta_codes.pt', map_location='cuda', weights_only=True)['codes']
        accepted = torch.load(folder / 'original_delta_constrained_codes.pt', map_location='cuda', weights_only=True)['codes']
        if raw.shape != (5, 18, 512) or accepted.shape != raw.shape or not torch.equal(raw[:, 8:], accepted[:, 8:]):
            raise ValueError('Parent code shape/layers mismatch')
        return rgb, x, raw, accepted
    mid = age.mid.flatten(); target = torch.tensor([.60, .75], device='cuda')
    distribution = torch.zeros(2, 9, device='cuda')
    for i, value in enumerate(target):
        high = min(int(torch.searchsorted(mid, value).item()), 8); low = max(0, high-1)
        weight = (value-mid[low])/(mid[high]-mid[low])
        distribution[i, low] = 1-weight; distribution[i, high] = weight
    if not torch.allclose(distribution@mid, target, atol=1e-7): raise ValueError('Changed requested age')
    write_json(run / 'target-distribution.json', dict(targets=[60, 75], probabilities=distribution.tolist()))

    def refine(row, x, raw, accepted, hard):
        started = time.monotonic()
        with torch.no_grad():
            own = identity.feats(x).expand(2, -1).detach()
            prior = resize(decode(raw[3:]), (256, 256))
            start_images = resize(decode(accepted[3:]), (256, 256))
            floor = (identity.feats(start_images)*own).sum(1)-.01
            baseline_pixels = [native(y) for y in decode(accepted)[3:]]
            png_floor = torch.cat([(identity.feats(tensor(cv2.resize(y, (256, 256), interpolation=cv2.INTER_AREA)))*own[:1]).sum(1)
                                   for y in baseline_pixels])-.01
            color_reference = F.avg_pool2d(x, 32).expand(2, -1, -1, -1)
        active = torch.tensor([row['source_age'] != 60, row['source_age'] != 75], device='cuda')[:, None, None]
        origin = accepted[3:].clone()
        delta = torch.zeros_like(origin[:, :8]).requires_grad_(True)
        optimizer = torch.optim.Adam([delta], lr=.012)
        def codes(value):
            return origin + torch.cat([value*active, torch.zeros_like(origin[:, 8:])], 1)
        def measure(value, full=True):
            y = resize(decode(codes(value)), (256, 256))
            similarities = (identity.feats(y)*own).sum(1)
            if not full: return similarities
            logit = logits(y); prediction = logit.softmax(1)@mid
            kl = F.kl_div(logit.log_softmax(1), distribution, reduction='none').sum(1)
            order = .25*((.075-(prediction[1]-prediction[0]))/.1).clamp_min(0)
            absolute_delta = (codes(value)-raw[3:])[:, :8]
            losses = .8*(floor-similarities).clamp_min(0) + .5*kl
            losses = losses + .6*perceptual(y, prior).reshape(2, -1).mean(1)
            losses = losses + .05*absolute_delta.square().mean((1, 2))
            losses = losses + .1*(F.avg_pool2d(y, 32)-color_reference).abs().mean((1, 2, 3)) + order
            return losses.mean(), similarities, prediction*100
        def quantized(value):
            all_codes = accepted.clone(); all_codes[3:] = codes(value)
            pixels = [native(y) for y in decode(all_codes)[3:]]
            return torch.cat([(identity.feats(tensor(cv2.resize(y, (256, 256), interpolation=cv2.INTER_AREA)))*own[:1]).sum(1)
                              for y in pixels])
        trace = []; rejected = 0; projected = 0; backtracks = 0; best_step = 0
        with torch.no_grad():
            start_loss, _, _ = measure(delta)
        best_loss = float(start_loss); best = delta.detach().clone()
        for step in range(180):
            optimizer.param_groups[0]['lr'] = .012*(.2+.8*(1-step/180))
            loss, sim, pred = measure(delta)
            identity_gradient = torch.autograd.grad(sim.sum(), delta, retain_graph=True)[0] if hard else None
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if not torch.isfinite(loss) or not torch.isfinite(delta.grad).all(): raise ValueError('Nonfinite gradient')
            previous = delta.detach().clone(); optimizer.step()
            if hard:
                with torch.no_grad():
                    direction = (delta-previous)*active
                    gradient = identity_gradient*active
                    predicted = sim.detach()+(gradient*direction).sum((1, 2))
                    correction = (floor+1e-5-predicted).clamp_min(0)
                    projected += int((correction > 0).sum())
                    direction += (correction/gradient.square().sum((1, 2)).clamp_min(1e-12))[:, None, None]*gradient
                    chosen = previous.clone(); pending = active.flatten().clone()
                    for trial, scale in enumerate([1., .5, .25, .125, .0625, .03125, .015625, .0078125]):
                        if not pending.any(): break
                        proposal = previous+scale*direction
                        candidate_sim = measure(proposal, full=False)
                        feasible = (candidate_sim >= floor) & pending
                        chosen[feasible] = proposal[feasible]
                        backtracks += int(feasible.sum())*trial
                        pending &= ~feasible
                    rejected += int(pending.sum()); delta.copy_(chosen)
                    checked = measure(delta, full=False)
                    if (checked < floor).any(): raise ValueError('Accepted hard update violated float budget')
            if (step+1) % 10 == 0:
                with torch.no_grad():
                    measured, sim_now, pred_now = measure(delta); png_sim = quantized(delta)
                    feasible = bool((sim_now >= floor).all() and (png_sim >= png_floor).all())
                    if float(measured) < best_loss and (not hard or feasible):
                        best_loss = float(measured); best = delta.detach().clone(); best_step = step+1
                    trace.append(dict(step=step+1, loss=float(measured), arcface=sim_now.tolist(),
                        quantized_arcface=png_sim.tolist(), shared_age=pred_now.tolist(), hard_feasible=feasible))
        with torch.no_grad():
            output = codes(best).detach(); final_loss, final_sim, final_age = measure(best); png_sim = quantized(best)
        if hard and ((final_sim < floor).any() or (png_sim < png_floor).any()): raise ValueError('Hard output violated budget')
        torch.cuda.synchronize()
        return output, dict(arm='hard_identity_budget' if hard else 'warm_soft_budget', seconds=time.monotonic()-started,
            best_step=best_step, initial_loss=float(start_loss), final_loss=float(final_loss),
            float_floor=floor.tolist(), png_floor=png_floor.tolist(), final_arcface=final_sim.tolist(),
            final_png_arcface=png_sim.tolist(), final_shared_age=final_age.tolist(),
            projected_proposals=projected, rejected_target_steps=rejected, accepted_backtrack_count=backtracks,
            trace=trace)

    rows = json.loads((run / 'selection.json').read_text())
    old_report = json.loads((PARENT / 'evaluation/report.json').read_text())
    old = {(o['name'], o['condition']): o['sha256'] for o in old_report['outputs'] if o['route'] == 'original_delta_constrained'}
    for row in rows:
        _, _, _, accepted = load(row)
        with torch.no_grad(): images = decode(accepted)
        for label, y in zip(['same', '20', '40', '60', '75'], images):
            if digest_image(native(y)) != old[row['name'], label]: raise ValueError('Baseline not reproduced')
    write_json(run / 'baseline-preflight.json', dict(cases=CASES, pngs=len(rows)*5, exact=True))
    print('IDENTITY_BUDGET_BASELINE_OK', flush=True)
    first = rows[0]; _, x, raw, accepted = load(first)
    repeat_codes, repeat_meta = refine(first, x, raw, accepted, hard=True)
    repeat_codes2, repeat_meta2 = refine(first, x, raw, accepted, hard=True)
    repeatable = torch.equal(repeat_codes, repeat_codes2)
    write_json(run / 'repeatability.json', dict(target_codes_equal=repeatable,
        seconds=[repeat_meta['seconds'], repeat_meta2['seconds']], best_steps=[repeat_meta['best_step'], repeat_meta2['best_step']]))
    if not repeatable: raise ValueError('Hard optimizer is not repeatable')
    print('IDENTITY_BUDGET_REPEATABLE', flush=True)
    detector = cv2.FaceDetectorYN.create(str(base / 'models/yunet/face_detection_yunet_2023mar.onnx'), '', (256, 256), .6, .3, 100)
    recognizer = cv2.FaceRecognizerSF.create(str(ROOT / 'deployment/gan-research/face_recognition_sface_2021dec.onnx'), '')
    def feature(rgb):
        bgr = cv2.cvtColor(cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
        _, faces = detector.detect(bgr)
        if faces is None: return None
        value = recognizer.feature(recognizer.alignCrop(bgr, max(faces, key=lambda face: face[2]*face[3]))).ravel()
        return value/(np.linalg.norm(value)+1e-12)
    out = run / 'evaluation'; out.mkdir(exist_ok=False)
    report = dict(records=rows, routes=ROUTES, outputs=[], timings=[], model_weights_frozen=True,
                  default_model_changed=False, review_title='执行身份预算的年龄编辑对照')
    for row in rows:
        rgb, x, raw, accepted = load(row); folder = out / row['name']; folder.mkdir()
        Image.fromarray(rgb).save(folder / 'reference.png'); own = feature(rgb)
        soft, soft_meta = refine(row, x, raw, accepted, hard=False)
        hard, hard_meta = (repeat_codes, repeat_meta) if row['name'] == first['name'] else refine(row, x, raw, accepted, hard=True)
        for route, high_codes, meta in [('accepted_baseline', None, None), ('warm_soft_budget', soft, soft_meta), ('hard_identity_budget', hard, hard_meta)]:
            code = accepted.clone()
            if high_codes is not None: code[3:] = high_codes
            with torch.no_grad(): images = decode(code)
            for y, label, requested in zip(images, ['same', '20', '40', '60', '75'], [row['source_age'], 20, 40, 60, 75]):
                pixels = native(y); path = folder / (route+'_'+label+'.png'); Image.fromarray(pixels).save(path)
                digest = sha256(path)
                if (route == 'accepted_baseline' or label in ['same', '20', '40'] or requested == row['source_age']) and digest != old[row['name'], label]:
                    raise ValueError('Protected PNG changed')
                colored, _ = preserve_source_chroma(rgb, pixels)
                preserved = path.with_name(path.stem+'_color.png'); Image.fromarray(colored).save(preserved)
                f = feature(pixels)
                report['outputs'].append(dict(name=row['name'], cohort=row['cohort'], route=route, condition=label,
                    target=requested, native=str(path.relative_to(out)), preserved=str(preserved.relative_to(out)),
                    sha256=digest, source_cosine=float(own@f) if own is not None and f is not None else None, detected=f is not None))
            torch.save(dict(codes=code.cpu(), conditions=['same', 20, 40, 60, 75]), folder / (route+'_codes.pt'))
            if meta: report['timings'].append(dict(name=row['name'], **meta))
        write_json(out / 'report.partial.json', report)
        print('IDENTITY_BUDGET_CASE', row['name'], len(report['outputs']), flush=True)
    after = fingerprint()
    write_json(run / 'frozen-verification.json', dict(before=frozen, after=after, equal=frozen == after))
    if frozen != after: raise ValueError('Model state changed')
    report.update(total_seconds=sum(t['seconds'] for t in report['timings']), cuda_peak_bytes=torch.cuda.max_memory_allocated())
    write_json(out / 'report.json', report); (run / 'DONE').write_text('Identity budget pilot complete\n')
    print('IDENTITY_BUDGET_DONE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'submit', 'worker']); parser.add_argument('run', type=Path, nargs='?')
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.run is None: parser.error('run required')
    elif args.action == 'submit': submit(args.run.resolve())
    else: worker(args.run.resolve())

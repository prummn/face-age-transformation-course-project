"""Freeze and queue the upper-tail source-age ablation on one Yicloud GPU."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from probe_inversion import sha256, write_json
from train_gan_reconstruction import rows

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'integration'))
from deploy_yicloud import IMAGE, client, save

STOPPED_PHASES = 'Pending,Running,Inqueue,Starting,Stopping'


def freeze_calibration(previous, run):
    manifest = previous / 'manifests'
    original = rows(manifest / 'train.jsonl')
    heldout = rows(manifest / 'eval.jsonl') + rows(manifest / 'external.jsonl')
    if {r['sha256'] for r in original} & {r['sha256'] for r in heldout}:
        raise ValueError('Training and evaluation pixels overlap')
    knots = np.array([0., 45., 50., 55., 60., 65., 70., 75.])
    inputs = np.array([r['source_age'] for r in original])
    low, high = np.array([r['age_interval'] for r in original]).T
    counts = Counter(tuple(r['age_interval']) for r in original)
    weights = np.array([r['age_confidence'] / counts[tuple(r['age_interval'])] for r in original])
    upper = np.where(knots <= 45, knots, np.minimum(knots + 8, 75))
    penalty = .15

    def loss(values):
        estimates = np.interp(inputs, knots, values)
        errors = estimates - np.clip(estimates, low, high)
        return np.average(errors**2, weights=weights) + penalty * np.mean((values-knots)**2)

    fit = minimize(loss, knots, method='SLSQP', bounds=Bounds(knots, upper),
                   constraints=[LinearConstraint(np.diff(np.eye(len(knots)), axis=0), 0, np.inf)],
                   options={'ftol': 1e-10, 'maxiter': 500})
    if not fit.success or np.any(np.diff(fit.x) < -1e-6):
        raise RuntimeError(str(fit.message))

    def adjust(age):
        return round(float(np.interp(age, knots, fit.x))) if 45 < age < 75 else age

    groups = defaultdict(list)
    for row in heldout:
        record = dict(name=row['name'], before=row['source_age'], after=adjust(row['source_age']))
        if 'age_interval' in row:
            lo, hi = row['age_interval']
            error = lambda age: age - float(np.clip(age, lo, hi))
            older = lo >= 70
        else:
            error = lambda age: age - row['apparent_age']
            older = row['publisher_age'] >= 60
        record.update(before_error=error(record['before']), after_error=error(record['after']))
        groups[row['cohort']].append(record)
        if older:
            groups[row['cohort']+'/older'].append(record)
    diagnostics = {}
    for name, group in sorted(groups.items()):
        diagnostics[name] = dict(n=len(group), changed=sum(x['before'] != x['after'] for x in group))
        for route in ('before', 'after'):
            errors = [r[route+'_error'] for r in group]
            diagnostics[name][route] = dict(mae=float(np.mean(np.abs(errors))), bias=float(np.mean(errors)))

    folder = run / 'manifests'
    folder.mkdir()
    converted = [dict(row, source_age=adjust(row['source_age'])) for row in original]
    changed = sum(a['source_age'] != b['source_age'] for a, b in zip(original, converted))
    if not changed:
        raise ValueError('Fitted mapping is the identity on all training data')
    for name in ('eval.jsonl', 'external.jsonl'):
        shutil.copy2(manifest / name, folder / name)
    (folder/'train.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in converted))
    report = dict(knots=knots.tolist(), fitted_knots=fit.x.tolist(), penalty=penalty, training_size=len(original),
                  changed=changed, change_histogram=dict(Counter(b['source_age']-a['source_age'] for a,b in zip(original, converted))),
                  training_original_sha256=sha256(manifest/'train.jsonl'), heldout_diagnostics=diagnostics,
                  heldout_records=groups, diagnostics='Rounded source conditions. FFHQ interval and APPA apparent-age signed errors.',
                  inference='Keep the original source condition in both checkpoint evaluations to isolate the effect of training.')
    write_json(run/'calibration.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous', type=Path)
    args = parser.parse_args()
    previous = args.previous.resolve()
    if not (previous/'DONE').exists():
        raise ValueError('Control training and evaluation did not complete')
    prior = json.loads((previous/'experiment.json').read_text())
    if sha256(previous/'source/train_gan_reconstruction.py') != prior['source_sha256']['train_gan_reconstruction.py']:
        raise ValueError('Historical training snapshot differs')
    for path, digest in json.loads((previous/'runtime-source.json').read_text()).items():
        if sha256(ROOT/path) != digest:
            raise ValueError(f'Runtime differs from control: {path}')
    for item in json.loads((previous/'assets.json').read_text()):
        if sha256(ROOT/item['path']) != item['sha256']:
            raise ValueError(f'Model differs from control: {item["path"]}')
    job, ctx = client()
    if job.list_jobs(ctx, job.models.ListJobsReq(Project='fdj-infra', ZoneId='cn-shanghai-a',
            JobNames='face-age', Self=True, Limit=100, Phases=STOPPED_PHASES)).Items:
        raise RuntimeError('An existing project GPU job is active')
    now = datetime.now(timezone.utc)
    stamp = now.strftime('%Y%m%d-%H%M%S')
    run = ROOT/'deployment/training-runs'/stamp
    source = run/'source'
    source.mkdir(parents=True, exist_ok=False)
    for name in ('train_gan_reconstruction.py','evaluate_gan_training.py','probe_inversion.py'):
        shutil.copy2(previous/'source'/name, source/name)
    for name in ('run_source_age_calibration.sh', Path(__file__).name):
        shutil.copy2(Path(__file__).with_name(name), source/name)
    shutil.copy2(previous/'data-audit.json', run/'data-audit.json')
    freeze_calibration(previous, run)
    control_files = ['expanded_encoder/train.jsonl','expanded_encoder/encoder.pt','evaluation/report.json']
    save(run/'experiment.json', dict(control=str(previous),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        control_sha256={name:sha256(previous/name) for name in control_files},
        source_sha256={p.name:sha256(p) for p in source.iterdir()},
        manifest_sha256={p.name:sha256(p) for p in (run/'manifests').iterdir()},
        seed=20261123, steps=1000, batch_size=2, gpu=1, older_gate=45,
        promotion_criteria=dict(older_appa_mae_reduction_min=1, overall_appa_mae_increase_max=.25,
           asian_cross_sface_drop_max=.01, independent_age_span_ratio_min=.90,
           requires_visually_credible_aging_and_wplus_identity_comparison=True)))
    name = 'face-age-source-age-'+stamp
    payload = dict(Name=name, Project='fdj-infra', ZoneId='cn-shanghai-a', QuotaGroup='fdj_gpu_taskpool',
        Priority='3', UsePrivateMachine=False, AutoDeleteDuration='168h', TaskSpecs={'experiment':dict(
        Command=['bash',str(source/'run_source_age_calibration.sh'),str(ROOT),str(run),str(previous)],
        Env={}, Image=IMAGE, Replicas=1, CPU='4', Memory='48Gi', GPU='1', Internet=False,
        MaxRunningDuration='25m', MaxWaitDuration='10m', RestartPolicy='Never', MountGPFS='gpfs://gpfs2/haobinzhang:/root')})
    save(run/'request.json', payload)
    spec = {key:job.models.CreateJobReqTaskSpec(**value) for key,value in payload['TaskSpecs'].items()}
    try:
        result = job.create_job(ctx, job.models.CreateJobReq(**(payload | {'TaskSpecs':spec})))
    except Exception:
        existing = job.list_jobs(ctx, job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',
            JobNames=name,Self=True,Limit=100))
        matching = [j for j in existing.Items or [] if j.Name == name]
        if not matching:
            raise
        result = matching[0]
    receipt = dict(job_id=result.JobId, run_dir=str(run), name=name, gpus=1)
    save(run/'receipt.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

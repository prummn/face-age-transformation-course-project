"""Finish the high-age GAN study on a single bounded allocation."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

from probe_inversion import sha256
from train_gan_reconstruction import rows
from evaluate_quality_training import choose_cases

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'integration'))
from deploy_yicloud import IMAGE, client, save

ACTIVE = 'Pending,Running,Inqueue,Starting,Stopping'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('data',type=Path)
    p.add_argument('control',type=Path,help='Completed original-source calibration run')
    a = p.parse_args()
    data, control = a.data.resolve(), a.control.resolve()
    if not (data/'DONE').exists() or not (control/'DONE').exists():
        raise ValueError('Frozen data preparation and control run must complete first')
    audit = json.loads((data/'audit.json').read_text())
    for name,digest in audit['files'].items():
        if sha256(data/name) != digest:
            raise ValueError(f'Prepared manifest changed: {name}')
    checkpoint = control/'calibrated_encoder/encoder.pt'
    done = json.loads(checkpoint.with_name('completed.json').read_text())
    if sha256(checkpoint) != done['encoder_sha256']:
        raise ValueError('Control checkpoint digest differs')
    weights = json.loads((ROOT/'integration/quality/baselines/sam-wplus-v1.json').read_text())['weights']
    files = {}
    for item in weights:
        path = ROOT/item['path']
        if sha256(path) != item['sha256']:
            raise ValueError(f'Release weight changed: {path}')
        files[str(path)] = item['sha256']
    job,ctx = client()
    active = job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',
        JobNames='face-age',Self=True,Limit=100,Phases=ACTIVE))
    if active.Items:
        raise RuntimeError('One-card limit; another project task is active')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    run = ROOT/'deployment/training-runs'/stamp
    source = run/'source'
    source.mkdir(parents=True,exist_ok=False)
    for name in ('annotate_gan_training.py','real_age_guide.py','train_gan_reconstruction.py','evaluate_gan_training.py',
                 'evaluate_quality_training.py','deterministic_quality_ops.py','probe_inversion.py',
                 'run_quality_training.sh','submit_quality_training.py'):
        shutil.copy2(Path(__file__).with_name(name),source/name)
    for name in ('latent_quality.py','source_color.py'):
        shutil.copy2(ROOT/'integration'/name,source/name)
    shutil.copy2(data/'audit.json',run/'data-audit.json')
    packed = run/'data'
    packed.mkdir()
    for name in audit['files']:
        shutil.copy2(data/name,packed/name)
    fast_control = run/'calibrated_encoder'
    fast_control.mkdir()
    (fast_control/'encoder.pt').symlink_to(checkpoint)
    files[str(checkpoint)] = done['encoder_sha256']
    tracked = ['integration/ascend/src/'+name for name in ('sam_npu.py','sam_losses.py','faceattr.py')]
    tracked += ['deployment/gan-research/face_recognition_sface_2021dec.onnx']
    paths = [ROOT/path for path in tracked]+list(source.iterdir())+list(packed.iterdir())
    selection = run/'wplus-selection.json'
    save(selection,[dict(name=r['name'],cohort=r['cohort'],sha256=r['sha256']) for r in choose_cases(run)])
    paths.append(selection)
    files.update({str(path):sha256(path) for path in paths})
    training = rows(packed/'train.pre-age.jsonl')
    evaluation = rows(packed/'eval.pre-age.jsonl')+rows(packed/'external.jsonl')
    if {r['sha256'] for r in training}&{r['sha256'] for r in evaluation}:
        raise ValueError('Train and evaluation input hashes overlap')
    gates = dict(sface_drop_max=.02,high_age_mae_improvement_min=1.0,detection_increase_max=0,
        age_span_ratio_min=.95,dated_target_sface_drop_max=.03,dated_apparent_error_increase_max=2,
        visually_check_max_regressions_and_native_images=True)
    save(run/'experiment.json',dict(profile='human_age_supervision_and_wplus',frozen_inputs=files,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        training=len(training),eval=len(evaluation),head='60 epochs; group-disjoint internal dev; frozen features; apparent-age classes',
        steps=3000,batch_size=2,seed=20261123,editor_checkpoints=[1000,2000,3000],
        wplus_routes=['wplus_base','wplus_real_age','wplus_trained_editor'],
        promotion_gates=gates,age_gates_use_independent_model=True,
        resources=dict(gpu=1,cpu=4,memory='48Gi',max_running='58m')))
    name = 'face-age-real-guide-'+stamp
    payload = dict(Name=name,Project='fdj-infra',ZoneId='cn-shanghai-a',QuotaGroup='fdj_gpu_taskpool',
        Priority='3',UsePrivateMachine=False,AutoDeleteDuration='168h',TaskSpecs={'experiment':dict(
        Command=['bash',str(source/'run_quality_training.sh'),str(ROOT),str(run)],Env={},
        Image=IMAGE,Replicas=1,CPU='4',Memory='48Gi',GPU='1',Internet=False,MaxRunningDuration='58m',
        MaxWaitDuration='10m',RestartPolicy='Never',MountGPFS='gpfs://gpfs2/haobinzhang:/root')})
    save(run/'request.json',payload)
    spec = {k:job.models.CreateJobReqTaskSpec(**v) for k,v in payload['TaskSpecs'].items()}
    try:
        result = job.create_job(ctx,job.models.CreateJobReq(**(payload|{'TaskSpecs':spec})))
    except Exception:
        found = job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames=name,Self=True,Limit=100))
        exact = [x for x in found.Items or [] if x.Name == name]
        if not exact:
            raise
        result = exact[0]
    receipt = dict(job_id=result.JobId,name=name,run_dir=str(run),gpus=1)
    save(run/'receipt.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()

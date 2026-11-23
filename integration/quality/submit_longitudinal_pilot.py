"""Snapshot the dated evaluation pairs and run two directions per pair on one GPU."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from probe_inversion import sha256
from submit_source_age_calibration import client, IMAGE, ROOT, save, STOPPED_PHASES

WORKER = """#!/usr/bin/env bash
set -euo pipefail
repo="$1"
run="$2"
cd "$repo"
exec >> "$run/worker.log" 2>&1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export FACE_AGE_REPO="$repo" FACE_AGE_BASE="$repo/deployment/model-files/face-age"
export FACE_AGE_DEVICE=cuda:0 FACE_AGE_EXPERIMENT_ALLOCATION=1 HF_HUB_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TORCH_HOME="$FACE_AGE_BASE/torch_cache"
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -m venv --system-site-packages /tmp/face-age-longitudinal
py=/tmp/face-age-longitudinal/bin/python
"$py" -m pip install --no-index --no-deps "$repo"/deployment/wheels/*.whl
"$py" "$run/source/evaluate_longitudinal_pilot.py" "$run"
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pairs', type=Path)
    parser.add_argument('calibration_run', type=Path)
    args = parser.parse_args()
    candidate_run = args.calibration_run.resolve()
    if not (candidate_run/'DONE').exists():
        raise ValueError('Complete the calibration ablation first')
    experiment = json.loads((candidate_run/'experiment.json').read_text())
    control_run = Path(experiment['control'])
    checkpoints = dict(expanded_encoder=str(control_run/'expanded_encoder/encoder.pt'),
                       calibrated_encoder=str(candidate_run/'calibrated_encoder/encoder.pt'))
    digests = {}
    for route, filename in checkpoints.items():
        source = Path(filename)
        known = json.loads(source.with_name('completed.json').read_text())['encoder_sha256']
        if sha256(source) != known:
            raise ValueError(f'Unexpected digest for {route}')
        digests[filename] = known
    for item in json.loads((control_run/'assets.json').read_text()):
        path = ROOT/item['path']
        if sha256(path) != item['sha256']:
            raise ValueError(f'Unexpected release weight digest: {path}')
        digests[str(path)] = item['sha256']
    for name, digest in json.loads((control_run/'runtime-source.json').read_text()).items():
        digests[str(ROOT/name)] = digest

    pairs = json.loads(args.pairs.read_text())
    if not 1 <= len(pairs) <= 4:
        raise ValueError('Bounded pilot supports one to four pairs')
    known_hashes = {row['sha256'] for row in map(json.loads, (candidate_run/'manifests/train.jsonl').read_text().splitlines())}
    used = set()
    for pair in pairs:
        for timepoint in ('early','late'):
            row = pair[timepoint]
            for source in ('original','aligned'):
                path = row[source]
                digest = row[source+'_sha256']
                if digest in known_hashes or digest in used or sha256(path) != digest:
                    raise ValueError(f'Repeated, training, or changed portrait: {path}')
                used.add(digest)
                digests[path] = digest

    job,ctx = client()
    active = job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',
        JobNames='face-age',Self=True,Limit=100,Phases=STOPPED_PHASES))
    if active.Items:
        raise RuntimeError('An existing project GPU job is active')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    run = ROOT/'deployment/training-runs'/stamp
    source = run/'source'
    source.mkdir(parents=True,exist_ok=False)
    for file in [Path(__file__).with_name(n) for n in ('probe_inversion.py','evaluate_longitudinal_pilot.py',Path(__file__).name)] + [ROOT/'integration/latent_quality.py']:
        dest = source/file.name
        shutil.copy2(file,dest)
        digests[str(dest)] = sha256(dest)
    calibration = candidate_run/'calibration.json'
    recognizer = ROOT/'deployment/gan-research/face_recognition_sface_2021dec.onnx'
    digests[str(calibration)] = sha256(calibration)
    digests[str(recognizer)] = sha256(recognizer)
    save(run/'protocol.json',dict(pairs=pairs,checkpoints=checkpoints,sha256=digests,calibration=str(calibration)))
    (source/'worker.sh').write_text(WORKER)
    name = 'face-age-longitudinal-'+stamp
    payload = dict(Name=name,Project='fdj-infra',ZoneId='cn-shanghai-a',QuotaGroup='fdj_gpu_taskpool',Priority='3',
        UsePrivateMachine=False,AutoDeleteDuration='168h',TaskSpecs={'experiment':dict(
        Command=['bash',str(source/'worker.sh'),str(ROOT),str(run)],
        Env={},Image=IMAGE,Replicas=1,CPU='4',Memory='48Gi',GPU='1',Internet=False,
        MaxRunningDuration='12m',MaxWaitDuration='10m',RestartPolicy='Never',MountGPFS='gpfs://gpfs2/haobinzhang:/root')})
    save(run/'request.json',payload)
    specs = {key:job.models.CreateJobReqTaskSpec(**value) for key,value in payload['TaskSpecs'].items()}
    try:
        result = job.create_job(ctx,job.models.CreateJobReq(**(payload | {'TaskSpecs':specs})))
    except Exception:
        found = job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames=name,Self=True,Limit=100))
        matching = [item for item in found.Items or [] if item.Name == name]
        if not matching:
            raise
        result = matching[0]
    receipt = dict(job_id=result.JobId,run_dir=str(run),name=name,gpus=1)
    save(run/'receipt.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main()

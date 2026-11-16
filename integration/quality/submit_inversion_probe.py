"""Submit/status a bounded experiment; never modify serving deployment state."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'integration'))
from deploy_yicloud import IMAGE, client, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['submit', 'status'])
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--previous-results', type=Path, help='Run the detail follow-up on this completed first trial')
    args = parser.parse_args()
    service, ctx = client()
    if args.action == 'status':
        if args.receipt is None:
            parser.error('--receipt is required for status')
        state = json.loads(args.receipt.read_text())
        result = service.get_job(ctx, service.models.GetJobReq(Project='fdj-infra', JobId=state['job_id']))
        print(json.dumps(dict(**state, phase=result.Phase, message=result.Message), ensure_ascii=False, indent=2))
        return
    if args.previous_results and not (args.previous_results / 'DONE').exists():
        parser.error('--previous-results must contain a completed experiment')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    run = ROOT / 'deployment/quality-runs' / stamp
    run.mkdir(parents=True, exist_ok=False)
    snapshot = run / 'source'
    snapshot.mkdir()
    sources = {}
    for name in ['probe_inversion.py', 'run_inversion_probe.sh', 'submit_inversion_probe.py',
                 'probe_inversion_detail.py', 'run_detail_probe.sh']:
        path = Path(__file__).parent / name
        shutil.copy2(path, snapshot / name)
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    name = 'face-age-quality-' + stamp
    entry = 'run_detail_probe.sh' if args.previous_results else 'run_inversion_probe.sh'
    command_args = ['bash', str(snapshot / entry), str(ROOT), str(run)]
    if args.previous_results:
        command_args.append(str(args.previous_results.resolve()))
    command = shlex.join(command_args)
    payload = dict(Name=name, Project='fdj-infra', ZoneId='cn-shanghai-a', QuotaGroup='fdj_gpu_taskpool',
                   Priority='3', UsePrivateMachine=False, AutoDeleteDuration='168h',
                   TaskSpecs={'probe': dict(Command=['bash', '-c', command], Env={}, Image=IMAGE,
                     Replicas=1, CPU='8', Memory='32Gi', GPU='1', Internet=False,
                     MaxRunningDuration='1h', MaxWaitDuration='30m', RestartPolicy='Never',
                     MountGPFS='gpfs://gpfs2/haobinzhang:/root')})
    save(run / 'request.json', payload)
    request = dict(payload)
    request['TaskSpecs'] = {k: service.models.CreateJobReqTaskSpec(**v) for k,v in payload['TaskSpecs'].items()}
    try:
        result = service.create_job(ctx, service.models.CreateJobReq(**request))
    except Exception:
        matches = service.list_jobs(ctx, service.models.ListJobsReq(
            Project='fdj-infra', ZoneId='cn-shanghai-a', JobNames=name, Self=True, Limit=100))
        exact = [x for x in matches.Items or [] if x.Name == name]
        if not exact:
            raise
        result = exact[0]
    state = dict(job_id=result.JobId, name=name, run_dir=str(run), sources=sources,
                 submitted_at=datetime.now(timezone.utc).isoformat(), image=IMAGE)
    save(run / 'receipt.json', state)
    print(json.dumps(state, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

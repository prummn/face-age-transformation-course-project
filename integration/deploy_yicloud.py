"""Start/status/stop only this demo's dedicated GPU job and local gateway."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / 'deployment'
STATE = DEPLOY / 'current.json'
SDK = Path(os.environ.get('YICLOUD_SDK_ROOT', str(DEPLOY / 'sdk')))
IMAGE = ('registry.gate.yicloud.com.cn/library/pytorch-training-nvidia:'
         '2.6.0-py3.10-cuda12.9-h20-ubuntu22.04-v20260710')


def client():
    sys.path.insert(0, str(SDK))
    import yicloud.base as base
    from yicloud.base import auth, log, msgs
    from yicloud.services import job
    cfg = base.new_config()
    cfg.host, cfg.region_id = 'https://gate.yicloud.com.cn', 'cn-shanghai'
    cfg.timeout, cfg.max_retries, cfg.log_level = 45, 0, log.Level.ERROR
    instance = base.new_client(base.with_config(cfg), base.with_credential(auth.Credential(
        os.environ['YICLOUD_PUBLIC_KEY'], os.environ['YICLOUD_SECRET_KEY'])))
    instance.http_client.trust_env = False
    job.use_client(instance)
    return job, msgs.new_meta_ctx()[0]


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def spawn(name, command, logfile):
    pidfile = DEPLOY / (name + '.pid')
    if pidfile.exists():
        pid = int(pidfile.read_text())
        try:
            cmdline = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
            if all(part in cmdline for part in command[1:]):
                return
        except FileNotFoundError:
            pass
    with logfile.open('ab') as output:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
                                stderr=subprocess.STDOUT, start_new_session=True)
    pidfile.write_text(str(proc.pid))


def health():
    try:
        with urlopen('http://127.0.0.1:18765/api/health', timeout=3) as r:
            return json.load(r)
    except Exception:
        return None


def recorded_job(job, ctx, job_id):
    from types import SimpleNamespace
    from yicloud.base.errs import ServerException
    try:
        return job.get_job(ctx, job.models.GetJobReq(Project='fdj-infra', JobId=job_id))
    except ServerException as error:
        # Platform AutoDeleteDuration can remove a completed job while its local
        # receipt remains. Network/auth/other errors must still stop deployment.
        if error.action != 'GetJob' or error.code != 105 or 'Job 不存在' not in error.message:
            raise
        return SimpleNamespace(Phase='Deleted', Message='Recorded job was deleted by the platform')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'status', 'stop', 'gateway'])
    parser.add_argument('--quality', choices=['original', 'anchor', 'constrained', 'texture', 'texture_robust', 'original_delta'], default='original',
                        help='Explicit opt-in to slower per-photo W+ optimization; original weights stay unchanged')
    parser.add_argument('--max-running-duration', default='24h',
                        help='Platform GPU runtime cap, e.g. 30m for a bounded quality preview')
    parser.add_argument('--stable-numerics', action='store_true',
                        help='Explicit deterministic CUDA W+ fitting; model weights and age objective unchanged')
    args = parser.parse_args()
    state = json.loads(STATE.read_text()) if STATE.exists() else None
    if args.action in ['start', 'status', 'stop']:
        job, ctx = client()
    if args.action == 'status':
        if not state:
            print('No deployment job recorded')
            return
        result = recorded_job(job, ctx, state['job_id'])
        print(json.dumps(dict(**state, phase=result.Phase, message=result.Message, health=health()), ensure_ascii=False, indent=2))
        return
    if args.action == 'stop':
        if state:
            # A dedicated receipt is the only source of the job ID to stop.
            result = recorded_job(job, ctx, state['job_id'])
            if result.Phase not in ['Stopped', 'Succeeded', 'Failed', 'Deleted']:
                job.stop_job(ctx, job.models.StopJobReq(Project='fdj-infra', JobId=state['job_id']))
            state['stop_requested'] = True
            save(STATE, state)
        for name, marker in [('gateway', str(ROOT / 'integration/launch.py')), ('sshd', str(DEPLOY / 'ssh/sshd_config'))]:
            pidfile = DEPLOY / (name + '.pid')
            if pidfile.exists():
                pid = int(pidfile.read_text())
                try:
                    if marker in Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode():
                        os.kill(pid, 15)
                except (FileNotFoundError, ProcessLookupError):
                    pass
                pidfile.unlink(missing_ok=True)
        print('Stopped this demo only; code, weights and logs retained')
        return
    if args.action == 'start':
        # A separate SSH listener avoids this workspace's GatewayPorts=yes.
        spawn('sshd', ['/usr/sbin/sshd', '-D', '-e', '-f', str(DEPLOY / 'ssh/sshd_config')], DEPLOY / 'logs/sshd.log')
        if state:
            result = recorded_job(job, ctx, state['job_id'])
            if state.get('stop_requested'):
                deadline = time.monotonic() + 90
                while result.Phase not in ['Stopped', 'Succeeded', 'Failed', 'Deleted']:
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Platform stop is still pending; retry start after this job stops')
                    time.sleep(2)
                    result = recorded_job(job, ctx, state['job_id'])
            if result.Phase in ['Running', 'Pending', 'Inqueue', 'Starting']:
                if state.get('quality', 'original') != args.quality:
                    raise RuntimeError('Active demo uses a different quality profile; stop it before changing profiles')
                if state.get('stable_numerics', False) != args.stable_numerics:
                    raise RuntimeError('Active demo uses different numerics; stop it before changing numerics')
                print('Reusing dedicated job', state['job_id'])
            else:
                state = None
        if not state:
            active = job.list_jobs(ctx, job.models.ListJobsReq(
                Project='fdj-infra', ZoneId='cn-shanghai-a', JobNames='face-age', Self=True,
                Limit=100, Phases='Pending,Running,Inqueue,Starting,Stopping'))
            if active.Items:
                names = ', '.join(f'{x.Name} ({x.JobId}, {x.Phase})' for x in active.Items)
                raise RuntimeError('One-card limit: wait for these face-age tasks to finish: ' + names)
            stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
            run_dir = DEPLOY / 'runs' / stamp
            run_dir.mkdir(parents=True)
            name = 'face-age-demo-' + stamp
            command = shlex.join(['bash', str(ROOT / 'integration/run_yicloud_worker.sh'), str(ROOT), str(run_dir)])
            payload = dict(Name=name, Project='fdj-infra', ZoneId='cn-shanghai-a',
                QuotaGroup='fdj_gpu_taskpool', Priority='3', UsePrivateMachine=False, AutoDeleteDuration='168h',
                TaskSpecs={'inference':dict(Command=['bash','-c',command], Env={'FACE_AGE_QUALITY':args.quality,'FACE_AGE_STABLE_NUMERICS':'1' if args.stable_numerics else '0'}, Image=IMAGE,
                    Replicas=1, CPU='4', Memory='32Gi', GPU='1', Internet=False,
                    MaxRunningDuration=args.max_running_duration, MaxWaitDuration='10m', RestartPolicy='Never',
                    MountGPFS='gpfs://gpfs2/haobinzhang:/root')})
            save(run_dir / 'request.json', payload)
            request = dict(payload)
            request['TaskSpecs'] = {k:job.models.CreateJobReqTaskSpec(**v) for k,v in payload['TaskSpecs'].items()}
            try:
                result = job.create_job(ctx, job.models.CreateJobReq(**request))
            except Exception:
                matches = job.list_jobs(ctx, job.models.ListJobsReq(Project='fdj-infra', ZoneId='cn-shanghai-a', JobNames=name, Self=True, Limit=100))
                exact = [x for x in matches.Items or [] if x.Name == name]
                if not exact:
                    raise
                result = exact[0]
            state = dict(job_id=result.JobId, name=name, run_dir=str(run_dir), image=IMAGE,
                         code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                         submitted_at=datetime.now(timezone.utc).isoformat(), max_running_duration=args.max_running_duration,
                         quality=args.quality, stable_numerics=args.stable_numerics)
            save(run_dir / 'receipt.json', state)
            save(STATE, state)
            print('Submitted dedicated GPU job', result.JobId)
    if args.action == 'start' or (health() or {}).get('ready'):
        spawn('gateway', [sys.executable, '-u', str(ROOT / 'integration/launch.py'), '--upstream',
              'http://127.0.0.1:18765', '--port', '65512', '--no-browser'], DEPLOY / 'logs/gateway.log')
        print('Gateway: http://localhost:65512/')
    else:
        print('Backend not ready yet; inspect worker logs, then run this command with gateway')


if __name__ == '__main__':
    main()

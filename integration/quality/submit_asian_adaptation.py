"""Queue the prepared HD data and generator ablations sequentially on one GPU."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

from probe_inversion import sha256
from train_gan_reconstruction import rows


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'integration'))
from deploy_yicloud import IMAGE, client, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    args = parser.parse_args()
    data = args.data.resolve()
    if not (data / 'DONE').is_file():
        raise ValueError('Run CPU preparation first')
    audit = json.loads((data / 'audit.json').read_text())
    for name, digest in audit['files'].items():
        if sha256(data / name) != digest:
            raise ValueError(f'Frozen manifest hash differs: {name}')
    control = rows(data / 'train-control.pre-age.jsonl')
    added = rows(data / 'train-added.pre-age.jsonl')
    weights = json.loads((ROOT/'integration/quality/baselines/sam-wplus-v1.json').read_text())['weights']
    for item in weights:
        if sha256(ROOT/item['path']) != item['sha256']:
            raise ValueError(f'Baseline model digest differs: {item["path"]}')

    job,context = client()
    active = job.list_jobs(context,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',
        JobNames='face-age',Self=True,Limit=100,Phases='Pending,Running,Inqueue,Starting,Stopping'))
    if active.Items:
        raise RuntimeError('Existing project task: '+','.join(x.JobId for x in active.Items))
    now = datetime.now(timezone.utc)
    stamp = now.strftime('%Y%m%d-%H%M%S')
    run = ROOT/'deployment/training-runs'/stamp
    source = run/'source'
    source.mkdir(parents=True,exist_ok=False)
    files = ('run_asian_adaptation.sh','submit_asian_adaptation.py','train_gan_reconstruction.py',
        'annotate_gan_training.py','evaluate_gan_training.py','summarize_asian_adaptation.py',
        'prepare_asian_adaptation.py','collect_regional_ffhq.py','probe_inversion.py')
    hashes = {}
    for name in files:
        path = Path(__file__).with_name(name)
        shutil.copy2(path,source/name)
        hashes[name] = sha256(path)
    runtime = ['integration/ascend/src/'+name for name in ['sam_npu.py','sam_losses.py','faceattr.py']]
    save(run/'runtime-source.json',{name:sha256(ROOT/name) for name in runtime})
    save(run/'assets.json',weights)
    shutil.copy2(data/'audit.json',run/'data-audit.json')
    packed = run/'data'
    packed.mkdir()
    for name in ('eval.pre-age.jsonl','external.jsonl'):
        shutil.copy2(data/name,packed/name)
    shutil.copy2(data/'train-control.pre-age.jsonl',packed/'control.jsonl')
    (packed/'train.pre-age.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in control+added))
    save(run/'experiment.json',dict(profile='asian_hd_data_20261123',
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256=hashes, inputs={p.name:sha256(p) for p in packed.glob('*.jsonl')},
        arms=['control_encoder','expanded_encoder','expanded_generator'],control=len(control),added=len(added),
        steps_per_arm=1000,batch_size=2,seed=20261123,source_condition='Frozen FairFace numerical pseudo ages',
        train_targets=[20,75],eval_conditions=['same',20,40,60,75],
        interpretation='B-A is the effect of adding selected HD portraits; C-B is the marginal effect of generator adaptation on identical batches.',
        limits=['Existing APPA32 is an external regression, previously inspected.',
                'FFHQ holdouts are fine-tuning holdouts and overlap base StyleGAN2 pretraining.',
                'Published Asian visual annotations are broad; capture location never establishes ethnicity.',
                'No color correction for evaluator. One GPU, sequential arms, no online promotion.']))
    name='face-age-asian-hd-'+stamp
    payload=dict(Name=name,Project='fdj-infra',ZoneId='cn-shanghai-a',QuotaGroup='fdj_gpu_taskpool',Priority='3',
        UsePrivateMachine=False,AutoDeleteDuration='168h',TaskSpecs={'experiment':dict(
        Command=['bash',str(source/'run_asian_adaptation.sh'),str(ROOT),str(run)],Env={},Image=IMAGE,
        Replicas=1,CPU='4',Memory='48Gi',GPU='1',Internet=False,MaxRunningDuration='40m',
        MaxWaitDuration='10m',RestartPolicy='Never',MountGPFS='gpfs://gpfs2/haobinzhang:/root')})
    save(run/'request.json',payload)
    request=dict(payload,TaskSpecs={k:job.models.CreateJobReqTaskSpec(**v) for k,v in payload['TaskSpecs'].items()})
    try:
        result=job.create_job(context,job.models.CreateJobReq(**request))
    except Exception:
        found=job.list_jobs(context,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames=name,Self=True,Limit=100))
        exact=[j for j in found.Items or [] if j.Name==name]
        if not exact:raise
        result=exact[0]
    receipt=dict(job_id=result.JobId,name=name,run_dir=str(run),created_at=now.isoformat(),gpus=1)
    save(run/'receipt.json',receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()

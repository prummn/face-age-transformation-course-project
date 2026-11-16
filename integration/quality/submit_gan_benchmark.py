"""Submit one dedicated, bounded GAN-only benchmark without changing serving."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'integration'))
from deploy_yicloud import IMAGE,client,save
sys.path.insert(0,str(Path(__file__).parent))
from probe_inversion import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'deployment/gan-research/diagnostic-manifest.json')
    parser.add_argument('--reuse-sam',type=Path,help='Reuse completed SAM directory; submit only CUSP')
    args=parser.parse_args()
    if args.reuse_sam:
        args.reuse_sam=args.reuse_sam.resolve()
        report=json.loads((args.reuse_sam/'report.json').read_text())
        if report['manifest_sha256']!=sha256(args.manifest):
            raise RuntimeError('Reused SAM inputs differ from requested manifest')
        if not (args.reuse_sam/'DONE').is_file():raise RuntimeError('SAM run is not complete')
    required=[ROOT/'deployment/gan-research/cusp_ffhq_ls_005780.pkl',
              ROOT/'deployment/gan-research/face_recognition_sface_2021dec.onnx',
              ROOT/'deployment/model-files/face-age/models/irse50/model_ir_se50.pth',
              ROOT/'deployment/model-files/face-age/torch_cache/hub/checkpoints/alexnet-owt-7be5be79.pth']
    for p in required:
        if not p.is_file():raise RuntimeError('Missing required asset: '+str(p))
    if not list((ROOT/'deployment/wheels').glob('lpips-0.1.4-*.whl')):raise RuntimeError('LPIPS wheel missing')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    run=ROOT/'deployment/gan-runs'/stamp;run.mkdir(parents=True,exist_ok=False)
    source=run/'source';source.mkdir()
    hashes={}
    for name in ['benchmark_gan.py','probe_inversion.py','run_gan_benchmark.sh','submit_gan_benchmark.py']:
        path=Path(__file__).parent/name;shutil.copy2(path,source/name);hashes[name]=sha256(path)
    shutil.copy2(args.manifest,run/'diagnostic-manifest.json')
    if args.reuse_sam:(run/'sam').symlink_to(args.reuse_sam,target_is_directory=True)
    save(run/'assets.json',[dict(path=str(p),sha256=sha256(p),bytes=p.stat().st_size) for p in required])
    name='face-age-gan-bench-'+stamp
    payload=dict(Name=name,Project='fdj-infra',ZoneId='cn-shanghai-a',QuotaGroup='fdj_gpu_taskpool',Priority='3',
                 UsePrivateMachine=False,AutoDeleteDuration='168h',TaskSpecs={'benchmark':dict(
                     Command=['bash',str(source/'run_gan_benchmark.sh'),str(ROOT),str(run),
                              'cusp-only' if args.reuse_sam else 'all'],Env={},Image=IMAGE,
                     Replicas=1,CPU='8',Memory='40Gi',GPU='1',Internet=False,MaxRunningDuration='1h',
                     MaxWaitDuration='30m',RestartPolicy='Never',MountGPFS='gpfs://gpfs2/haobinzhang:/root')})
    save(run/'request.json',payload)
    job,ctx=client();request=dict(payload)
    request['TaskSpecs']={k:job.models.CreateJobReqTaskSpec(**v) for k,v in payload['TaskSpecs'].items()}
    try:result=job.create_job(ctx,job.models.CreateJobReq(**request))
    except Exception:
        found=job.list_jobs(ctx,job.models.ListJobsReq(Project='fdj-infra',ZoneId='cn-shanghai-a',JobNames=name,Self=True,Limit=100))
        exact=[j for j in found.Items or [] if j.Name==name]
        if not exact:raise
        result=exact[0]
    receipt=dict(job_id=result.JobId,name=name,run_dir=str(run),source_sha256=hashes,
                 reused_sam=str(args.reuse_sam) if args.reuse_sam else None,
                 submitted_at=datetime.now(timezone.utc).isoformat(),max_running_duration='1h')
    save(run/'receipt.json',receipt);print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

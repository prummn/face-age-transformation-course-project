"""Validate, independently score and expose one bounded frozen-model diagnostic."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from probe_inversion import sha256,write_json

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--page',required=True);a=p.parse_args();run=a.run.resolve()
    if not a.page or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in a.page):raise ValueError('Simple local page name required')
    exp=json.loads((run/'experiment.json').read_text());receipt=json.loads((run/'receipt.json').read_text());sys.path.insert(0,str(ROOT/'integration'));from deploy_yicloud import client
    job,ctx=client();deadline=time.monotonic()+50*60;last=0
    while not (run/'DONE').exists():
        if time.monotonic()>deadline:raise TimeoutError('Diagnostic worker did not finish')
        if time.monotonic()-last>30:
            state=job.get_job(ctx,job.models.GetJobReq(Project='fdj-infra',JobId=receipt['job_id']));last=time.monotonic()
            if state.Phase in ('Failed','Stopped'):raise ValueError('Diagnostic job '+state.Phase+'; preserve worker.log')
        time.sleep(5)
    from PIL import Image
    r=json.loads((run/'evaluation/report.json').read_text());counts=dict(raw=0,color=0)
    for row in r['outputs']:
        for field,key in [('native','raw'),('preserved','color')]:
            path=run/'evaluation'/row[field]
            with Image.open(path) as im:im.load();assert im.size==(1024,1024) and im.format=='PNG'
            if field=='native' and sha256(path)!=row['sha256']:raise ValueError('Raw diagnostic output changed')
            counts[key]+=1
    write_json(run/'artifact-validation.json',dict(**counts,size=[1024,1024],all_decoded=True,raw_sha256_verified=True))
    def cmd(*parts,timeout=900):subprocess.run([str(x) for x in parts],cwd=ROOT,check=True,timeout=timeout)
    cmd(sys.executable,ROOT/'integration/quality/summarize_highres_texture.py',run)
    cmd(ROOT/'deployment/asia-adaptation/20261123/age-cpu-venv/bin/python',ROOT/'integration/quality/check_independent_age.py','--model',ROOT/'deployment/asia-adaptation/20261123/mivolo_v2_age.onnx','--sources',run/'independent-age-inputs.json','--out',run/'independent-age.json')
    cmd(sys.executable,ROOT/'integration/quality/summarize_highres_texture.py',run,'--age-report',run/'independent-age.json')
    cmd(sys.executable,ROOT/'integration/quality/measure_texture_changes.py',run)
    summary=json.loads((run/'summary.json').read_text());prior=exp['profile']=='frozen_pretrained_discriminator_prior'
    note=(f"{exp['cases']}图预训练判别器潜变量对照：所有网络冻结，两组各150步，仅细层W+可变。" if prior else f"{exp['cases']}图年龄差值分解：相同W+同龄锚点与生成器，比较原SAM年龄差值及180步跨龄约束。")+'独立指标完成，原生自然度尚待复核；未替换基线。'
    write_json(run/'decision.json',dict(selected='wplus_base',promotion=False,visually_reviewed=[],numeric_guard_pass=summary['numeric_guard_pass'],gallery_note=note))
    cmd(sys.executable,ROOT/'integration/quality/build_texture_review.py',run,run/'gallery')
    from finish_native_age import link
    link(a.page,run/'gallery')
    cmd(ROOT/'deployment/control-venv/bin/python',ROOT/'integration/quality/verify_texture_review_browser.py','http://127.0.0.1:65512/'+a.page+'/',run,'--texture',timeout=180)
    state=job.get_job(ctx,job.models.GetJobReq(Project='fdj-infra',JobId=receipt['job_id']));write_json(run/'platform-final.json',dict(job_id=receipt['job_id'],phase=state.Phase,allocated_gpus=1))
    write_json(run/'automatic-completion.json',dict(status='ready',native_review_pending=True,numeric_guard_pass=summary['numeric_guard_pass'],default_model_changed=False,gallery='/'+a.page+'/'));print('SMALL_DIAGNOSTIC_READY',run,flush=True)


if __name__=='__main__':main()

"""Observe one full native-data job, then score and expose all outputs on CPU.

Does not submit/retry training or change serving models. Native human review
remains necessary after numeric guards and browser checks complete.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from probe_inversion import sha256,write_json

ROOT=Path(__file__).resolve().parents[2]
PAGE='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>全量高清训练进度</title><style>body{font:17px/1.8 system-ui;max-width:860px;margin:40px auto;padding:20px;background:#eff2ee;color:#21312c}pre{white-space:pre-wrap}a{color:#285c4b}</style><h1>全量高清训练与纹理对照</h1><p>扩大原生高清训练池，训练侧W+匹配；共同重建与年龄保护，对照额外RGB局部GAN。最多一张GPU。候选需要原生纹理复核，数值通过不会自动替换模型。</p><pre id="status">读取中……</pre><p><a id="result" href="/age-native-full-review/" hidden>查看全部真实输出</a></p><script>async function tick(){try{const r=await fetch('status.json',{cache:'no-store'}),s=await r.json();document.getElementById('status').textContent=s.message+'\\n实际训练清单：'+s.train+'张；W+缓存：'+s.cached+'张\\n重建组：'+s.control_steps+'/4000；GAN组：'+s.gan_steps+'/4000\\n更新时间：'+s.updated;if(s.state==='ready'){document.getElementById('result').hidden=false}}catch(e){}}tick();setInterval(tick,15000)</script></html>'''


def link(name,target):
    p=ROOT/'原型'/name
    if p.is_symlink() and p.resolve()==target.resolve():return
    if p.exists() or p.is_symlink():raise ValueError('Preserving existing page: '+str(p))
    p.symlink_to(os.path.relpath(target,p.parent),target_is_directory=True)


def last_step(path):
    if not path.exists():return 0
    with path.open('rb') as f:f.seek(max(0,path.stat().st_size-9000));lines=f.read().decode().splitlines()
    for line in reversed(lines):
        try:return json.loads(line)['step']
        except (KeyError,ValueError):continue
    return 0


def merge(run):
    if (run/'evaluation-merge.json').exists():return
    main=run/'evaluation';additional=run/'evaluation-additional';a=json.loads((main/'report.json').read_text());b=json.loads((additional/'report.json').read_text())
    if a['routes']!=b['routes'] or {r['name'] for r in a['records']}&{r['name'] for r in b['records']}:raise ValueError('Unexpected evaluation merge')
    frozen=main/'report.fixed15.json';frozen.write_bytes((main/'report.json').read_bytes())
    receipt=dict(fixed15_sha256=sha256(frozen),additional_sha256=sha256(additional/'report.json'),fixed15_count=len(a['records']),additional_count=len(b['records']))
    for row in b['records']:
        source=additional/row['name'];dest=main/row['name'];source.rename(dest);source.symlink_to(os.path.relpath(dest,source.parent),target_is_directory=True)
    for field in ['records','outputs','timings']:a[field]+=b[field]
    a['total_seconds']+=b['total_seconds'];a['baseline_code_scope']='Original15reuse frozen codes with exact baseline PNG; additional regression uses separately frozen inputs and original-generator-only W+ fitting.'
    a['all_baselines_reused_exact']=False;a['reused_baseline_exact_cases']=15;write_json(main/'report.json',a);receipt['combined_sha256']=sha256(main/'report.json');write_json(run/'evaluation-merge.json',receipt)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();run=a.run.resolve();exp=json.loads((run/'experiment.json').read_text());receipt=json.loads((run/'receipt.json').read_text())
    if exp['profile']!='full_native_age_adaptation' or exp['probe']:raise ValueError('Full native study required')
    public=run/'public-status';public.mkdir(exist_ok=True);(public/'index.html').write_text(PAGE);link('age-native-full',public)
    status=dict(state='running',message='准备训练侧W+缓存',train=exp['train'],cached=0,control_steps=0,gan_steps=0)
    def update(**changes):status.update(changes);status['updated']=datetime.now(timezone.utc).isoformat();write_json(public/'status.json',status)
    def cmd(*parts,timeout=1800):subprocess.run([str(x) for x in parts],cwd=ROOT,check=True,timeout=timeout)
    sys.path.insert(0,str(ROOT/'integration'));from deploy_yicloud import client
    job,ctx=client();lastcheck=0;deadline=time.monotonic()+175*60;update()
    try:
        while not (run/'DONE').exists():
            count=len(list((run/'train-codes').glob('*.pt'))) if (run/'train-codes').exists() else 0
            c=last_step(run/'native_control/train.jsonl');g=last_step(run/'native_gan/train.jsonl')
            update(cached=count,control_steps=c,gan_steps=g,message='匹配全部训练侧W+代码' if count<exp['train'] else ('全量高清重建训练' if c<4000 else ('RGB局部GAN训练' if g<4000 else '生成固定及扩展回归1024输出')))
            if time.monotonic()>deadline:raise TimeoutError('Observer deadline; GPU task has its own150m cap')
            if time.monotonic()-lastcheck>60:
                j=job.get_job(ctx,job.models.GetJobReq(Project='fdj-infra',JobId=receipt['job_id']));lastcheck=time.monotonic()
                if j.Phase in ('Failed','Stopped'):raise ValueError('Platform task '+j.Phase+'; inspect worker.log')
            time.sleep(25)
        update(state='scoring',cached=exp['train'],control_steps=4000,gan_steps=4000,message='GPU完成，CPU核验采样、1024图片与独立年龄指标')
        logs=[[json.loads(s) for s in (run/arm/'train.jsonl').read_text().splitlines()] for arm in ('native_control','native_gan')]
        fields=['step','names','real','target','jitter','flip','contrast']
        if any(len(x)!=4000 for x in logs) or any(any(a[k]!=b[k] for k in fields) for a,b in zip(*logs)):raise ValueError('Paired sampling differs or incomplete')
        complete=[json.loads((run/arm/'completed.json').read_text()) for arm in ('native_control','native_gan')]
        if complete[0]['frozen_tensors_sha256']!=complete[1]['frozen_tensors_sha256'] or any(x['unique_images']!=exp['train'] for x in complete):raise ValueError('Incomplete full data coverage or different frozen models')
        write_json(run/'paired-verification.json',dict(steps=4000,batch=2,sampling_equal=True,frozen_parameters_equal=True,all_training_images_seen=True,no_automatic_promotion=True))
        merge(run)
        from PIL import Image
        report=json.loads((run/'evaluation/report.json').read_text());counts=dict(raw=0,color=0)
        for row in report['outputs']:
            for key,label in [('native','raw'),('preserved','color')]:
                path=run/'evaluation'/row[key]
                with Image.open(path) as im:im.load();assert im.size==(1024,1024) and im.format=='PNG'
                if key=='native' and sha256(path)!=row['sha256']:raise ValueError('PNG changed')
                counts[label]+=1
        write_json(run/'artifact-validation.json',dict(**counts,all_decoded=True,size=[1024,1024],raw_sha256_verified=True))
        cmd(sys.executable,ROOT/'integration/quality/summarize_highres_texture.py',run)
        cmd(ROOT/'deployment/asia-adaptation/20261123/age-cpu-venv/bin/python',ROOT/'integration/quality/check_independent_age.py','--model',ROOT/'deployment/asia-adaptation/20261123/mivolo_v2_age.onnx','--sources',run/'independent-age-inputs.json','--out',run/'independent-age.json')
        cmd(sys.executable,ROOT/'integration/quality/summarize_highres_texture.py',run,'--age-report',run/'independent-age.json')
        cmd(sys.executable,ROOT/'integration/quality/measure_texture_changes.py',run)
        summary=json.loads((run/'summary.json').read_text());write_json(run/'decision.json',dict(selected='wplus_base',promotion=False,visually_reviewed=[],numeric_guard_pass=summary['numeric_guard_pass'],gallery_note='全量原生高清训练对照：所有合格训练图均已遍历，两组各4000步／batch2。数值评估已完成，原生纹理尚待复核；未替换演示模型。'))
        cmd(sys.executable,ROOT/'integration/quality/build_texture_review.py',run,run/'gallery');link('age-native-full-review',run/'gallery')
        cmd(ROOT/'deployment/control-venv/bin/python',ROOT/'integration/quality/verify_texture_review_browser.py','http://127.0.0.1:65512/age-native-full-review/',run,'--texture',timeout=180)
        j=job.get_job(ctx,job.models.GetJobReq(Project='fdj-infra',JobId=receipt['job_id']));write_json(run/'platform-final.json',dict(job_id=receipt['job_id'],phase=j.Phase,allocated_gpus=1))
        write_json(run/'automatic-completion.json',dict(status='ready',native_review_pending=True,numeric_guard_pass=summary['numeric_guard_pass'],default_model_changed=False,gallery='/age-native-full-review/'))
        update(state='ready',message='全量训练、独立指标及网页完成；候选尚待原生纹理复核');print('NATIVE_STUDY_READY',run,flush=True)
    except Exception as error:
        update(state='failed',message=str(error)[:300]);write_json(run/'automatic-failure.json',dict(error=repr(error)));raise


if __name__=='__main__':main()

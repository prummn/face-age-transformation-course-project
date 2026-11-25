"""Finish the single submitted direction-matching job; never submit training."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from probe_inversion import sha256,write_json
from finish_small_quality import read_state

ROOT=Path(__file__).resolve().parents[2]
SLUG='age-direction-match'


def link(name,target):
    path=ROOT/'原型'/name
    if path.is_symlink() and path.resolve()==target.resolve():return
    if path.exists() or path.is_symlink():raise ValueError('Preserving existing page')
    path.symlink_to(os.path.relpath(target,path.parent),target_is_directory=True)


def step(path):
    if not path.exists():return 0
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-12000));lines=f.read().decode().splitlines()
    for line in reversed(lines):
        try:return json.loads(line)['step']
        except (ValueError,KeyError):pass
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();run=a.run.resolve()
    exp=json.loads((run/'experiment.json').read_text());receipt=json.loads((run/'receipt.json').read_text())
    if exp['profile']!='original_delta_matched_generator_pair':raise ValueError('Unexpected profile')
    public=run/'public-status';public.mkdir(exist_ok=True)
    (public/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>年龄方向匹配训练进度</title><style>body{font:17px/1.8 system-ui;max-width:850px;margin:40px auto;padding:16px;background:#eff2ee;color:#21312c}pre{white-space:pre-wrap}</style><h1>原SAM年龄方向匹配训练</h1><p>两组各2000步，共用训练图／锚点／损失，只改变训练年龄潜变量来源；35图固定代码评估。单卡任务。数值通过不等于自然度改善，不自动切换模型。</p><pre id="s">读取进度……</pre><p><a id="r" href="/age-direction-match-review/" hidden>查看完整真实输出</a></p><script>async function tick(){try{const s=await(await fetch('status.json',{cache:'no-store'})).json();document.querySelector('#s').textContent=s.message+'\\n训练代码：'+s.cached+'/60\\n微调编码器方向组：'+s.control_steps+'/2000\\n原SAM方向组：'+s.original_steps+'/2000\\n更新时间：'+s.updated;document.querySelector('#r').hidden=s.state!=='ready'}catch(e){}}tick();setInterval(tick,10000)</script></html>''')
    link(SLUG,public);status={}
    def update(**values):
        status.update(values);status['updated']=datetime.now(timezone.utc).isoformat();write_json(public/'status.json',status)
    def cmd(*parts,timeout=1200):subprocess.run([str(x) for x in parts],cwd=ROOT,check=True,timeout=timeout)
    sys.path.insert(0,str(ROOT/'integration'));from deploy_yicloud import client
    job,ctx=client();deadline=time.monotonic()+90*60;lastcheck=0
    update(state='running',message='已提交单卡任务',cached=0,control_steps=0,original_steps=0)
    try:
        while not (run/'DONE').exists():
            n=len(list((run/'train-codes-original').glob('*.pt'))) if (run/'train-codes-original').exists() else 0
            c,g=step(run/'control/train.jsonl'),step(run/'patch_gan/train.jsonl')
            message='缓存同锚点的两种年龄方向' if n<60 else ('训练微调编码器方向组' if c<2000 else ('训练原SAM方向组' if g<2000 else '生成35张回归的原生1024结果'))
            update(cached=n,control_steps=c,original_steps=g,message=message)
            if time.monotonic()>deadline:raise TimeoutError('Observer deadline; job has75m cap')
            if time.monotonic()-lastcheck>60:
                result=read_state(job,ctx,receipt['job_id']);lastcheck=time.monotonic()
                if result.Phase in ('Failed','Stopped'):raise RuntimeError('Platform '+result.Phase+'; inspect worker.log')
            time.sleep(15)
        update(state='scoring',cached=60,control_steps=2000,original_steps=2000,message='GPU输出完成，CPU独立估龄与采样核验')
        logs=[[json.loads(s) for s in (run/arm/'train.jsonl').read_text().splitlines()] for arm in ['control','patch_gan']]
        fields=['step','name','real','target','flip','jitter','patch_flip','contrast']
        if any(len(x)!=2000 for x in logs) or any(any(a[k]!=b[k] for k in fields) for a,b in zip(*logs)):raise ValueError('Paired sampling differs')
        completed=[json.loads((run/arm/'completed.json').read_text()) for arm in ['control','patch_gan']]
        if any(not c['frozen_verified'] for c in completed) or completed[0]['frozen_tensors_sha256']!=completed[1]['frozen_tensors_sha256']:raise ValueError('Frozen models differ')
        if any(c['unique_images']!=60 for c in completed):raise ValueError('Training coverage incomplete')
        write_json(run/'paired-verification.json',dict(steps=2000,all60_seen=True,sampling_equal=True,frozen_models_equal=True))
        report=json.loads((run/'evaluation/report.json').read_text());counts=dict(raw=0,color=0)
        if len(report['records'])!=35 or report['baseline_route']!='original_delta_constrained' or not report['reused_baseline_exact']:raise ValueError('Unexpected evaluation baseline')
        from PIL import Image
        for row in report['outputs']:
            for key,label in [('native','raw'),('preserved','color')]:
                path=run/'evaluation'/row[key]
                with Image.open(path) as im:im.load();assert im.format=='PNG' and im.size==(1024,1024)
                if key=='native' and sha256(path)!=row['sha256']:raise ValueError('PNG changed')
                counts[label]+=1
        write_json(run/'artifact-validation.json',dict(**counts,all_decoded=True,size=[1024,1024],raw_sha256_verified=True,baseline_png_exact=True))
        parent=Path(exp['fixed_code_run'])
        if (parent/'supplemental-publisher-labels.json').is_file():
            (run/'supplemental-publisher-labels.json').write_bytes((parent/'supplemental-publisher-labels.json').read_bytes())
        cmd(sys.executable,ROOT/'integration/quality/summarize_highres_texture.py',run)
        cmd(ROOT/'deployment/asia-adaptation/20261123/age-cpu-venv/bin/python',ROOT/'integration/quality/check_independent_age.py','--model',ROOT/'deployment/asia-adaptation/20261123/mivolo_v2_age.onnx','--sources',run/'independent-age-inputs.json','--out',run/'independent-age.json')
        cmd(sys.executable,ROOT/'integration/quality/summarize_highres_texture.py',run,'--age-report',run/'independent-age.json')
        cmd(sys.executable,ROOT/'integration/quality/measure_texture_changes.py',run)
        summary=json.loads((run/'summary.json').read_text())
        write_json(run/'decision.json',dict(selected='wplus_base',baseline_route='original_delta_constrained',promotion=False,visually_reviewed=[],numeric_guard_pass=summary['numeric_guard_pass'],gallery_note='当前原SAM差值＋约束作为固定基线。两支生成器训练仅改变训练潜变量方向；原生1024／原始和保色全部留档。35图为开发回归；数值已评估，原生自然度尚待复核，演示模型未替换。'))
        cmd(sys.executable,ROOT/'integration/quality/build_texture_review.py',run,run/'gallery');link(SLUG+'-review',run/'gallery')
        cmd(ROOT/'deployment/control-venv/bin/python',ROOT/'integration/quality/verify_texture_review_browser.py','http://127.0.0.1:65512/'+SLUG+'-review/',run,'--texture',timeout=180)
        result=read_state(job,ctx,receipt['job_id'])
        write_json(run/'platform-final.json',dict(job_id=receipt['job_id'],phase=result.Phase,allocated_gpus=1))
        write_json(run/'automatic-completion.json',dict(status='ready',native_review_pending=True,default_model_changed=False,gallery='/'+SLUG+'-review/'))
        update(state='ready',message='训练、独立评估、1024和网页核验完成；原生纹理待复核')
        print('DIRECTION_STUDY_READY',run,flush=True)
    except Exception as error:
        update(state='failed',message=str(error)[:300]);write_json(run/'automatic-failure.json',dict(error=repr(error)));raise


if __name__=='__main__':main()

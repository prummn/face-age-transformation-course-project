"""Watch one recorded single-GPU study; score on CPU and publish a local result page.

Never submits/retries GPU jobs, changes model defaults, or publishes to GitHub.
The status page remains useful after the interactive Codex session ends.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
PAGE='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>W+匹配训练对照进度</title><style>body{font:17px/1.8 system-ui;max-width:850px;margin:50px auto;padding:20px;background:#f0f3ef;color:#21312c}pre{white-space:pre-wrap}a{color:#285c4b}</style><h1>W+匹配训练对照</h1><p>比较直接编码潜变量训练与W+匹配训练。固定60张训练图，两组各1,000步，最多一张GPU。候选不会自动替换演示模型。</p><pre id="status">正在读取进度……</pre><p><a id="result" hidden href="/age-latent-match-review/">查看全部真实生成结果</a></p><p>页面每10秒更新；结果生成并通过检查后自动打开对照页。独立估龄只作保护项，不代表真实年龄。</p><script>async function tick(){try{const r=await fetch('status.json',{cache:'no-store'}),s=await r.json();document.getElementById('status').textContent=s.message+'\\nW+缓存：'+s.cached+'/60\\n直接编码训练：'+s.direct_steps+'/1000\\nW+匹配训练：'+s.matched_steps+'/1000\\n更新时间：'+s.updated;if(s.state==='ready'){document.getElementById('result').hidden=false;location.replace('/age-latent-match-review/')}}catch(e){document.getElementById('status').textContent='进度暂不可读，稍后自动重试。'}}tick();setInterval(tick,10000)</script>'''


def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def safe_link(name,target):
    link=ROOT/'原型'/name
    if link.is_symlink() and link.resolve()==target.resolve():return
    if link.exists() or link.is_symlink():raise ValueError('Preserving existing page: '+str(link))
    link.symlink_to(os.path.relpath(target,link.parent),target_is_directory=True)


def latest_step(path):
    if not path.exists():return 0
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-8000));lines=stream.read().decode().splitlines()
    for line in reversed(lines):
        try:return json.loads(line)['step']
        except (ValueError,KeyError):continue
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    parser.add_argument('--age-python',type=Path,default=ROOT/'deployment/asia-adaptation/20261123/age-cpu-venv/bin/python')
    parser.add_argument('--browser-python',type=Path,default=ROOT/'deployment/control-venv/bin/python')
    args=parser.parse_args();run=args.run.resolve()
    experiment=json.loads((run/'experiment.json').read_text())
    if experiment['profile'] not in ('matched_training_latent_distribution','conditional_age_preserved_texture'):raise ValueError('Unexpected study')
    conditional=experiment['profile']=='conditional_age_preserved_texture'
    arms=['control','patch_gan'] if conditional else ['direct_latent','matched_wplus']
    slug='age-conditional-texture' if conditional else 'age-latent-match'
    receipt=json.loads((run/'receipt.json').read_text())
    page=PAGE.replace('age-latent-match',slug)
    if conditional:
        page=page.replace('W+匹配训练对照','年龄条件纹理对照').replace('直接编码潜变量训练与W+匹配训练','年龄条件重建对照与额外局部GAN监督').replace('直接编码训练','条件重建训练').replace('W+匹配训练','局部GAN训练')
    folder=run/'public-status';folder.mkdir(exist_ok=True);(folder/'index.html').write_text(page)
    safe_link(slug,folder)
    status=dict(state='running',message='正在准备训练侧W+代码',cached=0,direct_steps=0,matched_steps=0)
    def update(**changes):
        status.update(changes);status['updated']=datetime.now(timezone.utc).isoformat();atomic(folder/'status.json',status)
    def command(*parts,timeout=900):subprocess.run([str(x) for x in parts],cwd=ROOT,check=True,timeout=timeout)
    sys.path.insert(0,str(ROOT/'integration'))
    from deploy_yicloud import client
    job,ctx=client();deadline=time.monotonic()+65*60;last_platform_check=0
    update()
    try:
        while not (run/'DONE').exists():
            if time.monotonic()>deadline:raise TimeoutError('Observer deadline; GPU task has its own48m cap')
            count=len(list((run/'train-codes').glob('*.pt'))) if (run/'train-codes').exists() else 0
            direct=latest_step(run/arms[0]/'train.jsonl');matched=latest_step(run/arms[1]/'train.jsonl')
            phase='拟合训练侧W+代码' if count<60 else ('训练共同对照组' if direct<1000 else ('训练实验组' if matched<1000 else '生成固定代码对照结果'))
            update(cached=count,direct_steps=direct,matched_steps=matched,message=phase)
            if time.monotonic()-last_platform_check>60:
                result=job.get_job(ctx,job.models.GetJobReq(Project='fdj-infra',JobId=receipt['job_id']));last_platform_check=time.monotonic()
                if result.Phase in ('Failed','Stopped'):raise RuntimeError('GPU task '+result.Phase+'; inspect worker.log')
            time.sleep(20)
        update(state='scoring',message='GPU生成完成，正在CPU独立评估',cached=60,direct_steps=1000,matched_steps=1000)
        logs=[[json.loads(x) for x in (run/name/'train.jsonl').read_text().splitlines()] for name in arms]
        if any(len(x)!=1000 for x in logs):raise ValueError('Incomplete paired training')
        fields=['step','name','real','target','flip','jitter','patch_flip','contrast']
        if any(any(a[k]!=b[k] for k in fields) for a,b in zip(*logs)):raise ValueError('Paired sampling differs')
        completions=[json.loads((run/name/'completed.json').read_text()) for name in arms]
        if completions[0]['frozen_tensors_sha256']!=completions[1]['frozen_tensors_sha256']:raise ValueError('Frozen models differ')
        atomic(run/'paired-verification.json',dict(steps=1000,sampling_equal=True,frozen_parameters_equal=True,no_automatic_promotion=True))
        if conditional:
            import numpy as np
            from PIL import Image
            report=json.loads((run/'evaluation/report.json').read_text())
            if report.get('conditional_decode_batch_size')!=5 or report.get('zero_gate_parameters_exact') is not True:raise ValueError('Conditional evaluation must use baseline batch shape and exact zero-gate parameters')
            baselines={(x['name'],x['condition']):x for x in report['outputs'] if x['route']=='wplus_base'}
            checks=[]
            for row in report['outputs']:
                if row['route']=='wplus_base' or row['condition'] not in ('20','40'):continue
                base=baselines[row['name'],row['condition']]
                with Image.open(run/'evaluation'/base['native']) as im:reference=np.asarray(im.convert('RGB')).astype(np.int16)
                with Image.open(run/'evaluation'/row['native']) as im:output=np.asarray(im.convert('RGB')).astype(np.int16)
                maximum=int(np.abs(reference-output).max())
                if maximum!=0:raise ValueError('Same-shape zero-gate PNG changed: '+row['name'])
                checks.append(dict(name=row['name'],route=row['route'],condition=row['condition'],max_rgb8_difference=maximum))
            if len(checks)!=60:raise ValueError('Missing young-age invariance checks')
            atomic(run/'young-age-invariance.json',dict(checked=len(checks),same_batch_shape=True,parameter_identity_exact=True,maximum_difference=max(x['max_rgb8_difference'] for x in checks),checks=checks))
        command(sys.executable,ROOT/'integration/quality/summarize_highres_texture.py',run)
        command(args.age_python,ROOT/'integration/quality/check_independent_age.py','--model',ROOT/'deployment/asia-adaptation/20261123/mivolo_v2_age.onnx',
            '--sources',run/'independent-age-inputs.json','--out',run/'independent-age.json')
        command(sys.executable,ROOT/'integration/quality/summarize_highres_texture.py',run,'--age-report',run/'independent-age.json')
        command(sys.executable,ROOT/'integration/quality/measure_texture_changes.py',run)
        summary=json.loads((run/'summary.json').read_text())
        note='年龄条件纹理对照：共同年龄条件参数适配与年龄响应保护，仅实验组增加256像素局部GAN。原始1024真实输出；数值评估完成，纹理自然度尚待人工复核，未替换演示模型。' if conditional else '潜变量匹配机制对照：原W+基线、直接编码训练、W+匹配训练。固定同一组验证代码与原生1024输出；数值结果已完成，纹理自然度尚待人工复核，未替换演示模型。'
        atomic(run/'decision.json',dict(selected='wplus_base',promotion=False,visually_reviewed=[],numeric_guard_pass=summary['numeric_guard_pass'],gallery_note=note))
        gallery=run/'gallery';command(sys.executable,ROOT/'integration/quality/build_texture_review.py',run,gallery)
        safe_link(slug+'-review',gallery)
        command(args.browser_python,ROOT/'integration/quality/verify_texture_review_browser.py','http://127.0.0.1:65512/'+slug+'-review/',run,'--texture',timeout=180)
        atomic(run/'automatic-completion.json',dict(status='ready',completed=datetime.now(timezone.utc).isoformat(),numeric_guard_pass=summary['numeric_guard_pass'],
            native_review_pending=True,default_model_changed=False,gallery='/'+slug+'-review/'))
        update(state='ready',message='训练、独立评估和网页检查完成；原生纹理尚待人工复核')
        print('MATCHED_STUDY_READY',run,flush=True)
    except Exception as error:
        update(state='failed',message='流程停止：'+str(error)[:300]);atomic(run/'automatic-failure.json',dict(error=repr(error)))
        raise


if __name__=='__main__':main()

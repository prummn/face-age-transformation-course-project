"""Observe one existing latent-editor job and finish CPU evaluation; no GPU submission."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from probe_inversion import sha256, write_json
from finish_small_quality import read_state
from finish_native_age import link, last_step

ROOT = Path(__file__).resolve().parents[2]
SLUG = 'age-latent-editor'
ARMS = ['latent_editor_control', 'latent_editor_gan']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    exp = json.loads((run / 'experiment.json').read_text())
    receipt = json.loads((run / 'receipt.json').read_text())
    if exp['profile'] != 'frozen_generator_low_rank_latent_editor_pair':
        raise ValueError('Unexpected profile')
    public = run / 'public-status'
    public.mkdir(exist_ok=True)
    (public / 'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>W+年龄残差实验</title><style>body{font:17px/1.8 system-ui;max-width:850px;margin:40px auto;padding:16px;background:#eff2ee;color:#21312c}pre{white-space:pre-wrap}</style><h1>W+年龄残差与高龄局部对抗</h1><p>冻结SAM全部编码器和生成器，两臂各2000步，仅一臂加入真实高龄局部GAN。35张开发回归；不自动替换演示模型。</p><pre id="s">读取进度……</pre><p><a id="r" href="/age-latent-editor-review/" hidden>查看完整真实输出</a></p><script>async function tick(){try{const s=await(await fetch('status.json',{cache:'no-store'})).json();document.querySelector('#s').textContent=s.message+'\\n无GAN组：'+s.control_steps+'/2000\\n高龄GAN组：'+s.gan_steps+'/2000\\n更新时间：'+s.updated;document.querySelector('#r').hidden=s.state!=='ready'}catch(e){}}tick();setInterval(tick,10000)</script></html>''')
    link(SLUG, public)
    status = {}

    def update(**values):
        status.update(values)
        status['updated'] = datetime.now(timezone.utc).isoformat()
        write_json(public / 'status.json', status)

    def cmd(*parts, timeout=1200):
        subprocess.run([str(x) for x in parts], cwd=ROOT, check=True, timeout=timeout)

    sys.path.insert(0, str(ROOT / 'integration'))
    from deploy_yicloud import client
    job, ctx = client()
    deadline, last = time.monotonic() + 55 * 60, 0
    update(state='running', message='单卡训练／输出正在进行', control_steps=0, gan_steps=0)
    try:
        while not (run / 'DONE').exists():
            c, g = [last_step(run / arm / 'train.jsonl') for arm in ARMS]
            update(control_steps=c, gan_steps=g, message='生成35图原生输出' if g == 2000 else '单卡成对训练')
            if time.monotonic() > deadline:
                raise TimeoutError('Observer deadline; platform job capped at45 minutes')
            if time.monotonic() - last > 45:
                state = read_state(job, ctx, receipt['job_id'])
                last = time.monotonic()
                if state.Phase in ('Failed', 'Stopped'):
                    raise RuntimeError('Platform ' + state.Phase + '; inspect worker.log')
            time.sleep(10)
        update(state='scoring', control_steps=2000, gan_steps=2000, message='CPU独立估龄、PNG与网页检查')
        logs = [[json.loads(s) for s in (run / arm / 'train.jsonl').read_text().splitlines()] for arm in ARMS]
        fields = ['step', 'name', 'target', 'real', 'jitter', 'flip', 'contrast']
        if any(len(log) != exp['steps'] for log in logs) or any(any(a[k] != b[k] for k in fields) for a, b in zip(*logs)):
            raise ValueError('Paired sampling differs')
        completed = [json.loads((run / arm / 'completed.json').read_text()) for arm in ARMS]
        if any(c['source_images'] != 60 for c in completed):
            raise ValueError('Training coverage incomplete')
        frozen = json.loads((run / 'frozen-verification.json').read_text())
        if frozen['before'] != frozen['after'] or not frozen['all_sam_parameters_unchanged']:
            raise ValueError('Frozen SAM mismatch')
        report = json.loads((run / 'evaluation/report.json').read_text())
        if len(report['records']) != 35 or len(report['outputs']) != 525:
            raise ValueError('Incomplete evaluation')
        from PIL import Image
        counts = dict(raw=0, color=0)
        baseline = {(o['name'], o['condition']): o for o in report['outputs'] if o['route'] == 'wplus_base'}
        protected = 0
        for row in report['outputs']:
            for field, key in [('native', 'raw'), ('preserved', 'color')]:
                path = run / 'evaluation' / row[field]
                with Image.open(path) as im:
                    im.load()
                    if im.size != (1024, 1024) or im.format != 'PNG':
                        raise ValueError('Invalid native PNG')
                if field == 'native' and sha256(path) != row['sha256']:
                    raise ValueError('Output SHA256 mismatch')
                counts[key] += 1
            if row['condition'] in ('same', '20', '40'):
                if row['sha256'] != baseline[row['name'], row['condition']]['sha256']:
                    raise ValueError('Protected same/young PNG differs')
                protected += 1
        write_json(run / 'artifact-validation.json', dict(**counts, size=[1024, 1024], all_decoded=True,
            raw_sha256_verified=True, protected_same_young_pngs=protected))
        write_json(run / 'paired-verification-cpu.json', dict(steps=exp['steps'], all60_seen=True, sampling_equal=True,
            actual_real_coverage=completed[1]['real_references_used_for_gradients'], all_sam_frozen=True))
        parent = Path(exp['parent_run'])
        labels = parent / 'supplemental-publisher-labels.json'
        if labels.is_file():
            (run / labels.name).write_bytes(labels.read_bytes())
        if not (run / 'independent-age-inputs.json').exists():
            cmd(sys.executable, ROOT / 'integration/quality/summarize_highres_texture.py', run)
        if not (run / 'independent-age.json').exists():
            cmd(ROOT / 'deployment/asia-adaptation/20261123/age-cpu-venv/bin/python',
                ROOT / 'integration/quality/check_independent_age.py', '--model',
                ROOT / 'deployment/asia-adaptation/20261123/mivolo_v2_age.onnx', '--sources',
                run / 'independent-age-inputs.json', '--out', run / 'independent-age.json')
        if not (run / 'summary.json').exists():
            cmd(sys.executable, ROOT / 'integration/quality/summarize_highres_texture.py', run, '--age-report', run / 'independent-age.json')
        if not (run / 'native-change-summary.json').exists():
            cmd(sys.executable, ROOT / 'integration/quality/measure_texture_changes.py', run)
        summary = json.loads((run / 'summary.json').read_text())
        write_json(run / 'decision.json', dict(selected='wplus_base', promotion=False, visually_reviewed=[],
            numeric_guard_pass=summary['numeric_guard_pass'], gallery_note='固定原SAM差值＋约束基线，两支小型W+残差编辑器共用全部冻结SAM。唯一成对差别是真实高龄局部GAN；same／20／40原始PNG逐文件不变。35张为已看过开发回归；数值完成、原生自然度待复核，不自动换线上模型。'))
        if not (run / 'gallery/index.html').is_file():
            cmd(sys.executable, ROOT / 'integration/quality/build_texture_review.py', run, run / 'gallery')
        link(SLUG + '-review', run / 'gallery')
        cmd(ROOT / 'deployment/control-venv/bin/python', ROOT / 'integration/quality/verify_texture_review_browser.py',
            'http://127.0.0.1:65512/' + SLUG + '-review/', run, '--texture', timeout=180)
        state = read_state(job, ctx, receipt['job_id'])
        write_json(run / 'platform-final.json', dict(job_id=receipt['job_id'], phase=state.Phase, allocated_gpus=1))
        write_json(run / 'automatic-completion.json', dict(status='ready', native_review_pending=True,
            default_model_changed=False, gallery='/' + SLUG + '-review/'))
        update(state='ready', message='训练、独立评估、1024与网页检查完成；原生纹理待复核')
        print('LATENT_EDITOR_READY', run, flush=True)
    except Exception as error:
        update(state='failed', message=str(error)[:300])
        write_json(run / 'automatic-failure.json', dict(error=repr(error)))
        raise


if __name__ == '__main__':
    main()

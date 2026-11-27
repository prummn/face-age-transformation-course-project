"""CPU-only artifact verification, independent age scoring and review gallery."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from probe_inversion import sha256, write_json

ROOT = Path(__file__).resolve().parents[2]

PAGE = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>StyleGANEX 与 SAM 同图对照</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f1f3ef;color:#24352c;font:16px/1.6 system-ui}main{max-width:1500px;margin:auto;padding:24px}h1{font-size:27px}.controls{display:flex;gap:12px;flex-wrap:wrap}label{display:grid}select{padding:9px;font:inherit;max-width:100%}#grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:22px}figure{margin:0;padding:9px;background:white;min-width:0}img{display:block;width:100%;aspect-ratio:1;object-fit:contain}a{color:#225b47}#note{padding:14px;background:#e0e9dd}.metric,small{font-size:13px}#credit{overflow-wrap:anywhere}@media(max-width:700px){main{padding:12px}#grid{grid-template-columns:1fr}}</style>
<main><h1>StyleGANEX 与 SAM 同图对照</h1><p id="note">固定8张开发图，原始结果全量保留。SAM是现有原SAM差值＋跨龄约束留档，按目标年龄生成；StyleGANEX按编辑强度生成，二者控制参数不等价。强度0是该编辑模型的重建，正负方向按实际结果判断；1.5／2为超过官方训练脚本强度范围的压力观察。尚未校准数值年龄，也未替换上传应用。</p><p><a href="/">应用</a> · <a href="/age-identity-budget-review/">上一项身份预算实验</a></p>
<div class="controls"><label>照片<select id="subject"></select></label><label>SAM目标年龄<select id="sam"><option value="same">同龄重建</option><option>20</option><option>40</option><option>60</option><option selected>75</option></select></label><label>StyleGANEX编辑强度<select id="scale"></select></label><label>区域<select id="region"><option value="full">全脸预览</option><option value="eye">原生眼周</option><option value="cheek">原生脸颊</option><option value="mouth">原生口周</option></select></label><label>颜色<select id="color"><option value="raw">原始模型输出</option><option value="color">按原照保色</option></select></label></div><p id="detail"></p><div id="grid"></div><p id="credit"></p><small>全脸预览缩至512显示；局部直接裁自1024PNG，下载始终为完整1024。外观估龄是独立模型诊断，不是照片真实年龄或未来容貌。8张均为看过的开发图，没有独立盲评。</small></main>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent),$=x=>document.getElementById(x);for(const r of data.cases)$('subject').add(new Option(r.name,r.name));for(const s of data.scales)$('scale').add(new Option(String(s),'scale_'+s));$('scale').value='scale_1';
function render(){const r=data.cases.find(x=>x.name==$('subject').value),sam=r.outputs.find(x=>x.route==='sam_baseline'&&x.condition===$('sam').value),ex=r.outputs.find(x=>x.route==='styleganex_age'&&x.condition===$('scale').value),color=$('color').value,region=$('region').value;const figures=[{title:'原照',asset:r.reference,age:r.source_age},...[[sam,'SAM · '+($('sam').value==='same'?'同龄':'目标'+$('sam').value)],[ex,'StyleGANEX · 强度'+ex.scale]].map(([o,title])=>({title,asset:o[color],age:o.age,id:o.identity}))];$('grid').replaceChildren();for(const item of figures){const f=document.createElement('figure'),im=document.createElement('img'),cap=document.createElement('figcaption'),m=document.createElement('p'),a=document.createElement('a');im.src=item.asset[region];im.alt=item.title;cap.textContent=item.title;m.className='metric';m.textContent=(item.age==null?'未检出年龄':'独立外观估龄 '+item.age.toFixed(1))+(item.id==null?'':' · 身份相似度 '+item.id.toFixed(3));a.href=item.asset.native;a.download='';a.textContent='下载完整1024 PNG';f.append(im,cap,m,a);$('grid').append(f)}$('detail').textContent='共用相同对齐256输入；高清参考1024。先将SAM设为同龄、EX设为0检查重建，再查看全部编辑强度。';$('credit').replaceChildren();if(r.credit){const a=document.createElement('a');a.href=r.credit;a.target='_blank';a.rel='noopener';a.textContent='原照来源与署名';$('credit').append(a)}}for(const id of ['subject','sam','scale','region','color'])$(id).onchange=render;render();</script></html>'''


def main():
    import cv2
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    from check_independent_age import MODEL_SHA256, MODEL_SOURCE, MEAN, STD
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('run',type=Path)
    p.add_argument('--profile',choices=['styleganex','sfe','agetrans'],default='styleganex'); args = p.parse_args()
    run = args.run.resolve(); report = json.loads((run/'evaluation/report.json').read_text())
    expected = {'styleganex':96,'sfe':152,'agetrans':120}[args.profile]
    profile=json.loads((run/'experiment.json').read_text()).get('profile')
    cpu_pilot = profile in ['official_sfe_cpu_pilot','official_agetrans_cpu_pilot']
    cases = 2 if cpu_pilot else 8
    if cpu_pilot and args.profile == 'sfe':
        if report['scales'] != [0.,5.,10.] or [r['name'] for r in report['records']] != ['ffhq_64376','ffhq_62644']:
            raise ValueError('Unexpected CPU pilot design')
        expected = 22
    elif cpu_pilot:
        if report['groups'] != list(range(10)) or [r['name'] for r in report['records']] != ['ffhq_64376','ffhq_62644']:
            raise ValueError('Unexpected CPU age-group pilot design')
        expected = 30
    if not (run/'DONE').exists(): raise ValueError('GPU run incomplete')
    if len(report['records']) != cases or len(report['outputs']) != expected: raise ValueError('Incomplete design')
    frozen = json.loads((run/'frozen-verification.json').read_text())
    repeat = json.loads((run/'repeatability.json').read_text())
    if not frozen['equal'] or not repeat['raw_pixels_equal']: raise ValueError('State/repetition check failed')
    model = ROOT/'deployment/asia-adaptation/20261123/mivolo_v2_age.onnx'
    if sha256(model) != MODEL_SHA256: raise ValueError('Independent age model changed')
    cv2.setNumThreads(2); options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model),options,providers=['CPUExecutionProvider'])
    det = cv2.FaceDetectorYN.create(str(ROOT/'deployment/model-files/face-age/models/yunet/face_detection_yunet_2023mar.onnx'),'',(256,256),.65,.3,100)
    body = (np.zeros((3,384,384),dtype=np.float32)-MEAN)/STD
    def predict(path):
        bgr = cv2.imread(str(path)); _, faces = det.detect(cv2.resize(bgr,(256,256),interpolation=cv2.INTER_AREA))
        if faces is None: return None
        x,y,w,h = max(faces,key=lambda f:f[2]*f[3])[:4]; sx,sy = bgr.shape[1]/256,bgr.shape[0]/256
        crop = bgr[max(0,round(y*sy)):round((y+h)*sy),max(0,round(x*sx)):round((x+w)*sx)]
        if not crop.size: return None
        h,w = crop.shape[:2]; ratio = 384/max(h,w); tw,th = round(w*ratio),round(h*ratio)
        image = cv2.resize(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB),(tw,th),interpolation=cv2.INTER_LINEAR)
        frame = np.zeros((384,384,3),dtype=np.uint8); left,top = (384-tw)//2,(384-th)//2
        frame[top:top+th,left:left+tw] = image
        face = (frame.transpose(2,0,1).astype(np.float32)/255-MEAN)/STD
        age = float(session.run(['age'],{'faces_bodies':np.concatenate([face,body],axis=0)[None]})[0].item())
        if not np.isfinite(age): raise ValueError('Nonfinite age')
        return age
    cache = run/'independent-age.json'; ages = json.loads(cache.read_text())['ages'] if cache.exists() else {}
    for item in report['outputs']:
        path = run/'evaluation'/item['native']
        if sha256(path) != item['sha256']: raise ValueError('Output hash changed')
        for field in ['native','preserved']:
            with Image.open(run/'evaluation'/item[field]) as im:
                im.load()
                if im.size != (1024,1024) or im.format != 'PNG': raise ValueError('Not native1024 PNG')
        if item['native'] not in ages:
            ages[item['native']] = predict(path)
            write_json(cache,dict(model_source=MODEL_SOURCE,model_sha256=MODEL_SHA256,ages=ages))
    sources = {r['name']:predict(run/'evaluation'/r['name']/'reference.png') for r in report['records']}
    groups = defaultdict(list)
    for item in report['outputs']: groups[item['route'],item['condition']].append(item)
    reconstruction = {(item['name'],item['route']):item for item in report['outputs']
                      if item['condition'] in ['same','scale_0']}
    def mean(values):
        v = [x for x in values if x is not None]; return statistics.mean(v) if v else None
    summary = {}
    for (route,condition),items in groups.items():
        summary[route+'/'+condition] = dict(cases=len(items),identity=mean(r['source_cosine'] for r in items),
            independent_age=mean(ages[r['native']] for r in items),missing_sface=sum(not r['detected'] for r in items),
            missing_age=sum(ages[r['native']] is None for r in items),reconstruction_l1=mean(r['reconstruction_l1'] for r in items),
            age_change_from_source=mean(ages[r['native']]-sources[r['name']]
                for r in items if ages[r['native']] is not None and sources[r['name']] is not None),
            age_change_from_reconstruction=mean(ages[r['native']]-ages[reconstruction[r['name'],route]['native']]
                for r in items if (r['name'],route) in reconstruction and ages[r['native']] is not None and ages[reconstruction[r['name'],route]['native']] is not None),
            identity_change_from_reconstruction=mean(r['source_cosine']-reconstruction[r['name'],route]['source_cosine']
                for r in items if (r['name'],route) in reconstruction and r['source_cosine'] is not None and reconstruction[r['name'],route]['source_cosine'] is not None))
    write_json(run/'summary.json',dict(groups=summary,independent_ages=ages,source_ages=sources,
        promotion=False,visual_review_pending=True,targets_not_calibrated=True,
        note=f'Fixed condition scan, not chronological-age accuracy. {cases} inspected development cases only.',cpu_pilot=cpu_pilot,
        device=json.loads((run/'device.json').read_text())))
    write_json(run/'artifact-validation.json',dict(raw=expected,color=expected,native_size=[1024,1024],all_decodable=True,
        raw_sha256_verified=True,model_frozen=True,repeatable=True,same_aligned_reference=True,
        same_aligned_input_pixels=args.profile == 'styleganex'))
    gallery = run/'gallery'; gallery.mkdir(exist_ok=False); cache_assets = {}
    def asset(path):
        if str(path) in cache_assets: return cache_assets[str(path)]
        if not path.resolve().is_relative_to(run/'evaluation'): raise ValueError('Asset outside evaluation')
        dest = gallery/'images'/path.relative_to(run/'evaluation'); dest.parent.mkdir(parents=True,exist_ok=True)
        dest.symlink_to(path); value = dict(native=str(dest.relative_to(gallery)))
        with Image.open(path) as im:
            image = im.convert('RGB'); preview = dest.with_suffix('.jpg')
            image.resize((512,512),Image.Resampling.LANCZOS).save(preview,quality=93); value['full'] = str(preview.relative_to(gallery))
            for key,center in [('eye',(.34,.40)),('cheek',(.32,.57)),('mouth',(.5,.69))]:
                left,top = round(1024*center[0])-128,round(1024*center[1])-128
                crop = dest.with_name(dest.stem+'_'+key+'.png'); image.crop((left,top,left+256,top+256)).save(crop)
                value[key] = str(crop.relative_to(gallery))
        cache_assets[str(path)] = value; return value
    cases = []
    for row in report['records']:
        case = dict(name=row['name'],reference=asset(run/'evaluation'/row['name']/'reference.png'),
            source_age=sources[row['name']],credit=row.get('source_metadata',{}).get('photo_url'),outputs=[])
        for item in report['outputs']:
            if item['name'] != row['name']: continue
            case['outputs'].append(dict(route=item['route'],condition=item['condition'],scale=item['scale'],
                group=item.get('group'),
                age=ages[item['native']],identity=item['source_cosine'],raw=asset(run/'evaluation'/item['native']),
                color=asset(run/'evaluation'/item['preserved'])))
        cases.append(case)
    data = dict(cases=cases,scales=report.get('scales',report.get('groups')))
    template = PAGE
    if args.profile == 'sfe':
        template = (Path(__file__).with_name('sfe_review.html')).read_text()
        if cpu_pilot:
            template = template.replace('固定8张开发图','固定2张亚洲男女开发图（CPU小样本，未占GPU）').replace('8张已看过的开发图','2张已看过的开发图')
    elif args.profile == 'agetrans':
        template = (Path(__file__).with_name('agetrans_review.html')).read_text()
        if cpu_pilot:
            template = template.replace('固定8张开发图','固定2张亚洲男女开发图（CPU小样本，未占GPU）').replace('8张已看过的开发图','2张已看过的开发图')
        elif profile == 'official_agetrans_cpu_full':
            template = template.replace('固定8张开发图','固定8张开发图（CPU执行，未占GPU）')
    (gallery/'index.html').write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('<','\\u003c')))
    print(args.profile.upper()+'_CPU_AND_GALLERY_READY',json.dumps(summary),flush=True)


if __name__ == '__main__': main()

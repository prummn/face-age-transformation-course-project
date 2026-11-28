"""CPU independent scoring and an all-cases review for SFE mechanism studies."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from probe_inversion import sha256,write_json
from check_independent_age import MODEL_SHA256,MODEL_SOURCE,MEAN,STD

ROOT=Path(__file__).resolve().parents[2]
PAGE=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>年龄编辑机制对照</title>
<style>body{font:16px/1.6 system-ui;background:#eef1ed;color:#24342c;margin:20px}main{max-width:1600px;margin:auto}.controls{display:flex;gap:14px;flex-wrap:wrap}label{display:grid}select{padding:8px;font:inherit}#grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin-top:20px}figure{margin:0;background:white;padding:10px}img{width:100%;aspect-ratio:1;object-fit:contain}small{display:block}a{color:#246044}#trajectory{overflow:auto;white-space:pre;font:14px/1.7 monospace}#note{padding:14px;background:#dce6d8}@media(max-width:600px){body{margin:10px}#grid{grid-template-columns:1fr}}</style>
<main><h1>年龄编辑机制对照</h1><p id="note"></p><a href="/age-sfe-review/">官方 SFE 旧对照</a> · <a href="/age-sfe-dense-review/">高龄年轻化密集扫描</a> · <a href="/age-sfe-conditioned-review/">条件训练对照</a>
<div class="controls"><label>照片<select id="subject"></select></label><label id="condition-label">条件<select id="condition"></select></label><label>颜色<select id="color"><option value="raw">原始模型</option><option value="color">原照保色</option></select></label><label>区域<select id="region"><option value="full">全脸</option><option value="eye">原生眼周</option><option value="cheek">原生脸颊</option></select></label></div>
<div id="grid"></div><p id="credit"></p><p>全轨迹（独立外观估龄；不是照片真实年龄或未来容貌）</p><div id="trajectory"></div><small>保色是可选后处理。全脸缩至512显示，局部直接裁自原生1024。所有条件与失败样例保留，未替换上传应用；这些开发图已看过，不是独立盲评。</small></main>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent),$=x=>document.getElementById(x);$('note').textContent=data.note;for(const c of data.cases)$('subject').add(new Option(c.name,c.name));for(const c of data.conditions)$('condition').add(new Option(c,c));$('condition').value=data.default_condition;
function number(v,d=1){return v==null?'未检出':v.toFixed(d)}
function render(){const c=data.cases.find(x=>x.name===$('subject').value),condition=$('condition').value,region=$('region').value,color=$('color').value;const figures=[{label:'原照',asset:c.reference,age:c.source_age},...data.routes.map(route=>{const o=c.outputs.find(x=>x.route===route&&x.condition===condition);return {label:data.labels[route],asset:o[color],age:o.age,id:o.identity,strength:o.scale}})];$('grid').replaceChildren();for(const o of figures){const f=document.createElement('figure'),im=document.createElement('img'),cap=document.createElement('figcaption'),m=document.createElement('small'),a=document.createElement('a');im.src=o.asset[region];im.alt=o.label;cap.textContent=o.label;m.textContent='外观估龄 '+number(o.age)+(o.id==null?'':' · 身份 '+number(o.id,3))+(o.strength==null?'':' · 内部强度 '+number(o.strength,2));a.href=o.asset.native;a.download='';a.textContent='下载完整1024 PNG';f.append(im,cap,m,a);$('grid').append(f)}$('trajectory').textContent=data.routes.map(r=>data.labels[r]+'\n'+c.outputs.filter(o=>o.route===r).map(o=>o.condition+': '+number(o.age)+' / ID '+number(o.identity,3)).join(' | ')).join('\n\n');$('credit').replaceChildren();if(c.credit){const a=document.createElement('a');a.href=c.credit;a.target='_blank';a.rel='noopener';a.textContent='原照来源与署名';$('credit').append(a)}}for(const id of ['subject','condition','color','region'])$(id).onchange=render;render();</script></html>'''


def main():
    import cv2
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--reuse-age',type=Path)
    p.add_argument('--original-sfe',type=Path,default=ROOT/'deployment/training-runs/sfe-20261127-110920')
    a=p.parse_args();run=a.run.resolve()
    exp=json.loads((run/'experiment.json').read_text());dense=exp['profile']=='official_sfe_dense_younger'
    projection=exp['profile']=='official_sfe_glasses_projected';scanned=dense or projection
    if not (run/'DONE').exists():raise ValueError('Incomplete worker')
    report=json.loads((run/'evaluation/report.json').read_text());records=report['records']
    expected=3*(5+2*len(exp['scales'])) if dense else (152 if projection else 168)
    if dense and exp['scales'] not in [[i/2 for i in range(-10,1)],[i/2 for i in range(-20,1)]]:
        raise ValueError('Unexpected dense scan protocol')
    if len(report['outputs'])!=expected or len(records)!=(3 if dense else 8):raise ValueError('Incomplete conditions')
    if not json.loads((run/'frozen-verification.json').read_text())['equal']:raise ValueError('Model tensors changed')
    path=ROOT/'deployment/asia-adaptation/20261123/mivolo_v2_age.onnx'
    if sha256(path)!=MODEL_SHA256:raise ValueError('Age model hash differs')
    opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),opt,providers=['CPUExecutionProvider']);cv2.setNumThreads(2)
    det=cv2.FaceDetectorYN.create(str(ROOT/'deployment/model-files/face-age/models/yunet/face_detection_yunet_2023mar.onnx'),'',(256,256),.65,.3,100)
    body=(np.zeros((3,384,384),dtype=np.float32)-MEAN)/STD
    def predict(path):
        bgr=cv2.imread(str(path));_,faces=det.detect(cv2.resize(bgr,(256,256),interpolation=cv2.INTER_AREA))
        if faces is None:return None
        x,y,w,h=max(faces,key=lambda f:f[2]*f[3])[:4];sx,sy=bgr.shape[1]/256,bgr.shape[0]/256
        crop=bgr[max(0,round(y*sy)):round((y+h)*sy),max(0,round(x*sx)):round((x+w)*sx)]
        if not crop.size:return None
        h,w=crop.shape[:2];ratio=384/max(h,w);tw,th=round(w*ratio),round(h*ratio)
        rgb=cv2.resize(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB),(tw,th));frame=np.zeros((384,384,3),dtype=np.uint8)
        left,top=(384-tw)//2,(384-th)//2;frame[top:top+th,left:left+tw]=rgb
        face=(frame.transpose(2,0,1).astype(np.float32)/255-MEAN)/STD
        result=float(session.run(['age'],{'faces_bodies':np.concatenate([face,body],0)[None]})[0].item())
        if not np.isfinite(result):raise ValueError('Non-finite independent age')
        return result
    agepath=run/'independent-age.json';ages=json.loads(agepath.read_text())['ages'] if agepath.exists() else {}
    if a.reuse_age:
        prior=json.loads((a.reuse_age/'independent-age.json').read_text())
        if prior['model_sha256']!=MODEL_SHA256:raise ValueError('Reused age model differs')
        prior_outputs={o['native']:o for o in json.loads((a.reuse_age/'evaluation/report.json').read_text())['outputs']}
        reused=[]
        for o in report['outputs']:
            old=prior_outputs.get(o['native'])
            if old and o['sha256']==old['sha256'] and o['native'] in prior['ages']:
                ages[o['native']]=prior['ages'][o['native']];reused.append(o['native'])
        write_json(run/'age-reuse.json',dict(source=str(a.reuse_age.resolve()),items=reused,count=len(reused),
            source_age_json_sha256=sha256(a.reuse_age/'independent-age.json'),same_native_sha256=True,model_sha256=MODEL_SHA256))
    for i,o in enumerate(report['outputs']):
        path=run/'evaluation'/o['native']
        if sha256(path)!=o['sha256']:raise ValueError('Output changed')
        for key in ['native','preserved']:
            with Image.open(run/'evaluation'/o[key]) as im:
                im.load()
                if im.size!=(1024,1024) or im.format!='PNG':raise ValueError('Invalid native PNG')
        if o['native'] not in ages:ages[o['native']]=predict(path)
        if (i+1)%20==0:
            write_json(agepath,dict(ages=ages,model_sha256=MODEL_SHA256,model_source=MODEL_SOURCE));print('SCORED',i+1,flush=True)
    write_json(agepath,dict(ages=ages,model_sha256=MODEL_SHA256,model_source=MODEL_SOURCE))
    sources={r['name']:predict(run/'evaluation'/r['name']/'reference.png') for r in records}
    routes=['sfe_age','e4e_age'] if scanned else report['routes']
    outputs=[o for o in report['outputs'] if o['route'] in routes]
    if not scanned:
        for record in records:
            same=[o['sha256'] for o in outputs if o['name']==record['name'] and o['condition']=='same']
            if len(same)!=3 or len(set(same))!=1:raise ValueError('Trained same-age gate changed the reconstruction')
    def mean(v):
        v=[x for x in v if x is not None];return statistics.mean(v) if v else None
    groups=defaultdict(list)
    for o in outputs:groups[o['route']+'/'+o['condition']].append(o)
    summary={k:dict(cases=len(v),identity=mean(o['source_cosine'] for o in v),independent_age=mean(ages[o['native']] for o in v),
        target_proxy_mae=mean(abs(ages[o['native']]-o['target']) for o in v if ages[o['native']] is not None) if not scanned else None,
        missing=sum(ages[o['native']] is None or not o['detected'] for o in v)) for k,v in groups.items()}
    aggregate={};lookup={(o['name'],o['route'],o['condition']):o for o in outputs}
    for route in routes:
        rows=[o for o in outputs if o['route']==route and o['condition']!='same']
        spans=[];violations=0
        if not scanned:
            for r in records:
                trajectory=[ages[lookup[r['name'],route,str(t)]['native']] for t in exp['targets']]
                if all(x is not None for x in trajectory):
                    spans.append(trajectory[-1]-trajectory[0]);violations+=sum(b<a for a,b in zip(trajectory,trajectory[1:]))
        old=[o for o in rows if o['name'] in ['ffhq_67140','ffhq_60764','ffhq_68335'] and o['condition'] in ['20','30','40']]
        aggregate[route]=dict(identity=mean(o['source_cosine'] for o in rows),span20to75=mean(spans),nonmonotonic_pairs=violations,
            target_proxy_mae=mean(abs(ages[o['native']]-o['target']) for o in rows if ages[o['native']] is not None) if not scanned else None,
            older_younger_proxy_mae=mean(abs(ages[o['native']]-o['target']) for o in old if ages[o['native']] is not None),
            missing=sum(ages[o['native']] is None or not o['detected'] for o in rows))
    original=a.original_sfe.resolve()
    original_outputs=[];original_ages={};original_groups={}
    if projection:
        original_report=json.loads((original/'evaluation/report.json').read_text())
        original_outputs=[o for o in original_report['outputs'] if o['route']=='sfe_age']
        original_ages=json.loads((original/'independent-age.json').read_text())['ages']
        original_groups={k:v for k,v in json.loads((original/'summary.json').read_text())['groups'].items() if k.startswith('sfe_age/')}
        if len(original_outputs)!=56:raise ValueError('Incomplete original reference')
        for o in original_outputs:
            if sha256(original/'evaluation'/o['native'])!=o['sha256']:raise ValueError('Original reference changed')
    write_json(run/'summary.json',dict(groups=summary,aggregate=aggregate,source_ages=sources,independent_ages=ages,original_groups=original_groups,
        promotion=False,visual_review_pending=True,targets_not_calibrated=True))
    write_json(run/'artifact-validation.json',dict(raw=expected,color=expected,size=[1024,1024],all_decodable=True,raw_hashes_verified=True))
    gallery=run/'gallery';gallery.mkdir(exist_ok=True);assets={}
    def asset(path):
        if str(path) in assets:return assets[str(path)]
        path=path.resolve()
        if path.is_relative_to(run/'evaluation'):relative=Path('images')/path.relative_to(run/'evaluation')
        elif projection and path.is_relative_to(original/'evaluation'):relative=Path('original')/path.relative_to(original/'evaluation')
        else:raise ValueError('Asset path escaped evaluations')
        dest=gallery/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():dest.symlink_to(path)
        value=dict(native=str(dest.relative_to(gallery)))
        with Image.open(path) as im:
            im=im.convert('RGB');preview=dest.with_suffix('.jpg');im.resize((512,512),Image.Resampling.LANCZOS).save(preview,quality=93)
            value['full']=str(preview.relative_to(gallery))
            for key,(cx,cy) in {'eye':(.34,.40),'cheek':(.32,.57)}.items():
                left,top=round(1024*cx)-128,round(1024*cy)-128;crop=dest.with_name(dest.stem+'_'+key+'.png')
                im.crop((left,top,left+256,top+256)).save(crop);value[key]=str(crop.relative_to(gallery))
        assets[str(path)]=value;return value
    cases=[]
    for r in records:
        case=dict(name=r['name'],source_age=sources[r['name']],reference=asset(run/'evaluation'/r['name']/'reference.png'),
            credit=r.get('source_metadata',{}).get('photo_url'),outputs=[])
        for o in outputs:
            if o['name']!=r['name']:continue
            case['outputs'].append(dict(route=o['route'],condition=o['condition'],scale=o['scale'],age=ages[o['native']],identity=o['source_cosine'],
                raw=asset(run/'evaluation'/o['native']),color=asset(run/'evaluation'/o['preserved'])))
        if projection:
            for o in original_outputs:
                if o['name']!=r['name']:continue
                case['outputs'].append(dict(route='official_sfe',condition=o['condition'],scale=o['scale'],age=original_ages[o['native']],identity=o['source_cosine'],
                    raw=asset(original/'evaluation'/o['native']),color=asset(original/'evaluation'/o['preserved'])))
        cases.append(case)
    data=dict(dense=dense,cases=cases,routes=['official_sfe']+routes if projection else routes,default_condition='scale_-3' if dense else ('scale_5' if projection else '40'),
        labels=dict(official_sfe='原版 SFE 年龄方向',sfe_age='去眼镜耦合 SFE' if projection else '官方 SFE',e4e_age='去耦合 e4e' if projection else '官方 e4e 方向对照',fixed_mapping='未训练的源龄差值映射',age_id_editor='目标年龄＋身份训练',protected_editor='增加眼周与属性保留约束'),
        conditions=['scale_'+format(s,'g') for s in report['scales']] if scanned else ['same']+[str(t) for t in exp['targets']],
        note=f"3张高龄开发图，强度{exp['scales'][0]:g}至0每0.5一步；强度不是目标年龄。用于检查此前粗扫描是否跳过中青年阶段。" if dense else
             ('8张开发图，将原年龄W+方向正交于官方眼镜边界并恢复原范数；固定7强度，不按人调参。正交不保证语义解耦，须同时检查实际老化、身份和原生图。' if projection else
              f"{exp['train']}个训练身份组，两组各{exp['steps_per_arm']}步，相同初始化和采样。"+('关闭W+和空间自由修正，仅学习条件强度。' if exp.get('adapter_mode')=='strength' else '包含W+和空间修正，已发现高龄纹理伪影，作为失败对照保留。')+'增加保留约束能否抑制无端眼镜需要逐图确认；目标条件未完成年龄校准。'))
    (gallery/'index.html').write_text(PAGE.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('<',chr(92)+'u003c')))
    print('FINISHED',json.dumps(aggregate,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

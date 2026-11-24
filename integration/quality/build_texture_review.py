"""Expose true native PNG and fixed local crops for a bounded generator study."""
import argparse
import json
from pathlib import Path
from PIL import Image

PAGE='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>高清纹理训练对照</title>
<style>*{box-sizing:border-box}body{margin:0;color:#21312c;background:#eff2ee;font:16px/1.6 system-ui}main{max-width:1500px;margin:auto;padding:24px}h1{font-size:28px}a{color:#285c4b}.controls{display:flex;gap:12px;flex-wrap:wrap}label{display:grid}select{padding:9px;font:inherit;max-width:100%}#grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:22px}figure{margin:0;background:white;padding:8px;min-width:0}img{width:100%;display:block;aspect-ratio:1;object-fit:contain}figcaption{font-weight:650}.metric,small{font-size:13px}#note{padding:12px;background:#dfe8df}#credit{overflow-wrap:anywhere}@media(max-width:700px){main{padding:12px}#grid{grid-template-columns:repeat(2,minmax(0,1fr))}}</style>
<main><h1>高清纹理训练对照</h1><p id="note"></p><p><a href="/age-quality-review/">上一版年龄监督对照</a> · <a href="/">演示首页</a></p>
<div class="controls"><label>照片<select id="subject"></select></label><label>目标<select id="age"><option value="same">同龄</option><option>20</option><option>40</option><option>60</option><option selected>75</option></select></label><label>查看区域<select id="region"><option value="full">全脸</option><option value="eye">眼周</option><option value="cheek">脸颊</option><option value="mouth">口周</option></select></label><label>颜色<select id="color"><option value="raw">原始模型输出</option><option value="color">保留原照颜色</option></select></label></div>
<p id="detail"></p><div id="grid"></div><p id="credit"></p><small>生成图均为原生 1024 PNG，局部图直接裁剪。外观估龄不是照片真实年龄；本页是开发回归，未做盲评。</small></main><script id="data" type="application/json">__DATA__</script>
<script>const data=JSON.parse(document.querySelector('#data').textContent),$=id=>document.getElementById(id);$('note').textContent=data.note;for(const r of data.cases)$('subject').add(new Option(r.name+' · '+(r.cohort==='new-hd-holdout'?'高清回归':'低清压力回归'),r.name));
function render(){const r=data.cases.find(x=>x.name==$('subject').value),region=$('region').value,color=$('color').value,age=$('age').value;const images=[{title:'原照',...r.reference},...r.outputs.filter(x=>x.condition==age).map(x=>({title:data.labels[x.route],metric:[x.identity!=null?'身份相似度 '+x.identity.toFixed(3):'',x.age!=null?'独立外观估龄 '+x.age.toFixed(1):''].filter(Boolean).join(' · '),...x[color]}))];$('detail').textContent='对齐原照 '+r.size.join('×')+(r.native_edge!=null?'；原始脸区短边约 '+Math.round(r.native_edge)+' 像素':'')+'；所有路线使用相同 W+ 潜变量；生成器适配可单独卸载。';$('grid').replaceChildren();for(const x of images){const f=document.createElement('figure'),img=document.createElement('img'),cap=document.createElement('figcaption'),m=document.createElement('p'),a=document.createElement('a');img.src=x[region];img.alt=x.title;cap.textContent=x.title;m.className='metric';m.textContent=x.metric||'';a.href=x.native;a.download='';a.textContent='下载完整 PNG';f.append(img,cap,m,a);$('grid').append(f)}$('credit').replaceChildren();if(r.credit){const a=document.createElement('a');a.href=r.credit;a.textContent='照片来源与署名';a.target='_blank';a.rel='noopener';$('credit').append(a)}}for(const id of ['subject','age','region','color'])$(id).onchange=render;render();</script></html>'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    run=a.run.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    r=json.loads((run/'evaluation/report.json').read_text());summary=json.loads((run/'summary.json').read_text())
    decision=json.loads((run/'decision.json').read_text());cases=[]
    asset_cache={}
    def asset(path):
        if str(path) in asset_cache:return asset_cache[str(path)]
        if not path.resolve().is_relative_to(run/'evaluation'):raise ValueError('Asset outside evaluation')
        relative=path.relative_to(run/'evaluation');dest=out/'images'/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(path)
        result=dict(native=str(dest.relative_to(out)))
        with Image.open(path) as im:
            im.load();image=im.convert('RGB');w,h=image.size
            preview=dest.with_suffix('.jpg');image.resize((512,512),Image.Resampling.LANCZOS).save(preview,quality=93);result['full']=str(preview.relative_to(out))
            for key,xy in [('eye',(.34,.40)),('cheek',(.32,.57)),('mouth',(.50,.69))]:
                # Generated native1024 crops retain all256 source pixels. Input
                # references smaller than1024 retain their actual native crop size.
                edge=round(min(w,h)*.25);left=round(w*xy[0])-edge//2;top=round(h*xy[1])-edge//2
                crop=dest.with_name(dest.stem+'_'+key+'.png');image.crop((left,top,left+edge,top+edge)).save(crop);result[key]=str(crop.relative_to(out))
        asset_cache[str(path)]=result
        return result
    anchors={(x['name'],x['route']):x for x in r['outputs'] if x['condition']=='same'}
    for row in r['records']:
        folder=run/'evaluation'/row['name'];reference=folder/'reference.png'
        with Image.open(reference) as im:size=list(im.size)
        item=dict(name=row['name'],cohort=row['cohort'],size=size,native_edge=row.get('source_face_short_edge',row.get('native_face_short_edge')),reference=asset(reference),outputs=[],credit=row.get('source_metadata',{}).get('photo_url'))
        if row['cohort'].startswith('new_appa'):item['credit']='https://chalearnlap.cvc.uab.cat/dataset/26/description/'
        for o in r['outputs']:
            if o['name']==row['name']:
                if o['condition']!='same' and o['target']==row['source_age']:
                    o=dict(anchors[o['name'],o['route']],condition=o['condition'],target=o['target'])
                item['outputs'].append(dict(route=o['route'],condition=o['condition'],age=summary['independent_ages'][o['native']],identity=o['source_cosine'],raw=asset(run/'evaluation'/o['native']),color=asset(run/'evaluation'/o['preserved'])))
        cases.append(item)
    labels=dict(wplus_base='W+ 基线',highres_control='高清重建适配',highres_patch_gan='高清重建＋局部 GAN',direct_latent='直接编码潜变量训练',matched_wplus='匹配 W+ 潜变量训练')
    labels.update(conditional_control='年龄条件适配＋年龄保护',conditional_gan='年龄条件适配＋年龄保护＋局部 GAN')
    data=dict(cases=cases,note=decision['gallery_note'],labels=labels)
    (out/'index.html').write_text(PAGE.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('<','\\u003c')))
    print('TEXTURE_GALLERY_READY',len(cases),out)

if __name__=='__main__':main()

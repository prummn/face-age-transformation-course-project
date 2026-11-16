"""Expose completed real results as a static CPU-only comparison page."""
import argparse
import json
from pathlib import Path
import statistics


TITLES=dict(fine_direct='原始直接路线',previous_identity='前轮身份候选',recon_encoder='加强重建：编码器',
            recon_generator='加强重建：编码器＋生成器',latent_anchor='单图重建修正',latent_refined='重建＋跨龄修正',
            latent_constrained60='细节约束：60步',latent_constrained180='细节约束：180步')

PAGE='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>质量改进对照</title>
<style>body{margin:22px;background:#f4f6f2;color:#233b34;font:16px/1.6 system-ui}main{max-width:1500px;margin:auto}h1{font-size:26px}select{padding:8px;font:inherit;max-width:95%;margin:5px}.controls{display:flex;flex-wrap:wrap;gap:12px}.grid{display:grid;gap:12px;grid-template-columns:repeat(var(--columns,4),minmax(0,1fr))}figure{margin:0;background:white;padding:8px;border-radius:8px}img{width:100%;display:block}figcaption{font-weight:600}.info{min-height:4em;font-size:14px}a{color:#17634f}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ccd6ce;text-align:left}.scroll{overflow-x:auto}@media(max-width:800px){body{margin:12px}.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}</style>
<main><h1>人脸年龄编辑：质量改进对照</h1><p>全部为SAM／StyleGAN2真实输出。原始模型、前轮候选与本轮方案分别保留；此页浏览不占GPU。</p>
<div class="controls"><label>方案<select id="family"></select></label><label>照片<select id="sample"></select></label><label>条件<select id="age"><option value="same">同龄重建</option><option>20</option><option>40</option><option>60</option><option>75</option></select></label></div>
<p id="note"></p><div class="grid" id="images"></div><h2>当前方案的分组结果</h2><div class="scroll"><table><thead><tr><th>评估组／方案</th><th>同龄身份↑</th><th>跨龄身份↑</th><th>同龄LPIPS↓</th><th>20→75参考跨度</th></tr></thead><tbody id="metrics"></tbody></table></div>
<p>身份为独立SFace相似度；LPIPS、参考估龄与训练网络共享，不能代表真实年龄准确度。单图修正只优化本次输入的潜变量，每张新照片都需额外计算；没有训练或更新通用网络权重。单图样例与固定回归集均为探索性比较。</p>
<p><a href="quality-summary.json">完整分组汇总与耗时</a> · <a href="review-data.json">逐图记录</a></p><p id="error"></p></main><script>
const $=x=>document.getElementById(x);let data;
function family(){return data.families[$('family').value]}
function samples(){const f=family();$('sample').replaceChildren(...f.records.map(r=>new Option(r.name==='uploaded'?'你上传的人像':r.name,r.name)));render()}
function render(){const f=family(),s=f.records.find(r=>r.name===$('sample').value);if(!s)return;const rows=f.outputs.filter(r=>r.name===s.name&&r.condition===$('age').value);$('images').replaceChildren();
const items=[{title:'输入参考',src:f.prefix+'/'+s.name+'/reference.png',download:f.prefix+'/'+s.name+'/reference.png',info:'源条件 '+s.source_age}];for(const route of f.routes){const r=rows.find(r=>r.route===route);if(r)items.push({title:data.titles[route]||route,src:f.prefix+'/'+r.preview,download:f.prefix+'/'+(r.native||r.preview),info:`身份 ${r.sface_cosine?.toFixed(3)}；共享参考估龄 ${r.shared_output_age.toFixed(1)}`})}
for(const item of items){const fig=document.createElement('figure'),cap=document.createElement('figcaption'),img=document.createElement('img'),info=document.createElement('p'),link=document.createElement('a');cap.textContent=item.title;img.src=item.src;img.alt=item.title;info.textContent=item.info;info.className='info';link.href=item.download;link.download='';link.textContent='下载PNG';fig.append(cap,img,info,link);$('images').append(fig)}$('images').style.setProperty('--columns',items.length);$('note').textContent=f.note+' · '+s.cohort;
$('metrics').replaceChildren();for(const [cohort,methods] of Object.entries(data.summary[$('family').value].cohorts))for(const [route,m] of Object.entries(methods)){const tr=document.createElement('tr');for(const value of [cohort+' / '+data.titles[route],m.same_identity,m.cross_identity,m.same_lpips,m.age_span]){const td=document.createElement('td');td.textContent=typeof value==='number'?value.toFixed(4):value??'待完成';tr.append(td)}$('metrics').append(tr)}}
fetch('review-data.json').then(r=>r.json()).then(d=>{data=d;for(const [key,f] of Object.entries(d.families))$('family').append(new Option(f.title,key));if(d.families.constrained)$('family').value='constrained';samples()}).catch(e=>$('error').textContent=e.message);$('family').onchange=samples;$('sample').onchange=render;$('age').onchange=render;
</script></html>'''


def summarize(report):
    result=dict(cohorts={},timings=report.get('timings'),total_seconds=report.get('total_seconds'))
    for cohort in sorted({r['cohort'] for r in report['records']}):
        result['cohorts'][cohort]={}
        for route in report['routes']:
            rows=[r for r in report['outputs'] if r['cohort']==cohort and r['route']==route]
            if not rows:continue
            same=[r for r in rows if r['condition']=='same'];cross=[r for r in rows if r['condition']!='same']
            def mean(key,items):
                values=[r[key] for r in items if r.get(key) is not None]
                return statistics.mean(values) if values else None
            ages={(r['name'],r['condition']):r['shared_output_age'] for r in rows}
            spans=[ages[(name,'75')]-ages[(name,'20')] for name in {r['name'] for r in rows} if (name,'75') in ages and (name,'20') in ages]
            result['cohorts'][cohort][route]=dict(cases=len(same),same_identity=mean('sface_cosine',same),cross_identity=mean('sface_cosine',cross),
                same_lpips=mean('lpips_shared',same),age_span=statistics.mean(spans) if spans else None)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    run=parser.parse_args().run;families={};summary={}
    for key,subfolder,title,note in [
        ('latent','latent-refinement','单图潜变量修正','网络权重冻结；同龄重建200步，四个目标联合修正120步。'),
        ('trained','evaluation','通用训练候选','两组800步配对；生成器组解冻64–1024分辨率卷积。'),
        ('constrained','latent-constrained','限制细节变化的单图修正','复用重建锚点；只优化前8层潜变量，并约束感知差异及颜色。')]:
        path=run/subfolder/'report.json'
        if not path.exists():continue
        report=json.loads(path.read_text());names={r['name'] for r in report['outputs']}
        families[key]=dict(title=title,note=note,prefix=subfolder,routes=report['routes'],outputs=report['outputs'],
            records=sorted([r for r in report['records'] if r['name'] in names],key=lambda r:(r['name']!='uploaded',r['cohort'],r['name'])))
        summary[key]=summarize(report)
    if not families:raise RuntimeError('No completed real results yet')
    (run/'review-data.json').write_text(json.dumps(dict(families=families,summary=summary,titles=TITLES),ensure_ascii=False)+'\n')
    (run/'quality-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (run/'index.html').write_text(PAGE)
    print(json.dumps({key:value['cohorts'] for key,value in summary.items()},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

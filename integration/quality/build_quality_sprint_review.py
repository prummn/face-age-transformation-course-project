"""Expose completed real results as a static CPU-only comparison page."""
import argparse
import json
from pathlib import Path
import statistics
import html


def sources_only(run):
    """An explicit input preview when external GPU inference is blocked."""
    from PIL import Image
    records=json.loads((run/'inputs.pre-age.json').read_text())
    dest=run/'source-inputs';dest.mkdir(exist_ok=True)
    cards=[]
    for r in records:
        image=dest/(r['name']+'.jpg')
        with Image.open(r['image_path']) as im:im.convert('RGB').resize((256,256)).save(image,quality=90)
        cards.append('<figure><img loading="lazy" width="256" height="256" src="source-inputs/'+image.name+'"><figcaption>'+html.escape(
            f"{r['name']}｜{r['annotation']['race']}｜发布方年龄 {r['publisher_age']:.0f}，表观 {r['apparent_age']:.1f}｜原始人脸短边 {r['native_face_short_edge']:.0f}px｜{r['review_split']}")+'</figcaption></figure>')
    (run/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>新增测试照片：仅输入预览</title><style>body{font:16px/1.6 system-ui;background:#f4f6f2;color:#233b34;margin:20px}main{max-width:1300px;margin:auto}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(256px,1fr));gap:16px}figure{margin:0;padding:8px;background:white}img{width:100%;height:auto}figcaption{font-size:14px}</style><main><h1>新增32张外部测试照片：仅输入预览</h1><p>这些是APPA-REAL官方测试照片的对齐参考，不是GAN生成结果。GPU调度尚未完成，因此此页不提供本轮模型对照结论。来源记录、真实／表观年龄和低分辨率样例全部保留。</p><p>24张发布方标注Asian、其他8张。26张原始人脸不足256px，1024参考可能经过上采样；新照片不等于已核实的独立身份。仅用于本地评价，不训练通用模型。</p><div class="grid">'''+''.join(cards)+'</div></main></html>')
    print('SOURCE_PREVIEW_ONLY',len(records),'no model output claimed')


TITLES=dict(fine_direct='原始直接路线',previous_identity='前轮身份候选',recon_encoder='加强重建：编码器',
            recon_generator='加强重建：编码器＋生成器',latent_anchor='单图重建修正',latent_refined='重建＋跨龄修正',
            latent_constrained60='细节约束：60步',latent_constrained180='细节约束：180步',
            baseline_v1='已确认基线v1',texture_half='减弱细节层年龄改动',robust_age='多视图年龄约束',
            texture_robust='细节控制＋多视图约束',texture_pair='确定性成对对照',texture_chroma='增加低频色度约束（试验）')

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
let sourceInfo='模型源条件 '+s.source_age;if(s.publisher_age!=null)sourceInfo+=`；发布方年龄 ${s.publisher_age}，表观年龄 ${s.apparent_age.toFixed(1)}；原始人脸短边 ${s.native_face_short_edge.toFixed(0)}px`;
const items=[{title:'输入参考',src:f.prefix+'/'+s.name+'/reference.png',download:f.prefix+'/'+s.name+'/reference.png',info:sourceInfo}];for(const route of f.routes){const r=rows.find(r=>r.route===route);if(r)items.push({title:data.titles[route]||route,src:f.prefix+'/'+r.preview,download:f.prefix+'/'+(r.native||r.preview),info:`身份 ${r.sface_cosine?.toFixed(3)??'无检测'}；共享参考估龄 ${r.shared_output_age.toFixed(1)}`})}
for(const item of items){const fig=document.createElement('figure'),cap=document.createElement('figcaption'),img=document.createElement('img'),info=document.createElement('p'),link=document.createElement('a');cap.textContent=item.title;img.src=item.src;img.alt=item.title;info.textContent=item.info;info.className='info';link.href=item.download;link.download='';link.textContent='下载PNG';fig.append(cap,img,info,link);$('images').append(fig)}$('images').style.setProperty('--columns',items.length);$('note').textContent=f.note+' · '+s.cohort+(s.review_split==='diagnostic'?' · 诊断组':s.review_split==='confirmation'?' · 复核组':'');
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
    parser.add_argument('--inputs-only',action='store_true',help='Explicit input-only page if external inference is unavailable')
    args=parser.parse_args();run=args.run;families={};summary={}
    if args.inputs_only:sources_only(run);return
    for key,subfolder,title,note in [
        ('latent','latent-refinement','单图潜变量修正','网络权重冻结；同龄重建200步，四个目标联合修正120步。'),
        ('trained','evaluation','通用训练候选','两组800步配对；生成器组解冻64–1024分辨率卷积。'),
        ('constrained','latent-constrained','限制细节变化的单图修正','复用重建锚点；只优化前8层潜变量，并约束感知差异及颜色。'),
        ('age_texture','age-texture','基线v1之后：纹理与年龄约束','冻结已确认基线；两因素对照：细节层年龄改动减半、多视图年龄约束。没有对输出做模糊。'),
        ('texture_validation','texture-validation','追加样例复核','实际后端代码在追加11张输入上的配对结果。'),
        ('external','external-results','新增32张外部测试','官方APPA测试照，保留低清晰度及旧照片；基线与细节控制候选分别保留。'),
        ('chroma','chroma-repair','新增测试：针对颜色漂移修正','同一锚点、相同步数，仅增加低频色度损失；原始GAN输出，没有粘贴或后处理。')]:
        path=run/subfolder/'report.json'
        if not path.exists():continue
        report=json.loads(path.read_text());names={r['name'] for r in report['outputs']}
        families[key]=dict(title=report.get('title',title),note=report.get('note',note),prefix=subfolder,routes=report['routes'],outputs=report['outputs'],
            records=sorted([r for r in report['records'] if r['name'] in names],key=lambda r:(r['name']!='uploaded',r['cohort'],r['name'])))
        summary[key]=summarize(report)
    if not families:raise RuntimeError('No completed real results yet')
    (run/'review-data.json').write_text(json.dumps(dict(families=families,summary=summary,titles=TITLES),ensure_ascii=False)+'\n')
    (run/'quality-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (run/'index.html').write_text(PAGE)
    print(json.dumps({key:value['cohorts'] for key,value in summary.items()},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

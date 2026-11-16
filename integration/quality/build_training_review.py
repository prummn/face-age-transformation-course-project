"""Summarize completed paired pilot and expose raw comparisons on existing private gateway."""
import argparse,csv,json,statistics
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

PAGE=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>100步 GAN 配对实验</title><style>*{box-sizing:border-box}body{margin:0;background:#f3f4f0;color:#203b36;font:15px/1.6 system-ui,sans-serif}main{max-width:1500px;padding:24px;margin:auto}h1{font-size:26px;margin:0}p{margin:10px 0}.muted{color:#52655f}.controls{display:flex;flex-wrap:wrap;gap:14px;padding:16px 0}label{display:grid;gap:4px}select{font:inherit;padding:8px;max-width:330px}.grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}figure{margin:0;background:white;padding:8px;border-radius:10px}.frame{overflow:hidden;aspect-ratio:1}.frame img{width:100%;height:100%;object-fit:contain;display:block}.crop .frame img{transform:scale(1.9);transform-origin:50% 51%}figcaption{min-height:3em;font-weight:600}.info{min-height:7em;font-size:13px}a{color:#176454}table{border-collapse:collapse;max-width:100%;font-size:13px}td,th{padding:7px;border-bottom:1px solid #bdcfc5;text-align:left}.scroll{overflow:auto}.compare{max-width:620px;margin:24px auto}.slider{position:relative;aspect-ratio:1;overflow:hidden}.slider img{width:100%;position:absolute;inset:0}#overlay{clip-path:inset(0 50% 0 0)}input[type=range]{width:100%}@media(max-width:800px){main{padding:14px}.grid{grid-template-columns:repeat(2,minmax(0,1fr))}figure:first-child{grid-column:1/-1;max-width:330px}.info{min-height:6em}h1{font-size:23px}}</style><main><h1>GAN 配对实验：编码器与有限生成器适配</h1><p>两个候选各训练100步；相同起点、数据、翻转、年龄条件和损失。生成器候选只额外更新高分辨率层的调制仿射参数。此页展示原始模型输出，线上默认权重尚未替换。</p><div class="controls"><label>评估组<select id="cohort"><option value="diagnostic">用户与作者诊断图</option><option value="appa_asian_test">APPA · Asian标注测试图</option><option value="ffhq_regression">FFHQ · 回归评估</option></select></label><label>照片<select id="sample"></select></label><label>年龄条件<select id="age"><option value="same">同龄重建（参考估龄条件）</option><option>20</option><option>40</option><option>60</option><option>75</option></select></label><label>查看范围<select id="crop"><option value="full">完整图</option><option value="face">五官放大</option></select></label></div><p id="note"></p><div class="grid" id="pictures"></div><div class="compare"><label>滑动对比：右侧为直接微调基线；左侧候选<select id="candidate"><option value="encoder_generator">编码器＋少量生成器参数</option><option value="encoder">仅编码器</option><option value="mixed">线上混合路线</option></select></label><div class="slider"><img id="underlay" alt="直接微调基线"><img id="overlay" alt="候选"></div><input id="split" aria-label="对比位置" type="range" min="0" max="100" value="50"></div><details open><summary>本评估组的汇总</summary><p class="muted">身份余弦来自独立 SFace；LPIPS和估龄仍复用训练网络。跨龄身份与年龄响应需要一起看，弱化变老不能算全面改善。</p><div class="scroll"><table><thead><tr><th>路线</th><th>同龄身份↑</th><th>跨龄身份↑</th><th>同龄LPIPS↓</th><th>同龄色偏↓</th><th>20→75估龄跨度</th></tr></thead><tbody id="summary"></tbody></table></div></details><p class="muted">色偏采用OpenCV8位Lab色度均值变化，非DeltaE00。FFHQ已被基础生成器用于训练，本组用于回归检查；APPA的Asian是宽泛发布者标注。保守相似度／来源分组不能保证严格身份隔离；源年龄是FairFace伪标签。同龄结果不是未来容貌真值。100步试验不能证明亚洲专项适配成功。</p><p><a href="evaluation/report.json" target="_blank">完整逐图记录</a> · <a href="summary.json" target="_blank">汇总与配对差值</a> · <a href="metrics.csv" download>数值 CSV</a> · <a id="overview" target="_blank">当前照片各年龄总览</a></p><p id="error" role="alert"></p></main><script>
const $=x=>document.getElementById(x),titles={mixed:'线上混合路线',fine_direct:'直接微调基线',encoder:'训练：仅编码器',encoder_generator:'训练：编码器＋有限生成器'};let review,summary;function current(){return review.records.find(r=>r.name===$('sample').value)}function samples(){if($('cohort').value==='appa_asian_test'&&review.external_alignment_failures.length)$('error').textContent=`12张固定APPA测试图中有 ${review.external_alignment_failures.length} 张对齐失败，失败名单保留在完整记录；下表仅统计可对齐部分。`;else $('error').textContent='';const list=review.records.filter(r=>r.cohort===$('cohort').value);$('sample').replaceChildren(...list.map(r=>new Option(r.name==='uploaded'?'你上传的人像':r.name,r.name)));render()}function img(src,alt){const el=document.createElement('img');el.src=src;el.alt=alt;return el}function render(){const s=current();if(!s)return;const rows=review.outputs.filter(o=>o.name===s.name&&o.condition===$('age').value);$('pictures').replaceChildren();const paths=[{title:'输入参考',path:`evaluation/${s.name}/reference.png`,info:`参考估龄 ${s.shared_reference_age.toFixed(1)}；源条件 ${s.source_age}。`,native:true}];for(const route of review.routes){const o=rows.find(o=>o.route===route);paths.push({title:titles[route],path:'evaluation/'+o.preview,download:'evaluation/'+(o.native||o.preview),native:!!o.native,info:`身份余弦 ${o.sface_cosine===null?'检测失败':o.sface_cosine.toFixed(3)}；共享估龄 ${o.shared_output_age.toFixed(1)}；色偏 ${o.face_chroma_shift.toFixed(2)}；推理 ${o.model_seconds.toFixed(3)}秒。`})}for(const item of paths){const f=document.createElement('figure'),cap=document.createElement('figcaption'),frame=document.createElement('div'),info=document.createElement('p'),a=document.createElement('a');cap.textContent=item.title;frame.className='frame';frame.append(img(item.path,item.title));info.className='info';info.textContent=item.info;a.href=item.download||item.path;a.download='';a.textContent=item.native?'下载原始 PNG':'下载256预览';f.append(cap,frame,info,a);$('pictures').append(f)}$('pictures').classList.toggle('crop',$('crop').value==='face');$('underlay').src='evaluation/'+rows.find(o=>o.route==='fine_direct').preview;$('overlay').src='evaluation/'+rows.find(o=>o.route===$('candidate').value).preview;$('note').textContent=s.annotation?`发布者实际年龄 ${s.annotation.real_age}；人评表观年龄 ${Number(s.annotation.apparent_age_avg).toFixed(1)}；源照片 ${s.original_size.join('×')}。`:'各目标由同一输入独立生成；同龄条件来自共享估龄网络。';$('overview').href='overview/'+s.name+'.jpg';$('summary').replaceChildren();for(const route of review.routes){const v=summary.cohorts[s.cohort][route],tr=document.createElement('tr');for(const value of [titles[route],v.same.sface_cosine,v.cross.sface_cosine,v.same.lpips_shared,v.same.face_chroma_shift,v.age_response20to75]){const td=document.createElement('td');td.textContent=typeof value==='number'?value.toFixed(3):value??'缺失';tr.append(td)}$('summary').append(tr)}}Promise.all([fetch('evaluation/report.json').then(r=>r.json()),fetch('summary.json').then(r=>r.json())]).then(([r,s])=>{review=r;summary=s;samples()}).catch(e=>$('error').textContent=e.message);$('cohort').onchange=samples;for(const id of ['sample','age','crop','candidate'])$(id).onchange=render;$('split').oninput=()=>$('overlay').style.clipPath=`inset(0 ${100-$('split').value}% 0 0)`;
</script></html>'''

def mean(values):
    values=[x for x in values if x is not None]
    return statistics.mean(values) if values else None

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();run=a.run.resolve()
    if not (run/'DONE').exists():raise RuntimeError('Wait for completed paired run')
    report=json.loads((run/'evaluation/report.json').read_text());outputs=report['outputs'];summary={'cohorts':{},'paired_against_fine_direct':{}}
    appa=[r for r in report['records'] if r['cohort']=='appa_asian_test']
    summary['source_age_diagnostic']=dict(cohort='appa_asian_test',attempted=12,aligned=len(appa),alignment_failed=report['external_alignment_failures'],
        shared_fairface_mae_real_age=mean([abs(r['shared_reference_age']-float(r['annotation']['real_age'])) for r in appa]),
        shared_fairface_mae_apparent_age=mean([abs(r['shared_reference_age']-float(r['annotation']['apparent_age_avg'])) for r in appa]),
        note='Unmodified source photo estimates only; small selected set, not population calibration or accuracy of generated target ages.')
    for cohort in sorted({r['cohort'] for r in report['records']}):
        summary['cohorts'][cohort]={};summary['paired_against_fine_direct'][cohort]={}
        cohortrows=[r for r in outputs if r['cohort']==cohort]
        base={(r['name'],r['condition']):r for r in cohortrows if r['route']=='fine_direct'}
        for route in report['routes']:
            subset=[r for r in cohortrows if r['route']==route];v={}
            for phase in ['same','cross']:
                rows=[r for r in subset if (r['condition']=='same')==(phase=='same')]
                v[phase]=dict(count=len(rows),missing_sface=sum(r['sface_cosine'] is None for r in rows),**{key:mean([r[key] for r in rows]) for key in ['sface_cosine','lpips_shared','rgb_mae','face_chroma_shift','shared_age_error','model_seconds']})
            byname={r['name'] for r in subset};response=[]
            for name in byname:
                t={r['condition']:r['shared_output_age'] for r in subset if r['name']==name}
                response.append(t['75']-t['20'])
            v['age_response20to75']=mean(response);summary['cohorts'][cohort][route]=v
            paired={}
            for phase in ['same','cross']:
                rows=[r for r in subset if (r['condition']=='same')==(phase=='same')]
                delta={}
                for key in ['sface_cosine','lpips_shared','face_chroma_shift','shared_age_error']:
                    differences=[r[key]-base[(r['name'],r['condition'])][key] for r in rows if r[key] is not None and base[(r['name'],r['condition'])][key] is not None]
                    delta[key]=dict(mean=mean(differences),count=len(differences),positive=sum(x>0 for x in differences))
                paired[phase]=delta
            summary['paired_against_fine_direct'][cohort][route]=paired
    (run/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');(run/'index.html').write_text(PAGE)
    with (run/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=outputs[0].keys());writer.writeheader();writer.writerows(outputs)
    folder=run/'overview';folder.mkdir(exist_ok=True);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    for s in report['records']:
        name=s['name'];cell=180;left=165;board=Image.new('RGB',(left+cell*6,208*4),'#edf1ec');draw=ImageDraw.Draw(board)
        for i,route in enumerate(report['routes']):
            y=i*208;draw.text((4,y+90),route,font=font,fill='black')
            ref=Image.open(run/'evaluation'/name/'reference.png').convert('RGB').resize((cell,cell));board.paste(ref,(left,y+28));draw.text((left,y+4),'source',font=font,fill='black')
            for j,condition in enumerate(['same','20','40','60','75'],1):
                r=next(r for r in outputs if r['name']==name and r['route']==route and r['condition']==condition)
                board.paste(Image.open(run/'evaluation'/r['preview']).convert('RGB').resize((cell,cell)),(left+j*cell,y+28))
                draw.text((left+j*cell,y+4),f"{condition} ID:{r['sface_cosine']:.2f}" if r['sface_cosine'] is not None else condition,font=font,fill='black')
        board.save(folder/(name+'.jpg'),quality=92)
    print(json.dumps(summary['cohorts'],ensure_ascii=False,indent=2))
if __name__=='__main__':main()

"""CPU summary of a completed identity-weight pilot, including paired uncertainty."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import statistics


def average(values):
    return statistics.mean(values) if values else None


def paired_interval(differences):
    """Resample input cases, retaining all age targets of each input together."""
    rng = random.Random(20261116)
    n = len(differences)
    means = sorted(average(rng.choices(differences, k=n)) for _ in range(4000))
    return dict(mean=average(differences), ci95=[means[100], means[3899]],
                cases=n, improved=sum(x > 0 for x in differences),
                method='4000 paired case bootstrap samples; targets averaged within case')


def audit_source_ages(run):
    bins=defaultdict(list)
    for line in (run/'manifests/train.jsonl').read_text().splitlines():
        row=json.loads(line);bins[tuple(row['age_interval'])].append(row['source_age'])
    report=dict(note='FairFace numeric pseudo conditions vs human apparent-age intervals; not chronological-age errors.',by_human_interval={})
    for interval,ages in sorted(bins.items()):
        report['by_human_interval'][str(interval)]=dict(count=len(ages),mean_pseudo_age=average(ages),range=[min(ages),max(ages)],
            below_interval=sum(a<interval[0] for a in ages),inside_interval=sum(interval[0]<=a<=interval[1] for a in ages),above_interval=sum(a>interval[1] for a in ages))
    (run/'source-age-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


PAGE = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>跨龄身份约束实验</title><style>body{font:16px/1.6 system-ui;margin:24px;background:#f7f8f5;color:#223b32}main{max-width:1400px;margin:auto}select{font:inherit;padding:8px;margin:8px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}figure{margin:0}img{width:100%}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ccc;text-align:left}@media(max-width:700px){.grid{grid-template-columns:repeat(2,1fr)}}pre{white-space:pre-wrap}</style><main>
<h1>跨龄身份约束：单卡配对实验</h1><p>两组均从原始Release开始，各500步；仅将跨龄身份损失系数从0.1提高至0.5。原始输出，无保色。线上默认模型未替换。</p>
<label>照片<select id="sample"></select></label><label>目标<select id="age"><option value="same">同龄</option><option>20</option><option>40</option><option>60</option><option>75</option></select></label>
<p id="note"></p><div class="grid" id="images"></div><h2>分组结果</h2><div style="overflow-x:auto"><table><thead><tr><th>组／路线</th><th>跨龄身份↑</th><th>同龄LPIPS↓</th><th>20→75共享估龄跨度</th></tr></thead><tbody id="metrics"></tbody></table></div>
<p>FFHQ为回归集；APPA为12张Asian标注测试图。身份由独立SFace计算；估龄和LPIPS复用训练网络，不能证明年龄准确。所有目标都从同一原图生成。上传人像仅用于诊断。</p>
<p><a href="summary.json">完整汇总与配对置信区间</a> · <a href="evaluation/report.json">逐图记录</a> · <a href="metrics.csv" download>下载CSV</a></p><p id="error"></p></main><script>
const $=id=>document.getElementById(id),titles={fine_direct:'Release直接路线',id_control:'500步：身份系数0.1',id_strong:'500步：身份系数0.5'};let report;
function render(){const sample=report.records.find(r=>r.name===$('sample').value);if(!sample)return;const rows=report.outputs.filter(r=>r.name===sample.name&&r.condition===$('age').value);$('images').replaceChildren();
for(const item of [{title:'输入参考',path:`evaluation/${sample.name}/reference.png`,note:'',native:true},...rows.map(r=>({title:titles[r.route],path:'evaluation/'+r.preview,download:'evaluation/'+(r.native||r.preview),native:!!r.native,note:`身份 ${r.sface_cosine?.toFixed(3)}；共享估龄 ${r.shared_output_age.toFixed(1)}`}))]){const f=document.createElement('figure'),im=document.createElement('img'),caption=document.createElement('figcaption'),a=document.createElement('a');im.src=item.path;im.alt=item.title;caption.textContent=item.title+' '+item.note;a.href=item.download||item.path;a.download='';a.textContent=item.native?'下载原始PNG':'下载256预览';f.append(im,caption,a);$('images').append(f)}$('note').textContent=`${sample.cohort}；参考条件 ${sample.source_age}。同龄条件是模型参考估龄。`}
Promise.all([fetch('evaluation/report.json').then(r=>r.json()),fetch('summary.json').then(r=>r.json())]).then(([r,s])=>{report=r;const records=[...r.records].sort((a,b)=>(a.name==='uploaded'?-1:0)-(b.name==='uploaded'?-1:0));for(const x of records)$('sample').append(new Option(x.name,x.name));for(const [c,arms] of Object.entries(s.cohorts))for(const [arm,m] of Object.entries(arms)){const tr=document.createElement('tr');for(const v of [c+' / '+titles[arm],m.cross_sface,m.same_lpips,m.age_span]){const td=document.createElement('td');td.textContent=typeof v==='number'?v.toFixed(4):v;tr.append(td)}$('metrics').append(tr)}render()}).catch(e=>$('error').textContent=e.message);$('sample').onchange=render;$('age').onchange=render;
</script></html>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args=parser.parse_args();run=args.run
    if not (run/'DONE').exists():
        raise RuntimeError('Wait for the complete run')
    report=json.loads((run/'evaluation/report.json').read_text())
    summary=dict(cohorts={}, paired={}, resource={}, criteria=json.loads((run/'experiment.json').read_text())['decision'])
    outputs=report['outputs']
    for cohort in sorted({r['cohort'] for r in report['records']}):
        summary['cohorts'][cohort]={};summary['paired'][cohort]={}
        rows=[r for r in outputs if r['cohort']==cohort]
        byroute={route:[r for r in rows if r['route']==route] for route in report['routes']}
        for route, subset in byroute.items():
            same=[r for r in subset if r['condition']=='same'];cross=[r for r in subset if r['condition']!='same']
            names=sorted({r['name'] for r in subset})
            ages={(r['name'],r['condition']):r['shared_output_age'] for r in subset}
            summary['cohorts'][cohort][route]=dict(cases=len(names),same_sface=average([r['sface_cosine'] for r in same]),
                cross_sface=average([r['sface_cosine'] for r in cross]),same_lpips=average([r['lpips_shared'] for r in same]),
                same_chroma=average([r['face_chroma_shift'] for r in same]),
                age_error=average([r['shared_age_error'] for r in cross]),
                age_span=average([ages[(name,'75')]-ages[(name,'20')] for name in names]))
        for baseline in ['fine_direct','id_control']:
            lookup={(r['name'],r['condition']):r for r in byroute[baseline]}
            changes={}
            for r in byroute['id_strong']:
                if r['condition']!='same':
                    changes.setdefault(r['name'],[]).append(r['sface_cosine']-lookup[(r['name'],r['condition'])]['sface_cosine'])
            summary['paired'][cohort]['id_strong_minus_'+baseline]=paired_interval([average(v) for v in changes.values()])
    ffhq=summary['cohorts']['ffhq_regression'];appa=summary['cohorts']['appa_asian_test']
    candidate=ffhq['id_strong'];control=ffhq['id_control'];release=ffhq['fine_direct']
    summary['numeric_gate']=dict(
        cross_identity_gain_over_control=candidate['cross_sface']-control['cross_sface']>=.005,
        case_bootstrap_lower_positive=summary['paired']['ffhq_regression']['id_strong_minus_id_control']['ci95'][0]>0,
        cross_identity_no_worse_than_release=candidate['cross_sface']>=release['cross_sface'],
        age_span_retained=candidate['age_span']>=.9*max(control['age_span'],release['age_span']),
        same_lpips_retained=candidate['same_lpips']<=min(control['same_lpips'],release['same_lpips'])+.01,
        appa_identity_direction=appa['id_strong']['cross_sface']>=max(appa['id_control']['cross_sface'],appa['fine_direct']['cross_sface']))
    summary['numeric_gate_passed']=all(summary['numeric_gate'].values())
    visual=run/'visual-review.json'
    summary['visual_review']=json.loads(visual.read_text()) if visual.exists() else 'Required; numerical screening alone does not promote a model.'
    summary['promotion']='Not promoted; keep the original online model.'
    for arm in ['id_control','id_strong']:
        summary['resource'][arm]=json.loads((run/arm/'completed.json').read_text())
    summary['resource']['evaluation_seconds']=report['total_seconds']
    summary['resource']['device']=json.loads((run/'device.json').read_text())
    summary['source_age_audit']=audit_source_ages(run)
    (run/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (run/'index.html').write_text(PAGE)
    with (run/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=outputs[0].keys());writer.writeheader();writer.writerows(outputs)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

"""Assemble completed GAN benchmark outputs into a private static viewer."""
import argparse
import csv
import json
from pathlib import Path
import statistics

PAGE=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>GAN 年龄编辑对照</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f5f4ef;color:#243234;font:16px/1.6 system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:24px}h1{font-size:28px;margin:0}p{margin:8px 0 18px}.muted{color:#59696c}.controls{display:flex;gap:16px;flex-wrap:wrap;background:white;padding:16px;border-radius:12px;margin:16px 0}label{display:grid;font-size:14px;gap:4px}select{font:inherit;padding:8px;border:1px solid #bbc6c7;border-radius:6px;max-width:340px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}figure{margin:0;min-width:0;background:white;padding:12px;border-radius:12px}.frame{aspect-ratio:1;overflow:hidden;background:#ddd}.frame img{width:100%;height:100%;object-fit:contain;display:block}.crop .frame img{transform:scale(1.9);transform-origin:50% 51%}figcaption{font-weight:600;margin-bottom:8px}.detail{font-size:14px;min-height:4.5em;margin-top:8px}a{color:#18645b}.compare{max-width:650px;margin:24px auto}.slider{position:relative;aspect-ratio:1;overflow:hidden}.slider img{width:100%;position:absolute;inset:0}#overlay{clip-path:inset(0 50% 0 0)}input[type=range]{width:100%}summary{cursor:pointer}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:7px 12px;border-bottom:1px solid #ccd4d0;text-align:left}@media(max-width:760px){main{padding:14px}.grid{grid-template-columns:1fr}h1{font-size:23px}}</style>
<main><h1>GAN 年龄编辑：固定样例对照</h1><p class="muted">当前线上混合路线、直接微调、原始 SAM、感知与身份约束反演，以及 CUSP 外部 GAN。12张诊断样例，来源、输入哈希与失败结果保留。所有网络权重保持冻结。</p>
<div class="controls"><label>照片<select id="sample"></select></label><label>年龄条件<select id="age"></select></label><label>对照方法<select id="route"><option value="perceptual_inversion">SAM · 感知与身份约束反演</option><option value="cusp">CUSP · 官方 FFHQ-LS 权重</option><option value="fine_direct">SAM · 直接微调编码器</option><option value="original">SAM · 原始模型</option></select></label><label>查看范围<select id="crop"><option value="full">完整图</option><option value="face">五官放大</option></select></label></div><p id="note"></p>
<div class="grid" id="pictures"><figure><figcaption>原始参考</figcaption><div class="frame"><img id="source" alt="参考"></div><div class="detail" id="sourceInfo"></div><a id="sourceDownload" download>下载参考</a></figure><figure><figcaption>当前线上混合路线</figcaption><div class="frame"><img id="baseline" alt="线上混合路线"></div><div class="detail" id="baselineInfo"></div><a id="baselineDownload" download>下载原始 PNG</a></figure><figure><figcaption id="candidateTitle">实验对照</figcaption><div class="frame"><img id="candidate" alt="GAN实验对照"></div><div class="detail" id="candidateInfo"></div><a id="candidateDownload" download>下载原始 PNG</a></figure></div>
<div class="compare"><b>左侧实验，右侧线上混合路线</b><div class="slider"><img id="underlay" alt="线上"><img id="overlay" alt="对照"></div><input id="split" aria-label="对比位置" type="range" min="0" max="100" value="50"></div>
<details><summary>当前目标的诊断数值</summary><p>独立 SFace 余弦只作配对诊断，未校准“同一人”阈值。FairFace 估龄仍是共享训练网络的辅助量，不是真实年龄。CUSP 的年龄档与 SAM 数值年龄不是同一控制口径。</p><table><thead><tr><th>方法</th><th>SFace余弦</th><th>共享网络估龄</th><th>原生尺寸</th></tr></thead><tbody id="metrics"></tbody></table><p><a href="review.json" target="_blank">完整结果记录</a> · <a href="metrics.csv" download>下载数值 CSV</a> · <a id="overview" target="_blank">当前照片全部方法总览</a></p></details><p class="muted">作者样例是否出现在各模型训练中未知，本次不是人群独立盲测。没有核实真实年龄或人群标签。照片只用于离线诊断和逐照片潜变量反演，不作为通用模型训练集。1024生成尺寸不代表恢复了原图缺失的细节。</p><p id="error" role="alert"></p>
<script>const $=id=>document.getElementById(id);let review;const titles={mixed:'当前混合',fine_direct:'直接微调',original:'原始SAM',perceptual_inversion:'感知与身份约束反演',cusp:'CUSP'};function sample(){return review.samples[+$('sample').value]}function ages(){const s=sample(),old=$('age').value;const list=[...new Set(s.outputs.map(o=>o.target))];$('age').replaceChildren(...list.map(a=>new Option(`${a}${a===s.source_condition?'（源条件／同龄）':''}`,a)));$('age').value=list.includes(+old)?old:s.source_condition;render()}function render(){const s=sample(),a=+$('age').value,route=$('route').value,base=s.outputs.find(o=>o.route==='mixed'&&o.target===a),candidate=s.outputs.find(o=>o.route===route&&o.target===a);for(const option of $('route').options)option.disabled=!s.outputs.some(o=>o.route===option.value&&o.target===a);if(!candidate){$('route').value='perceptual_inversion';return render()}
$('source').src=s.reference;$('baseline').src=base.path;$('candidate').src=candidate.path;$('underlay').src=base.path;$('overlay').src=candidate.path;$('sourceDownload').href=s.reference;$('baselineDownload').href=base.path;$('candidateDownload').href=candidate.path;$('candidateTitle').textContent='实验：'+titles[route];$('pictures').classList.toggle('crop',$('crop').value==='face');$('sourceInfo').textContent=`参考估龄 ${s.reference_age.toFixed(1)}；源条件 ${s.source_condition}；参考 ${s.original_size.join('×')}。`;
const info=o=>`SFace余弦 ${o.sface_cosine===null?'未检测到有效源脸':o.sface_cosine.toFixed(3)}；共享诊断估龄 ${o.shared_age_estimate.toFixed(1)}；原生 ${o.native_size.join('×')}。`;$('baselineInfo').textContent=info(base);$('candidateInfo').textContent=info(candidate);$('note').textContent=route==='cusp'?`CUSP实际条件是 ${candidate.age_bin.range.join('–')} 岁档；原生256，显示时会放大。`:route==='perceptual_inversion'?`本照片额外进行120步潜变量反演，耗时 ${s.fit_seconds.toFixed(1)} 秒，编码器和生成器权重冻结。`:'各目标均从同一输入独立生成。';$('overview').href='overview/'+s.name+'.jpg';$('metrics').replaceChildren();for(const o of s.outputs.filter(o=>o.target===a)){const tr=document.createElement('tr');for(const value of [titles[o.route],o.sface_cosine===null?'缺失':o.sface_cosine.toFixed(4),o.shared_age_estimate.toFixed(1),o.native_size.join('×')]){const td=document.createElement('td');td.textContent=value;tr.append(td)}$('metrics').append(tr)}}
fetch('review.json').then(r=>{if(!r.ok)throw new Error('记录读取失败');return r.json()}).then(r=>{review=r;review.samples.forEach((s,i)=>$('sample').add(new Option(s.name==='uploaded'?'你上传的人像':s.name,i)));ages()}).catch(e=>$('error').textContent=e.message);$('sample').onchange=ages;for(const id of ['age','route','crop'])$(id).onchange=render;$('split').oninput=()=>$('overlay').style.clipPath=`inset(0 ${100-$('split').value}% 0 0)`;</script></main></html>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);args=parser.parse_args()
    sam=json.loads((args.run/'sam/report.json').read_text())
    cusp=json.loads((args.run/'cusp/report.json').read_text())
    assert sam['manifest_sha256']==cusp['manifest_sha256']
    samples=[]
    for s in sam['samples']:
        other=next(t for t in cusp['samples'] if t['name']==s['name'])
        assert s['source_condition']==other['source_condition']
        item=dict(s,reference=f"sam/{s['name']}/reference.png",outputs=[])
        for prefix,record in [('sam',s),('cusp',other)]:
            item['outputs'] += [dict(o,path=prefix+'/'+o['path']) for o in record['outputs']]
        samples.append(item)
    review=dict(sam_model=sam['model'],cusp_model=cusp['model'],samples=samples)
    (args.run/'review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
    (args.run/'index.html').write_text(PAGE)
    rows=[dict(sample=s['name'],**{k:v for k,v in o.items() if k!='age_bin'},age_bin=json.dumps(o.get('age_bin')))
          for s in samples for o in s['outputs']]
    with (args.run/'metrics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summaries={}
    for route in ['original','fine_direct','mixed','perceptual_inversion','cusp']:
        summaries[route]={}
        for label,subset in [('same',[o for s in samples for o in s['outputs'] if o['route']==route and o['same_source_condition']]),
                             ('cross',[o for s in samples for o in s['outputs'] if o['route']==route and not o['same_source_condition']])]:
            valid=[o['sface_cosine'] for o in subset if o['sface_cosine'] is not None]
            summaries[route][label]=dict(count=len(subset),sface_valid=len(valid),mean_sface=statistics.mean(valid) if valid else None,
                mean_rgb_mae=statistics.mean(o['rgb_mae'] for o in subset),mean_face_mae=statistics.mean(o['face_mae'] for o in subset))
    (args.run/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    from PIL import Image,ImageDraw,ImageFont
    out=args.run/'overview';out.mkdir(exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    for sample in samples:
        routes=['original','fine_direct','mixed','perceptual_inversion','cusp'];ages=list(dict.fromkeys([sample['source_condition'],20,40,60,75]))
        cell=192;left=185;board=Image.new('RGB',(left+(len(ages)+1)*cell,len(routes)*(cell+28)),'#f0f0f0');draw=ImageDraw.Draw(board)
        for i,route in enumerate(routes):
            y=i*(cell+28);draw.text((5,y+90),route,font=font,fill='black')
            board.paste(Image.open(args.run/sample['reference']).convert('RGB').resize((cell,cell)),(left,y+28))
            draw.text((left+3,y+4),'source',font=font,fill='black')
            for j,age in enumerate(ages,1):
                result=next(o for o in sample['outputs'] if o['route']==route and o['target']==age)
                label=str(age)
                if route=='cusp':label='bin '+ '-'.join(map(str,result['age_bin']['range']))
                board.paste(Image.open(args.run/result['path']).convert('RGB').resize((cell,cell)),(left+j*cell,y+28))
                draw.text((left+j*cell+3,y+4),label,font=font,fill='black')
        board.save(out/(sample['name']+'.jpg'),quality=92)
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()

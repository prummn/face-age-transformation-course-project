"""Publish a small, attributed age-response diagnosis from existing public outputs."""
import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('diagnosis',type=Path)
    args=p.parse_args();run=args.diagnosis.resolve()
    audit=json.loads((run/'trajectory-audit.json').read_text())
    parent=Path(audit['parent_run'])
    report=json.loads((parent/'evaluation/report.json').read_text())
    out=run/'review';out.mkdir(exist_ok=True)
    names=['ffhq_64376','ffhq_65817','ffhq_67140']
    records=[]
    for name in names:
        record=next(r for r in audit['records'] if r['name']==name)
        meta=next(r for r in report['records'] if r['name']==name)
        record=dict(record,credit=meta['source_metadata'])
        folder=out/name;folder.mkdir(exist_ok=True)
        shutil.copyfile(parent/'evaluation'/name/'reference.png',folder/'reference.png')
        for route in ['original_delta','original_delta_constrained']:
            for age in ['20','60','75']:
                file=route+'_'+age+'.png'
                shutil.copyfile(parent/'evaluation'/name/file,folder/file)
        records.append(record)
    live=run/'live-64376'
    live_report=json.loads((live/'report.json').read_text())
    scores={r['name']:r['prediction'] for r in json.loads((live/'independent-age.json').read_text())['results']}
    for src,dest in [('reference.png','live-reference.png'),('60-raw.png','live-60.png'),('75-raw.png','live-75.png')]:
        shutil.copyfile(live/src,out/dest)
    (out/'data.json').write_text(json.dumps(dict(summary=audit['summary'],records=records,
        live=dict(age=scores,source_reference_age=live_report['source_reference_age'],outputs=live_report['outputs'])),ensure_ascii=False,indent=2)+'\n')
    (out/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>目标高龄不足：诊断对照</title><style>
body{font:16px/1.7 system-ui;max-width:1150px;margin:30px auto;padding:0 16px;background:#f3f4ef;color:#23342b}a{color:#315e47}h1{font-size:28px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px 7px;border-bottom:1px solid #cdd5cc;text-align:left}.scroll{overflow:auto}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}figure{margin:0}img{width:100%;border-radius:8px;display:block}figcaption{margin:8px 0}select{font:inherit;padding:8px;margin:12px 10px 20px 0}small{display:block}svg{width:100%;height:auto;background:white;border-radius:8px;margin:20px 0}section{margin:36px 0}@media(max-width:650px){.grid{grid-template-columns:1fr}figure{max-width:450px}h1{font-size:24px}}.note{background:#e5ece2;padding:14px;border-radius:8px}
</style><a href="/">返回时空镜</a> · <a href="/age-input-review/">输入链路对照</a><h1>纹路减轻后，为什么高龄还不够老？</h1><p>本页复核已保存的35图开发回归，另外用修正后网页输入真实生成一组同龄／60／75。没有训练、换权重或放大目标年龄。</p><p class="note">读图结论：当前约束恢复了部分眼嘴与本人特征，同时压弱年龄变化；年长源照在同龄重建时也会丢失部分真实皮肤细节。去掉约束虽然更老，但眼嘴和身份漂移，不能直接上线。</p><div class="scroll"><table><thead><tr><th>35图路线</th><th>目标60的平均外观估龄</th><th>目标75的平均外观估龄</th><th>60→75变化</th><th>跨龄身份相似度</th></tr></thead><tbody id="stats"></tbody></table></div><p><small>估龄来自独立MiVOLO；它不是照片真实年龄，也不能单独裁定皮肤自然度。35张均为已看过的开发回归，不是新盲测。</small></p><section><h2>同图：年龄变化与身份的取舍</h2><label>公开样本 <select id="subject"></select></label><label>目标 <select id="age"><option>75</option><option>60</option><option>20</option></select> 岁</label><div class="grid" id="grid"></div><svg id="curve" viewBox="0 0 720 300" role="img" aria-label="各目标年龄与两条路线外观估龄曲线"></svg><p id="credit"></p><p>曲线虚线为请求目标。实际外观允许个体差异，不能要求每张图准确等于目标；持续大幅偏低、60与75接近以及年轻化受限，都需要单独复核。</p></section><section><h2>修正后网页输入：新的一次真实API生成</h2><div class="grid"><figure><img src="live-reference.png"><figcaption>对齐原照</figcaption></figure><figure><img src="live-60.png"><figcaption id="live60"></figcaption></figure><figure><img src="live-75.png"><figcaption id="live75"></figcaption></figure></div><p>这组包含首次锚点拟合，模型／完整HTTP时间保存在报告中；与离线FFHQ输入不是相同像素，不做严格跨协议评分。用户本次实拍没有另存，因此这不是对其具体照片的判定。</p></section><section><h2>代码核对</h2><p>当前年龄项模仿的是旧未拟合锚点所生成图片的FairFace分布，并非直接要求目标75岁。相同身份、感知和颜色约束同时优化，尚未分别测量每项梯度贡献。FairFace最高年龄档代表值75，无法把80与90作为两个有可靠数值监督的年龄。</p><p>下一项应固定输入和数值实现，比较明确年龄段／年龄顺序约束与当前teacher，另检查身份保护的权衡；原生纹理、身份和年龄响应一起验收。不能用把75暗改90或单纯加皱纹来替代。</p><a href="/age-original-delta-full-review/">查看全部35图原始／保色留档</a></section><script>
fetch('data.json',{cache:'no-store'}).then(r=>r.json()).then(data=>{
const q=s=>document.querySelector(s),labels={wplus_base:'历史ft_l1差值＋约束',original_delta:'原SAM差值，无约束',original_delta_constrained:'当前原SAM差值＋约束'};
for(const [route,s] of Object.entries(data.summary)){const tr=document.createElement('tr');for(const value of [labels[route],s.mean_age60.toFixed(1),s.mean_age75.toFixed(1),s.mean_span60to75.toFixed(1),s.mean_cross_age_identity.toFixed(3)]){const td=document.createElement('td');td.textContent=value;tr.append(td);}q('#stats').append(tr);}
data.records.forEach(r=>{const o=document.createElement('option');o.value=r.name;o.textContent=r.name;q('#subject').append(o);});
function draw(){const r=data.records.find(r=>r.name===q('#subject').value),age=q('#age').value;q('#grid').replaceChildren();
for(const [route,title] of [['source','原照'],['original_delta','原SAM差值：无约束'],['original_delta_constrained','当前：原SAM差值＋约束']]){const f=document.createElement('figure'),im=new Image(),cap=document.createElement('figcaption'),link=document.createElement('a');im.src=r.name+'/'+(route==='source'?'reference.png':route+'_'+age+'.png');im.alt=title;cap.textContent=title+(route==='source'?'':' · 外观估龄 '+r.routes[route].ages[age].toFixed(1));link.href=im.src;link.download='';link.textContent='下载原始1024 PNG';f.append(im,cap,link);q('#grid').append(f);}
const c=r.credit;q('#credit').replaceChildren();for(const [text,url] of [[c.author,c.photo_url],[c.license,c.license_url]]){const a=document.createElement('a');a.textContent=text;a.href=url;q('#credit').append(a,document.createTextNode(' · '));}
const svg=q('#curve'),ns='http://www.w3.org/2000/svg',x=t=>60+(t-20)*600/55,y=t=>260-t*2.4;svg.replaceChildren();
function element(tag,attr,text){const e=document.createElementNS(ns,tag);Object.entries(attr).forEach(([k,v])=>e.setAttribute(k,v));if(text)e.textContent=text;svg.append(e);}
for(const t of [20,40,60,75]){element('line',{x1:60,y1:y(t),x2:660,y2:y(t),stroke:'#e4e7e2'});element('text',{x:20,y:y(t)+5,'font-size':14},String(t));element('text',{x:x(t)-10,y:285,'font-size':14},String(t));}
element('polyline',{points:[20,40,60,75].map(t=>x(t)+','+y(t)).join(' '),fill:'none',stroke:'#879188','stroke-dasharray':'6 5'});
for(const [route,color,title,ty] of [['original_delta','#b46e35','无约束',19],['original_delta_constrained','#315e47','当前约束',40]]){const a=r.routes[route].ages;element('polyline',{points:[20,40,60,75].map(t=>x(t)+','+y(a[String(t)])).join(' '),fill:'none',stroke:color,'stroke-width':3});element('text',{x:72,y:ty,fill:color,'font-size':14},title);}
element('text',{x:490,y:20,'font-size':13,fill:'#59625b'},'横轴：请求目标；纵轴：外观估龄');}
q('#subject').onchange=q('#age').onchange=draw;draw();q('#live60').textContent='目标60 · 外观估龄 '+data.live.age['60'].toFixed(1);q('#live75').textContent='目标75 · 外观估龄 '+data.live.age['75'].toFixed(1);
});</script></html>''')
    link=ROOT/'原型/age-response-review'
    if link.is_symlink():link.unlink()
    elif link.exists():raise ValueError('Refusing to replace non-symlink review')
    link.symlink_to(out,target_is_directory=True)
    print('/age-response-review/')


if __name__=='__main__':main()

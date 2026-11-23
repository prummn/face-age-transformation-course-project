"""Build a local browser gallery of the fixed native GAN comparisons."""
import argparse
import json
from pathlib import Path
import shutil

from PIL import Image

REVIEW = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>年龄编辑质量对照</title>
<style>
:root{color-scheme:light;color:#202b29;background:#eef1ed;font:16px/1.55 system-ui,sans-serif}
*{box-sizing:border-box}body{margin:0}main{max-width:1480px;margin:auto;padding:24px 20px 48px}
header{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:12px}
h1{font-size:clamp(24px,4vw,36px);margin:0;letter-spacing:-.035em}
a{color:#285b4f}h2{font-size:18px;margin-bottom:6px}
#decision{padding:10px 14px;background:#dfe9e0;border-radius:8px;margin:0 0 14px}
.controls{display:flex;flex-wrap:wrap;align-items:end;gap:12px;padding:14px 0}
label{display:grid;gap:4px;font-size:14px;color:#3c504c}
select,button{max-width:100%;min-height:40px;padding:8px;border:1px solid #b3c3ba;border-radius:6px;background:#fff;color:#202b29;font:inherit}
#grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin-top:10px}
figure{margin:0;padding:8px;background:white;border-radius:9px;min-width:0}
figure img{display:block;width:100%;aspect-ratio:1;object-fit:cover;border-radius:6px}
figcaption{font-weight:660;padding:6px 0 0}.metric{margin:3px 0;font-size:13px;color:#4d6059}
.credits{font-size:13px;color:#52635d}.back{white-space:nowrap}
#sizes{font-size:14px;color:#52635d}
@media(max-width:550px){main{padding:16px 10px 30px}header{align-items:start}.controls{gap:8px}
#grid{grid-template-columns:repeat(2,minmax(0,1fr));gap:7px}figure{padding:6px}a{overflow-wrap:anywhere}}
</style></head>
<body><main>
<header><h1>年龄编辑质量对照</h1><a class="back" href="/">回到演示</a></header>
<p id="decision"></p>
<div class="controls">
<label>照片组<select id="cohort"></select></label>
<label>照片<select id="subject"></select></label>
<label>目标<select id="age"></select></label>
<label>颜色<select id="color"><option value="raw">原始模型输出</option><option value="preserved">保留原照颜色</option></select></label>
</div>
<div id="sizes"></div><div id="grid" aria-live="polite"></div>
<p class="credits" id="credits"></p><p class="credits">外观年龄为估计值。PNG 链接提供本次模型生成的原文件，页面预览使用 512 像素图。</p>
</main>
<script id="records" type="application/json">__DATA__</script>
<script>
const {cases,decision}=JSON.parse(document.querySelector('#records').textContent);
const byId=id=>document.getElementById(id);
const labels={'new-hd-holdout':'新留出高清照片','annotated-ffhq':'既有高清回归','new_appa_asian':'外部 APPA 回归','dated-pairs':'有拍摄日期的同人配对'};
const routes={wplus_base:'W+ 基线',wplus_real_age:'W+ ＋ 实照年龄监督',wplus_trained_editor:'W+ ＋ 实照年龄监督与重训编码器'};
const ageLabels={same:'同龄重建',actual:'另一时点年龄'};
byId('decision').textContent=decision;
const cohort=byId('cohort'), subject=byId('subject'), age=byId('age'), color=byId('color');
for(const value of new Set(cases.map(r=>r.cohort))) cohort.add(new Option(labels[value]??value,value));
function selectCase(){
  const previous=subject.value;
  const choices=cases.filter(r=>r.cohort===cohort.value);
  subject.replaceChildren(...choices.map(r=>new Option(r.label,r.name)));
  if(choices.some(r=>r.name===previous))subject.value=previous;
  selectAge();
}
function selectAge(){
  const r=cases.find(r=>r.name===subject.value), desired=age.value||'75';
  age.replaceChildren(...Object.keys(r.results).map(value=>new Option(ageLabels[value]??value+' 岁',value)));
  if(Object.hasOwn(r.results,desired))age.value=desired;
  render();
}
function render(){
  const r=cases.find(r=>r.name===subject.value);
  const grid=byId('grid'); grid.replaceChildren();
  byId('sizes').textContent=[r.native_edge?'原始脸区短边约 '+r.native_edge+' px':'',r.source_age!=null?'模型源年龄 '+r.source_age+' 岁':'',r.actual?'目标拍摄年龄 '+r.actual.age+' 岁':''].filter(Boolean).join(' · ');
  const own=[{title:'原照',...r.reference}];
  if(r.actual)own.push({title:'另一时点真照',...r.actual});
  for(const output of r.results[age.value]){
    const variant=output[color.value]??output.raw;
    const metrics=[output.age!=null?'独立外观估计 '+output.age.toFixed(1)+' 岁':'',output.identity!=null?'源照身份相似度 '+output.identity.toFixed(3):''].filter(Boolean).join(' · ');
    own.push({title:routes[output.route]??output.route,metric:metrics,...variant});
  }
  for(const item of own){
    const fig=document.createElement('figure'),img=document.createElement('img'),
      heading=document.createElement('figcaption'), note=document.createElement('p'), link=document.createElement('a');
    heading.textContent=item.title;img.src=item.preview;img.alt=item.title;img.width=512;img.height=512;
    note.className='metric';note.textContent=item.metric??'';
    link.href=item.native;link.textContent='下载 PNG';link.download='';
    fig.append(img,heading,note,link);grid.append(fig);
  }
  const credits=byId('credits');credits.replaceChildren();
  for(const item of r.credits){
    const a=document.createElement('a');a.href=item.url;a.textContent=item.text;
    a.target='_blank';a.rel='noopener noreferrer';credits.append(a,document.createTextNode('  '));
  }
}
cohort.onchange=selectCase;subject.onchange=selectAge;age.onchange=render;color.onchange=render;
selectCase();
</script></body></html>'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run',type=Path)
    ap.add_argument('dest',type=Path)
    args = ap.parse_args()
    run,dest = args.run.resolve(),args.dest.resolve()
    result = json.loads((run/'wplus-evaluation/report.json').read_text())
    summary = json.loads((run/'selection-summary.json').read_text())
    decision = json.loads((run/'decision.json').read_text())
    photos = {r['name']:r for p in json.loads((run/'data/longitudinal.json').read_text()) for r in (p['early'],p['late'])}
    dest.mkdir(parents=True,exist_ok=False)
    cases = []

    def asset(relative):
        original = run/'wplus-evaluation'/relative
        if not original.is_file() or not original.resolve().is_relative_to(run/'wplus-evaluation'):
            raise ValueError(f'Unexpected image path: {original}')
        native = Path('images')/relative
        preview = native.with_suffix('.jpg')
        (dest/native).parent.mkdir(parents=True,exist_ok=True)
        (dest/native).symlink_to(original)
        with Image.open(original) as img:
            img.load()
            img.convert('RGB').resize((512,512),Image.Resampling.LANCZOS).save(dest/preview,quality=92)
        return dict(native=native.as_posix(),preview=preview.as_posix())

    for index,row in enumerate(result['records'],1):
        name = row['name']
        source = row.get('source_metadata',{})
        entry = dict(name=name,cohort=row['cohort'],source_age=row['source_age'],
            native_edge=row.get('source_face_short_edge',row.get('native_face_short_edge')),
            label=f'{index:02d} · '+('-'.join(map(str,row['age_interval']))+' 岁外观档' if 'age_interval' in row else str(row.get('publisher_age',''))+' 岁来源标注'),
            reference=asset(f'{name}/reference.png'),credits=[],results={})
        photo_url = source.get('photo_url','')
        if photo_url.startswith(('https://','http://')):
            note = ' · '.join(str(source[k]) for k in ('author','license') if source.get(k))
            entry['credits'].append(dict(text=note or '查看 FFHQ 原照片',url=photo_url))
        if row['cohort']=='new_appa_asian':
            entry['credits'].append(dict(text='APPA-REAL 官方评估集',url='https://chalearnlap.cvc.uab.cat/dataset/26/description/'))
        if row['cohort']=='dated-pairs':
            entry['actual'] = dict(asset(f'{name}/actual.png'),age=row['actual_target'])
            pair = photos[name]
            other = next(r for r in photos.values() if r['aligned']==row['target_photo'])
            for shot in (pair,other):
                entry['credits'].append(dict(text=shot['credit']+' · '+shot['source_licence'],url=shot['source']))
        for r in result['outputs']:
            if r['name'] != name:
                continue
            entry['results'].setdefault(r['condition'],[]).append(dict(route=r['route'],
                age=summary['independent_ages'][r['native']],identity=r['source_cosine'],
                raw=asset(r['native']),preserved=asset(r['preserved'])))
        cases.append(entry)
    page_data = json.dumps(dict(cases=cases,decision=decision['gallery_note']),ensure_ascii=False).replace('<','\\u003c')
    (dest/'index.html').write_text(REVIEW.replace('__DATA__',page_data))
    shutil.copy2(run/'decision.json',dest/'decision.json')
    print('GALLERY_READY',len(cases),dest)


if __name__=='__main__':
    main()

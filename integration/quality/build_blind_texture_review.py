"""Build route-hidden texture comparisons from verified native development PNGs.

No inference, training, pixel blending, or public release is performed. A private
key stays outside the served directory. Development inputs remain development
inputs even when the route labels are hidden from reviewers.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import secrets
import shutil

from probe_inversion import sha256, write_json

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / 'deployment/training-runs/personalized-generator-confirm-20261125-055616'
REGIONS = {'eyes': (220, 310, 804, 490), 'cheeks': (220, 490, 804, 670),
           'mouth': (340, 625, 684, 810)}
METRICS = ('texture', 'identity', 'age')

HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>高龄纹理路线隐藏评审</title><style>
body{font:16px system-ui;background:#f3f4f6;color:#17202a;margin:0}main{max-width:1320px;margin:auto;padding:18px}
button,select,input{font:inherit;padding:8px}button{cursor:pointer}button:disabled{cursor:default}p{line-height:1.6}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0;background:white;padding:10px;border-radius:8px}
img{max-width:100%;height:auto;display:block;margin:auto}figcaption{margin-bottom:8px}a{color:#165bad}fieldset{margin:12px 0;padding:12px;background:white}
label{display:inline-block;margin:6px 18px 6px 0}small{color:#52606d}.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
#notice{min-height:24px}details{margin:12px 0}textarea{width:95%;font:inherit}#thanks{display:none}
@media(max-width:720px){main{padding:10px}.grid{grid-template-columns:1fr}}
</style><main><h1>高龄纹理评审</h1>
<p>比较两张结果的皮肤自然度、本人特征和目标年龄外观。重点看眼下、脸颊、口周是否有刻痕、塑料感或涂抹；皱纹数量多并不等于更自然。两张结果使用同一输入和目标年龄。路线名称已隐藏。</p>
<details><summary>评审说明</summary><p>这是已有开发照片的路线隐藏比较，不能作为独立测试集结论。请独立作答，完成前不要打开其他有路线名称的对照页，也不要与其他评审讨论。部分题目重复用于检查一致性。年龄只评价外观，不是预测未来长相。</p>
<p>局部图直接裁剪原生1024 PNG，没有放大或保色。点击原图可查看完整像素。答案只保存在本浏览器；完成后下载JSON交给项目负责人。请勿在评审编号中填写姓名或联系方式。</p></details>
<div class="toolbar"><label>评审编号 <input id="rater" maxlength="40" placeholder="例如 reviewer01"></label>
<label>查看 <select id="region"><option value="full">完整人脸</option><option value="eyes">眼周原生裁剪</option><option value="cheeks">脸颊原生裁剪</option><option value="mouth">口周原生裁剪</option></select></label>
<button id="export">下载当前答案 JSON</button></div><p id="progress"></p>
<section id="task"><div class="grid" id="images"></div><div id="questions"></div>
<fieldset><legend>严重伪影（可多选；都无则不选）</legend><label><input type="checkbox" id="artifactLeft">左图有严重伪影</label><label><input type="checkbox" id="artifactRight">右图有严重伪影</label>
<p><textarea id="comment" rows="2" maxlength="500" placeholder="可选：刻痕、磨皮、偏色、五官变形等具体位置"></textarea></p></fieldset>
<div class="toolbar"><button id="prev">上一题</button><button id="save">保存并下一题</button></div></section>
<p id="notice" role="status"></p><p id="thanks">已完成全部题目，请下载答案 JSON。</p><details><summary>照片来源与署名</summary><div id="credits"></div></details></main>
<script id="study" type="application/json">__STUDY__</script><script>
const study=JSON.parse(document.querySelector('#study').textContent),storage='face-age-blind-'+study.study_id;
let state;try{state=JSON.parse(localStorage.getItem(storage))}catch{};
if(!state||state.study_id!==study.study_id)state={schema:1,study_id:study.study_id,rater:'',answers:{},index:0};
const $=s=>document.querySelector(s),metrics={texture:'哪张皮肤纹理更自然？',identity:'哪张更像原照中的本人？',age:'哪张更符合目标年龄的外观？'};
$('#rater').value=state.rater;let saving=true;
function persist(){try{localStorage.setItem(storage,JSON.stringify(state))}catch{saving=false;$('#notice').textContent='浏览器未能保存进度，请及时下载当前答案。'}}
function imagePath(base){const region=$('#region').value;return base+(region==='full'?'':'_'+region)+'.png'}
function render(){const t=study.trials[state.index],a=state.answers[t.id]||{};$('#progress').textContent=`第 ${state.index+1} / ${study.trials.length} 题 · 目标 ${t.target} 岁 · 已保存 ${Object.keys(state.answers).length} 题`;
 $('#images').replaceChildren();for(const [caption,key] of [['原照','reference'],['左图','left'],['右图','right']]){const fig=document.createElement('figure'),cap=document.createElement('figcaption'),img=document.createElement('img'),link=document.createElement('a');cap.textContent=caption;img.src=imagePath(t[key]);img.alt=caption;link.href=t[key]+'.png';link.target='_blank';link.rel='noopener';link.textContent='查看原生1024图';fig.append(cap,img,link);$('#images').append(fig)}
 $('#questions').replaceChildren();for(const [key,text] of Object.entries(metrics)){const f=document.createElement('fieldset'),l=document.createElement('legend');l.textContent=text;f.append(l);for(const [value,label] of [['left','左图'],['right','右图'],['tie','差不多'],['uncertain','无法判断']]){const wrap=document.createElement('label'),input=document.createElement('input');input.type='radio';input.name=key;input.value=value;input.checked=a[key]===value;wrap.append(input,document.createTextNode(' '+label));f.append(wrap)}$('#questions').append(f)}
 $('#artifactLeft').checked=!!a.artifact_left;$('#artifactRight').checked=!!a.artifact_right;$('#comment').value=a.comment||'';$('#prev').disabled=state.index===0;$('#save').textContent=state.index===study.trials.length-1?'保存并完成':'保存并下一题';$('#notice').textContent='';$('#thanks').style.display='none';persist()}
$('#rater').addEventListener('change',()=>{state.rater=$('#rater').value.trim();persist()});
$('#region').addEventListener('change',()=>{const t=study.trials[state.index];[...$('#images').querySelectorAll('img')].forEach((im,i)=>im.src=imagePath(t[['reference','left','right'][i]]))});
$('#prev').onclick=()=>{if(state.index>0){state.index--;render()}};
$('#save').onclick=()=>{state.rater=$('#rater').value.trim();if(!state.rater){$('#notice').textContent='请先填写不含个人信息的评审编号。';return}const a={};for(const key of Object.keys(metrics)){const el=document.querySelector(`input[name="${key}"]:checked`);if(!el){$('#notice').textContent='请完成三个选择；不能判断时请选择“无法判断”。';return}a[key]=el.value}Object.assign(a,{artifact_left:$('#artifactLeft').checked,artifact_right:$('#artifactRight').checked,comment:$('#comment').value,recorded_at:new Date().toISOString()});state.answers[study.trials[state.index].id]=a;persist();if(state.index<study.trials.length-1){state.index++;render();scrollTo(0,0)}else{render();$('#thanks').style.display='block';$('#notice').textContent='全部答案已保存，请下载 JSON。'}};
$('#export').onclick=()=>{state.rater=$('#rater').value.trim();persist();const blob=new Blob([JSON.stringify({...state,exported_at:new Date().toISOString()},null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='texture-review-'+(state.rater.replace(/[^a-zA-Z0-9_-]/g,'_')||'anonymous')+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
for(const c of study.credits){const p=document.createElement('p'),a=document.createElement('a');a.href=c.url;a.textContent=c.title;a.target='_blank';a.rel='noopener';p.append(document.createTextNode(c.author+' · '),a,document.createTextNode(' · '+c.license));$('#credits').append(p)}render();
</script></html>'''


def build(run, out):
    from PIL import Image
    out.mkdir(parents=True, exist_ok=False)
    public = out / 'public'
    public.mkdir()
    (public / 'images').mkdir()
    report_path = run / 'evaluation/report.json'
    report = json.loads(report_path.read_text())
    if len(report['records']) != 25:
        raise ValueError('This preregistration expects the existing 25 development inputs')
    groups=[r.get('identity_group') for r in report['records']]
    if None in groups or len(set(groups))!=25:
        raise ValueError('Expected 25 distinct conservative source groups, not verified identities')
    index = {(r['name'], r['route'], r['condition']): r for r in report['outputs']}
    seed = secrets.randbits(128)
    rng = random.Random(seed)
    study_id = 'texture-' + secrets.token_hex(6)
    aliases, hashes, credits = {}, {}, []

    def copy_image(path, expected=None):
        digest = sha256(path)
        if expected and digest != expected:
            raise ValueError('Native output hash differs: ' + str(path))
        if str(path) in aliases:
            return aliases[str(path)]
        name = 'images/' + secrets.token_hex(12)
        with Image.open(path) as image:
            image.load()
            if image.format != 'PNG' or image.size != (1024, 1024):
                raise ValueError('Expected native 1024 PNG: ' + str(path))
            shutil.copy2(path, public / (name + '.png'))
            for region, box in REGIONS.items():
                image.crop(box).save(public / (name + '_' + region + '.png'))
        aliases[str(path)] = name
        hashes[str(path)] = digest
        return name

    trials, keys = [], {}
    for row in report['records']:
        name = row['name']
        reference = copy_image(run / 'evaluation' / name / 'reference.png', row['reference_sha256'])
        meta = row['source_metadata']
        credits.append(dict(author=meta['author'], title=meta['photo_title'],
                            url=meta['photo_url'], license=meta['license']))
        for candidate in ['personalized_reconstruction', 'personalized_locality']:
            routes = ['wplus_base', candidate]
            rng.shuffle(routes)
            images = []
            for route in routes:
                spec = index[name, route, '75']
                images.append(copy_image(run / 'evaluation' / spec['native'], spec['sha256']))
            tid = secrets.token_hex(10)
            trials.append(dict(id=tid, reference=reference, left=images[0], right=images[1], target=75))
            keys[tid] = dict(subject=name, candidate=candidate, left=routes[0], right=routes[1], repeat_of=None)
    for trial in rng.sample(trials, 5):
        tid = secrets.token_hex(10)
        # Reverse sides so consistency cannot be scored as memorizing "left".
        trials.append(dict(trial, id=tid, left=trial['right'], right=trial['left']))
        k = keys[trial['id']]
        keys[tid] = dict(k, left=k['right'], right=k['left'], repeat_of=trial['id'])
    rng.shuffle(trials)
    public_study = dict(schema=1, study_id=study_id, trials=trials, credits=credits)
    data = json.dumps(public_study, ensure_ascii=False).replace('<', '\\u003c')
    (public / 'index.html').write_text(HTML.replace('__STUDY__', data))
    private = dict(schema=1, study_id=study_id, seed=seed, trial_key=keys,
                   run=str(run), report_sha256=sha256(report_path), input_sha256=hashes,
                   created_at=datetime.now(timezone.utc).isoformat(),
                   protocol=dict(primary='native 75-year texture preference', subjects=25,
                                 primary_trials=50, repeat_trials=5, raters_required=2,
                                 texture='candidate vs baseline preference, ties and uncertain reported separately',
                                 aggregation='Exclude repeats from efficacy. One vote per rater/subject/candidate. Report paired subject-level results.',
                                 secondary=['identity preference', 'age appearance preference', 'serious artifact flags'],
                                 limitations=['Route-hidden development review; inputs have been inspected previously.',
                                              'No independent new-subject or chronological-age accuracy claim.',
                                              'No automatic model promotion; no training data or weights changed.']))
    write_json(out / 'private-key.json', private)
    (out / 'private-key.json').chmod(0o600)
    write_json(out / 'build-receipt.json', dict(study_id=study_id, subjects=25, primary_trials=50,
                                               repeat_trials=5, native_images=len(aliases),
                                               derived_crops=len(aliases)*len(REGIONS),
                                               html_sha256=sha256(public / 'index.html'),
                                               no_training=True, no_model_change=True,
                                               private_key_outside_public=True))
    print(json.dumps(dict(output=str(out), study_id=study_id, trials=len(trials), native_images=len(aliases))))


def score(out, answer_paths):
    private = json.loads((out / 'private-key.json').read_text())
    keys = private['trial_key']
    raters, votes, artifacts, repeats = set(), defaultdict(list), defaultdict(list), []
    completion = {}
    for path in answer_paths:
        answer = json.loads(path.read_text())
        rater = answer.get('rater', '').strip()
        if not rater or rater in raters or answer.get('study_id') != private['study_id']:
            raise ValueError('Mismatched study or missing/duplicate rater: ' + str(path))
        raters.add(rater)
        answers = answer['answers']
        if set(answers) - set(keys):
            raise ValueError('Unknown trial id')
        completion[rater] = dict(primary=sum(not keys[t]['repeat_of'] for t in answers),
                                 repeat=sum(bool(keys[t]['repeat_of']) for t in answers))
        for tid, a in answers.items():
            k = keys[tid]
            if any(a.get(m) not in {'left', 'right', 'tie', 'uncertain'} for m in METRICS):
                raise ValueError('Invalid or missing preference')
            if any(type(a.get(f)) is not bool for f in ['artifact_left','artifact_right']):
                raise ValueError('Invalid artifact flag')
            def route_preference(key, value):
                return key[value] if value in {'left', 'right'} else value
            if k['repeat_of']:
                first = answers.get(k['repeat_of'])
                if first:
                    repeats.append(dict(rater=rater, trial=tid, agreement={
                        m: route_preference(k,a[m]) == route_preference(keys[k['repeat_of']], first[m]) for m in METRICS}))
                continue
            for metric in METRICS:
                value = route_preference(k, a[metric])
                votes[k['candidate'], k['subject'], metric].append(
                    'candidate' if value == k['candidate'] else 'baseline' if value == 'wplus_base' else value)
            for side in ['left', 'right']:
                artifacts[k['candidate'], k['subject'], k[side]].append(a['artifact_'+side])
    complete_raters = sum(r['primary']==50 and r['repeat']==5 for r in completion.values())
    result = dict(study_id=private['study_id'], raters=len(raters), completed_raters=complete_raters,
                  reviewer_completion=completion,
                  minimum_raters_met=len(raters)>=private['protocol']['raters_required'],
                  status='human_ratings_recorded' if complete_raters>=2 else 'insufficient_complete_reviews',
                  comparisons={}, repeats=repeats,
                  limitations=private['protocol']['limitations'], automatic_promotion=False)
    for candidate in sorted({k['candidate'] for k in keys.values()}):
        data = {}
        for metric in METRICS:
            counts = Counter()
            subjects = Counter()
            complete = 0
            for (arm, subject, m), values in votes.items():
                if arm != candidate or m != metric:continue
                counts.update(values)
                if len(values)==len(raters):complete+=1
                c = values.count('candidate'); b = values.count('baseline')
                if c>b:subjects['candidate']+=1
                elif b>c:subjects['baseline']+=1
                elif c or b or 'tie' in values:subjects['tie']+=1
                else:subjects['uncertain']+=1
            data[metric] = dict(votes=dict(counts), subject_majority=dict(subjects),
                                subjects_with_all_submitted_raters=complete)
        data['serious_artifacts'] = {route:dict(flagged=sum(sum(vals) for (arm,subject,r),vals in artifacts.items() if arm==candidate and r==route),
                                                     reviewed=sum(len(vals) for (arm,subject,r),vals in artifacts.items() if arm==candidate and r==route))
                                       for route in ['wplus_base',candidate]}
        result['comparisons'][candidate] = data
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['build','score'])
    parser.add_argument('output',type=Path)
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--answers',type=Path,nargs='+')
    args=parser.parse_args()
    if args.action=='build':build(args.run.resolve(),args.output.resolve())
    elif not args.answers:parser.error('score requires --answers')
    else:score(args.output.resolve(),args.answers)

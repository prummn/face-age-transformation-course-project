"""Acquire additional original FFHQ faces and extend the existing common audit.

Publisher demographic labels and coarse apparent ages remain annotations, not
inferred ethnicity or chronological ground truth. No protected rows are reused.
"""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys
import time
from urllib.parse import urlsplit
from probe_inversion import sha256,write_json

ROOT=Path(__file__).resolve().parents[2]
AUDIT=ROOT/'deployment/age-quality-final/common-audit-20261124'
SOURCE=ROOT/'deployment/data-research/20261113'


def lines(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def account(row):
    p=urlsplit(row.get('source_metadata',{}).get('photo_url','')).path.split('/')
    return row.get('flickr_account') or (p[2] if len(p)>2 and p[1]=='photos' else None)


def plan(out):
    if (out/'plan.json').exists():raise ValueError('Preserving existing frozen plan')
    m=json.loads((SOURCE/'ffhq-dataset-v2.json').read_text())
    with (ROOT/'deployment/asia-adaptation/20261123/ffhq-demographics.csv').open(encoding='utf-8-sig') as f:
        asian={int(Path(r['File']).stem) for r in csv.DictReader(f) if r['Ethnic'].strip().lower()=='asian'}
    mapping=json.loads((AUDIT/'common-groups.json').read_text());known={r['name'] for r in mapping}
    protected={r['name'] for split in ('eval','external') for r in lines(AUDIT/(split+'.jsonl'))}
    blocked={r['group'] for r in mapping if r['name'] in protected or r['origin']=='protected'}
    blocked_accounts={account(dict(source_metadata=m[str(int(r['name'].split('_')[1]))]['metadata'])) for r in mapping if r['name'].startswith('ffhq_') and r['group'] in blocked}
    pools=defaultdict(list);excluded=Counter()
    for r in csv.DictReader((SOURCE/'ffhq_aging_labels.csv').open()):
        n=int(r['image_number']);item=m[r['image_number']];lo,hi=map(int,r['age_group'].split('-'));rect=item['in_the_wild']['face_rect'];edge=min(rect[2]-rect[0],rect[3]-rect[1]);name=f'ffhq_{n:05d}'
        if item['category']!='training' or name in known or lo<20 or edge<512:continue
        if n not in asian and lo<70:continue
        if float(r['age_group_confidence'])<.6 or abs(float(r['head_yaw']))>35 or abs(float(r['head_pitch']))>25 or max(float(r['left_eye_occluded']),float(r['right_eye_occluded']))>15:continue
        record=dict(image_id=n,category='training',age_interval=[lo,hi],annotation=r,source_face_short_edge=edge,
            source_metadata=item['metadata'],image_spec=item['image'],published_demographic_label='Asian' if n in asian else None,
            demographic_label_source='Leyva et al., CVPR Workshops2024 broad publisher visual annotation' if n in asian else None)
        if account(record) in blocked_accounts:excluded[str((lo,n in asian))]+=1;continue
        pools[lo,n in asian].append(record)
    rng=random.Random(20261124);selected=[]
    for (lo,annotation),rows in sorted(pools.items()):
        rng.shuffle(rows);cap=150 if lo<50 else 1000
        selected.extend(rows[:cap])
    out.mkdir(parents=True,exist_ok=False);(out/'native1024').mkdir()
    write_json(out/'plan.json',dict(samples=selected,count=len(selected),counts=dict(Counter(str((r['age_interval'],r['published_demographic_label'])) for r in selected)),
        excluded_blocked_account=dict(excluded),metadata_sha256=sha256(SOURCE/'ffhq-dataset-v2.json'),labels_sha256=sha256(SOURCE/'ffhq_aging_labels.csv'),
        common_mapping_sha256=sha256(AUDIT/'common-groups.json'),criteria=dict(native_face_edge_min=512,confidence_min=.6,yaw_max=35,pitch_max=25,eye_occlusion_max=15),
        notes=['Only original official training split; source pixel/file MD5 will be checked.','New35degree yaw versus prior20 is disclosed; pixel threshold unchanged.','Candidates require native texture review and expanded common grouping before admission.']))
    print('EXPANSION_PLAN',len(selected),dict(Counter(str((r['age_interval'],r['published_demographic_label'])) for r in selected)),flush=True)


def collect(out):
    from http_zip import HTTPZip
    from PIL import Image
    selected=json.loads((out/'plan.json').read_text())['samples']
    archives=[HTTPZip.from_index(SOURCE/f'ffhq_zip{i}-index.json',max_transfer=2*1024**3) for i in (1,2)]
    for a in archives:a.resolve_download()
    lookup={Path(n).name:(a,n) for a in archives for n in a.entries if n.endswith('.png')}
    def fetch(row):
        name=f"{row['image_id']:05d}.png";path=out/'native1024'/name;spec=row['image_spec']
        for attempt in range(3):
            try:
                a,n=lookup[name];data=path.read_bytes() if path.exists() else a.read(n)
                if len(data)!=spec['file_size'] or hashlib.md5(data).hexdigest()!=spec['file_md5']:raise ValueError('Official file digest mismatch')
                with Image.open(io.BytesIO(data)) as im:
                    im.load()
                    if im.size!=(1024,1024) or im.mode!='RGB' or im.format!='PNG' or hashlib.md5(im.tobytes()).hexdigest()!=spec['pixel_md5']:raise ValueError('Official pixels mismatch')
                if not path.exists():path.write_bytes(data)
                return row|dict(status='verified',path=str(path),sha256=hashlib.sha256(data).hexdigest())
            except Exception as e:
                if attempt==2:return row|dict(status='failed',error=type(e).__name__)
                time.sleep(1)
    results=[];started=time.monotonic()
    with ThreadPoolExecutor(max_workers=6) as executor:
        for r in executor.map(fetch,selected):
            results.append(r)
            if len(results)%25==0:print('EXPANSION_COLLECTED',len(results),'/',len(selected),'verified',sum(x['status']=='verified' for x in results),flush=True)
    (out/'collection-manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results))
    write_json(out/'collection-report.json',dict(selected=len(selected),verified=sum(r['status']=='verified' for r in results),seconds=time.monotonic()-started,transfers=[a.stats() for a in archives]))
    if any(r['status']!='verified' for r in results):raise ValueError('Incomplete download; verified files retained for resume')


def prepare(out):
    import cv2
    import numpy as np
    from PIL import Image
    corpus=lines(out/'collection-manifest.jsonl')
    if any(r['status']!='verified' for r in corpus):raise ValueError('Unverified originals')
    folder=out/'prepared';folder.mkdir(exist_ok=False);(folder/'256').mkdir();(folder/'512').mkdir()
    cv2.setNumThreads(2);det=cv2.FaceDetectorYN.create(str(ROOT/'deployment/model-files/face-age/models/yunet/face_detection_yunet_2023mar.onnx'),'',(256,256),.6,.3,100)
    sf=cv2.FaceRecognizerSF.create(str(ROOT/'deployment/gan-research/face_recognition_sface_2021dec.onnx'),'')
    mapping=json.loads((AUDIT/'common-groups.json').read_text());features=np.load(AUDIT/'features.npz',allow_pickle=False)['features'];assert len(mapping)==len(features)
    prior={split:lines(AUDIT/(split+'.jsonl')) for split in ('train','eval','external')};protected={r['name'] for split in ('eval','external') for r in prior[split]}
    blocked={r['group'] for r in mapping if r['name'] in protected or r['origin']=='protected'}
    oldgroups=sorted({r['group'] for r in mapping});node={g:i for i,g in enumerate(oldgroups)};parent=list(range(len(oldgroups)+len(corpus)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def join(a,b):
        a,b=sorted((find(a),find(b)));parent[b]=a
    accounts={}
    meta=json.loads((SOURCE/'ffhq-dataset-v2.json').read_text())
    for r in mapping:
        if r['name'].startswith('ffhq_'):
            key=account(dict(source_metadata=meta[str(int(r['name'].split('_')[1]))]['metadata']))
            if key:accounts[key]=node[r['group']]
    accepted=[];newfeatures=[];reject=[]
    for index,r in enumerate(corpus):
        if sha256(r['path'])!=r['sha256']:raise ValueError('Original changed')
        rgb=np.asarray(Image.open(r['path']).convert('RGB'));small=cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA);bgr=cv2.cvtColor(small,cv2.COLOR_RGB2BGR);_,faces=det.detect(bgr)
        sharp=float(cv2.Laplacian(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)[55:220,45:210],cv2.CV_64F).var())
        if faces is None or sharp<60:reject.append(dict(image_id=r['image_id'],reason='face_or_sharpness',sharpness=sharp));continue
        face=max(faces,key=lambda f:f[2]*f[3]);v=sf.feature(sf.alignCrop(bgr,face)).ravel();v/=np.linalg.norm(v)+1e-12
        i=len(oldgroups)+index;key=account(r)
        if key:
            if key in accounts:join(i,accounts[key])
            else:accounts[key]=i
        for j in np.flatnonzero(features@v>=.4):join(i,node[mapping[int(j)]['group']])
        for j,f in newfeatures:
            if float(f@v)>=.4:join(i,j)
        newfeatures.append((i,v));name=f"ffhq_{r['image_id']:05d}";inp=folder/'256'/(name+'.png');ref=folder/'512'/(name+'.png')
        Image.fromarray(small).save(inp,compress_level=2);Image.fromarray(cv2.resize(rgb,(512,512),interpolation=cv2.INTER_AREA)).save(ref,compress_level=2)
        row=dict(name=name,image_id=r['image_id'],image_path=str(inp),sha256=sha256(inp),reference_path=r['path'],reference_sha256=r['sha256'],reference512_path=str(ref),reference512_sha256=sha256(ref),
            source_metadata=r['source_metadata'],flickr_account=key,training_authorized=True,official_split='training',source_face_short_edge=r['source_face_short_edge'],face_sharpness=sharp,
            age_interval=r['age_interval'],age_confidence=float(r['annotation']['age_group_confidence']),age_interval_kind='publisher apparent-age interval, not chronological age',
            publisher_gender=r['annotation']['gender'],published_demographic_label=r['published_demographic_label'],demographic_label_source=r['demographic_label_source'],cohort='expanded-native-training',
            collection='expanded-native-age-20261124',source_age=None,age_kind='pseudo')
        accepted.append((i,row))
        if (index+1)%50==0:print('EXPANSION_CPU_SCREEN',index+1,'/',len(corpus),flush=True)
    bad={find(node[g]) for g in blocked};byname={r['name']:node[r['group']] for r in mapping}
    train=[]
    for row in prior['train']:
        i=byname[row['name']]
        if find(i) in bad:reject.append(dict(name=row['name'],reason='new_transitive_protected_bridge'));continue
        if row.get('reference_path') and row.get('source_face_short_edge',row.get('native_face_short_edge',0))>=512:train.append((i,row))
    old_count=len(train)
    for i,row in accepted:
        if find(i) in bad:reject.append(dict(name=row['name'],reason='expanded_common_group_touches_protected'));continue
        train.append((i,row))
    label=lambda i:'expanded_'+hashlib.sha256(str(find(i)).encode()).hexdigest()[:16]
    train=[dict(r,identity_group=label(i)) for i,r in train]
    for split,records in prior.items():
        data=train if split=='train' else [dict(r,identity_group=label(byname[r['name']])) for r in records]
        (folder/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data))
    assert not {r['identity_group'] for r in train}&{label(byname[r['name']]) for split in ('eval','external') for r in prior[split]}
    write_json(folder/'audit.json',dict(train=len(train),prior_native_retained=old_count,new_training=len(train)-old_count,
        train_by_age=dict(Counter(str(r['age_interval']) for r in train)),train_by_annotation=dict(Counter(str((r['age_interval'],r.get('published_demographic_label'))) for r in train)),
        rejected=reject,common_rule='Extend previous6657-row universe with all new candidates, same account/hash/SFace>=.4 transitive groups; recompute blocked components and remove newly bridged old train rows.',
        files={s+'.jsonl':sha256(folder/(s+'.jsonl')) for s in prior},notes=['Conservative source/similarity grouping is not verified identity.','No relaxed native pixel threshold or recovered excluded old row.']))
    (folder/'DONE').write_text('Expanded common grouped native data ready; new source ages need frozen baseline inference\n');print('EXPANDED_DATA_READY',len(train),len(train)-old_count,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','collect','prepare']);p.add_argument('output',type=Path);a=p.parse_args();globals()[a.action](a.output.resolve())

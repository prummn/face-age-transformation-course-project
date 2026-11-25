"""Freeze new high-resolution validation candidates before inspecting GAN outputs."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import time
from PIL import Image
from probe_inversion import sha256,write_json
from prepare_quality_training import account
from http_zip import HTTPZip

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'deployment/data-research/20261113'
AUDIT=ROOT/'deployment/age-quality-final/common-audit-20261124'
NATIVE=ROOT/'deployment/training-runs/native-full-20261124-141846'


def plan(out):
    if hashlib.md5((SOURCE/'ffhq-dataset-v2.json').read_bytes()).hexdigest()!='425ae20f06a4da1d4dc0f46d40ba5fd6':raise ValueError('Official metadata changed')
    if sha256(SOURCE/'ffhq_aging_labels.csv')!='b99585b0ab9720a0ea55b71d677126b4a1fb1c6b7e362a48e6f00ac835cabc04':raise ValueError('Publisher age annotations changed')
    meta=json.loads((SOURCE/'ffhq-dataset-v2.json').read_text())
    mapping=json.loads((AUDIT/'common-groups.json').read_text());known={r['name'] for r in mapping}
    latest=[json.loads(s) for s in (NATIVE/'data/train.jsonl').read_text().splitlines()]
    known.update(r['name'] for r in latest)
    accounts={account(dict(source_metadata=meta[str(int(r['name'].split('_')[1]))]['metadata'])) for r in mapping if r['name'].startswith('ffhq_')}
    accounts.update(account(r) for r in latest);accounts.discard(None);accounts.discard('')
    for path in (ROOT/'deployment/training-runs').glob('*/evaluation/report.json'):
        for r in json.loads(path.read_text()).get('records',[]):known.add(r['name'])
    groups={}
    with (SOURCE/'ffhq_aging_labels.csv').open() as f:
        for label in csv.DictReader(f):
            n=int(label['image_number']);item=meta[str(n)];name=f'ffhq_{n:05d}';lo,hi=map(int,label['age_group'].split('-'))
            rect=item['in_the_wild']['face_rect'];edge=min(rect[2]-rect[0],rect[3]-rect[1])
            if item['category']!='validation' or name in known or lo<20 or edge<700 or account(dict(source_metadata=item['metadata'])) in accounts:continue
            if float(label['age_group_confidence'])<.6 or abs(float(label['head_yaw']))>20 or abs(float(label['head_pitch']))>15 or max(float(label['left_eye_occluded']),float(label['right_eye_occluded']))>15:continue
            groups.setdefault(lo,[]).append(dict(name=name,image_id=n,age_interval=[lo,hi],source_face_short_edge=edge,annotation=label,source_metadata=item['metadata'],image_spec=item['image']))
    rng=random.Random(20261125);selected=[]
    for age,count in [(70,8),(50,16),(40,24),(30,28),(20,28)]:
        pool=groups.get(age,[]);rng.shuffle(pool);selected+=pool[:count]
    out.mkdir(parents=True,exist_ok=False);(out/'native1024').mkdir()
    write_json(out/'plan.json',dict(seed=20261125,samples=selected,candidates=len(selected),counts=dict(Counter(str(r['age_interval']) for r in selected)),known_names=len(known),banned_accounts=len(accounts),mapping_sha256=sha256(AUDIT/'common-groups.json'),
        native_face_edge_min=700,scope='New official FFHQ validation IDs, no prior account or known evaluation IDs. Reserve only; no model output used in selection. FFHQ generator pretraining overlap still prevents population-independence claims. No demographic inference or balanced Asian70+ claim.'))
    print('FRESH_HOLDOUT_PLANNED',len(selected),out,flush=True)


def collect(out):
    records=json.loads((out/'plan.json').read_text())['samples']
    archives=[HTTPZip.from_index(SOURCE/f'ffhq_zip{i}-index.json',max_transfer=512*1024**2) for i in [1,2]]
    for archive in archives:archive.resolve_download()
    lookup={Path(name).name:(archive,name) for archive in archives for name in archive.entries if name.endswith('.png')}
    def download(row):
        path=out/'native1024'/f"{row['image_id']:05d}.png";spec=row['image_spec'];last=None
        for _ in range(2):
            try:
                arc,member=lookup[path.name];data=path.read_bytes() if path.exists() else arc.read(member)
                if len(data)!=spec['file_size'] or hashlib.md5(data).hexdigest()!=spec['file_md5']:raise ValueError('Official size/MD5 differs')
                with Image.open(io.BytesIO(data)) as im:
                    im.load()
                    if im.size!=(1024,1024) or im.mode!='RGB' or hashlib.md5(im.tobytes()).hexdigest()!=spec['pixel_md5']:raise ValueError('Official pixel MD5 differs')
                if not path.exists():path.write_bytes(data)
                return dict(row,status='verified',path=str(path),sha256=hashlib.sha256(data).hexdigest())
            except Exception as error:last=type(error).__name__
        return dict(row,status='failed',error=last)
    results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for row in pool.map(download,records):
            results.append(row);print('FRESH_DOWNLOADED',len(results),'/',len(records),row['status'],flush=True)
    (out/'collection-manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results))
    write_json(out/'collection-report.json',dict(selected=len(records),verified=sum(r['status']=='verified' for r in results),transfers=[a.stats() for a in archives]))


def screen(out):
    import cv2
    import numpy as np
    cv2.setNumThreads(2)
    det=cv2.FaceDetectorYN.create(str(ROOT/'deployment/model-files/face-age/models/yunet/face_detection_yunet_2023mar.onnx'),'',(256,256),.6,.3,100)
    sf=cv2.FaceRecognizerSF.create(str(ROOT/'deployment/gan-research/face_recognition_sface_2021dec.onnx'),'')
    def describe(rgb):
        small=cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA);bgr=cv2.cvtColor(small,cv2.COLOR_RGB2BGR);_,faces=det.detect(bgr)
        sharp=float(cv2.Laplacian(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)[55:220,45:210],cv2.CV_64F).var())
        if faces is None:return small,sharp,None
        face=max(faces,key=lambda f:f[2]*f[3]);v=sf.feature(sf.alignCrop(bgr,face)).ravel();v/=np.linalg.norm(v)+1e-12
        return small,sharp,v
    mapping=json.loads((AUDIT/'common-groups.json').read_text());features=list(np.load(AUDIT/'features.npz',allow_pickle=False)['features']);known={r['name'] for r in mapping}
    for row in [json.loads(s) for s in (NATIVE/'data/train.jsonl').read_text().splitlines()]:
        if row['name'] not in known:
            if sha256(row['reference_path'])!=row['reference_sha256']:raise ValueError('Latest training input changed')
            with Image.open(row['reference_path']) as im:_,_,v=describe(np.asarray(im.convert('RGB')))
            if v is not None:features.append(v)
    matrix=np.stack(features);rows=[json.loads(s) for s in (out/'collection-manifest.jsonl').read_text().splitlines()]
    label_file=ROOT/'deployment/asia-adaptation/20261123/ffhq-demographics.csv'
    with label_file.open(encoding='utf-8-sig') as f:
        publisher_labels={int(Path(r['File']).stem):r['Ethnic'] for r in csv.DictReader(f)}
    accepted=[];rejected=[];vectors=[];accounts=set();folder=out/'prepared';folder.mkdir(exist_ok=False);(folder/'256').mkdir()
    for row in rows:
        if row['status']!='verified':rejected.append(dict(name=row['name'],reason='download_failed'));continue
        if sha256(row['path'])!=row['sha256']:raise ValueError('New portrait changed')
        with Image.open(row['path']) as im:rgb=np.asarray(im.convert('RGB'))
        small,sharp,v=describe(rgb);key=account(row)
        maximum=float((matrix@v).max()) if v is not None else None
        newmaximum=max((float(u@v) for u in vectors),default=-1) if v is not None else None
        if v is None or sharp<60 or maximum>=.4 or newmaximum>=.4 or (key and key in accounts):
            rejected.append(dict(name=row['name'],reason='face_sharpness_or_prior_or_internal_similarity',sharpness=sharp,prior_maximum=maximum,new_maximum=newmaximum));continue
        vectors.append(v)
        if key:accounts.add(key)
        path=folder/'256'/(row['name']+'.png');Image.fromarray(small).save(path)
        accepted.append(dict(name=row['name'],image_id=row['image_id'],cohort='fresh-hd-reserve',image_path=str(path),sha256=sha256(path),reference_path=row['path'],reference_sha256=row['sha256'],source_metadata=row['source_metadata'],source_face_short_edge=row['source_face_short_edge'],face_sharpness=sharp,
            age_interval=row['age_interval'],age_confidence=float(row['annotation']['age_group_confidence']),publisher_gender=row['annotation']['gender'],published_demographic_label=publisher_labels.get(row['image_id']),demographic_label_source='Leyva et al. published broad visual category, not self-report',source_age=None,age_kind='pseudo_pending_frozen_FairFace',training_authorized=False,official_split='validation',identity_group='fresh_'+row['name'],prior_sface_maximum=maximum))
    write_json(folder/'reserve.json',accepted)
    write_json(folder/'audit.json',dict(downloaded=sum(r['status']=='verified' for r in rows),accepted=len(accepted),age_counts=dict(Counter(str(r['age_interval']) for r in accepted)),publisher_labels=dict(Counter(str(r['published_demographic_label']) for r in accepted)),publisher_labels_sha256=sha256(label_file),rejected=rejected,prior_features=len(matrix),selection_before_model_outputs=True,
        limitations=['No generated outputs or model selection on these photos yet.','Conservative source/hash/SFace screening does not verify true identities.','FFHQ pretraining overlap; not an independent population test.','Asian high-age coverage is not established.','Human native texture review and final blinded route scoring still pending.']))
    (folder/'DONE').write_text('Fresh candidate reserve frozen; no GAN inference or new training\n');print('FRESH_RESERVE_READY',len(accepted),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','collect','screen']);p.add_argument('output',type=Path);a=p.parse_args();globals()[a.action](a.output.resolve())

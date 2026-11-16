"""Collect a fixed FFHQ-Aging adult pilot via verified, bounded ZIP members."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import threading
import time

from PIL import Image

from collect_ffhq_data_pilot import digest
from http_zip import HTTPZip


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--extend-from',type=Path,help='Preserve prior frozen1280+80 selection and add400 training candidates')
    args=p.parse_args()
    if not 1<=args.workers<=4:p.error('workers must be1..4')
    source=args.source.resolve();out=args.output.resolve()
    meta_path=source/'ffhq-dataset-v2.json'
    if digest(meta_path,'md5')!='425ae20f06a4da1d4dc0f46d40ba5fd6':raise ValueError('Metadata MD5')
    metadata=json.loads(meta_path.read_text())
    labels=list(csv.DictReader((source/'ffhq_aging_labels.csv').open()))
    rng=random.Random(20261113);selected=[]
    for category,n in [('training',256),('validation',16)]:
        for age_bin in ['20-29','30-39','40-49','50-69','70-120']:
            for gender in ['male','female']:
                candidates=[]
                for row in labels:
                    if row['age_group']!=age_bin or row['gender']!=gender:continue
                    item=metadata[row['image_number']];rect=item['in_the_wild']['face_rect']
                    edge=min(rect[2]-rect[0],rect[3]-rect[1])
                    if (item['category']==category and edge>=512 and float(row['age_group_confidence'])>=.6
                        and abs(float(row['head_yaw']))<=20 and abs(float(row['head_pitch']))<=15
                        and max(float(row['left_eye_occluded']),float(row['right_eye_occluded']))<=15):
                        candidates.append((row,item,edge))
                candidates.sort(key=lambda r:int(r[0]['image_number']));rng.shuffle(candidates)
                if len(candidates)<n//2:raise ValueError('Not enough candidates')
                for row,item,edge in candidates[:n//2]:
                    selected.append(dict(image_id=int(row['image_number']),category=category,
                        age_interval=list(map(int,age_bin.split('-'))),annotation=row,
                        age_label_kind='Appen human apparent-age interval; not chronological age',
                        source_face_short_edge=edge,source_metadata=item['metadata'],image_spec=item['image']))
    if args.extend_from:
        previous=json.loads((args.extend_from/'collection-plan.json').read_text())
        if previous['samples']!=selected:raise ValueError('Extension requires the unchanged original1280+80 plan')
        used={r['image_id'] for r in selected};extra_rng=random.Random(20261114)
        for age_bin in ['20-29','30-39','40-49','50-69','70-120']:
            for gender in ['male','female']:
                extra=[]
                for row in labels:
                    if row['age_group']!=age_bin or row['gender']!=gender or int(row['image_number']) in used:continue
                    item=metadata[row['image_number']];rect=item['in_the_wild']['face_rect'];edge=min(rect[2]-rect[0],rect[3]-rect[1])
                    if (item['category']=='training' and edge>=512 and float(row['age_group_confidence'])>=.6
                        and abs(float(row['head_yaw']))<=20 and abs(float(row['head_pitch']))<=15
                        and max(float(row['left_eye_occluded']),float(row['right_eye_occluded']))<=15):extra.append((row,item,edge))
                extra.sort(key=lambda x:int(x[0]['image_number']));extra_rng.shuffle(extra)
                if len(extra)<40:raise ValueError('Insufficient unchanged-rule reserve')
                for row,item,edge in extra[:40]:
                    selected.append(dict(image_id=int(row['image_number']),category='training',
                        age_interval=list(map(int,age_bin.split('-'))),annotation=row,
                        age_label_kind='Appen human apparent-age interval; not chronological age',
                        source_face_short_edge=edge,source_metadata=item['metadata'],image_spec=item['image']))
    out.mkdir(parents=True,exist_ok=True);(out/'native1024').mkdir(exist_ok=True)
    plan=dict(seed=20261113,training_selected=sum(r['category']=='training' for r in selected),validation_selected=80,
              metadata_sha256=digest(meta_path),labels_sha256=digest(source/'ffhq_aging_labels.csv'),samples=selected)
    if args.extend_from:plan.update(extension_seed=20261114,extended_from_sha256=digest(args.extend_from/'collection-plan.json'))
    planpath=out/'collection-plan.json'
    if planpath.exists() and json.loads(planpath.read_text())!=plan:raise ValueError('Existing plan differs')
    planpath.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    archives=[HTTPZip.from_index(source/f'ffhq_zip{i}-index.json',max_transfer=3*1024**3) for i in [1,2]]
    for archive in archives:archive.resolve_download()
    lookup={Path(name).name:(archive,name) for archive in archives for name in archive.entries if name.endswith('.png')}
    reusable={p.name:p for p in (source/'ffhq-pilot32/images').glob('*.png')}
    if args.extend_from:reusable.update({p.name:p for p in (args.extend_from/'native1024').glob('*.png')})
    lock=threading.Lock();started=time.monotonic();results=[];logpath=out/'collection.jsonl'
    def fetch(row):
        name=f"{row['image_id']:05d}.png";path=out/'native1024'/name;spec=row['image_spec'];result=dict(row)
        try:
            if path.exists():data=path.read_bytes();delivery='existing verified file'
            elif name in reusable:data=reusable[name].read_bytes();delivery='existing diagnostic collection'
            else:
                archive,member=lookup[name];data=archive.read(member);delivery=archive.url
            if len(data)!=spec['file_size'] or hashlib.md5(data).hexdigest()!=spec['file_md5']:
                raise ValueError('File MD5/size mismatch')
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                if im.size!=(1024,1024) or im.mode!='RGB' or im.format!='PNG':raise ValueError('Image format/size')
                if hashlib.md5(im.tobytes()).hexdigest()!=spec['pixel_md5']:raise ValueError('Pixel MD5')
            if not path.exists():
                temp=path.with_suffix('.tmp');temp.write_bytes(data);temp.replace(path)
            result.update(status='verified',path=str(path),sha256=hashlib.sha256(data).hexdigest(),delivery=delivery)
        except Exception as e:result.update(status='failed',error=type(e).__name__,detail=str(e)[:180])
        with lock:
            with logpath.open('a') as stream:stream.write(json.dumps(result,ensure_ascii=False)+'\n')
        return result
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending=[pool.submit(fetch,row) for row in selected]
        for future in as_completed(pending):
            result=future.result();results.append(result)
            if len(results)%40==0 or result['status']!='verified':
                print('COLLECT',len(results),'/',len(selected),'verified',sum(r['status']=='verified' for r in results),
                      'seconds',round(time.monotonic()-started,1),
                      'last_error',result.get('detail'),flush=True)
    results.sort(key=lambda r:r['image_id'])
    (out/'collection-manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results))
    report=dict(selected=len(selected),verified=sum(r['status']=='verified' for r in results),
                failed=sum(r['status']=='failed' for r in results),seconds=time.monotonic()-started,
                transfers=[a.stats() for a in archives])
    (out/'collection-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    if report['failed']:raise SystemExit(1)


if __name__=='__main__':main()

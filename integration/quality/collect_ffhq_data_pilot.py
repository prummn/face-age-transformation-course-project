"""Download a small, reproducible FFHQ-Aging adult data audit, without training.

Requires the upstream FFHQ v2 metadata and FFHQ-Aging CSV. Images retain their
individual attribution/license records. Official age intervals are not exact
chronological ages, and official FFHQ splits are not identity-disjoint.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import time
from urllib.parse import parse_qs,urlparse

import requests
from PIL import Image


def digest(path,kind='sha256'):
    h=hashlib.new(kind)
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--per-bin',type=int,default=8)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--resume',action='store_true',help='Verify existing images, retry missing images')
    parser.add_argument('--mirror-index',type=Path,action='append',default=[],
                        help='Use selected HTTP ZIP members; verify against official file and pixel MD5')
    args=parser.parse_args()
    if args.per_bin<2 or args.per_bin>32 or args.per_bin%2:
        parser.error('per-bin must be even and within2..32')
    if not 1<=args.workers<=4:parser.error('workers must be1..4')
    if digest(args.metadata,'md5')!='425ae20f06a4da1d4dc0f46d40ba5fd6':
        raise RuntimeError('FFHQ metadata differs from publisher canonical MD5')
    metadata=json.loads(args.metadata.read_text())
    rows=list(csv.DictReader(args.labels.open()))
    selected=[];rng=random.Random(20261113)
    for age_bin in ['20-29','40-49','50-69','70-120']:
        for gender in ['female','male']:
            candidates=[]
            for row in rows:
                if row['age_group']!=age_bin or row['gender']!=gender:continue
                record=metadata[row['image_number']]
                rect=record['in_the_wild']['face_rect']
                face_edge=min(rect[2]-rect[0],rect[3]-rect[1])
                if (record['category']=='training' and face_edge>=512
                    and float(row['age_group_confidence'])>=.6
                    and abs(float(row['head_yaw']))<=20
                    and abs(float(row['head_pitch']))<=15
                    and max(float(row['left_eye_occluded']),float(row['right_eye_occluded']))<=15):
                    candidates.append((row,record,face_edge))
            candidates.sort(key=lambda item:int(item[0]['image_number']))
            rng.shuffle(candidates)
            if len(candidates)<args.per_bin//2:raise RuntimeError('Insufficient candidates')
            selected.extend(candidates[:args.per_bin//2])
    args.output.mkdir(parents=True,exist_ok=args.resume)
    (args.output/'images').mkdir(exist_ok=args.resume)
    if args.resume:
        old=[json.loads(line) for line in (args.output/'manifest.jsonl').read_text().splitlines()]
        if [row['image_id'] for row in old]!=[int(row['image_number']) for row,_,_ in selected]:
            raise RuntimeError('Resume selection differs from existing manifest')
    mirrors=[]
    if args.mirror_index:
        from http_zip import HTTPZip
        mirrors=[HTTPZip.from_index(path) for path in args.mirror_index]

    def download(item):
        row,record,edge=item;spec=record['image'];name=f"{int(row['image_number']):05d}.png"
        path=args.output/'images'/name
        drive_id=parse_qs(urlparse(spec['file_url']).query)['id'][0]
        url='https://drive.usercontent.google.com/download'
        result=dict(image_id=int(row['image_number']),image_path=str(path.resolve()),
            age_interval=list(map(int,row['age_group'].split('-'))),
            age_label_kind='Appen human apparent-age group',
            age_confidence=float(row['age_group_confidence']),
            gender_annotation=row['gender'],gender_confidence=float(row['gender_confidence']),
            annotation=row,source_face_short_edge=edge,ffhq_category=record['category'],
            source_metadata=record['metadata'],source_image_spec=spec,
            identity_group=None,chronological_age=None,
            generator_pretraining_overlap='FFHQ was used by the base generator; not an independent test set',
            allowed_use='Noncommercial research, retaining per-image attribution and dataset BY-NC-SA terms',
            role='Candidate training data audit only; no train/eval split or weight updates')
        started=time.monotonic()
        try:
            if path.exists():data=path.read_bytes()
            elif mirrors:
                matches=[(archive,name_in_zip) for archive in mirrors for name_in_zip in archive.entries
                         if Path(name_in_zip).name==name]
                if len(matches)!=1:raise RuntimeError('Image missing or ambiguous in mirror ZIP index')
                archive,name_in_zip=matches[0];data=archive.read(name_in_zip)
                result['delivery']=dict(kind='independently verified mirror ZIP member',url=archive.url,member=name_in_zip)
            else:
                response=requests.get(url,params={'id':drive_id,'export':'download','confirm':'t'},timeout=(8,25))
                response.raise_for_status();data=response.content
            if len(data)!=spec['file_size'] or hashlib.md5(data).hexdigest()!=spec['file_md5']:
                raise RuntimeError('Publisher file size/MD5 mismatch')
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                if im.format!='PNG' or list(im.size)!=spec['pixel_size'] or im.mode!='RGB':
                    raise RuntimeError('Unexpected native PNG format/size/mode')
                if hashlib.md5(im.tobytes()).hexdigest()!=spec['pixel_md5']:
                    raise RuntimeError('Publisher pixel MD5 mismatch')
            path.write_bytes(data)
            result.update(status='verified',sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),
                          native_size=spec['pixel_size'],publisher_file_and_pixel_md5_verified=True)
        except Exception as e:
            result.update(status='failed',error=type(e).__name__,detail=str(e)[:200])
        result['download_seconds']=time.monotonic()-started
        print(json.dumps({k:result[k] for k in ['image_id','age_interval','status','download_seconds'] if k in result}),flush=True)
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(download,selected))
    with (args.output/'manifest.jsonl').open('w') as stream:
        for row in results:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    report=dict(seed=20261113,selected=len(results),verified=sum(r['status']=='verified' for r in results),
                failed=sum(r['status']=='failed' for r in results),
                metadata_sha256=digest(args.metadata),labels_sha256=digest(args.labels),
                training_performed=False,training_ready=False,
                limitations=['Age intervals are not verified birth ages','No identity labels or Asian subgroup labels',
                             'Base generator FFHQ pretraining overlap','Small convenience audit, not a benchmark'])
    report['mirror_transfer']=[archive.stats() for archive in mirrors]
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if report['failed']:raise SystemExit(1)


if __name__=='__main__':main()

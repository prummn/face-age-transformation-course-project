#!/usr/bin/env python3
"""Build new manifests only; never mutate the original dataset or split.

Perceptual-hash groups are conservative near-duplicate candidates. If any group
intersects the old eval set, all its train members are quarantined. The old eval
set remains a frozen regression set, not a newly independent blind test set.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import pandas as pd


def digest(path):
    hasher=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):
            hasher.update(block)
    return hasher.hexdigest()


def frozen_write(path, text):
    if path.exists():
        if path.read_text()!=text:
            raise RuntimeError(f'Refusing to change frozen artifact: {path}. Choose a NEW experiment output directory.')
    else:
        path.write_text(text)


def flickr_id(url):
    from urllib.parse import urlsplit
    parts=urlsplit(str(url))
    if 'flickr' not in (parts.hostname or ''):
        return None
    value=Path(parts.path).name.split('_')[0]
    return value if value.isdigit() else None


class Groups:
    def __init__(self,keys):
        self.parent={key:key for key in keys}
    def find(self,key):
        while self.parent[key]!=key:
            self.parent[key]=self.parent[self.parent[key]]
            key=self.parent[key]
        return key
    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:
            self.parent[max(a,b)]=min(a,b)


def make_groups(rows,features,threshold=6):
    groups=Groups([r['key'] for r in rows])
    exact={}
    for row in rows:
        for kind,value in [('raw_dhash',row.get('dhash')),('flickr_photo',flickr_id(row['url']))]:
            if not value:
                continue
            signature=(kind,value)
            if signature in exact:
                groups.join(row['key'],exact[signature])
            else:
                exact[signature]=row['key']
    # For <=6 differing bits across four 16-bit chunks, at least one chunk
    # differs in <=1 bit. This index finds all such pairs without O(N^2) scans.
    buckets=[defaultdict(list) for _ in range(4)]
    hashes={}
    edges=[]
    for row in rows:
        key=row['key']
        if key not in features:
            continue
        value=int(features[key]['phash64'],16)
        candidates=set()
        chunks=[(value>>(i*16))&65535 for i in range(4)]
        for i,chunk in enumerate(chunks):
            for neighbor in [chunk]+[chunk^(1<<bit) for bit in range(16)]:
                candidates.update(buckets[i].get(neighbor,[]))
        for prior in sorted(candidates):
            distance=(value^hashes[prior]).bit_count()
            if distance<=threshold:
                groups.join(key,prior)
                edges.append({'a':prior,'b':key,'phash_hamming':distance})
        hashes[key]=value
        for i,chunk in enumerate(chunks):
            buckets[i][chunk].append(key)
    raw_buckets=[defaultdict(list) for _ in range(8)]
    raw_hashes={}
    for row in rows:
        key=row['key']; value=int(row['dhash'],16)
        chunks=[(value>>(i*32))&4294967295 for i in range(8)]
        candidates=set()
        for i,chunk in enumerate(chunks):
            for neighbor in [chunk]+[chunk^(1<<bit) for bit in range(32)]:
                candidates.update(raw_buckets[i].get(neighbor,[]))
        for prior in sorted(candidates):
            distance=(value^raw_hashes[prior]).bit_count()
            if distance<=12:
                groups.join(key,prior)
                edges.append({'a':prior,'b':key,'raw_dhash_hamming':distance})
        raw_hashes[key]=value
        for i,chunk in enumerate(chunks): raw_buckets[i][chunk].append(key)
    return groups,edges


def main(args):
    base,output=Path(args.base).resolve(),Path(args.output).resolve()
    if output==base or base in output.parents:
        raise ValueError("Output must be outside the original face-age project; the source is strictly read-only.")
    output.mkdir(parents=True,exist_ok=True)
    frame=pd.read_parquet(base/'data/index.parquet')
    rows=frame.to_dict('records')
    features=json.loads(Path(args.features).read_text())
    groups,edges=make_groups(rows,features)
    members=defaultdict(list)
    for row in rows:
        members[groups.find(row['key'])].append(row)
    cross=[{'group':gid,'train':[r['key'] for r in items if r['split']=='train'],'eval':[r['key'] for r in items if r['split']=='eval']} for gid,items in sorted(members.items()) if len({r['split'] for r in items})>1]
    eval_groups={groups.find(row['key']) for row in rows if row['split']=='eval'}
    rejected=Counter()
    train=[]
    evaluation=[]
    for row in sorted(rows,key=lambda r:r['key']):
        key=row['key']; group=groups.find(key)
        path=base/'data/aligned'/row['split']/(key+'.png')
        if key not in features or not path.exists():
            rejected['missing_'+row['split']]+=1
            continue
        common={'key':key,'image_path':str(path),'duplicate_candidate_group':group,'pseudo_age':round(float(row['age_exp']),5),'pseudo_age_source':'FairFace_classification_expectation_NOT_ground_truth','source_face_short_edge':round(float(row['fw']),4),'aligned_laplacian_variance':round(float(features[key]['blur_laplacian']),4)}
        if row['split']=='eval':
            # Do not prune old evaluation according to output quality; freeze all
            # surviving original eval photos, including hard/low-resolution cases.
            evaluation.append({**common,'split':'frozen_regression_eval','sha256':digest(path)})
            continue
        if group in eval_groups:
            rejected['near_duplicate_candidate_of_eval']+=1
            continue
        if float(row['fw'])<args.min_face_px:
            rejected['source_face_too_small']+=1
            continue
        if float(features[key]['blur_laplacian'])<args.min_laplacian:
            rejected['aligned_blur_below_gate']+=1
            continue
        train.append({**common,'split':'train'})
    # Keep one deterministic best-quality member from each remaining candidate
    # group; this does not purport to remove different photos of the same person.
    by_group=defaultdict(list)
    for row in train:
        by_group[row['duplicate_candidate_group']].append(row)
    train=[]
    for gid,items in by_group.items():
        items.sort(key=lambda r:(-r['source_face_short_edge'],-r['aligned_laplacian_variance'],r['key']))
        train.append(items[0])
        rejected['within_train_near_duplicate_candidates']+=len(items)-1
    train.sort(key=lambda r:r['key'])
    for row in train:
        decade=max(20,min(70,int(row['pseudo_age']//10)*10))
        row['pseudo_age_bin']=f'{decade}-{decade+9}'
    counts=Counter(row['pseudo_age_bin'] for row in train)
    for row in train:
        row['sampling_weight']=round(min(args.weight_cap,len(train)/(max(1,len(counts))*counts[row['pseudo_age_bin']])),6)
    if not train or not evaluation:
        raise RuntimeError('No eligible train or eval images; inspect the quality gates.')
    meta={'source_index_sha256':digest(base/'data/index.parquet'),'original_index_count':len(rows),'train_count':len(train),'frozen_eval_count':len(evaluation),'rejected':dict(rejected),'pseudo_age_bins':dict(sorted(counts.items())),'weight_cap':args.weight_cap,'quality_gates':{'source_face_short_edge_ge':args.min_face_px,'aligned_laplacian_variance_ge':args.min_laplacian},'near_duplicate_method':'raw dHash256 Hamming<=12 / same Flickr photo id / aligned pHash64 Hamming<=6 (candidate groups; not identity recognition)','cross_split_candidate_groups':cross,'near_duplicate_candidate_edges':edges,'limitations':['The frozen eval photos have already been used in historical model selection; this is a regression set, NOT a fresh blind test set.','There are no subject identity labels. Different photos of one person can still occur on both sides.','Pseudo-age bins are classifier estimates, not known chronological ages.','Capped resampling repeats existing samples and does not add missing older adults or children.','Low-resolution source photos are excluded, not restored by resizing or super-resolution.']}
    frozen_write(output/'train_manifest.jsonl',''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in train))
    frozen_write(output/'eval_frozen.jsonl',''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in evaluation))
    frozen_write(output/'manifest_report.json',json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({key:meta[key] for key in ['train_count','frozen_eval_count','rejected','pseudo_age_bins','weight_cap']},ensure_ascii=False,indent=2))
    print('MANIFEST_READY '+str(output))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',default='/data/home/1120231322/scow/face-age')
    parser.add_argument('--features',required=True,help='aligned_features.json produced by audit_dataset.py')
    parser.add_argument('--output',required=True,help='NEW experiment directory; existing artifacts are immutable')
    parser.add_argument('--min-face-px',type=float,default=192)
    parser.add_argument('--min-laplacian',type=float,default=40)
    parser.add_argument('--weight-cap',type=float,default=3)
    args=parser.parse_args()
    if args.weight_cap<=0:
        parser.error('weight-cap must be positive')
    main(args)

"""Prepare frozen train arms and held-out evaluation for the HD data ablation."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np
from PIL import Image

from prepare_gan_training import jsonl
from probe_inversion import sha256, write_json
from train_gan_reconstruction import rows


ROOT = Path(__file__).resolve().parents[2]


def bucket(record):
    return '-'.join(map(str, record['age_interval']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    args = parser.parse_args()
    work = args.work.resolve()
    out = work / 'prepared'
    if (out / 'DONE').exists():
        raise FileExistsError('Prepared experiment already frozen')
    known = ROOT / 'deployment/training-runs/20261113-040634/manifests'
    old_train = rows(known / 'train.jsonl')
    old_eval = rows(known / 'eval.jsonl')
    original_external = rows(known / 'external.jsonl')
    external_root = ROOT / 'deployment/external-tests/20261117-appa32/prepared'
    appa = json.loads((external_root / 'inputs.pre-age.json').read_text())
    for row in appa:
        row['identity_group'] = row['similarity_group']
    demo = json.loads((ROOT / 'deployment/demo-hd/20261117-reviewed/inputs.json').read_text())
    collections = ['regional-ffhq', 'annotated-ffhq']
    new_rows = []
    for collection in collections:
        records = rows(work / collection / 'collection-manifest.jsonl')
        if any(r['status'] != 'verified' for r in records):
            raise ValueError('Collection has missing or unverified images')
        new_rows.extend(dict(r, collection=collection) for r in records)
    if len({r['image_id'] for r in new_rows}) != len(new_rows):
        raise ValueError('New cohorts must have distinct FFHQ IDs')
    cv2.setNumThreads(2)
    base = ROOT / 'deployment/model-files/face-age'
    detector_path = base / 'models/yunet/face_detection_yunet_2023mar.onnx'
    sface_path = ROOT / 'deployment/gan-research/face_recognition_sface_2021dec.onnx'
    detector = cv2.FaceDetectorYN.create(str(detector_path), '', (256,256), .6, .3, 100)
    detector.setInputSize((256,256))
    recognizer = cv2.FaceRecognizerSF.create(str(sface_path), '')

    def describe(rgb):
        small = cv2.resize(rgb, (256,256), interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
        _, faces = detector.detect(bgr)
        if faces is None:
            return small, None, 0.0
        face = max(faces, key=lambda f: f[2]*f[3])
        vector = recognizer.feature(recognizer.alignCrop(bgr, face)).ravel()
        vector /= np.linalg.norm(vector) + 1e-12
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)[55:220,45:210]
        return small, vector, float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def rgb_at(path, expected):
        if sha256(path) != expected:
            raise ValueError('Image input digest differs')
        with Image.open(path) as image:
            return np.asarray(image.convert('RGB'))

    out.mkdir(exist_ok=True)
    entries, vectors, rejected = [], [], []

    def add(row, split, vector):
        entries.append((row,split))
        vectors.append(vector if vector is not None else np.zeros(128,np.float32))

    cache_dir = ROOT / 'deployment/training-data/ffhq-adult-v2/prepared/screening-cache'
    for records, split in [(old_train,'old_train'),(old_eval,'old_eval'),(original_external+appa+demo,'protected')]:
        for row in records:
            cached = cache_dir / (row.get('reference_sha256','')+'.npz')
            if cached.is_file() and split != 'protected':
                with np.load(cached,allow_pickle=False) as loaded:
                    vector = loaded['feature']
                    vector = vector if vector.size else None
            else:
                _,vector,_ = describe(rgb_at(row['image_path'],row['sha256']))
            add(row, split, vector)
    for size in [256,512]:
        (out / str(size)).mkdir(exist_ok=True)
    for record in new_rows:
        rgb = rgb_at(record['path'],record['sha256'])
        small,vector,sharpness = describe(rgb)
        split = 'new_train' if record['category'] == 'training' else 'new_eval'
        name = f"ffhq_{record['image_id']:05d}"
        if split == 'new_train' and (vector is None or sharpness < 60):
            rejected.append(dict(name=name, reason='training_quality', sharpness=sharpness));continue
        dest = out / '256' / (name+'.png')
        ref512 = out / '512' / (name+'.png')
        Image.fromarray(small).save(dest)
        Image.fromarray(cv2.resize(rgb,(512,512),interpolation=cv2.INTER_AREA)).save(ref512)
        meta = record['source_metadata']
        account = urlsplit(meta['photo_url']).path.split('/')[2]
        row = dict(name=name,image_id=record['image_id'],image_path=str(dest),sha256=sha256(dest),
                   reference_path=record['path'],reference_sha256=record['sha256'],
                   reference512_path=str(ref512),reference512_sha256=sha256(ref512),
                   source='FFHQ + FFHQ-Aging; '+record['collection'],source_metadata=meta,
                   flickr_account=account,training_authorized=True,official_split=record['category'],
                   source_face_short_edge=record['source_face_short_edge'],face_sharpness=sharpness,
                   age_interval=record['age_interval'],age_confidence=float(record['annotation']['age_group_confidence']),
                   age_interval_kind='publisher apparent-age interval',publisher_gender=record['annotation']['gender'],
                   group_kind='conservative_source_similarity_component',cohort=record['collection'],
                   published_demographic_label=record.get('published_demographic_label'),
                   demographic_label_source=record.get('demographic_label_source'),
                   collection=record['collection'])
        add(row,split,vector)

    parent = list(range(len(entries)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i,j):
        a,b = sorted([find(i),find(j)])
        parent[b] = a

    seen = {}
    for i,(row,split) in enumerate(entries):
        metadata = row.get('source_metadata',{})
        path = urlsplit(metadata.get('photo_url','')).path.split('/')
        account = row.get('flickr_account') or (path[2] if len(path)>2 and path[1]=='photos' else None)
        for key in [('account',account),('file',row.get('reference_sha256',row['sha256'])),('prior_group',row.get('identity_group'))]:
            if not key[1]:continue
            if key in seen:union(i,seen[key])
            else:seen[key]=i
    vectors = np.stack(vectors)
    for i,j in zip(*np.where(np.triu(vectors@vectors.T>=.4,1))):
        union(int(i),int(j))
    component_splits = {}
    for i,(_,split) in enumerate(entries):
        component_splits.setdefault(find(i),set()).add(split)
    control,mixed_add,eval_rows = [],[],[]
    seen_hashes = set()
    for i,(row,split) in enumerate(entries):
        group = find(i)
        kinds = component_splits[group]
        if split=='protected':continue
        if split in ('old_train','new_train') and kinds & {'old_eval','protected','new_eval'}:
            rejected.append(dict(name=row['name'],reason='conservative_group_touches_eval_or_protected'));continue
        if split=='new_eval' and kinds & {'old_eval','protected'}:
            rejected.append(dict(name=row['name'],reason='regional_eval_group_touches_other_eval_or_protected'));continue
        if row['sha256'] in seen_hashes:
            rejected.append(dict(name=row['name'],reason='duplicate_input'));continue
        seen_hashes.add(row['sha256'])
        row = dict(row,identity_group='hd_'+hashlib.sha256(entries[group][0]['name'].encode()).hexdigest()[:16])
        {'old_train':control,'new_train':mixed_add,'old_eval':eval_rows,'new_eval':eval_rows}[split].append(row)
    if len(mixed_add)<100 or sum(r['collection']=='annotated-ffhq' for r in mixed_add)<50:
        raise RuntimeError('Too few HD annotated samples after leakage screening')
    if {bucket(r) for r in control}!={bucket(r) for r in control+mixed_add}:
        raise ValueError('Paired training arms must support the same sampling strata')
    jsonl(out/'train-control.pre-age.jsonl',control)
    jsonl(out/'train-added.pre-age.jsonl',mixed_add)
    jsonl(out/'eval.pre-age.jsonl',eval_rows)
    jsonl(out/'external.jsonl',appa)
    report = dict(control=len(control),added=len(mixed_add),added_by_source=dict(Counter(r['collection'] for r in mixed_add)),
                  added_by_age=dict(Counter(bucket(r) for r in mixed_add)),
                  evaluation=dict(Counter(r['cohort'] for r in eval_rows+appa)),rejected=rejected,
                  external_alignment_failures=json.loads((external_root/'audit.json').read_text()).get('failures',[]),
                  method='Exact image hash, shared Flickr account, previous groups and SFace cosine >= .4 components',
                  grouping_limit='Conservative duplicate screening; publisher verified subject identifiers unavailable',
                  files={p.name:sha256(p) for p in out.glob('*.jsonl')},
                  upstream=dict(base_train=sha256(known/'train.jsonl'),base_eval=sha256(known/'eval.jsonl'),
                                ap=f'{sha256(external_root/"inputs.pre-age.json")}',
                                collection={c:sha256(work/c/'collection-manifest.jsonl') for c in collections}))
    write_json(out/'audit.json',report)
    (out/'DONE').write_text('Fixed CPU preparation and held-out input manifest\n')
    print(json.dumps({key:report[key] for key in ['control','added','added_by_source','added_by_age','evaluation']},ensure_ascii=False))


if __name__ == '__main__':
    main()

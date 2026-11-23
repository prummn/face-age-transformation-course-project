"""Extend the frozen Asian-age study with new grouped HD train and holdout images."""
import argparse
from collections import Counter
import hashlib
from itertools import chain
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


def account(record):
    path = urlsplit(record.get('source_metadata', {}).get('photo_url', '')).path.split('/')
    return record.get('flickr_account') or (path[2] if len(path) > 2 and path[1] == 'photos' else None)


def load_rgb(path, digest):
    if sha256(path) != digest:
        raise ValueError(f'Input digest changed: {path}')
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'))


def prepare(collections, prior, pairs, output):
    if (output/'DONE').exists():
        raise FileExistsError('Dataset already frozen')
    output.mkdir(parents=True, exist_ok=True)
    for key in ('256','512'):
        (output/key).mkdir(exist_ok=True)
    old = {split:rows(prior/'manifests'/f'{split}.jsonl') for split in ('train','eval','external')}
    corpus = [dict(r, collection=folder.name) for folder in collections for r in rows(folder/'collection-manifest.jsonl')]
    if any(r['status'] != 'verified' for r in corpus):
        raise ValueError('Collection contains unverified assets')
    if len({r['image_id'] for r in corpus}) != len(corpus):
        raise ValueError('Repeated FFHQ source IDs')
    demo = json.loads((ROOT/'deployment/demo-hd/20261117-reviewed/inputs.json').read_text())
    longitudinal = json.loads(pairs.read_text())
    pair_photos = []
    for pair in longitudinal:
        for side in ('early','late'):
            r = pair[side]
            pair_photos.append(dict(name=r['name'], image_path=r['aligned'], sha256=r['aligned_sha256']))
    diagnostics = rows(ROOT/'deployment/training-runs/20261113-040634/manifests/external.jsonl')
    cv2.setNumThreads(2)
    detector = cv2.FaceDetectorYN.create(str(ROOT/'deployment/model-files/face-age/models/yunet/face_detection_yunet_2023mar.onnx'),'',(256,256),.6,.3,100)
    sface = cv2.FaceRecognizerSF.create(str(ROOT/'deployment/gan-research/face_recognition_sface_2021dec.onnx'),'')
    entries, embeddings, rejected = [], [], []

    def add(record, split, bgr):
        _, faces = detector.detect(bgr)
        if faces is None:
            embedding = None
        else:
            face = max(faces,key=lambda f: f[2]*f[3])
            embedding = sface.feature(sface.alignCrop(bgr,face)).ravel()
            embedding /= np.linalg.norm(embedding)+1e-12
        entries.append((record, split))
        embeddings.append(embedding if embedding is not None else np.zeros(128,np.float32))
        return embedding is not None

    for split, records in chain(old.items(), [('protected', demo + pair_photos + diagnostics)]):
        for r in records:
            rgb = load_rgb(r['image_path'], r['sha256'])
            bgr = cv2.cvtColor(cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2BGR)
            add(r,split,bgr)
    print('EXISTING_INPUTS',len(entries),flush=True)
    for index, r in enumerate(corpus):
        name = f"ffhq_{r['image_id']:05d}"
        rgb = load_rgb(r['path'], r['sha256'])
        small = cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(small,cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)[55:220,45:210]
        sharpness = float(cv2.Laplacian(gray,cv2.CV_64F).var())
        dest = output/'256'/f'{name}.png'
        ref = output/'512'/f'{name}.png'
        record = dict(name=name,image_id=r['image_id'],image_path=str(dest),
            reference_path=r['path'],reference_sha256=r['sha256'],reference512_path=str(ref),
            source='FFHQ and FFHQ-Aging with published visual annotations',source_metadata=r['source_metadata'],
            training_authorized=True,official_split=r['category'],collection=r['collection'],
            cohort='new-hd-holdout',age_interval=r['age_interval'],age_confidence=float(r['annotation']['age_group_confidence']),
            publisher_gender=r['annotation']['gender'],published_demographic_label=r.get('published_demographic_label'),
            source_face_short_edge=r['source_face_short_edge'],face_sharpness=sharpness,
            group_kind='conservative_source_similarity_component')
        split = 'new_train' if r['category'] == 'training' else 'new_eval'
        found = add(record,split,bgr)
        if not found or sharpness < 60:
            entries.pop()
            embeddings.pop()
            rejected.append(dict(name=name,reason='native-face-screen',sharpness=sharpness,detected=found))
        else:
            Image.fromarray(small).save(dest,compress_level=2)
            Image.fromarray(cv2.resize(rgb,(512,512),interpolation=cv2.INTER_AREA)).save(ref,compress_level=2)
            record.update(sha256=sha256(dest),reference512_sha256=sha256(ref))
        if (index+1)%100 == 0:
            print('CPU_SCREENED',index+1,len(corpus),flush=True)

    parent = list(range(len(entries)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(i, j):
        first, second = sorted((find(i),find(j)))
        parent[second] = first

    seen = {}
    for i,(row,_) in enumerate(entries):
        keys = [('account',account(row)),('group',row.get('identity_group')),
                ('pixels',row['sha256']),('source',row.get('reference_sha256',row['sha256']))]
        for field,value in keys:
            if value:
                key = (field,value)
                if key in seen:
                    join(i,seen[key])
                else:
                    seen[key] = i
    features = np.stack(embeddings)
    for first in range(0,len(entries),256):
        similarity = features[first:first+256] @ features.T
        for local,j in zip(*np.where(similarity >= .4)):
            if first+local < j:
                join(first+int(local),int(j))
    groups = {}
    for i,(_,split) in enumerate(entries):
        groups.setdefault(find(i),set()).add(split)
    blocked = {'eval','external','protected'}
    training, holdout, used_new_eval, used_train = [], [], set(), set()
    for i,(row,split) in enumerate(entries):
        group = find(i)
        grouped = groups[group]
        identifier = 'hd_'+hashlib.sha256(entries[group][0]['name'].encode()).hexdigest()[:16]
        if split in ('train','new_train'):
            if grouped & blocked or ('new_eval' in grouped and 'train' not in grouped):
                rejected.append(dict(name=row['name'],reason='group-touches-holdout'))
            elif row['sha256'] not in used_train:
                training.append(dict(row,identity_group=identifier))
                used_train.add(row['sha256'])
        elif split in ('eval','new_eval'):
            if split == 'new_eval' and (grouped & (blocked|{'train'}) or group in used_new_eval):
                rejected.append(dict(name=row['name'],reason='new-holdout-not-independent-of-other-eval-or-old-train'))
            else:
                holdout.append(dict(row,identity_group=identifier))
                if split == 'new_eval':
                    used_new_eval.add(group)
    collection_names = {p.name for p in collections}
    additions = [r for r in training if r.get('collection') in collection_names]
    fresh_holdout = [r for r in holdout if r['cohort'] == 'new-hd-holdout']
    if len(additions) < 350 or len(fresh_holdout) < 20:
        raise RuntimeError(f'Too few new train or holdout photos: {len(additions)}, {len(fresh_holdout)}')
    jsonl(output/'train.pre-age.jsonl',training)
    jsonl(output/'eval.pre-age.jsonl',holdout)
    jsonl(output/'external.jsonl',old['external'])
    write_json(output/'longitudinal.json',longitudinal)
    report = dict(train=len(training), added=len(additions), added_by_age=Counter(str(r['age_interval']) for r in additions),
        added_by_source=Counter(r['collection'] for r in additions),
        added_by_gender=Counter(r['publisher_gender'] for r in additions),
        train_by_age=Counter(str(r['age_interval']) for r in training), evaluation=Counter(r['cohort'] for r in holdout+old['external']),
        external_alignment_failures=json.loads((prior/'data-audit.json').read_text())['external_alignment_failures'],
        rejected=rejected, files={p.name:sha256(p) for p in output.glob('*.json*')},
        grouping='Hash, photographer account, previous groups, and SFace cosine ≥.4; no published verified subject splits')
    write_json(output/'audit.json',report)
    (output/'DONE').write_text('Grouped CPU preparation complete\n')
    print('DATA_READY',json.dumps({k:v for k,v in report.items() if k not in ('files','rejected')},ensure_ascii=False),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('collection',type=Path)
    parser.add_argument('prior',type=Path)
    parser.add_argument('pairs',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--additional-collection',type=Path,action='append',default=[])
    args = parser.parse_args()
    prepare([p.resolve() for p in [args.collection,*args.additional_collection]],args.prior.resolve(),args.pairs.resolve(),args.output.resolve())


if __name__ == '__main__':
    main()

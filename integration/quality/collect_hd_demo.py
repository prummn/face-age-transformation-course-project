"""Collect source-quality-screened FFHQ demonstration portraits, before inference.

No generated result is used for selection. Apparent age/gender are publisher
annotations; luminance diversity is not an ethnicity label. No training occurs.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
from http_zip import HTTPZip
from probe_inversion import sha256, write_json

ROOT = Path(__file__).resolve().parents[2]
GROUPS = ['7-9', '15-19', '20-29', '40-49', '50-69', '70-120']


def quality(path):
    rgb = np.asarray(Image.open(path).convert('RGB'))
    gray = cv2.cvtColor(cv2.resize(rgb, (512, 512)), cv2.COLOR_RGB2GRAY)[150:380, 150:365]
    # Two cheek windows only; used to spread source appearances, not to assign race.
    lab = cv2.cvtColor(cv2.resize(rgb, (256, 256)).astype(np.float32)/255, cv2.COLOR_RGB2LAB)
    cheeks = np.concatenate([lab[135:153, 86:103, 0].ravel(), lab[135:153, 153:170, 0].ravel()])
    return dict(sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                brightness=float(gray.mean()), clipped=float(((gray < 5) | (gray > 250)).mean()),
                cheek_luminance=float(np.median(cheeks)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pool-cache', type=Path, help='Reuse previously hash-verified source pool')
    parser.add_argument('--exclude-source', action='append', default=[], metavar='ID=REASON',
                        help='Document source-only visual rejection before inference')
    parser.add_argument('--replace-source', action='append', default=[], metavar='OLD=NEW=REASON',
                        help='Source-only visual review; replacement must meet the same hard criteria and stratum')
    a = parser.parse_args(); out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out/'inputs.json').exists():
        raise RuntimeError('Selection already frozen; use a new output directory to revise it')
    source = ROOT/'deployment/data-research/20261113'
    meta_path = source/'ffhq-dataset-v2.json'
    assert hashlib.md5(meta_path.read_bytes()).hexdigest() == '425ae20f06a4da1d4dc0f46d40ba5fd6'
    labels_path = source/'ffhq_aging_labels.csv'
    assert sha256(labels_path) == 'b99585b0ab9720a0ea55b71d677126b4a1fb1c6b7e362a48e6f00ac835cabc04'
    metadata = json.loads(meta_path.read_text()); labels = list(csv.DictReader(labels_path.open()))
    local = ROOT/'deployment/training-data/ffhq-adult-v2/native1024'
    prepared = local.parent/'prepared'
    prior_train = {json.loads(l)['image_id'] for l in (prepared/'train.pre-age.jsonl').read_text().splitlines()}
    excluded = {int(s.split('=',1)[0]):s.split('=',1)[1] for s in a.exclude_source}
    candidates = []
    for row in labels:
        if row['age_group'] not in GROUPS: continue
        item = metadata[row['image_number']]; rect = item['in_the_wild']['face_rect']
        edge = min(rect[2]-rect[0], rect[3]-rect[1])
        if (edge < 700 or float(row['age_group_confidence']) < .65 or row['glasses'] != 'None'
            or max(abs(float(row[k])) for k in ['head_yaw', 'head_pitch', 'head_roll']) > 10
            or max(float(row['left_eye_occluded']), float(row['right_eye_occluded'])) >= 5): continue
        number = int(row['image_number'])
        # Avoid the recent course fine-tuning manifest. FFHQ still overlaps GAN pretraining.
        if number in prior_train: continue
        candidates.append(dict(image_id=number, age_group=row['age_group'], publisher_gender=row['gender'],
            source_face_short_edge=edge, annotation=row, source_metadata=item['metadata'],
            image_spec=item['image'], official_split=item['category']))
    pool = []; rng = random.Random(20261117)
    for age in GROUPS:
        for gender in ['female', 'male']:
            rows = [r for r in candidates if r['age_group'] == age and r['publisher_gender'] == gender]
            cached = [r for r in rows if (local/f"{r['image_id']:05d}.png").exists()]
            if len(cached) >= 6: pool.extend(cached)
            else:
                # Frozen random sample among well-sized frontal photos; never result-based.
                rows.sort(key=lambda r: r['image_id']); rng.shuffle(rows)
                pool.extend(rows[:12])
    write_json(out/'pool-plan.json', dict(seed=20261117, samples=pool,
        metadata_sha256=sha256(meta_path), labels_sha256=sha256(labels_path)))
    cache = a.pool_cache.resolve() if a.pool_cache else out/'pool'; cache.mkdir(parents=True,exist_ok=True)
    archives = []
    if any(not (local/f"{r['image_id']:05d}.png").exists() and not (cache/f"{r['image_id']:05d}.png").exists() for r in pool):
        archives = [HTTPZip.from_index(source/f'ffhq_zip{i}-index.json', max_transfer=100*1024**2) for i in [1, 2]]
        for archive in archives: archive.resolve_download()
    lookup = {Path(n).name: (z, n) for z in archives for n in z.entries if n.endswith('.png')}
    def fetch(row):
        name = f"{row['image_id']:05d}.png"; path = cache/name; existing = local/name
        if path.exists(): data = path.read_bytes()
        elif existing.exists(): data = existing.read_bytes()
        else:
            for attempt in range(3):
                try:
                    data = lookup[name][0].read(lookup[name][1]); break
                except Exception:
                    if attempt == 2: raise
                    time.sleep(1)
        spec = row['image_spec']
        assert len(data) == spec['file_size'] and hashlib.md5(data).hexdigest() == spec['file_md5']
        with Image.open(io.BytesIO(data)) as im:
            im.load(); assert im.size == (1024, 1024) and im.mode == 'RGB' and im.format == 'PNG'
            assert hashlib.md5(im.tobytes()).hexdigest() == spec['pixel_md5']
        if not path.exists(): path.write_bytes(data)
        return dict(row, image_path=str(path), sha256=hashlib.sha256(data).hexdigest(), quality=quality(path))
    with ThreadPoolExecutor(max_workers=4) as executor:
        verified = list(executor.map(fetch, pool))
    write_json(out/'pool-manifest.json', verified)
    selected = []
    for age in GROUPS:
        for gender in ['female', 'male']:
            rows = [r for r in verified if r['image_id'] not in excluded and r['age_group'] == age and r['publisher_gender'] == gender
                    and r['quality']['sharpness'] >= 140 and 60 < r['quality']['brightness'] < 210
                    and r['quality']['clipped'] < .06]
            if len(rows) < 2: raise RuntimeError(f'Insufficient sharp sources: {age}/{gender}; {len(rows)}')
            # Choose from the sharpest half, then sample a darker/lighter source.
            rows.sort(key=lambda r: (-r['quality']['sharpness'], r['image_id']))
            rows = rows[:max(2, (len(rows)+1)//2)]
            rows.sort(key=lambda r: (r['quality']['cheek_luminance'], r['image_id']))
            selected.extend([rows[0], rows[-1]])
    replacements = []
    for spec in a.replace_source:
        old,new,reason=spec.split('=',2);old=int(old);new=int(new)
        pos=next(i for i,r in enumerate(selected) if r['image_id']==old)
        replacement=next(r for r in verified if r['image_id']==new)
        assert new not in {r['image_id'] for r in selected} and new not in excluded
        assert all(replacement[k]==selected[pos][k] for k in ['age_group','publisher_gender'])
        q=replacement['quality'];assert q['sharpness']>=140 and 60<q['brightness']<210 and q['clipped']<.06
        selected[pos]=replacement;replacements.append(dict(old=old,new=new,reason=reason))
    originals = out/'originals'; originals.mkdir(exist_ok=True)
    for row in selected:
        row['name'] = f"ffhq_{row['image_id']:05d}"
        dest = originals/(row['name']+'.png'); shutil.copy2(row['image_path'], dest)
        row.update(image_path=str(dest), age_label_kind='Appen human apparent-age range, not chronological age',
                   race_label=None, selection_kind='Source-only HD demonstration, not independent benchmark',
                   recent_finetuning_manifest_overlap=False)
    write_json(out/'inputs.json', selected)
    protocol = dict(samples=len(selected), groups=GROUPS, each_gender_per_group=2, selection_uses_generated_results=False,
        source_visual_exclusions=excluded,
        source_visual_replacements=replacements,
        source_edge_min=700, source_face_laplacian_min=140, pose_abs_max=10, native_size=[1024,1024],
        by_age=dict(Counter(r['age_group'] for r in selected)), by_gender=dict(Counter(r['publisher_gender'] for r in selected)),
        inputs_sha256=sha256(out/'inputs.json'), transfers=[z.stats() for z in archives],
        limitations=['No reliable ethnicity labels; cheek luminance also depends on lighting.',
                     'FFHQ was GAN pretraining data; demo quality is not blind generalization evidence.',
                     'Children/teens exceed adult-dominant course training distribution; no chronological accuracy claim.'])
    write_json(out/'selection.json', protocol)
    sheet = Image.new('RGB', (6*240, 4*270), '#eeeeee'); draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(selected):
        x, y = i%6*240, i//6*270
        sheet.paste(Image.open(row['image_path']).resize((240,240)), (x,y))
        draw.text((x+4,y+243), f"{row['name']} {row['age_group']} {row['publisher_gender']}", fill='black')
    sheet.save(out/'source-contact.jpg', quality=92)
    (out/'DONE').write_text('Verified HD source selection frozen before model inference.\n')
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()

"""Collect verified HD FFHQ candidates using published metadata or labels."""

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

from collect_ffhq_data_pilot import digest
from http_zip import HTTPZip
from probe_inversion import sha256, write_json


ROOT = Path(__file__).resolve().parents[2]
COUNTRIES = {
    'china', 'hong kong', 'macao', 'macau', 'taiwan', 'japan', 'south korea', 'korea',
    'mongolia', 'vietnam', 'thailand', 'singapore', 'malaysia', 'indonesia',
    'philippines', 'cambodia', 'laos', 'myanmar', 'brunei', 'timor-leste',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--demographics', type=Path, help='Author-published FFHQ_Demographics.csv')
    parser.add_argument('--exclude-plan', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    source = ROOT / 'deployment/data-research/20261113'
    original = ROOT / 'deployment/training-data/ffhq-adult-v2'
    metadata_file = source / 'ffhq-dataset-v2.json'
    labels = source / 'ffhq_aging_labels.csv'
    if digest(metadata_file, 'md5') != '425ae20f06a4da1d4dc0f46d40ba5fd6':
        raise ValueError('Official FFHQ metadata digest differs')
    if sha256(labels) != 'b99585b0ab9720a0ea55b71d677126b4a1fb1c6b7e362a48e6f00ac835cabc04':
        raise ValueError('FFHQ-Aging CSV digest differs')
    metadata = json.loads(metadata_file.read_text())
    exclude = {r['image_id'] for r in json.loads((original / 'collection-plan.json').read_text())['samples']}
    demo = ROOT / 'deployment/demo-hd/20261117-reviewed/inputs.json'
    exclude.update(r['image_id'] for r in json.loads(demo.read_text()))
    if args.exclude_plan:
        exclude.update(r['image_id'] for r in json.loads(args.exclude_plan.read_text())['samples'])
    demographic_asian = set()
    if args.demographics:
        with args.demographics.open(encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            columns = {c.strip().lower(): c for c in reader.fieldnames}
            label_column = columns.get('ethnicity') or columns['ethnic']
            for row in reader:
                if row[label_column].strip().casefold() == 'asian':
                    demographic_asian.add(int(Path(row[columns['file']].strip()).stem))
    groups = {}
    with labels.open() as f:
        for row in csv.DictReader(f):
            number = int(row['image_number'])
            low, high = map(int, row['age_group'].split('-'))
            item = metadata[row['image_number']]
            info = item['metadata']
            rect = item['in_the_wild']['face_rect']
            edge = min(rect[2] - rect[0], rect[3] - rect[1])
            source_match = (number in demographic_asian if args.demographics
                            else info['country'].strip().casefold() in COUNTRIES)
            if (number in exclude or low < 20 or edge < 512
                or not source_match
                or float(row['age_group_confidence']) < .6
                or abs(float(row['head_yaw'])) > 20 or abs(float(row['head_pitch'])) > 15
                or max(float(row['left_eye_occluded']), float(row['right_eye_occluded'])) > 15):
                continue
            record = dict(image_id=number, category=item['category'], age_interval=[low, high],
                annotation=row, source_face_short_edge=edge, source_metadata=info, image_spec=item['image'])
            if args.demographics:
                record['published_demographic_label'] = 'Asian'
                record['demographic_label_source'] = 'Leyva et al., CVPR Workshops 2024; visual annotation'
            groups.setdefault((item['category'], low, row['gender']), []).append(record)
    rng = random.Random(20261123)
    selected = []
    for (split, age, gender), rows in sorted(groups.items()):
        rows.sort(key=lambda r: r['image_id'])
        rng.shuffle(rows)
        cap = (36 if split == 'training' else 8) if args.demographics else (68 if split == 'training' else 16)
        selected.extend(rows[:cap])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'native1024').mkdir(exist_ok=True)
    plan = dict(seed=20261123, scope='Capture-location metadata; no ethnicity inference',
                metadata_sha256=sha256(metadata_file), labels_sha256=sha256(labels), samples=selected)
    if args.demographics:
        plan.update(scope='Published broad Asian visual annotation; no self-identified ethnicity',
                    demographics_sha256=sha256(args.demographics))
    plan_path = out / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Frozen selection changed')
    write_json(plan_path, plan)
    print('SELECTED', len(selected), dict(Counter(r['category'] for r in selected)), flush=True)
    archives = [HTTPZip.from_index(source / f'ffhq_zip{i}-index.json', max_transfer=2 * 1024**3) for i in [1, 2]]
    for archive in archives:
        archive.resolve_download()
    lookup = {Path(name).name: (archive, name) for archive in archives for name in archive.entries if name.endswith('.png')}

    def fetch(row):
        name = f"{row['image_id']:05d}.png"
        dest = out / 'native1024' / name
        spec = row['image_spec']
        for attempt in range(3):
            try:
                archive, member = lookup[name]
                data = dest.read_bytes() if dest.exists() else archive.read(member)
                if len(data) != spec['file_size'] or hashlib.md5(data).hexdigest() != spec['file_md5']:
                    raise ValueError('File MD5/size mismatch')
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                    if (image.size != (1024, 1024) or image.mode != 'RGB' or image.format != 'PNG'
                        or hashlib.md5(image.tobytes()).hexdigest() != spec['pixel_md5']):
                        raise ValueError('Image pixels/format differ from FFHQ metadata')
                if not dest.exists():
                    temp = dest.with_suffix('.tmp')
                    temp.write_bytes(data)
                    temp.replace(dest)
                return dict(row, status='verified', path=str(dest), sha256=hashlib.sha256(data).hexdigest(), archive=archive.url)
            except Exception as error:
                if attempt == 2:
                    return dict(row, status='failed', error=type(error).__name__)
                time.sleep(1)

    results = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        for result in executor.map(fetch, selected):
            results.append(result)
            if len(results) % 25 == 0:
                print('COLLECTED', len(results), 'verified', sum(r['status'] == 'verified' for r in results), flush=True)
    verified = [r for r in results if r['status'] == 'verified']
    report = dict(selected=len(selected), verified=len(verified), failed=len(selected)-len(verified),
                  by_split=dict(Counter(r['category'] for r in verified)),
                  train_by_age=dict(Counter('-'.join(map(str, r['age_interval'])) for r in verified if r['category'] == 'training')),
                  transfers=[archive.stats() for archive in archives])
    write_json(out / 'collection-report.json', report)
    (out / 'collection-manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in results))
    if len(verified) != len(selected):
        raise SystemExit('Verified files are cached. Retry to fill missing members.')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

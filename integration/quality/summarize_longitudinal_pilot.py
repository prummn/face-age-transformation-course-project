"""Check the native paired outputs, then compare them to the two real timepoints."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from probe_inversion import sha256, write_json

ROUTES = ('wplus', 'expanded_encoder', 'calibrated_encoder', 'wplus_calibrated_source')


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--independent-age', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    report = read(run/'report.json')
    pairs = {p['id']:p for p in read(run/'protocol.json')['pairs']}
    originals = {r['name']:r for pair in pairs.values() for r in (pair['early'],pair['late'])}
    inputs = [dict(name=r['name'], image_path=r['aligned'], sha256=r['aligned_sha256'],
                   cohort='real_photo', publisher_age=r['age']) for r in originals.values()]
    cases = {r['pair']+'-'+r['direction']:r for r in report['cases']}
    if len(cases) != 2*len(pairs):
        raise ValueError('Missing a cross-age direction')
    outputs = defaultdict(dict)
    for row in report['outputs']:
        path = run/'results'/row['path']
        with Image.open(path) as im:
            im.load()
            if im.size != (1024,1024) or im.format != 'PNG' or sha256(path) != row['sha256']:
                raise ValueError(f'Invalid native output: {path}')
        outputs[row['case']][row['route']] = (path,row)
        case = cases[row['case']]
        inputs.append(dict(name=path.stem, image_path=str(path),sha256=row['sha256'],
                           cohort=case['tier']+'/'+row['route'], publisher_age=case['target_age']))
    for name,case in cases.items():
        expected = set(ROUTES if case['fairface_source'] != case['calibrated_source'] else ROUTES[:3])
        if set(outputs[name]) != expected:
            raise ValueError(f'Required routes differ for {name}')
    write_json(run/'independent-age-inputs.json',inputs)
    age = {}
    if args.independent_age:
        age = {r['name']:r['prediction'] for r in read(args.independent_age)['results']}
        if age.keys() != {i['name'] for i in inputs}:
            raise ValueError('Independent age records do not cover all source and generated images')
    by_group = defaultdict(list)
    for name,case in cases.items():
        for route,(path,row) in outputs[name].items():
            estimate = age.get(path.stem)
            target_estimate = age.get(case['target'])
            row.update(apparent_age=estimate, target_photo_apparent_age=target_estimate,
                       apparent_age_error=abs(estimate-target_estimate) if estimate is not None and target_estimate is not None else None)
            by_group[case['tier'],route].append(row)
    summary = defaultdict(dict)
    for (tier,route),rows in by_group.items():
        fields = {}
        for field in ('source_cosine','target_cosine','apparent_age_error','seconds'):
            scores = [r[field] for r in rows if r[field] is not None]
            fields[field] = dict(mean=float(np.mean(scores)) if scores else None, n=len(scores))
        summary[tier][route] = fields
    shifts = []
    for name,case in cases.items():
        if 'wplus_calibrated_source' in outputs[name]:
            a,b = [outputs[name][r][1] for r in ('wplus','wplus_calibrated_source')]
            shifts.append(dict(case=name,tier=case['tier'],original=case['fairface_source'],calibrated=case['calibrated_source'],
                source_cosine_delta=b['source_cosine']-a['source_cosine'] if a['source_cosine'] is not None and b['source_cosine'] is not None else None,
                target_cosine_delta=b['target_cosine']-a['target_cosine'] if a['target_cosine'] is not None and b['target_cosine'] is not None else None,
                apparent_age_error_delta=b['apparent_age_error']-a['apparent_age_error'] if a['apparent_age_error'] is not None and b['apparent_age_error'] is not None else None))
    for tier in {r['tier'] for r in cases.values()}:
        chosen = [(name,case) for name,case in cases.items() if case['tier'] == tier]
        tile = 224
        picture = Image.new('RGB',(6*tile,35+len(chosen)*(tile+29)),'white')
        pen = ImageDraw.Draw(picture)
        for i,title in enumerate(('Real source','Real target','W+ baseline','Encoder control','Encoder calibrated','W+ calibrated source')):
            pen.text((i*tile+8,10),title,fill='black')
        for index,(name,case) in enumerate(chosen):
            y = 35 + index*(tile+29)
            tiles = [(Path(originals[case['source']]['aligned']),f'{name}, age {case["source_age"]}'),
                     (Path(originals[case['target']]['aligned']),f'age {case["target_age"]}')]
            for route in ROUTES:
                path,row = outputs[name].get(route,(None,{}))
                score = row.get('target_cosine')
                caption = f'Target SFace {score:.3f}' if score is not None else 'Source estimate unchanged' if path is None else 'No target score'
                tiles.append((path,caption))
            for col,(path,caption) in enumerate(tiles):
                x = col*tile
                if path:
                    with Image.open(path) as im:
                        picture.paste(im.convert('RGB').resize((tile,tile),Image.Resampling.LANCZOS),(x,y))
                pen.text((x+6,y+tile+5),caption,fill='black')
        picture.save(run/f'panel-{tier}.jpg',quality=94)
    write_json(run/'summary.json',dict(pairs=len(pairs),outputs=len(report['outputs']),
        aggregates=summary,source_age_changes=shifts,real_photo_apparent_ages={n:age.get(n) for n in originals},
        cases=report['cases'],per_output=report['outputs']))
    print(json.dumps(dict(outputs=len(report['outputs']),pairs=len(pairs),aggregates=summary),indent=2))


if __name__ == '__main__':
    main()

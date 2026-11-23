"""Compare a calibrated checkpoint to the frozen control with matched photographs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from probe_inversion import sha256, write_json
from summarize_asian_adaptation import mean, paired_ci, stats

BASE = 'expanded_encoder'
TEST = 'calibrated_encoder'
CONDITIONS = ('same', '20', '40', '60', '75')


def load(path):
    return json.loads(path.read_text())


def combine_metrics(before, after):
    if before['records'] != after['records']:
        raise ValueError('Evaluation records changed between arms')
    by_photo = defaultdict(dict)
    for report, route in ((before, BASE), (after, TEST)):
        bins = defaultdict(list)
        for row in report['outputs']:
            if row['route'] == route:
                bins[row['name']].append(row)
        for record in after['records']:
            name = record['name']
            conditions = bins[name]
            if len(conditions) != 5 or {r['condition'] for r in conditions} != set(CONDITIONS):
                raise ValueError(f'Incomplete case {route}/{name}')
            by_photo[name][route] = dict(stats(conditions), conditions={r['condition']:r for r in conditions})
    for routes in by_photo.values():
        for condition in CONDITIONS:
            b, a = (routes[route]['conditions'][condition] for route in (BASE, TEST))
            if (b['target'], b['source_age']) != (a['target'], a['source_age']):
                raise ValueError('Evaluation conditioning differs')
    return by_photo


def independent_age(before, after, group):
    def collect(report, route):
        result = defaultdict(dict)
        for row in report['results']:
            if row['route'] == route and row['name'] in group and row['prediction'] is not None:
                result[row['name']][row['condition']] = row['prediction']
        return result
    a, b = collect(before, BASE), collect(after, TEST)
    common = sorted(n for n in a.keys() & b.keys() if {'20','75'} <= a[n].keys() & b[n].keys())
    span_f = lambda mapping: mean([mapping[n]['75'] - mapping[n]['20'] for n in common])
    return dict(n=len(common), before_span=span_f(a), after_span=span_f(b),
                before_75=mean([a[n]['75'] for n in common]), after_75=mean([b[n]['75'] for n in common]))


def review_sheet(run, control, cases, by_photo):
    tile = 216
    w = 5*tile
    info_height = 38
    canvas = Image.new('RGB', (w, info_height + (tile+28)*len(cases)), '#f5f5f5')
    draw = ImageDraw.Draw(canvas)
    columns = ('Input', 'Control 20', 'Calibrated 20', 'Control 75', 'Calibrated 75')
    for col,label in enumerate(columns):
        draw.text((col*tile+9,12), label, fill='black')
    for i,name in enumerate(cases):
        cases_row = [(run/'evaluation'/name/'reference.png', name)]
        for age in ('20','75'):
            for route, root in [(BASE,control),(TEST,run)]:
                item = by_photo[name][route]['conditions'][age]
                identity = item['sface_cosine']
                label = 'SFace missing' if identity is None else f'SFace {identity:.3f}'
                cases_row.append((root/'evaluation'/(item['native'] or item['preview']), label))
        y = info_height+i*(tile+28)
        for j,(path,label) in enumerate(cases_row):
            with Image.open(path) as im:
                canvas.paste(im.convert('RGB').resize((tile,tile), Image.Resampling.LANCZOS), (j*tile,y))
            draw.text((j*tile+8,y+tile+6),label,fill='black')
    canvas.save(run/'older-review.jpg',quality=94)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    run = parser.parse_args().run
    experiment = load(run/'experiment.json')
    control = Path(experiment['control'])
    for name, digest in experiment['control_sha256'].items():
        if sha256(control/name) != digest:
            raise ValueError(f'Frozen control file differs: {name}')
    training = load(run/TEST/'completed.json')
    if sha256(run/TEST/'encoder.pt') != training['encoder_sha256'] or training['generator_max_abs_update']:
        raise ValueError('Candidate weight verification failed')
    a = load(control/'evaluation/report.json')
    b = load(run/'evaluation/report.json')
    by_photo = combine_metrics(a, b)
    age_a = load(control/'independent-age.json')
    age_b = load(run/'independent-age.json')
    if age_a['model_sha256'] != age_b['model_sha256']:
        raise ValueError('Independent age model differs')
    groups = defaultdict(list)
    for row in b['records']:
        groups[row['cohort']].append(row)
        if row['cohort'] == 'new_appa_asian' and row['publisher_age'] >= 60:
            groups['new_appa_asian/older'].append(row)
    out = {}
    metrics = ('cross_sface','same_sface','same_lpips','same_rgb_mae','color_shift','shared_age_span')
    for cohort, group in sorted(groups.items()):
        names = {r['name'] for r in group}
        out[cohort] = dict(n=len(group), independent_age=independent_age(age_a, age_b, names),
            metrics={route:{metric:mean([by_photo[n][route][metric] for n in names]) for metric in metrics}
                     for route in (BASE, TEST)},
            missing={route:sum(by_photo[n][route]['missing_sface'] for n in names) for route in (BASE, TEST)},
            paired={metric:paired_ci(group,by_photo,BASE,TEST,metric) for metric in metrics})
    cases = sorted(r['name'] for r in groups['new_appa_asian/older'])
    review_sheet(run, control, cases, by_photo)
    wplus = load(control/'against-saved-wplus.json')
    if set(wplus['records']) != {r['name'] for r in groups['new_appa_asian']}:
        raise ValueError('W+ comparison uses different photographs')
    write_json(run/'paired-summary.json',dict(cohorts=out,
        wplus_asian_cross_sface=wplus['routes']['baseline_v1']['cross_sface'],
        wplus_asian_same_sface=wplus['routes']['baseline_v1']['same_sface'],
        output_count=len(b['outputs']), native_count=sum(bool(r['native']) for r in b['outputs']),
        run_seconds=b['total_seconds'],training=training, review_cases=cases))
    print(json.dumps({g:dict(n=out[g]['n'],independent=out[g]['independent_age'],
                   metrics=out[g]['metrics'],cross_sface_delta=out[g]['paired']['cross_sface'])
                   for g in ['new_appa_asian','new_appa_asian/older','annotated-ffhq']},indent=2))


if __name__ == '__main__':
    main()

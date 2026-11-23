"""Evaluate ages independently and check the frozen promotion criteria."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import numpy as np

from probe_inversion import sha256, write_json

BASE = 'wplus_base'
PILOT = 'dated-pairs'
AGES = ('20','40','60','75')
PRIMARY = ('new-hd-holdout','appa-60plus')


def read(path):
    return json.loads(path.read_text())


def average(values):
    scores = [v for v in values if v is not None]
    return statistics.mean(scores) if scores else None


def scores(items, ages):
    ages_out = {(r['name'],r['condition']):ages[r['native']] for r in items}
    cross = [r for r in items if r['condition'] in AGES]
    higher = [r for r in cross if r['condition'] in ('60','75')]
    actual = [r for r in items if r['condition'] == 'actual']
    count = len({r['name'] for r in items})
    names = {r['name'] for r in cross}
    def age_error(items, target):
        errors = []
        for r in items:
            generated, wanted = ages[r['native']], target(r)
            if generated is not None and wanted is not None:
                errors.append(abs(generated-wanted))
        return average(errors),len(errors)
    high_mae,valid = age_error(higher,lambda r:r['target'])
    dated_mae,dated_valid = age_error(actual,lambda r:ages[r['name']+'/actual.png'])
    span = [ages_out[n,'75']-ages_out[n,'20'] for n in names if ages_out[n,'75'] is not None and ages_out[n,'20'] is not None]
    return dict(cases=count, results=len(items), missing_faces=sum(not r['detected'] for r in items),
        average_source_identity=average([r['source_cosine'] for r in cross or actual]),
        real_target_identity=average([r['target_cosine'] for r in actual]),
        high_age_error=high_mae, valid_high_age=valid,
        age_span=average(span), age_span_n=len(span),
        mean_age75=average([ages_out[n,'75'] for n in names]),
        dated_apparent_error=dated_mae,dated_apparent_n=dated_valid,
        face_chroma=average([r['chroma'] for r in cross or actual]))


def gates_for(summary, route, criteria):
    checks = {}
    # Evaluate both cohorts individually before summarizing the high-age error.
    for cohort in PRIMARY:
        a,b = (summary[cohort][r] for r in (BASE,route))
        checks[cohort+'-identity'] = (b['average_source_identity'] is not None and a['average_source_identity'] is not None
            and b['average_source_identity']-a['average_source_identity'] >= -criteria['sface_drop_max'])
        checks[cohort+'-detection'] = b['missing_faces']-a['missing_faces'] <= criteria['detection_increase_max']
        checks[cohort+'-span'] = (b['age_span'] is not None and a['age_span'] is not None
            and b['age_span_n'] >= a['age_span_n'] and b['age_span'] >= a['age_span']*criteria['age_span_ratio_min'])
        checks[cohort+'-valid-high-age'] = b['valid_high_age'] >= a['valid_high_age']
    a,b = (summary['primary-pooled'][r] for r in (BASE,route))
    checks['high-age-mae'] = (b['high_age_error'] is not None and a['high_age_error'] is not None
        and a['high_age_error']-b['high_age_error'] >= criteria['high_age_mae_improvement_min'])
    a,b = (summary[PILOT][r] for r in (BASE,route))
    checks['dated-identity'] = (b['real_target_identity'] is not None and a['real_target_identity'] is not None
        and b['real_target_identity']-a['real_target_identity'] >= -criteria['dated_target_sface_drop_max'])
    checks['dated-apparent-age'] = (b['dated_apparent_error'] is not None and a['dated_apparent_error'] is not None
        and b['dated_apparent_n']==a['dated_apparent_n'] and b['dated_apparent_error']-a['dated_apparent_error'] <= criteria['dated_apparent_error_increase_max'])
    return checks


def paired_high_age(groups, metadata, ages, route):
    """Pair common age scores first, then resample whole conservative groups."""
    control = {(r['name'],r['condition']):r for r in groups['primary-pooled',BASE]}
    estimates = defaultdict(list)
    for cand in groups['primary-pooled',route]:
        if cand['condition'] not in ('60','75'):
            continue
        orig = control[cand['name'],cand['condition']]
        original_age, candidate_age = ages[orig['native']], ages[cand['native']]
        if original_age is not None and candidate_age is not None:
            record = metadata[cand['name']]
            group = record.get('identity_group',record['name'])
            estimates[group].append(abs(candidate_age-cand['target'])-abs(original_age-orig['target']))
    draws = np.array([statistics.mean(values) for values in estimates.values()])
    if not len(draws):
        return dict(groups=0,delta=None,interval95=None)
    rng = np.random.default_rng(20261123)
    samples = rng.choice(draws,size=(5000,len(draws)),replace=True).mean(1)
    return dict(groups=len(draws),delta=float(draws.mean()),interval95=np.quantile(samples,[.025,.975]).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--age-report',type=Path)
    parser.add_argument('--fast-age-report',type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    folder = run/'wplus-evaluation'
    report = read(folder/'report.json')
    inputs = []
    for row in report['records']:
        if row['cohort'] == PILOT:
            image = folder/row['name']/'actual.png'
            inputs.append(dict(name=str(image.relative_to(folder)),image_path=str(image),sha256=sha256(image),cohort='real-pair-reference'))
    for r in report['outputs']:
        inputs.append(dict(name=r['native'],image_path=str(folder/r['native']),sha256=r['sha256'],
                           cohort=r['cohort'],publisher_age=r['target'] if r['condition']!='same' else None))
    if len({r['name'] for r in inputs}) != len(inputs):
        raise ValueError('Repeated output path')
    write_json(run/'independent-age-inputs.json',inputs)
    if args.age_report is None:
        return
    age = {r['name']:r['prediction'] for r in read(args.age_report)['results']}
    if set(age) != {r['name'] for r in inputs}:
        raise ValueError('Independent estimator output coverage differs')
    grouped = defaultdict(list)
    metadata = {r['name']:r for r in report['records']}
    for row in report['outputs']:
        cohort = row['cohort']
        grouped[cohort,row['route']].append(row)
        is_old_appa = cohort == 'new_appa_asian' and metadata[row['name']]['publisher_age'] >= 60
        if is_old_appa:
            grouped['appa-60plus',row['route']].append(row)
        if cohort == PRIMARY[0] or is_old_appa:
            grouped['primary-pooled',row['route']].append(row)
    summary = defaultdict(dict)
    for (cohort,route),items in sorted(grouped.items()):
        summary[cohort][route] = scores(items,age)
    criteria = read(run/'experiment.json')['promotion_gates']
    gates = {r:gates_for(summary,r,criteria) for r in report['routes'] if r != BASE}
    eligible = [r for r,checks in gates.items() if all(checks.values())]
    result = dict(cohorts=summary,gates=gates,numeric_eligible=eligible,independent_ages=age,
        paired_primary_mae={r:paired_high_age(grouped,metadata,age,r) for r in gates},
        cases=len(report['records']),outputs=len(report['outputs']),wplus_seconds=report['total_seconds'],
        caveat='Numerical gates precede native image review. Independent age estimates are outward appearance proxies.')
    if (run/'evaluation/report.json').exists():
        fast = read(run/'evaluation/report.json')
        fast_scores = defaultdict(list)
        fast_inputs = []
        fast_age_scores = defaultdict(dict)
        fast_ages = {r['name']:r['prediction'] for r in read(args.fast_age_report)['results']} if args.fast_age_report else {}
        for row in fast['outputs']:
            if row['condition'] != 'same':
                fast_scores[row['cohort'],row['route']].append(row['sface_cosine'])
            if row['name'] in metadata and row['condition'] in ('20','75'):
                relative = row['native'] or row['preview']
                path = run/'evaluation'/relative
                fast_inputs.append(dict(name=relative,image_path=str(path),sha256=sha256(path),cohort=row['cohort'],publisher_age=row['target']))
                if fast_ages:
                    key = (row['cohort'],row['route'],row['name'])
                    fast_age_scores[key][row['condition']] = fast_ages[relative]
        write_json(run/'fast-age-inputs.json',fast_inputs)
        fast_groups = defaultdict(dict)
        for (cohort,route),values in fast_scores.items():
            selected = [v for (c,r,_),v in fast_age_scores.items() if (c,r)==(cohort,route)]
            age_spans = [v['75']-v['20'] for v in selected if v.get('75') is not None and v.get('20') is not None]
            fast_groups[cohort][route] = dict(cross_sface=average(values),valid=sum(v is not None for v in values),
                independent_age_span=average(age_spans), independent_span_cases=len(age_spans))
        result['feedforward'] = dict(output_count=len(fast['outputs']), metrics=fast_groups)
    write_json(run/'selection-summary.json',result)
    print(json.dumps(dict(cases=result['cases'],outputs=result['outputs'],numeric_eligible=eligible,
                  primary=summary['primary-pooled'],gates=gates),indent=2))


if __name__=='__main__':
    main()

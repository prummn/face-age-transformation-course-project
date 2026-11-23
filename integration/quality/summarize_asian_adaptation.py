"""Compare adult HD data and generator arms on frozen cohorts."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from probe_inversion import write_json


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def stats(conditions):
    cross = [r for r in conditions if r['condition'] != 'same']
    same = next(r for r in conditions if r['condition'] == 'same')
    by_age = {r['condition']:r for r in conditions}
    return dict(cross_sface=mean([r['sface_cosine'] for r in cross]),same_sface=same['sface_cosine'],
        same_lpips=same['lpips_shared'],same_rgb_mae=same['rgb_mae'],
        shared_age_span=by_age['75']['shared_output_age']-by_age['20']['shared_output_age'],
        high_age_proxy=by_age['75']['shared_output_age'],color_shift=mean([r['face_chroma_shift'] for r in cross]),
        missing_sface=sum(r['sface_cosine'] is None for r in conditions),
        detected=sum(r['output_face_detected'] for r in conditions))


def paired_ci(records, by_photo, a, b, metric):
    groups = defaultdict(list)
    for row in records:
        name = row['name']
        # Use common per-target identities when output detection differs.
        val_a = by_photo[name][a][metric]
        val_b = by_photo[name][b][metric]
        if metric == 'cross_sface':
            outputs_a = by_photo[name][a]['conditions']
            outputs_b = by_photo[name][b]['conditions']
            common = [outputs_b[c]['sface_cosine']-outputs_a[c]['sface_cosine'] for c in ['20','40','60','75']
                      if outputs_a[c]['sface_cosine'] is not None and outputs_b[c]['sface_cosine'] is not None]
            delta = mean(common)
        else:
            delta = val_b-val_a if val_a is not None and val_b is not None else None
        if delta is not None:
            groups[row.get('identity_group',name)].append(delta)
    group_means = np.array([np.mean(x) for x in groups.values()])
    if not group_means.size:
        return dict(delta=None, interval95=None, groups=0)
    rng = np.random.default_rng(20261123)
    draws = rng.choice(group_means,(5000,len(group_means)),replace=True).mean(1)
    return dict(delta=float(group_means.mean()), interval95=np.quantile(draws,[.025,.975]).tolist(), groups=len(groups))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    run = parser.parse_args().run
    raw = json.loads((run/'evaluation/report.json').read_text())
    by_photo = defaultdict(lambda:defaultdict(list))
    for output in raw['outputs']:
        by_photo[output['name']][output['route']].append(output)
    for routes in by_photo.values():
        for route,conditions in routes.items():
            if {r['condition'] for r in conditions}!={'same','20','40','60','75'} or len(conditions)!=5:
                raise ValueError('Each pose must have all five conditions')
            routes[route]=dict(stats(conditions),conditions={r['condition']:r for r in conditions})
    cohorts=defaultdict(list)
    for row in raw['records']:
        cohorts[row['cohort']].append(row)
    comparisons=[('control_encoder','expanded_encoder'),('expanded_encoder','expanded_generator'),
                 ('fine_direct','expanded_encoder'),('fine_direct','expanded_generator')]
    aggregates={}
    deltas={}
    for cohort,records in sorted(cohorts.items()):
        aggregates[cohort]={}
        for route in raw['routes']:
            values=[by_photo[row['name']][route] for row in records]
            aggregates[cohort][route]=dict(cases=len(records),**{key:mean([x[key] for x in values])
                for key in values[0] if key not in ['conditions','missing_sface','detected']},
                missing_sface=sum(x['missing_sface'] for x in values),detected=sum(x['detected'] for x in values))
        deltas[cohort]={f'{b}_minus_{a}':{field:paired_ci(records,by_photo,a,b,field)
            for field in ['cross_sface','same_sface','same_lpips','shared_age_span','color_shift']} for a,b in comparisons}
    arms={}
    for arm in ['control_encoder','expanded_encoder','expanded_generator']:
        log=[json.loads(s) for s in (run/arm/'train.jsonl').read_text().splitlines()]
        arms[arm]=dict(json.loads((run/arm/'completed.json').read_text()),
            sampled_by_collection=dict(Counter(c for step in log for c in step['collections'])))
    report=dict(aggregates=aggregates,paired=deltas,training=arms,native_outputs=sum(x['native'] is not None for x in raw['outputs']),
        outputs=len(raw['outputs']),evaluation_seconds=raw['total_seconds'],
        limitations=['SFace uses separate detection and alignment for output and input. Missing scores are enumerated.',
                     'Intervals use paired common-condition scores and bootstrap conservative photo groups.',
                     'Shared FairFace age-span only measures response to target, with teacher bias.',
                     'Direct checkpoint comparison; frozen per-photo W+ constrained/texture are separate serving baselines.'])
    write_json(run/'asian-adaptation-summary.json',report)
    print('ASIAN_ADAPTATION_DONE',len(raw['records']),len(raw['outputs']),flush=True)


if __name__=='__main__':
    main()

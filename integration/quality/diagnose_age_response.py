"""Audit saved SAME-input age trajectories without GPU or new model fitting.

Uses the original-delta paired development regression, not a new blinded test.
MiVOLO estimates appearance; no generated image has chronological ground truth.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
ROUTES = ('wplus_base', 'original_delta', 'original_delta_constrained')
CONDITIONS = ('same', '20', '40', '60', '75')


def mean(items):
    return statistics.mean(items)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=ROOT/'deployment/training-runs/original-delta-20261124-160055')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    run = args.run.resolve()
    evaluation = json.loads((run/'evaluation/report.json').read_text())
    age_report = json.loads((run/'independent-age.json').read_text())
    ages = {row['name']:row['prediction'] for row in age_report['results']}
    if len(ages) != len(age_report['results']):
        raise ValueError('Duplicate independent output scores')
    outputs = {(row['name'],row['route'],row['condition']):row for row in evaluation['outputs']}
    if len(outputs) != len(evaluation['outputs']):
        raise ValueError('Duplicate paired outputs')
    records = []
    for source in evaluation['records']:
        name = source['name']
        routes = {}
        for route in ROUTES:
            predictions = {}
            identity = []
            for condition in CONDITIONS:
                output = outputs[name,route,condition]
                value = ages[output['native']]
                if value is None or not output['detected']:
                    raise ValueError('Missing face/independent score; cannot silently drop case')
                predictions[condition] = value
                if condition != 'same':
                    if output['source_cosine'] is None:
                        raise ValueError('Missing identity score')
                    identity.append(output['source_cosine'])
            routes[route] = dict(ages=predictions,mean_cross_age_identity=mean(identity),
                span60to75=predictions['75']-predictions['60'],
                span20to75=predictions['75']-predictions['20'])
        records.append(dict(name=name,source_reference_age=source['source_age'],
            cohort=source['cohort'],published_demographic_label=source.get('published_demographic_label'),
            routes=routes))

    def describe(rows, route):
        items = [row['routes'][route] for row in rows]
        return dict(cases=len(items),mean_age60=mean(x['ages']['60'] for x in items),
            mean_age75=mean(x['ages']['75'] for x in items),
            mean_span60to75=mean(x['span60to75'] for x in items),
            span60to75_below5=sum(x['span60to75']<5 for x in items),
            reversed60to75=sum(x['span60to75']<0 for x in items),
            age75_below60=sum(x['ages']['75']<60 for x in items),
            mean_cross_age_identity=mean(x['mean_cross_age_identity'] for x in items))

    source_groups = defaultdict(list)
    for row in records:
        age = row['source_reference_age']
        label = 'under40' if age<40 else '40to59' if age<60 else '60plus'
        source_groups[label].append(row)
    groups = {label:dict(cases=len(rows),
        mean_age20=mean(r['routes']['original_delta_constrained']['ages']['20'] for r in rows),
        **{k:v for k,v in describe(rows,'original_delta_constrained').items() if k!='cases'})
        for label,rows in source_groups.items()}
    report = dict(parent_run=str(run),new_inference=False,
        summary={route:describe(records,route) for route in ROUTES},
        groups_by_model_source_age=groups,records=records,
        caveats=['Previously inspected35-image development regression, not a new blinded test.',
            'Age predictions are independent MiVOLO appearance estimates, not chronological ground truth.',
            'Source groups use FairFace pseudo ages and are descriptive; source-age confounding prevents causal claims.',
            'Unconstrained vs constrained routes isolate the complete refinement package, not a single loss term.',
            'Absolute appearance and per-photo age response must be checked in addition to relative guard ratios.'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report['summary'],ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()

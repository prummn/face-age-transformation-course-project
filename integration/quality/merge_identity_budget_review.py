"""Merge disjoint pilot/confirmation artifacts without rerunning inference."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import statistics

from probe_inversion import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pilot', type=Path); parser.add_argument('confirmation', type=Path)
    parser.add_argument('output', type=Path); args = parser.parse_args()
    sources = [args.pilot.resolve(), args.confirmation.resolve()]
    output = args.output.resolve(); (output / 'evaluation').mkdir(parents=True, exist_ok=False)
    reports = [json.loads((run/'evaluation/report.json').read_text()) for run in sources]
    summaries = [json.loads((run/'summary.json').read_text()) for run in sources]
    frozen = [json.loads((run/'frozen-verification.json').read_text()) for run in sources]
    if not all(row['equal'] for row in frozen) or frozen[0]['before'] != frozen[1]['before']:
        raise ValueError('Runs do not use the same frozen network states')
    records = reports[0]['records'] + reports[1]['records']
    if len(records) != 35 or len({r['name'] for r in records}) != 35:
        raise ValueError('Expected disjoint6+29 cases')
    scores = {}; outputs = []
    for run, report, summary in zip(sources, reports, summaries):
        for row in report['records']:
            folder = output/'evaluation'/row['name']; folder.mkdir()
            # Keep gallery assets physically inside evaluation, as required by
            # its containment check. Hard links reuse existing PNG bytes; a
            # directory symlink would resolve outside the merged run.
            os.link(run/'evaluation'/row['name']/'reference.png', folder/'reference.png')
        for row in report['outputs']:
            if sha256(run/'evaluation'/row['native']) != row['sha256']:
                raise ValueError('Output changed since scoring')
            for field in ['native', 'preserved']:
                os.link(run/'evaluation'/row[field], output/'evaluation'/row[field])
        scores.update(summary['independent_ages']); outputs += report['outputs']
    if len(outputs) != 380 or len(scores) != 380: raise ValueError('Incomplete combined design')
    report = dict(records=records, routes=reports[0]['routes'], outputs=outputs,
        timings=[row for r in reports for row in r['timings']], default_model_changed=False,
        review_title='执行身份预算的年龄编辑：完整开发回归',
        selection_note='Six matched soft/hard diagnostics plus29 fixed hard-only confirmations; no fresh holdout consumed.')
    write_json(output/'evaluation/report.json', report)
    grouped = defaultdict(list); lookup = {row['name']: row for row in records}
    diagnostic = {row['name'] for row in reports[0]['records']}
    for row in outputs:
        scope = 'diagnostic6' if row['name'] in diagnostic else 'confirmation29'
        for group in [scope, 'pooled', 'cohort:'+row['cohort']]: grouped[group, row['route']].append(row)
        published = lookup[row['name']].get('published_demographic_label')
        if published == 'Asian' or row['cohort'] == 'new_appa_asian':
            grouped['publisher_Asian', row['route']].append(row)
    def mean(values):
        valid = [v for v in values if v is not None]
        return statistics.mean(valid) if valid else None
    groups = defaultdict(dict)
    baseline = {(o['name'], o['condition']): o for o in outputs if o['route'] == 'accepted_baseline'}
    for (group, route), rows in grouped.items():
        names = {r['name'] for r in rows}; old = [r for r in rows if r['condition'] in ['60', '75']]
        by = {(r['name'], r['condition']): scores[r['native']] for r in rows}
        spans = [by[n, '75']-by[n, '60'] for n in names if by[n, '75'] is not None and by[n, '60'] is not None]
        groups[group][route] = dict(cases=len(names), high_age_identity=mean(r['source_cosine'] for r in old),
            high_age_error=mean(abs(scores[r['native']]-r['target']) for r in old if scores[r['native']] is not None),
            mean_age75=mean(by[n, '75'] for n in names), age60to75_span=mean(spans),
            weak60to75=sum(v < 5 for v in spans),
            age20to75_span=mean(by[n, '75']-by[n, '20'] for n in names if by[n, '75'] is not None and by[n, '20'] is not None),
            missing_faces=sum(not r['detected'] for r in rows), independent_missing=sum(scores[r['native']] is None for r in rows),
            identity_drop_over02=sum(r['source_cosine'] is not None and baseline[r['name'], r['condition']]['source_cosine'] is not None
                and r['source_cosine'] < baseline[r['name'], r['condition']]['source_cosine']-.02 for r in old))
    checks = {scope: summary['checks'] for scope, summary in zip(['diagnostic6', 'confirmation29'], summaries)}
    write_json(output/'summary.json', dict(groups=groups, checks=checks, independent_ages=scores,
        all_numeric_checks_pass=all(all(check.values()) for check in checks.values()), visual_review_pending=True,
        promotion=False, note='Development regression, not an independent population test. Published labels only.'))
    write_json(output/'merged-inputs.json', dict(sources=[str(run) for run in sources],
        reports_sha256={str(run/'evaluation/report.json'): sha256(run/'evaluation/report.json') for run in sources},
        no_gpu_inference=True, common_frozen_state=frozen[0]['before']))
    print('IDENTITY_BUDGET_MERGED', json.dumps(groups['pooled']), flush=True)


if __name__ == '__main__': main()

"""Paired CPU analysis; no age accuracy or automatic promotion claims."""
import argparse
import json
from pathlib import Path
import random
import statistics
from probe_inversion import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    run=p.parse_args().run
    report=json.loads((run/'age-texture/report.json').read_text())
    rows=report['outputs'];routes=report['routes'];summary={}
    for cohort in sorted({r['cohort'] for r in rows}):
        summary[cohort]={}
        names=sorted({r['name'] for r in rows if r['cohort']==cohort})
        indexed={(r['name'],r['route'],r['condition']):r for r in rows if r['cohort']==cohort}
        for route in routes:
            subset=[r for r in rows if r['cohort']==cohort and r['route']==route and r['condition']!='same']
            means={key:statistics.mean(r[key] for r in subset) for key in ['sface_cosine','age_view_spread','face_high_frequency_energy']}
            span=statistics.mean(indexed[n,route,'75']['shared_output_age']-indexed[n,route,'20']['shared_output_age'] for n in names)
            age75=statistics.mean(indexed[n,route,'75']['shared_output_age'] for n in names)
            delta=[statistics.mean(indexed[n,route,c]['sface_cosine']-indexed[n,'baseline_v1',c]['sface_cosine'] for c in ['20','40','60','75']) for n in names]
            rng=random.Random(20261117)
            boot=sorted(statistics.mean(rng.choices(delta,k=len(delta))) for _ in range(10000))
            base_span=statistics.mean(indexed[n,'baseline_v1','75']['shared_output_age']-indexed[n,'baseline_v1','20']['shared_output_age'] for n in names)
            base75=statistics.mean(indexed[n,'baseline_v1','75']['shared_output_age'] for n in names)
            result=dict(cases=len(names),cross_identity=means['sface_cosine'],paired_identity_gain=statistics.mean(delta),
                improved_cases=sum(v>0 for v in delta),case_gains=dict(zip(names,delta)),
                paired_case_bootstrap95=[boot[250],boot[9749]] if len(delta)>1 else None,shared_age_span=span,shared_age75=age75,
                shared_age_span_fraction=span/base_span,shared_age75_change=age75-base75,
                age_view_spread=means['age_view_spread'],face_high_frequency_energy=means['face_high_frequency_energy'],
                numeric_guard_pass=statistics.mean(delta)>=-.01 and span>=.9*base_span and age75>=base75-3,
                status='Exploratory diagnostic; visual review required; not age accuracy or a promotion decision.')
            summary[cohort][route]=result
    outputs={route:[{k:r[k] for k in ['condition','sface_cosine','shared_output_age','age_views','age_view_spread','face_high_frequency_energy']}
                   for r in rows if r['name']=='uploaded' and r['route']==route] for route in routes}
    timing_rows=[r for r in report['timings'] if r.get('condition')!='same']
    baseline={(r['name'],r['condition']):r for r in rows if r['route']=='baseline_v1'}
    failures={}
    for route in routes[1:]:
        changes=[dict(name=r['name'],condition=r['condition'],identity_gain=r['sface_cosine']-baseline[r['name'],r['condition']]['sface_cosine'])
                 for r in rows if r['route']==route and r['condition']!='same']
        failures[route]=dict(improved_targets=sum(r['identity_gain']>0 for r in changes),total_targets=len(changes),
                             worst_targets=sorted(changes,key=lambda r:r['identity_gain'])[:5])
    result=dict(cohorts=summary,uploaded=outputs,target_changes=failures,
        baseline_max_png_difference=max(r.get('baseline_png_max_difference',0) for r in rows),
        timings={route:statistics.mean(r['edit_seconds'] for r in timing_rows if r['route']==route)
                 for route in routes if any(r['route']==route for r in timing_rows)},
        timing_scope='Per-target serving helper, excluding same-age anchor' if any('condition' in r for r in report['timings']) else 'Joint optimization of four age targets; cached source anchor excluded',
        total_seconds=report['total_seconds'])
    write_json(run/'paired-summary.json',result)
    print(json.dumps(dict(cohorts=summary,timings=result['timings'],baseline_max_png_difference=result['baseline_max_png_difference']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()

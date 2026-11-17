"""Assess fixed confirmation guardrails without tuning on the confirmation set."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    run=p.parse_args().run;summary=json.loads((run/'chroma-summary.json').read_text())
    guard=json.loads((run/'experiment.json').read_text())['guardrails']
    s=summary['strata']['review_split:confirmation'];b=s['texture_half'];c=s['texture_chroma']
    pairs=[r for r in summary['pairs'] if r['review_split']=='confirmation']
    deltas=[r['identity_delta'] for r in pairs if r['identity_delta'] is not None]
    measures=dict(cases=b['cases'],mean_identity_delta=c['cross_identity']-b['cross_identity'],
        worst_identity_delta=min(deltas),missing_identity_conditions=len(pairs)-len(deltas),
        mean_chroma_ratio=c['chroma_mae']/b['chroma_mae'],
        shared_age_span_retained=c['shared_age_span']/b['shared_age_span'],
        shared_age75_shift=c['shared_age75']-b['shared_age75'],
        chroma_improved_conditions=sum(r['chroma_delta']<0 for r in pairs),conditions=len(pairs))
    checks=dict(mean_identity=measures['mean_identity_delta']>=-guard['confirmation_mean_identity_drop_max'],
        worst_condition_identity=measures['worst_identity_delta']>=-guard['confirmation_single_condition_drop_max'],
        no_missing_identity=measures['missing_identity_conditions']==0,
        mean_chroma=measures['mean_chroma_ratio']<=guard['confirmation_mean_chroma_ratio_max'],
        shared_age_span=measures['shared_age_span_retained']>=guard['confirmation_shared_age_span_retained_min'],
        shared_age75_shift=abs(measures['shared_age75_shift'])<=guard['confirmation_shared_age75_shift_max'])
    result=dict(measures=measures,checks=checks,numeric_guardrails_pass=all(checks.values()),
        worst_confirmation_identity=sorted(pairs,key=lambda r:r['identity_delta'] if r['identity_delta'] is not None else float('inf'))[:5],
        worst_confirmation_color=sorted(pairs,key=lambda r:-r['chroma_delta'])[:5],
        promotion='No automatic promotion; inspect representative and failed outputs. No retuning on confirmation.')
    (run/'chroma-assessment.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

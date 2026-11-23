"""Fit a monotone apparent-age calibration from training intervals and score frozen inputs."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from probe_inversion import sha256, write_json
from train_gan_reconstruction import rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    run=p.parse_args().run
    folder=run/'manifests'
    training=rows(folder/'train.jsonl')
    heldout=rows(folder/'eval.jsonl')+rows(folder/'external.jsonl')
    if {r['sha256'] for r in training}&{r['sha256'] for r in heldout}:
        raise ValueError('Calibration train and held-out pixels overlap')
    knots=np.array([0.,20.,35.,50.,60.,70.,75.])
    raw=np.array([r['shared_reference_age'] for r in training])
    lo,hi=np.array([r['age_interval'] for r in training]).T
    counts=Counter(tuple(r['age_interval']) for r in training)
    weights=np.array([r['age_confidence']/counts[tuple(r['age_interval'])] for r in training])
    regularizer=.15

    def loss(values):
        estimates=np.interp(raw,knots,values)
        outside=np.maximum(lo-estimates,0)+np.maximum(estimates-hi,0)
        penalty=np.average(outside**2,weights=weights)
        return penalty+regularizer*np.mean((values-knots)**2)

    increasing=np.diff(np.eye(len(knots)),axis=0)
    fit=minimize(loss,knots,method='SLSQP',bounds=Bounds(0,90),
        constraints=[LinearConstraint(increasing,0,np.inf)],options={'ftol':1e-10,'maxiter':500})
    if not fit.success or np.any(np.diff(fit.x)<-1e-6):raise RuntimeError(str(fit.message))
    output=[]
    for r in heldout:
        before=r['shared_reference_age']
        after=float(np.interp(before,knots,fit.x))
        labels={k:r[k] for k in ['age_interval','publisher_age','apparent_age'] if k in r}
        output.append(dict(name=r['name'],cohort=r['cohort'],original=before,calibrated=after,**labels))
    groups=defaultdict(list)
    for r in output:
        groups[r['cohort']].append(r)
        if r.get('publisher_age',0)>=60 or r.get('age_interval',[0])[0]>=70:
            groups[r['cohort']+'/older'].append(r)
    summary={}
    for name,samples in sorted(groups.items()):
        entry=dict(n=len(samples))
        for route in ['original','calibrated']:
            error=[]
            for r in samples:
                if 'age_interval' in r:
                    low,high=r['age_interval']
                    error.append(float(r[route]-np.clip(r[route],low,high)))
                else:error.append(r[route]-r['apparent_age'])
            entry[route]=dict(mae=float(np.mean(np.abs(error))),mean_signed_error=float(np.mean(error)))
        summary[name]=entry
    write_json(run/'source-age-calibration.json',dict(knots=knots.tolist(),calibrated_knots=fit.x.tolist(),
        objective='Confidence-weighted squared distance to apparent-age interval, equal weight by age bin, plus identity-map penalty',
        training=len(training),training_sha256=sha256(folder/'train.jsonl'),regularizer=regularizer,
        heldout=summary,outputs=output,
        note='FFHQ diagnostic is error to the interval, APPA diagnostic is error to mean apparent age. No image generation or model weights affected.'))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()

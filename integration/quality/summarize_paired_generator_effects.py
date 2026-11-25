"""Case-level paired effects; bootstrap intervals describe this development panel."""
import argparse
import json
from pathlib import Path
import random
import statistics

from probe_inversion import write_json


def describe(values):
    if not values:return None
    rng=random.Random(20261125)
    draws=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(5000))
    return dict(cases=len(values),mean=statistics.mean(values),median=statistics.median(values),
                minimum=min(values),maximum=max(values),positive=sum(v>0 for v in values),
                negative=sum(v<0 for v in values),bootstrap95=[draws[125],draws[4874]])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();run=a.run.resolve()
    report=json.loads((run/'evaluation/report.json').read_text())
    ages=json.loads((run/'summary.json').read_text())['independent_ages']
    rows={(r['name'],r['route'],r['condition']):r for r in report['outputs']}
    routes=report['routes'];effects={};summaries={}
    for route in routes[1:]:
        records=[]
        for case in report['records']:
            name=case['name']
            def sample(label,condition):
                row=rows[name,label,condition]
                if condition!='same' and row['target']==case['source_age']:row=rows[name,label,'same']
                return row
            def metrics(label):
                cross=[sample(label,c)['source_cosine'] for c in ['20','40','60','75']]
                first,last=(ages[sample(label,c)['native']] for c in ['20','75'])
                high=[abs(ages[sample(label,c)['native']]-int(c)) for c in ['60','75'] if ages[sample(label,c)['native']] is not None]
                return dict(identity=statistics.mean(cross) if all(v is not None for v in cross) else None,
                            same_identity=sample(label,'same')['source_cosine'],
                            age_span=last-first if first is not None and last is not None else None,
                            high_age_error=statistics.mean(high) if len(high)==2 else None)
            base,candidate=metrics(routes[0]),metrics(route)
            records.append(dict(name=name,cohort=case['cohort'],baseline=base,candidate=candidate,
                                delta={k:candidate[k]-base[k] if candidate[k] is not None and base[k] is not None else None for k in base}))
        effects[route]=records
        summaries[route]={k:describe([r['delta'][k] for r in records if r['delta'][k] is not None]) for k in records[0]['delta']}
    result=dict(bootstrap_unit='Photo / conservative source group, not verified independent identity',bootstrap_samples=5000,
                note='Paired descriptive effects on previously inspected development inputs. Identity higher is better; high-age error lower is better. Intervals do not establish skin realism or population generalization.',
                summaries=summaries,effects=effects)
    write_json(run/'paired-effect-summary.json',result);print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()

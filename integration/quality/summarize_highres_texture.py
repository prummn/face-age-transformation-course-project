"""Independent age/identity guards and native texture review for the paired pilot."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
from probe_inversion import sha256,write_json


def mean(v):
    v=[x for x in v if x is not None];return statistics.mean(v) if v else None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--age-report',type=Path)
    a=p.parse_args();run=a.run.resolve();folder=run/'evaluation';r=json.loads((folder/'report.json').read_text())
    inputs=[dict(name=o['native'],image_path=str(folder/o['native']),sha256=o['sha256'],cohort=o['cohort']) for o in r['outputs']]
    write_json(run/'independent-age-inputs.json',inputs)
    if not a.age_report:return
    ages={x['name']:x['prediction'] for x in json.loads(a.age_report.read_text())['results']}
    if set(ages)!={x['name'] for x in inputs}:raise ValueError('Independent age coverage differs')
    groups=defaultdict(list)
    for o in r['outputs']:
        groups[o['cohort'],o['route']].append(o);groups['pooled',o['route']].append(o)
    result=defaultdict(dict);criteria=json.loads((run/'experiment.json').read_text())['criteria']
    for (cohort,route),items in groups.items():
        cross=[o for o in items if o['condition']!='same'];same=[o for o in items if o['condition']=='same']
        paired={(o['name'],o['condition']):ages[o['native']] for o in items}
        names={o['name'] for o in items}
        spans=[paired[n,'75']-paired[n,'20'] for n in names if paired[n,'75'] is not None and paired[n,'20'] is not None]
        result[cohort][route]=dict(cases=len(names),outputs=len(items),missing_faces=sum(not o['detected'] for o in items),
            identity=mean(o['source_cosine'] for o in cross),same_identity=mean(o['source_cosine'] for o in same),
            high_age_error=mean(abs(ages[o['native']]-o['target']) for o in cross if o['condition'] in ('60','75') and ages[o['native']] is not None),
            age20to75_span=mean(spans),span_valid=len(spans),mean_age75=mean(ages[o['native']] for o in cross if o['condition']=='75'))
    checks={}
    for route in r['routes'][1:]:
        checks[route]={}
        for cohort,values in result.items():
            base,cand=values['wplus_base'],values[route]
            checks[route][cohort+'-identity']=cand['identity'] is not None and cand['identity']>=base['identity']-criteria['identity_drop_max']
            checks[route][cohort+'-same-identity']=cand['same_identity'] is not None and cand['same_identity']>=base['same_identity']-criteria['identity_drop_max']
            checks[route][cohort+'-detection']=cand['missing_faces']<=base['missing_faces']
            checks[route][cohort+'-age-span']=cand['span_valid']>=base['span_valid'] and cand['age20to75_span'] is not None and cand['age20to75_span']>=base['age20to75_span']*criteria['age_span_min_ratio']
    summary=dict(cohorts=result,checks=checks,independent_ages=ages,
        numeric_guard_pass=[route for route,flags in checks.items() if all(flags.values())],
        promotion=False,note='Texture naturalness requires native review. Passing numeric guards does not establish improvement; this small previously reviewed panel is development/regression, not a new blind test.')
    write_json(run/'summary.json',summary);print(json.dumps(dict(pooled=result['pooled'],checks=checks),indent=2))

if __name__=='__main__':main()

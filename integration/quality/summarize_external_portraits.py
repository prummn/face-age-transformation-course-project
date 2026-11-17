"""Summarize paired new-photo results; default review excludes confirmation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import numpy as np
from PIL import Image, ImageDraw


def mean(values):
    values=[float(x) for x in values if x is not None and np.isfinite(x)]
    return statistics.mean(values) if values else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('--split',choices=['diagnostic','all'],default='diagnostic')
    a=p.parse_args();folder=a.run/'external-results'
    report=json.loads((folder/('report.json' if (folder/'report.json').exists() else 'report.partial.json')).read_text())
    outputs=[r for r in report['outputs'] if a.split=='all' or r['review_split']=='diagnostic']
    completed={r['name'] for r in outputs if r['condition']=='75' and r['route']=='texture_half'}
    records=[r for r in report['records'] if r['name'] in completed]
    outputs=[r for r in outputs if r['name'] in completed];index={(r['name'],r['route'],r['condition']):r for r in outputs}
    paired=[]
    for r in records:
        for condition in ['20','40','60','75']:
            b=index[r['name'],'baseline_v1',condition];c=index[r['name'],'texture_half',condition]
            paired.append(dict(name=r['name'],condition=condition,review_split=r['review_split'],
                identity_delta=None if b['sface_cosine'] is None or c['sface_cosine'] is None else c['sface_cosine']-b['sface_cosine'],
                chroma_delta=c['chroma_mae']-b['chroma_mae'],shared_age_delta=c['shared_output_age']-b['shared_output_age']))
    def aggregate(rows):
        names={r['name'] for r in rows};out={}
        for route in report['routes']:
            cross=[o for o in outputs if o['name'] in names and o['route']==route and o['condition']!='same']
            same=[o for o in outputs if o['name'] in names and o['route']==route and o['condition']=='same']
            out[route]=dict(cases=len(names),cross_identity=mean(o['sface_cosine'] for o in cross),
                missing_identity=sum(o['sface_cosine'] is None for o in cross),same_identity=mean(o['sface_cosine'] for o in same),
                chroma_mae=mean(o['chroma_mae'] for o in cross),lpips_shared=mean(o['lpips_shared'] for o in cross),
                shared_age_span=mean(index[n,route,'75']['shared_output_age']-index[n,route,'20']['shared_output_age'] for n in names),
                shared_age75=mean(index[n,route,'75']['shared_output_age'] for n in names))
        # Resample conservative similarity groups, keeping all photos/targets together.
        groups={r['similarity_group'] for r in rows};deltas=[]
        for g in sorted(groups):
            ns={r['name'] for r in rows if r['similarity_group']==g}
            d=mean(x['identity_delta'] for x in paired if x['name'] in ns)
            if d is not None:deltas.append(d)
        ci=None
        if len(deltas)>1:
            rng=np.random.default_rng(20261117);v=np.asarray(deltas)
            ci=np.percentile(v[rng.integers(len(v),size=(5000,len(v)))].mean(1),[2.5,97.5]).tolist()
        out['paired']=dict(similarity_groups=len(groups),group_mean_identity_delta=mean(deltas),group_bootstrap95=ci,
            improved_conditions=sum(x['identity_delta'] is not None and x['identity_delta']>0 for x in paired if x['name'] in names),
            conditions=4*len(names))
        return out
    strata={'all':records}
    for key in ['review_split','cohort','publisher_age_bin']:
        for value in sorted({r[key] for r in records}):strata[key+':'+value]=[r for r in records if r[key]==value]
    strata.update(low_resolution=[r for r in records if r['native_face_short_edge']<128],
        native_face256plus=[r for r in records if r['native_face_short_edge']>=256],
        near_monochrome=[r for r in records if r['source_quality']['chroma_rgb_std']<3],
        dark_source=[r for r in records if r['source_quality']['brightness']<50],
        no_prior_similarity_match=[r for r in records if not r['old_similarity_matches']])
    summary=dict(split=a.split,completed=len(records),counts=dict(Counter(r['cohort'] for r in records)),
        strata={key:aggregate(rows) for key,rows in strata.items() if rows},
        pairs=paired,source_reference=dict(
            shared_vs_publisher_real_mae=mean(abs(r['shared_reference_age']-r['publisher_age']) for r in records),
            shared_vs_apparent_mae=mean(abs(r['shared_reference_age']-r['apparent_age']) for r in records)),
        warning='New photographs, not verified new identities. Small stress sample, not population accuracy. Shared age network is not independent. No confirmation tuning.')
    summary['worst_identity_conditions']=sorted([x for x in paired if x['identity_delta'] is not None],key=lambda x:x['identity_delta'])[:8]
    summary['largest_color_drift']=sorted([o for o in outputs if o['route']=='texture_half' and o['condition']!='same'],key=lambda x:-x['chroma_mae'])[:8]
    dest=a.run/('external-summary.json' if a.split=='all' else 'external-summary.diagnostic.json')
    dest.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    # A few failure/edge cases, never selecting the confirmation set in diagnostic mode.
    names=[]
    picks=summary['worst_identity_conditions'][:2]+summary['largest_color_drift'][:2]
    picks += [dict(name=r['name']) for r in records if r['source_quality']['chroma_rgb_std']<3][:1]
    for row in picks:
        if row['name'] not in names:names.append(row['name'])
    sheets=a.run/'representatives'/a.split;sheets.mkdir(parents=True,exist_ok=True)
    for name in names:
        canvas=Image.new('RGB',(6*220,3*250),'#eef0ec');draw=ImageDraw.Draw(canvas)
        columns=['source','same','20','40','60','75']
        for y,route in enumerate(['reference','baseline_v1','texture_half']):
            for x,condition in enumerate(columns):
                path=folder/name/('reference.png' if route=='reference' or condition=='source' else route+'_'+condition+'.png')
                with Image.open(path) as image:canvas.paste(image.convert('RGB').resize((212,212)),(x*220+4,y*250+4))
                label=route+' '+condition
                if route!='reference' and condition!='source':
                    m=index[name,route,condition];label=route+' '+condition+' id '+str(round(m['sface_cosine'],3) if m['sface_cosine'] is not None else None)
                draw.text((x*220+4,y*250+217),label,fill='black')
        canvas.save(sheets/(name+'.jpg'),quality=92)
    print(json.dumps(dict(path=str(dest),completed=len(records),all=summary['strata'].get('all'),representatives=names),ensure_ascii=False,indent=2))


if __name__=='__main__':main()

"""Rebuild one conservative grouping universe before combining dataset versions."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from prepare_quality_training import account
from probe_inversion import sha256,write_json

ROOT=Path(__file__).resolve().parents[2]

def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('prior',type=Path);p.add_argument('prepared',type=Path)
    p.add_argument('output',type=Path);p.add_argument('--collection',action='append',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    run,prior,prepared=(x.resolve() for x in (a.run,a.prior,a.prepared))
    old={k:rows(prior/'manifests'/f'{k}.jsonl') for k in ('train','eval','external')}
    current={k:rows(run/'manifests'/f'{k}.jsonl') for k in ('train','eval','external')}
    entries=[]
    for split,records in old.items():
        entries.extend((r,'prior-'+split) for r in records)
    demo=json.loads((ROOT/'deployment/demo-hd/20261117-reviewed/inputs.json').read_text())
    pairs=json.loads((prepared/'longitudinal.json').read_text())
    protected=demo+rows(ROOT/'deployment/training-runs/20261113-040634/manifests/external.jsonl')
    for pair in pairs:
        for side in ('early','late'):
            r=pair[side];protected.append(dict(name=r['name'],image_path=r['aligned'],sha256=r['aligned_sha256']))
    entries.extend((r,'protected') for r in protected)
    for collection in a.collection:
        for r in rows(collection/'collection-manifest.jsonl'):
            name=f"ffhq_{r['image_id']:05d}"
            path=prepared/'256'/(name+'.png')
            if not path.is_file():continue
            entries.append((dict(name=name,image_path=str(path),sha256=sha256(path),
                reference_sha256=r['sha256'],source_metadata=r['source_metadata']), 'candidate'))
    cv2.setNumThreads(2)
    detector=cv2.FaceDetectorYN.create(str(ROOT/'deployment/model-files/face-age/models/yunet/face_detection_yunet_2023mar.onnx'),'',(256,256),.6,.3,100)
    sf=cv2.FaceRecognizerSF.create(str(ROOT/'deployment/gan-research/face_recognition_sface_2021dec.onnx'),'')
    cache={};features=[];missing=[]
    for i,(r,split) in enumerate(entries):
        key=r['sha256']
        if key not in cache:
            rgb=np.asarray(Image.open(r['image_path']).convert('RGB'))
            small=cv2.cvtColor(cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2BGR)
            _,faces=detector.detect(small)
            if faces is None:
                v=np.zeros(128,np.float32);missing.append(r['name'])
            else:
                face=max(faces,key=lambda f:f[2]*f[3]);v=sf.feature(sf.alignCrop(small,face)).ravel();v/=np.linalg.norm(v)+1e-12
            cache[key]=v
        features.append(cache[key])
        if (i+1)%500==0:print('COMMON_FEATURES',i+1,len(entries),flush=True)
    f=np.stack(features);np.savez_compressed(out/'features.npz',features=f)
    parent=list(range(len(entries)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    def join(i,j):
        x,y=sorted((find(i),find(j)));parent[y]=x
    seen={}
    for i,(r,split) in enumerate(entries):
        keys=[('account',account(r)),('pixels',r['sha256']),('source',r.get('reference_sha256',r['sha256']))]
        if split.startswith('prior-'):keys.append(('prior-group',r.get('identity_group')))
        for kind,value in keys:
            if value:
                key=(kind,value)
                if key in seen:join(i,seen[key])
                else:seen[key]=i
    for first in range(0,len(entries),256):
        similarity=f[first:first+256]@f.T
        for local,j in zip(*np.where(similarity>=.4)):
            if first+local<j:join(first+int(local),int(j))
    lookup={r['sha256']:i for i,(r,_) in enumerate(entries)}
    assigned=defaultdict(set);group_records=defaultdict(list)
    for split,records in current.items():
        for r in records:
            if r['sha256'] not in lookup:raise ValueError('Missing current record from audit universe: '+r['name'])
            g=find(lookup[r['sha256']]);assigned[g].add(split);group_records[g].append((split,r['name']))
    protected_groups={find(i) for i,(_,split) in enumerate(entries) if split=='protected'}
    blocked={g for g,s in assigned.items() if s&{'eval','external'}}|protected_groups
    affected={k:[r['name'] for r in v if find(lookup[r['sha256']]) in ({g for g,s in assigned.items() if 'train' in s} if k!='train' else blocked)] for k,v in current.items()}
    strict={}
    for split,records in current.items():
        strict[split]=[]
        for r in records:
            g=find(lookup[r['sha256']])
            if split=='train' and g in blocked:continue
            label='common_'+hashlib.sha256(entries[g][0]['name'].encode()).hexdigest()[:16]
            strict[split].append(dict(r,identity_group=label))
        (out/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in strict[split]))
    mapping=[dict(name=r['name'],sha256=r['sha256'],origin=split,group=find(i)) for i,(r,split) in enumerate(entries)]
    write_json(out/'common-groups.json',mapping)
    report=dict(run=str(run),universe=len(entries),unique_pixels=len(cache),groups=len({find(i) for i in range(len(entries))}),
        rule='Same global universe: account, prior conservative groups, image/source hashes, transitive SFace cosine >= .4',
        missing_face=missing,affected=affected,affected_counts={k:len(v) for k,v in affected.items()},
        affected_eval_cohorts=dict(Counter(r['cohort'] for r in current['eval'] if r['name'] in set(affected['eval']))),
        original_counts={k:len(v) for k,v in current.items()},strict_counts={k:len(v) for k,v in strict.items()},
        strict_train_by_age=dict(Counter(str(r['age_interval']) for r in strict['train'])),
        cross_groups=[dict(group=g,records=group_records[g]) for g,s in assigned.items() if 'train' in s and s&{'eval','external'}],
        caveat='Conservative account/similarity components are not verified identities. Affected means possible relationship, not confirmed same person.',
        files={k+'.jsonl':sha256(out/(k+'.jsonl')) for k in strict})
    write_json(out/'audit.json',report);(out/'DONE').write_text('Common grouping audit complete\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('affected','cross_groups','missing_face','files')},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()

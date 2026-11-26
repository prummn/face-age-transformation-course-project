"""CPU-only incremental scoring and validation for an already submitted study."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import time

from probe_inversion import sha256,write_json

ROOT=Path(__file__).resolve().parents[2]


def main():
    import cv2
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    from check_independent_age import MODEL_SHA256,MODEL_SOURCE,MEAN,STD
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);args=p.parse_args();run=args.run.resolve()
    model=ROOT/'deployment/asia-adaptation/20261123/mivolo_v2_age.onnx'
    if sha256(model)!=MODEL_SHA256:raise ValueError('Independent model changed')
    cv2.setNumThreads(2);opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(model),opts,providers=['CPUExecutionProvider'])
    detector=cv2.FaceDetectorYN.create(str(ROOT/'deployment/model-files/face-age/models/yunet/face_detection_yunet_2023mar.onnx'),'',(256,256),.65,.3,100)
    body=(np.zeros((3,384,384),dtype=np.float32)-MEAN)/STD
    def predict(path):
        bgr=cv2.imread(str(path))
        if bgr is None:raise ValueError('Invalid image '+str(path))
        _,faces=detector.detect(cv2.resize(bgr,(256,256),interpolation=cv2.INTER_AREA))
        if faces is None:return None
        x,y,w,h=max(faces,key=lambda f:f[2]*f[3])[:4];sx,sy=bgr.shape[1]/256,bgr.shape[0]/256
        crop=bgr[max(0,round(y*sy)):round((y+h)*sy),max(0,round(x*sx)):round((x+w)*sx)]
        if not crop.size:return None
        h,w=crop.shape[:2];ratio=384/max(w,h);tw,th=round(w*ratio),round(h*ratio)
        image=cv2.resize(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB),(tw,th),interpolation=cv2.INTER_LINEAR)
        frame=np.zeros((384,384,3),dtype=np.uint8);left,top=(384-tw)//2,(384-th)//2;frame[top:top+th,left:left+tw]=image
        face=(frame.transpose(2,0,1).astype(np.float32)/255-MEAN)/STD
        value=float(session.run(['age'],{'faces_bodies':np.concatenate([face,body],axis=0)[None]})[0].item())
        if not np.isfinite(value):raise ValueError('Nonfinite independent age')
        return value
    score_file=run/'independent-age.json';scores=json.loads(score_file.read_text())['ages'] if score_file.exists() else {}
    exp=json.loads((run/'experiment.json').read_text());deadline=time.monotonic()+50*60;last_count=-1
    while True:
        final=run/'evaluation/report.json';partial=run/'evaluation/report.partial.json'
        path=final if final.exists() else partial
        if not path.exists():
            if time.monotonic()>deadline:raise TimeoutError('No GPU outputs before deadline')
            time.sleep(5);continue
        try:report=json.loads(path.read_text())
        except json.JSONDecodeError:time.sleep(2);continue
        for output in report['outputs']:
            key=output['native']
            if key in scores:continue
            native=run/'evaluation'/key
            if sha256(native)!=output['sha256']:raise ValueError('Native output hash mismatch')
            for field in ['native','preserved']:
                with Image.open(run/'evaluation'/output[field]) as image:
                    image.load()
                    if image.format!='PNG' or image.size!=(1024,1024):raise ValueError('Invalid native PNG')
            scores[key]=predict(native)
        write_json(score_file,dict(model_source=MODEL_SOURCE,model_sha256=MODEL_SHA256,ages=scores,
            outputs=len(scores),note='Independent CPU appearance estimates only, not chronological ground truth.'))
        if len(scores)!=last_count:print('AGE_OBJECTIVE_CPU_SCORED',len(scores),flush=True);last_count=len(scores)
        if final.exists() and (run/'DONE').exists():break
        if time.monotonic()>deadline:raise TimeoutError('GPU study did not complete')
        time.sleep(5)
    if len(report['outputs'])!=585 or len(scores)!=585:raise ValueError('Incomplete585-output design')
    frozen=json.loads((run/'frozen-verification.json').read_text());repeat=json.loads((run/'repeatability.json').read_text())
    if not frozen['equal'] or not all(repeat[k] for k in ['anchor_codes_equal','target_codes_equal','pngs_equal']):raise ValueError('Frozen/stability check failed')
    baseline={(o['name'],o['condition']):o for o in report['outputs'] if o['route']=='accepted_baseline'}
    protected=0
    for o in report['outputs']:
        source=next(r['source_age'] for r in report['records'] if r['name']==o['name'])
        if o['condition'] in ['same','20','40'] or o['target']==source:
            reference=baseline[o['name'],o['condition']]
            if o['sha256']!=reference['sha256']:raise ValueError('Protected output changed')
            if sha256(run/'evaluation'/o['preserved'])!=sha256(run/'evaluation'/reference['preserved']):
                raise ValueError('Protected color output changed')
            protected+=1
    groups=defaultdict(list)
    for o in report['outputs']:
        scope='diagnostic6' if o['name'] in exp['diagnostic_cases'] else 'confirmation29'
        groups[scope,o['route']].append(o);groups['pooled',o['route']].append(o)
    def mean(values):
        valid=[x for x in values if x is not None];return statistics.mean(valid) if valid else None
    def describe(items):
        old=[o for o in items if o['condition'] in ['60','75']];cross=[o for o in items if o['condition']!='same']
        by={(o['name'],o['condition']):scores[o['native']] for o in items};names={o['name'] for o in items}
        spans=[by[n,'75']-by[n,'20'] for n in names if by[n,'75'] is not None and by[n,'20'] is not None]
        high_spans=[by[n,'75']-by[n,'60'] for n in names if by[n,'75'] is not None and by[n,'60'] is not None]
        return dict(cases=len(names),missing_faces=sum(not o['detected'] for o in items),
            independent_missing=sum(scores[o['native']] is None for o in items),
            identity=mean(o['source_cosine'] for o in cross),high_age_identity=mean(o['source_cosine'] for o in old),
            high_age_error=mean(abs(scores[o['native']]-o['target']) for o in old if scores[o['native']] is not None),
            mean_age75=mean(by[n,'75'] for n in names),age20to75_span=mean(spans),age60to75_span=mean(high_spans),
            weak60to75=sum(v<5 for v in high_spans),reversed60to75=sum(v<0 for v in high_spans))
    summary=defaultdict(dict)
    for (scope,route),items in groups.items():summary[scope][route]=describe(items)
    checks={};paired={}
    for scope in ['diagnostic6','confirmation29','pooled']:
        checks[scope]={};paired[scope]={}
        candidate=summary[scope]['target_age_order_floor']
        for reference in ['accepted_baseline','teacher_control']:
            control=summary[scope][reference]
            checks[scope][reference]=dict(identity=candidate['identity']>=control['identity']-.02,
                high_age_identity=candidate['high_age_identity']>=control['high_age_identity']-.02,
                relative_span=candidate['age20to75_span']>=.9*control['age20to75_span'],
                high_age_error=candidate['high_age_error']<control['high_age_error'],
                high_age_span=candidate['age60to75_span']>=control['age60to75_span'],
                detection=candidate['missing_faces']<=control['missing_faces'],
                independent_detection=candidate['independent_missing']<=control['independent_missing'])
            candidates={(o['name'],o['condition']):o for o in groups[scope,'target_age_order_floor']
                        if o['condition'] in ['60','75']}
            controls={(o['name'],o['condition']):o for o in groups[scope,reference]
                      if o['condition'] in ['60','75']}
            rows=[]
            for key,o in candidates.items():
                ref=controls[key];age=scores[o['native']];ref_age=scores[ref['native']]
                rows.append(dict(name=o['name'],condition=o['condition'],
                    identity_delta=o['source_cosine']-ref['source_cosine']
                        if o['source_cosine'] is not None and ref['source_cosine'] is not None else None,
                    age_error_delta=abs(age-o['target'])-abs(ref_age-o['target'])
                        if age is not None and ref_age is not None else None))
            paired[scope][reference]=dict(conditions=len(rows),
                identity_drop_over02=sum(r['identity_delta'] is not None and r['identity_delta']<-.02 for r in rows),
                age_error_improved=sum(r['age_error_delta'] is not None and r['age_error_delta']<0 for r in rows),
                rows=rows)
    write_json(run/'artifact-validation.json',dict(raw=585,color=585,all_decodable=True,native_size=[1024,1024],
        raw_sha256_verified=True,protected_same_young_or_source_exact=protected,protected_color_exact=True,
        all_models_unchanged=True,repeatable=True))
    write_json(run/'summary.json',dict(groups=summary,checks=checks,paired=paired,independent_ages=scores,all_numeric_checks_pass=all(v for scopes in checks.values() for comp in scopes.values() for v in comp.values()),
        visual_review_pending=True,promotion=False,note='Development/regression with fixed recipe; explicit age compliance does not guarantee skin naturalness.'))
    print('AGE_OBJECTIVE_CPU_COMPLETE',json.dumps(summary['pooled']),flush=True)


if __name__=='__main__':main()

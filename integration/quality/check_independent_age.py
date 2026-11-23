"""Evaluate output age response on CPU with a frozen face-only MiVOLO v2 export."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort

from probe_inversion import sha256, write_json


ROOT = Path(__file__).resolve().parents[2]
MODEL_SHA256 = '2db5e05be33b3f120518a86f29a4a65eb0a01c2967219b8787a7dead9b796951'
MODEL_SOURCE = 'https://huggingface.co/Sam-Apostel/mivolo-v2-age-onnx/tree/8eb4cd8f5dd4bd28df2c9a43b24bc54d6d139d36'
MEAN = np.array([.485,.456,.406],dtype=np.float32)[:,None,None]
STD = np.array([.229,.224,.225],dtype=np.float32)[:,None,None]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    kind=p.add_mutually_exclusive_group(required=True)
    kind.add_argument('--eval',type=Path,help='Evaluation directory containing report.json')
    kind.add_argument('--sources',type=Path,help='JSON list containing source image records')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--routes',nargs='+',default=['fine_direct','control_encoder','expanded_encoder','expanded_generator'],
                   help='Evaluation routes to score')
    p.add_argument('--follow',action='store_true',help='Score newly completed outputs while the GPU evaluator runs')
    a=p.parse_args()
    if sha256(a.model) != MODEL_SHA256:raise ValueError('Independent model hash differs')
    cv2.setNumThreads(2)
    cfg=ort.SessionOptions()
    cfg.intra_op_num_threads=2
    cfg.inter_op_num_threads=1
    session=ort.InferenceSession(str(a.model),cfg,providers=['CPUExecutionProvider'])
    detector_path=ROOT/'deployment/model-files/face-age/models/yunet/face_detection_yunet_2023mar.onnx'
    detector=cv2.FaceDetectorYN.create(str(detector_path),'',(256,256),.65,.3,100)
    detector.setInputSize((256,256))
    body=(np.zeros((3,384,384),dtype=np.float32)-MEAN)/STD

    def predict(path):
        bgr=cv2.imread(str(path))
        if bgr is None:raise ValueError('Could not decode '+str(path))
        resized=cv2.resize(bgr,(256,256),interpolation=cv2.INTER_AREA)
        _,faces=detector.detect(resized)
        if faces is None:return None
        x,y,w,h=max(faces,key=lambda f:f[2]*f[3])[:4]
        sx,sy=bgr.shape[1]/256,bgr.shape[0]/256
        cropped=bgr[max(0,round(y*sy)):round((y+h)*sy),max(0,round(x*sx)):round((x+w)*sx)]
        if not cropped.size:return None
        h,w=cropped.shape[:2]
        ratio=384/max(w,h)
        tw,th=round(w*ratio),round(h*ratio)
        image=cv2.resize(cv2.cvtColor(cropped,cv2.COLOR_BGR2RGB),(tw,th),interpolation=cv2.INTER_LINEAR)
        frame=np.zeros((384,384,3),dtype=np.uint8)
        left,top=(384-tw)//2,(384-th)//2
        frame[top:top+th,left:left+tw]=image
        face=(frame.transpose(2,0,1).astype(np.float32)/255-MEAN)/STD
        x=np.concatenate([face,body],axis=0)[None]
        result=float(session.run(['age'],{'faces_bodies':x})[0].item())
        if not np.isfinite(result):raise ValueError('Independent estimator returned non-finite age')
        return result

    def read_report():
        final = a.eval/'report.json'
        path = a.eval/'report.partial.json' if a.follow and not final.exists() else final
        return json.loads(path.read_text())

    def select_records(report):
        records=[]
        selected=defaultdict(int)
        for row in sorted(report['records'],key=lambda x:x['name']):
            if row['cohort'].startswith('new_appa_'):
                records.append(row)
            elif row['cohort']=='annotated-ffhq':
                age=tuple(row['age_interval'])
                if selected[age]<3:
                    records.append(row)
                    selected[age]+=1
        return records

    deadline=time.monotonic()+45*60
    if a.sources:
        records=json.loads(a.sources.read_text())
    else:
        while True:
            try:
                report=read_report()
                break
            except (FileNotFoundError,json.JSONDecodeError):
                if not a.follow or time.monotonic()>=deadline:raise
                time.sleep(4)
        records=select_records(report)
    results=[]
    for row in records:
        path=Path(row.get('reference_path',row['image_path']))
        if sha256(path)!=row.get('reference_sha256',row['sha256']):raise ValueError('Source hash differs')
        results.append(dict(name=row['name'],cohort=row['cohort'],route='input',condition='source',
            prediction=predict(path),published_age=row.get('publisher_age'),apparent_age=row.get('apparent_age')))
    done=set()
    if a.eval:
        ids={r['name'] for r in records}
        routes=set(a.routes)
        while True:
            try:
                report=read_report()
            except json.JSONDecodeError:
                time.sleep(2)
                continue
            for row in report['outputs']:
                key=(row['name'],row['route'],row['condition'])
                if row['name'] not in ids or row['route'] not in routes or row['condition'] not in ['20','75'] or key in done:
                    continue
                path=a.eval/(row.get('native') or row['preview'])
                results.append({k:row[k] for k in ['name','cohort','route','condition','target']} | dict(prediction=predict(path)))
                done.add(key)
                if len(done)%50==0:print('MIVOLO_CHECKED',len(done),flush=True)
            if 'total_seconds' in report:break
            if time.monotonic()>=deadline:raise TimeoutError('GPU evaluator did not complete')
            time.sleep(4)
    grouped=defaultdict(dict)
    for row in results:
        grouped[(row['cohort'],row['route'],row['name'])][row['condition']]=row
    aggregate=defaultdict(lambda:defaultdict(list))
    for (cohort,route,name),conditions in grouped.items():
        if route=='input':
            item=conditions['source']
            if item['prediction'] is not None:
                for field in ['published_age','apparent_age']:
                    if item[field] is not None:aggregate[(cohort,route)][field+'_mae'].append(abs(item['prediction']-float(item[field])))
        else:
            first=conditions.get('20',{}).get('prediction')
            last=conditions.get('75',{}).get('prediction')
            if first is not None and last is not None:aggregate[(cohort,route)]['span20to75'].append(last-first)
            if last is not None:aggregate[(cohort,route)]['age75'].append(last)
    summary=defaultdict(dict)
    for (cohort,route),fields in sorted(aggregate.items()):
        summary[cohort][route]={key:dict(mean=float(np.mean(v)),valid=len(v)) for key,v in fields.items()}
    write_json(a.out,dict(model_source=MODEL_SOURCE,model_sha256=MODEL_SHA256,inputs=len(records),
        generated=len(done),results=results,summary=summary,missing_face=sum(r['prediction'] is None for r in results),
        note='Independent of GAN training losses. Face-only prediction of age appearance, not generated chronological ground truth. Native output used when available, else stored 256.'))
    print('INDEPENDENT_AGE_DONE',len(records),len(done),flush=True)


if __name__=='__main__':
    main()

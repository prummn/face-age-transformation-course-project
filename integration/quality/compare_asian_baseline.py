"""Rescore saved direct and W+ results on exactly the same APPA images."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np

from probe_inversion import sha256, write_json
from summarize_asian_adaptation import mean, paired_ci, stats


ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('previous',type=Path)
    args=parser.parse_args()
    current=args.run/'evaluation'
    previous=args.previous/'external-results'
    old=json.loads((previous/'report.json').read_text())
    new=json.loads((current/'report.json').read_text())
    records=[r for r in new['records'] if r['cohort']=='new_appa_asian']
    ids={r['name'] for r in records}
    prior={r['name']:r for r in old['records']}
    for row in records:
        if prior[row['name']]['sha256'] != row['sha256'] or prior[row['name']]['source_age'] != row['source_age']:
            raise ValueError('Prior and current source image or condition differ for '+row['name'])
    cv2.setNumThreads(2)
    assets=ROOT/'deployment/model-files/face-age/models/yunet/face_detection_yunet_2023mar.onnx'
    detector=cv2.FaceDetectorYN.create(str(assets),'',(256,256),.6,.3,100)
    detector.setInputSize((256,256))
    recognizer=cv2.FaceRecognizerSF.create(str(ROOT/'deployment/gan-research/face_recognition_sface_2021dec.onnx'),'')

    def describe(path):
        bgr=cv2.imread(str(path))
        if bgr is None:raise ValueError('Could not decode '+str(path))
        small=cv2.resize(bgr,(256,256),interpolation=cv2.INTER_AREA)
        _,faces=detector.detect(small)
        if faces is None:return None
        face=max(faces,key=lambda f:f[2]*f[3])
        x=recognizer.feature(recognizer.alignCrop(small,face)).ravel()
        return x/(np.linalg.norm(x)+1e-12)

    features={row['name']:describe(row['image_path']) for row in records}
    output=[]
    for source,base in [(old,previous),(new,current)]:
        for row in source['outputs']:
            if row['name'] not in ids:continue
            if not row['native']:raise ValueError('Native output required for paired APPA review')
            face=describe(base/row['native'])
            ref=features[row['name']]
            value=float(face@ref) if face is not None and ref is not None else None
            output.append(dict(row,sface_cosine=value,output_face_detected=face is not None))
    by_photo=defaultdict(lambda:defaultdict(list))
    for row in output:
        by_photo[row['name']][row['route']].append(row)
    for routes in by_photo.values():
        for route,conditions in routes.items():
            # Earlier W+ reports store face/chroma metrics with different field names.
            for condition in conditions:condition.setdefault('face_chroma_shift',None)
            routes[route]=dict(stats(conditions),conditions={r['condition']:r for r in conditions})
    names=sorted({row['route'] for row in output})
    summaries={}
    for route in names:
        cases=[r[route] for r in by_photo.values()]
        summaries[route]=dict(cross_sface=mean([r['cross_sface'] for r in cases]),
            same_sface=mean([r['same_sface'] for r in cases]),missing=sum(r['missing_sface'] for r in cases),
            against_saved_wplus=paired_ci(records,by_photo,'baseline_v1',route,'cross_sface'))
    result=dict(cases=len(records),records=sorted(ids),routes=summaries,
        input_digests=dict(current=sha256(current/'report.json'),prior=sha256(previous/'report.json')),
        note='APPA Asian labels are publisher annotations. Earlier offline batched W+ saved outputs, not new live API benchmark. Same aligned input and independent crop per result for SFace.')
    write_json(args.run/'against-saved-wplus.json',result)
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':
    main()

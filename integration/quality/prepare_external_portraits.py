"""Align new APPA crops and retain failures, native quality and duplicate screens."""
import argparse
from collections import Counter
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from probe_inversion import sha256,write_json
from prepare_gan_training import align


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('collection',type=Path)
    root=Path(__file__).resolve().parents[2];collection=p.parse_args().collection.resolve()
    out=collection/'prepared';out.mkdir(exist_ok=True)
    if (out/'DONE').exists():raise RuntimeError('Prepared external inputs already frozen')
    cv2.setNumThreads(3)
    detector=cv2.FaceDetectorYN.create(str(root/'deployment/model-files/face-age/models/yunet/face_detection_yunet_2023mar.onnx'),'',(256,256),.6,.3,100)
    recognizer=cv2.FaceRecognizerSF.create(str(root/'deployment/gan-research/face_recognition_sface_2021dec.onnx'),'')
    def describe(image):
        small=cv2.resize(image,(256,256),interpolation=cv2.INTER_AREA);bgr=cv2.cvtColor(small,cv2.COLOR_RGB2BGR)
        detector.setInputSize((256,256));_,faces=detector.detect(bgr)
        if faces is None:return small,None,None
        face=max(faces,key=lambda f:f[2]*f[3]);v=recognizer.feature(recognizer.alignCrop(bgr,face)).ravel();v/=np.linalg.norm(v)+1e-12
        return small,face,v
    old=root/'deployment/training-data/ffhq-adult-v2/prepared'
    prior=[json.loads(s) for split in ['train.pre-age','eval.pre-age','external'] for s in (old/(split+'.jsonl')).read_text().splitlines()]
    prior_hashes={r['sha256'] for r in prior}|{r.get('reference_sha256') for r in prior}
    # Reuse saved training/eval features, retaining this conservative screen's limits.
    cache=old/'screening-cache';prior_vectors=[];prior_names=[]
    for r in prior:
        h=r.get('reference_sha256',r['sha256']);path=cache/(h+'.npz')
        if path.exists():v=np.load(path,allow_pickle=False)['feature']
        else:
            image=np.asarray(Image.open(r.get('reference_path',r['image_path'])).convert('RGB'));_,_,v=describe(image)
        if v is not None and len(v):prior_vectors.append(v);prior_names.append(r['name'])
    matrix=np.stack(prior_vectors);records=[];failures=[];vectors=[]
    source=[json.loads(s) for s in (collection/'collection-manifest.jsonl').read_text().splitlines()]
    assert len(source)==32 and all(r['status']=='verified' for r in source)
    for row in source:
        original,crop=row['files'];assert sha256(crop['path'])==crop['sha256'] and sha256(original['path'])==original['sha256']
        rgb=np.asarray(Image.open(crop['path']).convert('RGB'))
        h,w=rgb.shape[:2];detector.setInputSize((w,h));_,faces=detector.detect(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        aligned=align(rgb,detector)
        if aligned is None:
            failures.append(dict(name=row['name'],reason='YuNet_no_face_on_publisher_crop',annotation=row['annotation'],files=row['files']));continue
        nativeface=max(faces,key=lambda f:f[2]*f[3])
        folder=out/row['name'];folder.mkdir(exist_ok=True);imagepath=folder/'aligned1024.png';Image.fromarray(aligned).save(imagepath)
        small,face,v=describe(aligned)
        gray=cv2.cvtColor(small,cv2.COLOR_RGB2GRAY);central=gray[65:215,55:205];chroma=small[65:215,55:205].astype(float).std(axis=2).mean()
        matches=[]
        if v is not None:
            for idx in np.where(matrix@v>=.4)[0]:matches.append(dict(name=prior_names[idx],cosine=float(matrix[idx]@v)))
        assert crop['sha256'] not in prior_hashes and original['sha256'] not in prior_hashes,'Exact old input found'
        annotation=row['annotation'];age=float(annotation['real_age']);agebin='20-39' if age<40 else '40-59' if age<60 else '60-74' if age<75 else '75+'
        r=dict(name=row['name'],cohort='new_appa_asian' if annotation['race']=='asian' else 'new_appa_other',
            image_path=str(imagepath),sha256=sha256(imagepath),reference_path=str(imagepath),reference_sha256=sha256(imagepath),
            source=row['source_archive'],source_original=original,source_crop=crop,training_authorized=False,aligned=True,
            annotation=annotation,publisher_age=float(annotation['real_age']),apparent_age=float(annotation['apparent_age_avg']),
            publisher_age_bin=agebin,native_crop_size=crop['size'],native_face_short_edge=float(min(nativeface[2:4])),
            source_quality=dict(brightness=float(central.mean()),dark_fraction=float((central<45).mean()),
                sharpness=float(cv2.Laplacian(central,cv2.CV_64F).var()),chroma_rgb_std=float(chroma),
                upscaled_reference=bool(min(nativeface[2:4])<256)),
            alignment='YuNet+FFHQ on publisher crop; largest detected face within that crop; generated1024 reference may upscale native pixels.',
            old_similarity_matches=matches,publisher_crop_detected_faces=len(faces),aligned_face_detected=face is not None)
        records.append(r);vectors.append(v)
        print('PREPARED_EXTERNAL',r['name'],r['native_crop_size'],len(matches),flush=True)
    # Conservative same-person grouping from SFace only; no verified subject IDs.
    parent=list(range(len(records)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    pairs=[]
    for i,v in enumerate(vectors):
        if v is None:continue
        for j,u in enumerate(vectors[:i]):
            if u is not None and float(v@u)>=.4:
                parent[find(i)]=find(j);pairs.append(dict(a=records[i]['name'],b=records[j]['name'],cosine=float(v@u)))
    assignments={};counts=Counter()
    for i,r in enumerate(records):
        key=find(i);r['similarity_group']='external_'+records[key]['name']
        if key not in assignments:
            stratum=(r['cohort'],r['publisher_age_bin']);assignments[key]='diagnostic' if counts[stratum]%2==0 else 'confirmation';counts[stratum]+=1
        r['review_split']=assignments[key]
    write_json(out/'inputs.pre-age.json',records)
    report=dict(selected=32,aligned=len(records),failures=failures,cohorts=dict(Counter(r['cohort'] for r in records)),
        split=dict(Counter(r['review_split'] for r in records)),similarity_pairs=pairs,
        prior_matches=sum(bool(r['old_similarity_matches']) for r in records),
        low_native_face=sum(r['native_face_short_edge']<256 for r in records),
        source_manifest_sha256=sha256(collection/'collection-manifest.jsonl'),inputs_sha256=sha256(out/'inputs.pre-age.json'),
        warning='Exact new photos only; approximate SFace screen does not establish subject independence. All32 selection/failures retained; APPA is evaluation-only, not new model training.')
    write_json(out/'audit.json',report);(out/'DONE').write_text('External inputs frozen; pseudo source ages pending GPU\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

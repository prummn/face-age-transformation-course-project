"""Audit FFHQ pilot and freeze conservative groups; no GPU and no age invention."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import shutil
from urllib.parse import urlparse
import cv2
import numpy as np
from PIL import Image
from probe_inversion import sha256,write_json

ROOT=Path(__file__).resolve().parents[2]

def jsonl(path,records):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))

def align(rgb,detector,size=1024):
    h,w=rgb.shape[:2];detector.setInputSize((w,h))
    _,faces=detector.detect(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    if faces is None:return None
    f=max(faces,key=lambda x:x[2]*x[3]);p=f[4:14].reshape(5,2)
    el,er=sorted(p[:2],key=lambda x:x[0]);ml,mr=sorted(p[3:5],key=lambda x:x[0])
    ea=(el+er)*.5;ee=er-el;em=(ml+mr)*.5-ea
    x=ee-np.flipud(em)*[-1,1];x/=np.hypot(*x);x*=max(np.hypot(*ee)*2,np.hypot(*em)*1.8)
    y=np.flipud(x)*[-1,1];c=ea+em*.1
    quad=np.stack([c-x-y,c-x+y,c+x+y,c+x-y]).astype(np.float32)
    mat=cv2.getPerspectiveTransform(quad,np.array([[0,0],[0,size],[size,size],[size,0]],np.float32))
    return cv2.warpPerspective(rgb,mat,(size,size),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_REFLECT)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reuse',type=Path,help='Reuse prior native-derived256/512 images, still rehash source and screen faces')
    p.add_argument('--collection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'DONE').exists():raise RuntimeError('Prepared dataset already frozen')
    collection=[json.loads(x) for x in (args.collection/'collection-manifest.jsonl').read_text().splitlines()]
    if any(r['status']!='verified' for r in collection):raise RuntimeError('Retry failed downloads before freezing')
    base=ROOT/'deployment/model-files/face-age';cv2.setNumThreads(4)
    det=cv2.FaceDetectorYN.create(str(base/'models/yunet/face_detection_yunet_2023mar.onnx'),'',(256,256),.6,.3,100)
    sface=ROOT/'deployment/gan-research/face_recognition_sface_2021dec.onnx'
    assert sha256(sface)=='0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79'
    rec=cv2.FaceRecognizerSF.create(str(sface),'')
    def describe(rgb):
        small=cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA);bgr=cv2.cvtColor(small,cv2.COLOR_RGB2BGR)
        det.setInputSize((256,256));_,faces=det.detect(bgr)
        if faces is None:return small,None,None
        f=max(faces,key=lambda x:x[2]*x[3]);v=rec.feature(rec.alignCrop(bgr,f)).ravel();v/=np.linalg.norm(v)+1e-12
        return small,v,f
    records=[];vectors=[];rejected=[]
    # Completed earlier audit verified these deterministic downsampling outputs.
    completed_cache_audit=(out/'audit.json').exists()
    (out/'screening-cache').mkdir(exist_ok=True)
    for i,r in enumerate(collection):
        if sha256(r['path'])!=r['sha256']:raise ValueError('Changed source')
        dest=out/'input256'/f"{r['image_id']:05d}.png";ref512=out/'reference512'/dest.name
        cache=out/'screening-cache'/(r['sha256']+'.npz')
        reuse_transform=completed_cache_audit and cache.exists() and dest.exists() and ref512.exists()
        if reuse_transform:
            small=np.asarray(Image.open(dest).convert('RGB'));reference=np.asarray(Image.open(ref512).convert('RGB'))
            assert small.shape==(256,256,3) and reference.shape==(512,512,3),'Invalid completed transform cache'
            rgb=None
        else:
            rgb=np.asarray(Image.open(r['path']).convert('RGB'));assert rgb.shape==(1024,1024,3)
        if cache.exists():
            cached=np.load(cache,allow_pickle=False)
            if not reuse_transform:small=cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA)
            v=cached['feature'];f=cached['face'];v=None if not v.size else v;f=None if not f.size else f
        else:
            small,v,f=describe(rgb)
            np.savez(cache,feature=np.array([]) if v is None else v,face=np.array([]) if f is None else f)
        gray=cv2.cvtColor(small,cv2.COLOR_RGB2GRAY)
        sharp=float(cv2.Laplacian(gray[55:220,45:210],cv2.CV_64F).var())
        # Minimum diagnostic threshold only: retain every validation case, including failures.
        bad=(v is None or sharp<25)
        if r['category']=='training' and bad:
            rejected.append(dict(image_id=r['image_id'],reason='no_face' if v is None else 'face_laplacian_below25',sharpness=sharp));continue
        dest=out/'input256'/f"{r['image_id']:05d}.png";dest.parent.mkdir(exist_ok=True)
        reuse=args.reuse/'input256'/dest.name if args.reuse else None
        if dest.exists():
            assert np.array_equal(np.asarray(Image.open(dest).convert('RGB')),small)
        elif reuse and reuse.exists():
            assert np.array_equal(np.asarray(Image.open(reuse).convert('RGB')),small),'Cached256 differs from source'
            shutil.copy2(reuse,dest)
        else:Image.fromarray(small).save(dest)
        ref512=out/'reference512'/dest.name;ref512.parent.mkdir(exist_ok=True)
        if not reuse_transform:reference=cv2.resize(rgb,(512,512),interpolation=cv2.INTER_AREA)
        reuse=args.reuse/'reference512'/dest.name if args.reuse else None
        if ref512.exists():
            assert np.array_equal(np.asarray(Image.open(ref512).convert('RGB')),reference)
        elif reuse and reuse.exists():
            assert np.array_equal(np.asarray(Image.open(reuse).convert('RGB')),reference),'Cached512 differs from source'
            shutil.copy2(reuse,ref512)
        else:Image.fromarray(reference).save(ref512)
        author=urlparse(r['source_metadata']['photo_url']).path.split('/')[2]
        a=dict(name=f"ffhq_{r['image_id']:05d}",image_id=r['image_id'],image_path=str(dest),sha256=sha256(dest),
               reference_path=r['path'],reference_sha256=r['sha256'],reference512_path=str(ref512),reference512_sha256=sha256(ref512),
               source='FFHQ + FFHQ-Aging; official '+r['category'],training_authorized=True,
               official_split=r['category'],age_interval=r['age_interval'],age_interval_kind=r['age_label_kind'],
               age_confidence=float(r['annotation']['age_group_confidence']),publisher_gender=r['annotation']['gender'],
               source_metadata=r['source_metadata'],flickr_account=author,face_sharpness=sharp,
               face_detected=v is not None,group_kind='conservative_source_similarity_component',cohort='ffhq_regression')
        records.append(a);vectors.append(np.zeros(128,np.float32) if v is None else v)
        if (i+1)%100==0:print('AUDIT',i+1,len(collection),flush=True)
    # Similarity and photographer-account grouping are leakage screens, not verified subjects.
    n=len(records);parent=list(range(n))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b:parent[max(a,b)]=min(a,b)
    seen={}
    for i,r in enumerate(records):
        k=r['flickr_account']
        if k in seen:union(i,seen[k])
        else:seen[k]=i
    mat=np.stack(vectors);sim=mat@mat.T;edges=[]
    for i,j in zip(*np.where(np.triu(sim>=.4,1))):
        union(int(i),int(j));edges.append(dict(a=records[i]['name'],b=records[j]['name'],cosine=float(sim[i,j])))
    diagnostic=json.loads((ROOT/'deployment/gan-research/diagnostic-manifest.json').read_text())['samples']
    external=[]
    for d in diagnostic:
        assert sha256(d['path'])==d['sha256']
        rgb=np.asarray(Image.open(d['path']).convert('RGB'));small,v,_=describe(rgb)
        external.append(dict(name=d['name'],cohort='diagnostic',image_path=d['path'],sha256=d['sha256'],
                             source=d['source'],training_authorized=False,aligned=True,original_size=list(rgb.shape[:2])))
        if v is not None:
            for idx in np.where(mat@v>=.4)[0]:records[int(idx)]['diagnostic_similarity_exclusion']=True
    # APPA official test: alignment failures recorded, never silently remove difficult cases.
    failures=[]
    for line in (ROOT/'deployment/data-research/20261113/appa-asian-pilot/manifest.jsonl').read_text().splitlines():
        d=json.loads(line);original=d['files'][0];assert sha256(original['path'])==original['sha256']
        rgb=np.asarray(Image.open(original['path']).convert('RGB'));aligned=align(rgb,det)
        if aligned is None:
            failures.append(dict(name=d['annotation']['file_name'],reason='YuNet_no_face',annotation=d['annotation']));continue
        dest=out/'external'/('appa_'+Path(original['path']).stem+'.png');dest.parent.mkdir(exist_ok=True)
        Image.fromarray(aligned).save(dest)
        external.append(dict(name=dest.stem,cohort='appa_asian_test',image_path=str(dest),sha256=sha256(dest),
            source=d['source_archive'],training_authorized=False,aligned=True,annotation=d['annotation'],
            original_sha256=original['sha256'],original_size=original['size']))
        _,v,_=describe(aligned)
        if v is not None:
            for idx in np.where(mat@v>=.4)[0]:records[int(idx)]['diagnostic_similarity_exclusion']=True
    validation_groups={find(i) for i,r in enumerate(records) if r['official_split']=='validation'}
    forbidden_groups={find(i) for i,r in enumerate(records) if r.get('diagnostic_similarity_exclusion')}
    training=[];evaluation=[]
    for i,r in enumerate(records):
        r['identity_group']='candidate_'+hashlib.sha256(records[find(i)]['name'].encode()).hexdigest()[:16]
        if r['official_split']=='validation':evaluation.append(r)
        elif find(i) in validation_groups|forbidden_groups:
            rejected.append(dict(image_id=r['image_id'],reason='conservative_group_overlaps_evaluation'))
        else:training.append(r)
    jsonl(out/'train.pre-age.jsonl',training);jsonl(out/'eval.pre-age.jsonl',evaluation);jsonl(out/'external.jsonl',external)
    write_json(out/'audit.json',dict(training=len(training),evaluation=len(evaluation),external=len(external),
        rejected=rejected,external_alignment_failures=failures,similarity_edges=edges,
        groups=len({r['identity_group'] for r in records}),
        train_age_bins=dict(Counter(str(r['age_interval']) for r in training)),
        protocol='same Flickr account OR independent SFace cosine>=0.40 connected components; remove train components touching eval/diagnostics; minimum train face sharpness25',
        caveats=['No verified subject IDs; conservative grouping reduces but cannot prove absence of same-person overlap.',
                 'FFHQ was used by base StyleGAN2: evaluation is regression, not pretraining-independent validation.',
                 'APPA test is evaluation only; no unknown-license training or redistribution.',
                 'FFHQ human apparent-age intervals retained; source numeric condition will be explicitly shared FairFace pseudo-age.'],
        collection_manifest_sha256=sha256(args.collection/'collection-manifest.jsonl')))
    if len(training)<1000:raise RuntimeError(f'Only {len(training)} after grouping; audit saved; extend data rather than relax thresholds')
    (out/'DONE').write_text('Frozen CPU preparation; age annotations pending\n')
    print('PREPARED',len(training),len(evaluation),len(external),flush=True)

if __name__=='__main__':main()

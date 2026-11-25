"""Screen named, source-attributed senior photographs into quarantine, never training.

Consumes frozen Commons search responses and Wikidata biography responses. Names
come from source metadata, not face recognition. Dates/DOB conflicts remain
intervals. Existing common-group similarity matches are only conservative flags.
"""
import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlencode,urlsplit,urlunsplit
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from probe_inversion import sha256,write_json


def fetch(params):
    url='https://commons.wikimedia.org/w/api.php?'+urlencode(dict(format='json',**params))
    with urlopen(Request(url,headers={'User-Agent':'FaceAgeCourseResearch/1.0 (source and licence audit)'}),timeout=35) as r:return json.load(r)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--search',action='append',type=Path,required=True)
    p.add_argument('--biographies',type=Path,required=True);p.add_argument('--biography-ids',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'raw').mkdir();(out/'aligned').mkdir()
    root=Path(__file__).resolve().parents[2]
    sources=[a.biographies,a.biography_ids,*a.search]
    for i,source in enumerate(sources):
        import shutil
        shutil.copy2(source,out/f'source-{i}.json')
    bio=json.loads(a.biographies.read_text())['entities']
    mapping={x['title']:x['pageprops']['wikibase_item'] for x in json.loads(a.biography_ids.read_text())['query']['pages'].values() if 'pageprops' in x}
    candidates=[];rejected=[];seen=set()
    for source in a.search:
        for subject in json.loads(source.read_text()):
            q=mapping.get(subject['name'])
            if not q:continue
            claims=[x for x in bio[q].get('claims',{}).get('P569',[]) if x.get('rank')!='deprecated']
            preferred=[x for x in claims if x.get('rank')=='preferred'];claims=preferred or claims
            births=[]
            for claim in claims:
                value=claim['mainsnak'].get('datavalue',{}).get('value',{})
                if value.get('precision',0)>=11:births.append(date.fromisoformat(value['time'][1:11]))
            if not births:continue
            for page in subject['pages']:
                info=page.get('imageinfo',[{}])[0];meta=info.get('extmetadata',{})
                value=lambda key:re.sub('<[^>]+>','',meta.get(key,{}).get('value',''))
                if page['pageid'] in seen:continue
                seen.add(page['pageid']);title=page['title'];when=value('DateTimeOriginal')
                item=dict(subject_id=q,subject_name=subject['name'],pageid=page['pageid'],title=title)
                reason=None
                if re.search('signature|autograph|poster|painting|sculpture|statue',title,re.I):reason='not_photographic_portrait'
                elif not re.match(r'^\d{4}-\d\d-\d\d',when):reason='no_exact_published_photo_date'
                elif info.get('mime') not in ('image/jpeg','image/png'):reason='unsupported_original_format'
                elif min(info.get('width',0),info.get('height',0))<800:reason='original_short_edge_below800'
                elif info.get('size',0)>25*1024**2:reason='single_original_over25MiB'
                elif not re.fullmatch(r'CC BY(?:-SA)? [1234]\.0|CC0|Public domain|KOGL Type 1',value('LicenseShortName')):reason='licence_needs_separate_review'
                if reason:rejected.append(item|dict(reason=reason));continue
                photographed=date.fromisoformat(when[:10]);ages=[photographed.year-b.year-((photographed.month,photographed.day)<(b.month,b.day)) for b in births]
                if min(ages)<55 or max(ages)>95:rejected.append(item|dict(reason='outside55to95'));continue
                candidates.append(item|dict(info=info,photo_date=photographed.isoformat(),published_birth_dates=[b.isoformat() for b in births],
                    age_interval=[min(ages),max(ages)],date_conflict=len(set(births))>1,licence=value('LicenseShortName'),licence_url=value('LicenseUrl'),
                    attribution=value('Artist'),description=value('ImageDescription'),source_page=info['descriptionurl']))
    # One source per person/day. Original files precede cropped derivatives.
    candidates.sort(key=lambda r:('cropped' in r['title'].lower(),-r['info']['width']*r['info']['height']))
    selected=[];events=set()
    for row in candidates:
        key=(row['subject_id'],row['photo_date'])
        if key in events:rejected.append({k:row[k] for k in ('title','subject_id')}|dict(reason='same_subject_day_duplicate'));continue
        events.add(key);selected.append(row)
    selected=selected[:24]
    write_json(out/'selection.json',selected)
    if not selected:write_json(out/'rejected.json',rejected);raise SystemExit('No eligible source metadata')
    pages=fetch(dict(action='query',pageids='|'.join(str(r['pageid']) for r in selected),prop='imageinfo',iiprop='sha1|size|url'))
    write_json(out/'official-files.json',pages)
    official={int(k):v['imageinfo'][0] for k,v in pages['query']['pages'].items()}
    media=fetch(dict(action='wbgetentities',ids='|'.join('M'+str(r['pageid']) for r in selected),props='claims'))
    write_json(out/'mediainfo.json',media)
    import cv2
    import numpy as np
    from PIL import Image,ImageDraw
    cv2.setNumThreads(2)
    sys.path.insert(0,str(root/'integration'))
    detector=cv2.FaceDetectorYN.create(str(root/'deployment/model-files/face-age/models/yunet/face_detection_yunet_2023mar.onnx'),'',(256,256),.7,.3,100)
    recognizer=cv2.FaceRecognizerSF.create(str(root/'deployment/gan-research/face_recognition_sface_2021dec.onnx'),'')
    old=np.load(root/'deployment/age-quality-final/common-audit-20261124/features.npz',allow_pickle=False)['features']
    records=[];vectors=[];verified_downloads=[]
    for row in selected:
        try:
            spec=official[row['pageid']];url=urlsplit(spec['url']);url=urlunsplit(url._replace(query=''))
            suffix='.jpg' if row['info']['mime']=='image/jpeg' else '.png';path=out/'raw'/(str(row['pageid'])+suffix)
            with urlopen(Request(url,headers={'User-Agent':'FaceAgeCourseResearch/1.0 (source and licence audit)'}),timeout=45) as response:
                data=response.read(25*1024**2+1)
            if len(data)!=spec['size'] or hashlib.sha1(data).hexdigest()!=spec['sha1']:raise ValueError('Official SHA1/size mismatch')
            path.write_bytes(data)
            verified_downloads.append(dict(pageid=row['pageid'],path=str(path),sha256=sha256(path),bytes=len(data)))
            with Image.open(path) as im:im.load();rgb=np.asarray(im.convert('RGB'))
            h,w=rgb.shape[:2];scale=min(1,1600/max(h,w));small=cv2.resize(rgb,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
            detector.setInputSize((small.shape[1],small.shape[0]));_,faces=detector.detect(cv2.cvtColor(small,cv2.COLOR_RGB2BGR))
            if faces is None:raise ValueError('No detected face')
            faces=sorted(faces,key=lambda f:f[2]*f[3],reverse=True);f=faces[0].copy();native_edge=float(min(f[2:4])/scale)
            if len(faces)>1 and faces[1][2]*faces[1][3]>f[2]*f[3]/6:raise ValueError('Multiple prominent faces; identity needs manual selection')
            f[:14]/=scale;points=f[4:14].reshape(5,2);el,er=sorted(points[:2],key=lambda v:v[0]);ml,mr=sorted(points[3:5],key=lambda v:v[0])
            ea=(el+er)*.5;ee=er-el;em=(ml+mr)*.5-ea
            axis=ee-np.flipud(em)*[-1,1];axis/=np.hypot(*axis);axis*=max(np.hypot(*ee)*2,np.hypot(*em)*1.8)
            other=np.flipud(axis)*[-1,1];center=ea+em*.1
            quad=np.stack([center-axis-other,center-axis+other,center+axis+other,center+axis-other]).astype(np.float32)
            matrix=cv2.getPerspectiveTransform(quad,np.array([[0,0],[0,1024],[1024,1024],[1024,0]],np.float32))
            aligned=cv2.warpPerspective(rgb,matrix,(1024,1024),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_REFLECT)
            preview=cv2.resize(aligned,(256,256),interpolation=cv2.INTER_AREA);bgr=cv2.cvtColor(preview,cv2.COLOR_RGB2BGR)
            detector.setInputSize((256,256));_,again=detector.detect(bgr)
            if again is None:raise ValueError('Aligned face not detected')
            af=max(again,key=lambda z:z[2]*z[3]);vector=recognizer.feature(recognizer.alignCrop(bgr,af)).ravel();vector/=np.linalg.norm(vector)+1e-12
            sharp=float(cv2.Laplacian(cv2.cvtColor(preview,cv2.COLOR_RGB2GRAY)[65:215,55:205],cv2.CV_64F).var())
            dest=out/'aligned'/(str(row['pageid'])+'.png');Image.fromarray(aligned).save(dest)
            statements=media.get('entities',{}).get('M'+str(row['pageid']),{})
            depicts=statements.get('statements',statements.get('claims',{})).get('P180',[])
            depicted_ids=[d.get('mainsnak',{}).get('datavalue',{}).get('value',{}).get('id') for d in depicts]
            record={k:v for k,v in row.items() if k!='info'}|dict(original=str(path),original_sha256=sha256(path),official_sha1_verified=True,
                original_size=[w,h],aligned=str(dest),aligned_sha256=sha256(dest),native_face_short_edge=native_edge,sharpness256=sharp,
                metadata_depicts_match=row['subject_id'] in depicted_ids,old_similarity_max=float((old@vector).max()),
                quality_candidate=native_edge>=512 and sharp>=35,training_admitted=False,
                review_required='Check depicted identity/date/source fidelity, retouching and local texture; run common-group audit before any split. No ethnicity inferred.')
            records.append(record);vectors.append(vector)
            print('COMMONS_INTAKE',row['subject_name'],row['age_interval'],round(native_edge),round(sharp,1),record['quality_candidate'],flush=True)
        except Exception as error:
            rejected.append({k:row[k] for k in ('title','subject_id')}|dict(reason=str(error)[:220]));print('COMMONS_REJECT',row['title'],str(error)[:160],flush=True)
            if isinstance(error,HTTPError) and error.code==429:
                write_json(out/'download-blocked.json',dict(status=429,retry_after=error.headers.get('Retry-After'),remaining_not_attempted=True))
                print('STOP: source download rate-limited; no further originals requested',flush=True)
                break
        time.sleep(.5)
    write_json(out/'records.json',records);write_json(out/'rejected.json',rejected)
    if vectors:np.savez(out/'private-similarity-screen.npz',features=np.stack(vectors))
    good=[r for r in records if r['quality_candidate']]
    sheet=Image.new('RGB',(4*300,max(1,(len(records)+3)//4)*340),'white');draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(records):
        with Image.open(r['aligned']) as im:sheet.paste(im.resize((300,300)),((i%4)*300,(i//4)*340))
        draw.text(((i%4)*300+5,(i//4)*340+302),str(r['pageid'])+' age '+str(r['age_interval'])+' edge '+str(round(r['native_face_short_edge'])),fill='black')
    sheet.save(out/'review-contact-sheet.jpg',quality=92)
    write_json(out/'verified-downloads.json',verified_downloads)
    report=dict(downloaded=len(verified_downloads),aligned_records=len(records),quality_candidates=len(good),candidate_subjects=len({r['subject_id'] for r in good}),
        candidate_age_intervals=[r['age_interval'] for r in good],training_admitted=0,prior_similarity_ge_point4=sum(r['old_similarity_max']>=.4 for r in records),
        note='Quarantine only. Public figure photographs do not constitute a balanced Asian cohort. Birth/date claims and source licences are preserved, not assumed ground truth.')
    write_json(out/'audit.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()

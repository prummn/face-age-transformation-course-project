"""Bounded, resumable Commons source intake with date evidence and native-face QC.

Subjects are source metadata, never identified from pixels. Public photos remain
quarantined until source/texture review and source-group isolation are complete.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import date
import hashlib
import html
import json
from pathlib import Path
import re
import time
import unicodedata
from urllib.error import HTTPError
from urllib.parse import urlencode,urlsplit,unquote
from urllib.request import Request,urlopen

from probe_inversion import sha256,write_json

ROOT=Path(__file__).resolve().parents[2]
UA='FaceAgeCourseResearch/1.1 (noncommercial source and quality audit)'
# Selection context is a geographic/source-search stratum, not an ethnicity label.
SEEDS={
 'Japan':['Tetsuko Kuroyanagi','Yoji Yamada','Nobuko Miyamoto','Sayuri Yoshinaga','Miyoko Asada','Chieko Baisho','Hayao Miyazaki','Toshiyuki Nishida','Masahiro Motoki','Syukuro Manabe','Isamu Akasaki','Toshihide Maskawa','Masatoshi Koshiba','Akira Suzuki','Ei-ichi Negishi','Susumu Tonegawa','Tasuku Honjo','Yoshinori Ohsumi'],
 'Korea':['Na Moon-hee','Kim Young-ok','Kim Hae-sook','Youn Yuh-jung','Go Doo-shim','Park Geun-hye','Kim Jung-sook','Kim Yoon-ok','Moon Jae-in','Lee Myung-bak','Ahn Sung-ki','Park Keun-hyong','Choi Bool-am'],
 'Taiwan-China':['Tsai Chin (actress)','Lisa Lu','Brigitte Lin','Sylvia Chang','Sally Yeh','Annette Lu','Chen Chu','Tsai Ing-wen','Lien Chan','James Soong','Chang Chun-hsiung','Tang Fei','Yang Chen-Ning','Tu Youyou','Yuan Longping','Wang Gungwu'],
 'South-Southeast-Asia':['Sharmila Tagore','Waheeda Rehman','Asha Parekh','Helen (actress)','Jaya Bachchan','Hema Malini','Sonia Gandhi','Droupadi Murmu','Narendra Modi','Manmohan Singh','Mallikarjun Kharge','Amitabh Bachchan','Shabana Azmi','Naseeruddin Shah','Lee Hsien Loong','Ho Ching','Goh Chok Tong','Mahathir Mohamad'],
 'General-control':['Judi Dench','Maggie Smith','Helen Mirren','Morgan Freeman','Ian McKellen','Jane Fonda','Vanessa Redgrave','Michael Caine']}


def clean(value):return html.unescape(re.sub('<[^>]+>',' ',value or '')).strip()


def request(url,limit=12*1024**2,attempts=2):
    for attempt in range(attempts):
        try:
            with urlopen(Request(url,headers={'User-Agent':UA}),timeout=22) as response:
                data=response.read(limit+1)
            if len(data)>limit:raise ValueError('Response exceeds transfer budget')
            return data
        except HTTPError as error:
            if error.code==429:raise  # Stop this source, never work around rate limits.
            if attempt==attempts-1:raise
        except OSError:
            if attempt==attempts-1:raise
        time.sleep(1)


def api(host,params,cache,offline=False):
    if cache.exists():return json.loads(cache.read_text())
    if offline:raise FileNotFoundError('No cached response: '+cache.name)
    obj=json.loads(request('https://'+host+'/w/api.php?'+urlencode(dict(format='json',**params))))
    if 'error' in obj:raise ValueError(str(obj['error'])[:250])
    write_json(cache,obj);return obj


def dates(text):
    text=clean(text)
    # Never interpret publication dates explicitly labelled unknown as shooting dates.
    if re.search('unknown|uploaded|date of the article|before|circa|between',text,re.I):return None
    text=re.sub(r'^Taken on\s+','',text,flags=re.I).strip()
    match=re.match(r'^(\d{4})-(\d{2})-(\d{2})(?:[ T]|$)',text)
    if match:
        try:d=date(*map(int,match.groups()))
        except ValueError:return None
        return d,d,'published_day'
    from datetime import datetime
    for fmt in ['%d %B %Y','%d %b %Y','%B %d, %Y']:
        try:d=datetime.strptime(text,fmt).date();return d,d,'published_day'
        except ValueError:pass
    if re.fullmatch(r'\d{4}',text):return date(int(text),1,1),date(int(text),12,31),'published_year'
    return None


def file_mime(info):
    # Older imageinfo caches omitted iiprop=mime. Extension only screens the
    # candidate; downloaded bytes must still decode and match the official hash.
    if info.get('mime'):return info['mime']
    suffix=Path(unquote(urlsplit(info.get('url','')).path)).suffix.lower()
    return {'.jpg':'image/jpeg','.jpeg':'image/jpeg','.png':'image/png'}.get(suffix)


def title_key(text):
    return ''.join(c for c in unicodedata.normalize('NFKD',text.split(' (')[0].casefold()) if not unicodedata.combining(c))


def plan(out,offline=False,cached_searches=()):
    (out/'metadata').mkdir(parents=True,exist_ok=True)
    people=[dict(title=n,selection_context=k) for k,names in SEEDS.items() for n in names]
    write_json(out/'source-search-seeds.json',people)
    entities={};biography_errors=[];fallback={}
    for path in [ROOT/'deployment/age-quality-final/commons-biographies-20261124.json',*sorted((out/'metadata').glob('biographies*.json'))]:
        if path.exists():entities.update(json.loads(path.read_text()).get('entities',{}))
    known={e.get('sitelinks',{}).get('enwiki',{}).get('title') for e in entities.values()}
    missing=[r for r in people if r['title'] not in known]
    for i in range(0,len(missing),4):
        batch=missing[i:i+4];stamp=hashlib.sha256('|'.join(r['title'] for r in batch).encode()).hexdigest()[:12]
        try:
            cache=out/'metadata'/f'wikipedia-birth-{stamp}.json'
            d=api('en.wikipedia.org',dict(action='query',titles='|'.join(r['title'] for r in batch),prop='pageprops|revisions',ppprop='wikibase_item',rvprop='ids|content',rvslots='main',redirects=1),cache,offline)
            redirects={r['from']:r['to'] for r in d.get('query',{}).get('redirects',[])}
            for page in d.get('query',{}).get('pages',{}).values():
                q=page.get('pageprops',{}).get('wikibase_item');revs=page.get('revisions',[])
                if not q or not revs:continue
                text=revs[0]['slots']['main']['*'];match=re.search(r'\|\s*birth_date\s*=([^\n]+)',text,re.I)
                if not match:continue
                template=re.search(r'\{\{\s*(?:birth date(?: and age)?|bda)\s*\|([^}]+)\}\}',match[1],re.I)
                if not template:continue
                numbers=[int(n.strip()) for n in template[1].split('|') if re.fullmatch(r'\d+',n.strip())]
                if len(numbers)!=3:continue
                birth=date(*numbers).isoformat()
                for row in batch:
                    if redirects.get(row['title'],row['title'])==page['title']:
                        fallback[row['title']]=dict(subject_id=q,source_label=page['title'],birth_dates=[birth],birth_evidence=dict(kind='Wikipedia infobox date template; source claim',page_title=page['title'],revision_id=revs[0].get('revid'),template=match[1].strip(),cached_response=f'metadata/wikipedia-birth-{stamp}.json',cached_sha256=sha256(cache)))
            print('BIOGRAPHIES_BATCH',i//4+1,'/',(len(missing)+3)//4,flush=True)
        except HTTPError as error:
            if error.code==429:raise
            biography_errors.append(dict(titles=[r['title'] for r in batch],reason=str(error)[:180]))
        except Exception as error:biography_errors.append(dict(titles=[r['title'] for r in batch],reason=str(error)[:180]));print('BIOGRAPHIES_UNAVAILABLE',i//4+1,flush=True)
    write_json(out/'biography-errors.json',biography_errors)
    subjects=[]
    for row in people:
        if row['title'] in fallback:
            subjects.append(dict(row,**fallback[row['title']]));continue
        entity=next((e for e in entities.values() if e.get('sitelinks',{}).get('enwiki',{}).get('title')==row['title']),None)
        if not entity:continue
        claims=[r for r in entity.get('claims',{}).get('P569',[]) if r.get('rank')!='deprecated']
        claims=[r for r in claims if r.get('rank')=='preferred'] or claims
        births=[]
        for r in claims:
            v=r.get('mainsnak',{}).get('datavalue',{}).get('value',{})
            if v.get('precision',0)>=11:
                try:births.append(date.fromisoformat(v['time'][1:11]))
                except ValueError:pass
        if not births:continue
        subjects.append(dict(row,subject_id=entity['id'],source_label=entity.get('labels',{}).get('en',{}).get('value',row['title']),birth_dates=sorted({b.isoformat() for b in births}),birth_evidence=dict(kind='Wikidata P569 date claim',entity_id=entity['id'])))
    write_json(out/'subjects.json',subjects)
    cached_pages={}
    for path in cached_searches:
        for row in json.loads(path.read_text()):
            if row.get('name'):cached_pages.setdefault(title_key(row['name']),[]).extend(row.get('pages',[]))
    def search(row):
        q=row['title'].split(' (')[0]
        cache=out/'metadata'/(row['subject_id']+'-search.json')
        extra=cached_pages.get(title_key(row['title']),[])
        if offline and not cache.exists() and extra:d={}
        else:d=api('commons.wikimedia.org',dict(action='query',generator='search',gsrsearch='intitle:"'+q+'" filew:>1000 fileh:>1000',gsrnamespace=6,gsrlimit=18,prop='imageinfo',iiprop='url|size|sha1|mime|extmetadata',iiurlwidth=1400),cache,offline)
        if extra:
            pages=d.setdefault('query',{}).setdefault('pages',{})
            for p in extra:pages.setdefault(str(p['pageid']),p)
        return row,d
    candidates=[];seen=set();errors=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks=[pool.submit(search,r) for r in subjects]
        for future in as_completed(tasks):
            try:row,d=future.result()
            except HTTPError as error:
                if error.code==429:
                    write_json(out/'rate-limit-stop.json',dict(stage='metadata_search',status=429,retry_after=error.headers.get('Retry-After'),source='commons.wikimedia.org'))
                    for t in tasks:t.cancel()
                    raise
                errors.append(str(error)[:160]);continue
            except Exception as error:errors.append(str(error)[:160]);continue
            group=[]
            for p in d.get('query',{}).get('pages',{}).values():
                info=p.get('imageinfo',[{}])[0];meta=info.get('extmetadata',{});v=lambda key:clean(meta.get(key,{}).get('value',''))
                if p['pageid'] in seen:continue
                if file_mime(info) not in ('image/jpeg','image/png') or info.get('size',0)>25*1024**2:continue
                info['mime']=file_mime(info)
                if not re.fullmatch(r'CC BY(?:-SA)? [1234]\.0|CC0|Public domain|KOGL Type 1',v('LicenseShortName')):continue
                if re.search('signature|poster|painting|sculpture|statue|wax figure|screenshot|screen shot|video frame|mural',p['title']+' '+v('ImageDescription'),re.I):continue
                dt=dates(v('DateTimeOriginal'))
                if dt is None:continue
                lower,upper,precision=dt
                if lower>date.today():continue
                # A scan's EXIF timestamp is not the original portrait's date.
                title_years=[int(y) for y in re.findall(r'(?<!\d)(?:19|20)\d{2}(?!\d)',p['title'])]
                if title_years and lower.year>min(title_years)+2:continue
                def age(d,b):return d.year-b.year-((d.month,d.day)<(b.month,b.day))
                ages=[age(day,date.fromisoformat(b)) for b in row['birth_dates'] for day in [lower,upper]]
                if min(ages)<55 or max(ages)>100 or max(ages)-min(ages)>2:continue
                group.append(dict(row,pageid=p['pageid'],file_title=p['title'],photo_date_claim=v('DateTimeOriginal'),photo_date_interval=[lower.isoformat(),upper.isoformat()],date_precision=precision,age_interval=[min(ages),max(ages)],birth_date_conflict=len(row['birth_dates'])>1,licence=v('LicenseShortName'),licence_url=v('LicenseUrl'),author=v('Artist'),description=v('ImageDescription'),source_claim=v('Credit'),info=info))
            # Prefer portrait-shaped derivatives, distinct dates, and larger face potential.
            group.sort(key=lambda r:(not bool(re.search('portrait|cropped|headshot',r['file_title'],re.I)),-r['info']['width']*r['info']['height']))
            days=set()
            for r in group:
                if r['photo_date_interval'][0] in days:continue
                days.add(r['photo_date_interval'][0]);seen.add(r['pageid']);candidates.append(r)
                if len(days)>=3:break
            print('SOURCE_SEARCH',row['subject_id'],row['title'],'eligible',min(len(days),3),flush=True)
    # Freeze before looking at face quality or model output.
    candidates.sort(key=lambda r:(r['subject_id'],r['photo_date_interval'][0],r['pageid']))
    write_json(out/'candidates.json',candidates);write_json(out/'search-errors.json',errors)
    write_json(out/'plan-audit.json',dict(offline_cache_only=offline,cached_source_inputs=[dict(path=str(p),sha256=sha256(p)) for p in cached_searches],seed_subjects=len(people),resolved_subjects=len(subjects),candidate_files=len(candidates),candidate_subjects=len({r['subject_id'] for r in candidates}),search_errors=len(errors),max_originals=120,max_bytes=1024**3,training_admitted=0,source_note='Names/countries are source-search context; no ethnicity inferred. Dates and DOB are metadata claims, not independently verified age.'))
    print('DOCUMENTED_PLAN',len(candidates),'files',len({r['subject_id'] for r in candidates}),'source subjects',flush=True)


def collect(out):
    if (out/'rate-limit-stop.json').exists():
        raise SystemExit('Source rate-limit stop is recorded. Do not retry until its access condition is resolved; offline plan remains available.')
    import cv2
    import numpy as np
    from PIL import Image
    from io import BytesIO
    cv2.setNumThreads(2)
    for folder in ['previews','raw','aligned','records','screen']: (out/folder).mkdir(exist_ok=True)
    detector=cv2.FaceDetectorYN.create(str(ROOT/'deployment/model-files/face-age/models/yunet/face_detection_yunet_2023mar.onnx'),'',(1400,1400),.7,.3,100)
    rows=json.loads((out/'candidates.json').read_text());used=sum(p.stat().st_size for p in (out/'raw').iterdir());attempted=len(list((out/'raw').iterdir()))
    def detect(rgb):
        h,w=rgb.shape[:2];scale=min(1,1400/max(h,w));small=cv2.resize(rgb,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA);detector.setInputSize((small.shape[1],small.shape[0]));_,faces=detector.detect(cv2.cvtColor(small,cv2.COLOR_RGB2BGR))
        if faces is None:raise ValueError('No detected face')
        faces=sorted(faces,key=lambda f:f[2]*f[3],reverse=True)
        if len(faces)>1 and faces[1][2]*faces[1][3]>faces[0][2]*faces[0][3]/6:raise ValueError('Multiple prominent faces; no automatic identity selection')
        f=faces[0].copy();f[:14]/=scale;return f
    for row in rows:
        key=str(row['pageid']);record=out/'records'/(key+'.json');screen=out/'screen'/(key+'.json')
        if record.exists() or screen.exists():continue
        if attempted>=120 or used>=1024**3:print('TRANSFER_BUDGET_REACHED',flush=True);break
        try:
            info=row['info'];preview=out/'previews'/(key+'.jpg')
            if not all(info.get(k) for k in ['url','sha1','size']):raise ValueError('Cached metadata lacks official hash/size; refresh source after access is restored')
            if used+info['size']>1024**3:print('TRANSFER_BUDGET_REACHED',flush=True);break
            if not preview.exists():preview.write_bytes(request(info.get('thumburl',info['url']),8*1024**2))
            with Image.open(preview) as im:im.load();rgb=np.asarray(im.convert('RGB'))
            f=detect(rgb);estimate=min(f[2:4])*min(info['width']/rgb.shape[1],info['height']/rgb.shape[0])
            if estimate<512:raise ValueError('Preview-estimated native face below512: '+str(round(estimate)))
            attempted+=1;suffix='.png' if info['mime']=='image/png' else '.jpg';raw=out/'raw'/(key+suffix)
            if raw.exists():data=raw.read_bytes()
            else:data=request(info['url'],25*1024**2);used+=len(data)
            if len(data)!=info['size'] or hashlib.sha1(data).hexdigest()!=info['sha1']:raise ValueError('Official original SHA1/size differs')
            if not raw.exists():raw.write_bytes(data)
            with Image.open(BytesIO(data)) as im:im.load();rgb=np.asarray(im.convert('RGB'))
            f=detect(rgb);edge=float(min(f[2:4]));points=f[4:14].reshape(5,2);el,er=sorted(points[:2],key=lambda v:v[0]);ml,mr=sorted(points[3:5],key=lambda v:v[0]);ea=(el+er)*.5;ee=er-el;em=(ml+mr)*.5-ea
            axis=ee-np.flipud(em)*[-1,1];axis/=np.hypot(*axis);axis*=max(np.hypot(*ee)*2,np.hypot(*em)*1.8);other=np.flipud(axis)*[-1,1];center=ea+em*.1
            quad=np.stack([center-axis-other,center-axis+other,center+axis+other,center+axis-other]).astype(np.float32);matrix=cv2.getPerspectiveTransform(quad,np.array([[0,0],[0,1024],[1024,1024],[1024,0]],np.float32));aligned=cv2.warpPerspective(rgb,matrix,(1024,1024),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_REFLECT)
            path=out/'aligned'/(key+'.png');Image.fromarray(aligned).save(path)
            # Eye/nose geometry only detects pose risk; not a calibrated yaw estimate.
            eye_gap=float(np.hypot(*(er-el)));roll=abs(float(np.degrees(np.arctan2(*(er-el)[::-1]))));nose=points[2];asym=abs(float(np.linalg.norm(nose-el)-np.linalg.norm(nose-er)))/(eye_gap+1e-9)
            small=cv2.resize(aligned,(256,256),interpolation=cv2.INTER_AREA);g=cv2.cvtColor(small,cv2.COLOR_RGB2GRAY);sharp=float(cv2.Laplacian(g[65:215,55:205],cv2.CV_64F).var())
            accepted=edge>=512 and roll<=15 and asym<=.35 and sharp>=35
            fields={k:v for k,v in row.items() if k!='info'}
            result=dict(fields,original=str(raw),original_sha256=sha256(raw),official_sha1=info['sha1'],official_size_verified=True,aligned=str(path),aligned_sha256=sha256(path),source_url=info['descriptionurl'],original_url=info['url'],original_size=[rgb.shape[1],rgb.shape[0]],native_face_short_edge=edge,roll_degrees=roll,nose_eye_asymmetry=asym,sharpness256=sharp,automatic_quality_candidate=accepted,training_admitted=False,manual_review='pending',same_person_source_claim=True)
            write_json(record,result);print('DOCUMENTED_ORIGINAL',row['subject_id'],key,row['age_interval'],round(edge),accepted,flush=True)
        except HTTPError as error:
            write_json(screen,dict(pageid=row['pageid'],subject_id=row['subject_id'],reason=str(error)[:180]))
            if error.code==429:write_json(out/'rate-limit-stop.json',dict(status=429,retry_after=error.headers.get('Retry-After')));break
        except Exception as error:
            write_json(screen,dict(pageid=row['pageid'],subject_id=row['subject_id'],reason=str(error)[:180]));print('DOCUMENTED_REJECT',key,str(error)[:120],flush=True)
        time.sleep(.25)
    records=[json.loads(p.read_text()) for p in sorted((out/'records').glob('*.json'))];write_json(out/'records.json',records)
    write_json(out/'collection-audit.json',dict(original_files=len(list((out/'raw').iterdir())),original_bytes=sum(p.stat().st_size for p in (out/'raw').iterdir()),aligned_records=len(records),automatic_quality_candidates=sum(r['automatic_quality_candidate'] for r in records),candidate_subjects=len({r['subject_id'] for r in records if r['automatic_quality_candidate']}),screen_rejections=len(list((out/'screen').iterdir())),training_admitted=0,manual_review_pending=True))
    print('DOCUMENTED_COLLECTION_COMPLETE',len(records),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','collect']);p.add_argument('output',type=Path);p.add_argument('--offline',action='store_true',help='Plan only from existing source caches; no network requests');p.add_argument('--cached-search',action='append',type=Path,default=[]);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if a.offline and a.action!='plan':p.error('--offline is only valid for plan')
    if a.action=='plan':plan(out,a.offline,a.cached_search)
    else:collect(out)

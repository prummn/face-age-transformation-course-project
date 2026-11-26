"""Verify deterministic real CUDA API fitting on one public photo, twice."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[2]


def main():
    from PIL import Image
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--url',default='http://127.0.0.1:65512');args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    def call(path,data=None,method=None):
        req=Request(args.url+path,data=json.dumps(data).encode() if data is not None else None,
            method=method,headers={'Content-Type':'application/json'})
        with urlopen(req,timeout=120) as r:return json.load(r)
    health=call('/api/health')
    if not health['ready'] or health.get('numerics')!='stable' or 'original SAM' not in health['model']:
        raise ValueError('Requires actual ready stable original_delta backend')
    photo=ROOT/'deployment/age-quality-final/expanded-ffhq/native1024/64376.png'
    body=dict(image='data:image/png;base64,'+base64.b64encode(photo.read_bytes()).decode(),
        face=dict(x=.1,y=.1,w=.8,h=.85,id='public-stability-check'))
    report=dict(health=health,source_sha256=hashlib.sha256(photo.read_bytes()).hexdigest(),repeats=[])
    for repeat in range(2):
        session=None
        try:
            tick=time.monotonic();prepared=call('/api/prepare',body);session=prepared['session_id']
            prepare_seconds=time.monotonic()-tick
            tick=time.monotonic();result=call('/api/transform',dict(session_id=session,target_age=75,
                request_id='public-stable-'+str(repeat),preserve_color=True,enhance=False))
            row=dict(prepare_seconds=prepare_seconds,request_seconds=time.monotonic()-tick,
                reference_age=prepared['age']['age'],meta=result['meta'])
            for field in ['image','raw_image']:
                raw=base64.b64decode(result[field].split(',',1)[1])
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
                    if image.format!='PNG' or image.size!=(1024,1024):raise ValueError('Invalid1024 PNG')
                (args.out/(str(repeat)+'-'+field+'.png')).write_bytes(raw)
                row[field+'_sha256']=hashlib.sha256(raw).hexdigest()
            report['repeats'].append(row);print('STABLE_API_REPEAT',repeat,row['request_seconds'],flush=True)
        finally:
            if session is not None:call('/api/sessions/'+session,method='DELETE')
        try:call('/api/transform',dict(session_id=session,target_age=75))
        except HTTPError as error:
            if error.code!=410:raise
            report['post_delete_status']=410
        else:raise ValueError('Deleted session still accepted')
    a,b=report['repeats']
    report['raw_png_exact']=a['raw_image_sha256']==b['raw_image_sha256']
    report['preserved_png_exact']=a['image_sha256']==b['image_sha256']
    report['source_reference_age_exact']=a['reference_age']==b['reference_age']
    report['passed']=all(report[k] for k in ['raw_png_exact','preserved_png_exact','source_reference_age_exact'])
    (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if not report['passed']:raise ValueError('API fitting is not repeatable')
    print('STABLE_INFERENCE_PASSED',flush=True)


if __name__=='__main__':main()

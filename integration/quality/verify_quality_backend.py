"""Actual quality-mode API inference, PNG checks and cancellation on a GPU task."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import time
from urllib.error import HTTPError,URLError
from urllib.request import Request,urlopen


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--subject',default='uploaded');p.add_argument('--url',default='http://127.0.0.1:8765');a=p.parse_args();run=a.run
    from urllib.parse import urlsplit
    endpoint=urlsplit(a.url)
    if endpoint.scheme!='http' or endpoint.hostname not in ('127.0.0.1','localhost') or endpoint.path not in ('','/') or endpoint.query or endpoint.fragment or endpoint.username or endpoint.password:raise ValueError('Loopback HTTP backend URL required')
    import cv2
    import numpy as np
    from PIL import Image
    def call(path,body=None,method=None):
        req=Request(a.url.rstrip('/')+path,data=json.dumps(body).encode() if body is not None else None,
            headers={'Content-Type':'application/json'},method=method)
        try:
            with urlopen(req,timeout=150) as response:return response.status,json.load(response)
        except HTTPError as error:return error.code,json.load(error)
    deadline=time.monotonic()+120
    while True:
        try:
            status,health=call('/api/health')
            if health.get('ready'):break
            if health.get('status')=='failed':raise RuntimeError(health)
        except URLError:pass
        if time.monotonic()>deadline:raise RuntimeError('Quality backend did not become ready')
        time.sleep(.5)
    assert 'W+' in health['model']
    row=next(x for x in json.loads((run/'latent-inputs.json').read_text()) if x['name']==a.subject)
    filename=row.get('reference_path',row['image_path']);raw=Path(filename).read_bytes()
    rgb=np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'));h,w=rgb.shape[:2]
    import os
    detector=cv2.FaceDetectorYN.create(os.environ['FACE_AGE_BASE']+'/models/yunet/face_detection_yunet_2023mar.onnx','',(w,h),.6,.3,100)
    _,faces=detector.detect(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR));assert faces is not None
    f=max(faces,key=lambda x:x[2]*x[3]);left=max(0,float(f[0]));top=max(0,float(f[1]));right=min(w,float(f[0]+f[2]));bottom=min(h,float(f[1]+f[3]))
    body=dict(image='data:image/png;base64,'+base64.b64encode(raw).decode(),face=dict(x=left/w,y=top/h,w=(right-left)/w,h=(bottom-top)/h))
    status,prepared=call('/api/prepare',body);assert status==200,prepared
    out=run/'api-validation';out.mkdir(exist_ok=True)
    report=dict(health=health,source_condition=prepared['age']['age'],outputs=[],
                protocol='Actual prepare/detection/alignment and transform; input is re-aligned by API, not pixel-identical to offline aligned benchmark.')
    session=prepared['session_id']
    for condition,age in [('same',prepared['age']['age'])]+[(str(age),age) for age in [20,40,60,75]]:
        started=time.monotonic();status,result=call('/api/transform',dict(session_id=session,target_age=age,preserve_color=age==75))
        assert status==200,result
        for field in ['image','raw_image']:
            content=base64.b64decode(result[field].split(',',1)[1])
            with Image.open(io.BytesIO(content)) as image:image.load();assert image.size==(1024,1024) and image.format=='PNG'
            (out/(condition+'_'+field+'.png')).write_bytes(content)
        report['outputs'].append(dict(condition=condition,request_seconds=time.monotonic()-started,meta=result['meta']))
        print('QUALITY_API',condition,report['outputs'][-1]['request_seconds'],flush=True)
    assert call('/api/sessions/'+session,method='DELETE')[0]==200
    assert call('/api/transform',dict(session_id=session,target_age=60))[0]==410
    _,prepared=call('/api/prepare',body);session=prepared['session_id']
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending=pool.submit(call,'/api/transform',dict(session_id=session,target_age=75))
        time.sleep(.25);assert call('/api/sessions/'+session,method='DELETE')[0]==200
        status,result=pending.result();assert status==410,(status,result)
    report['cancelled_inflight_status']=status
    _,health=call('/api/health');assert health['active_sessions']==0
    report['active_sessions_after']=0
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('REAL_QUALITY_API_VERIFIED',flush=True)


if __name__=='__main__':main()

"""Compare live preprocessing on the same public photo; never read user sessions.

Uses the already running API/GPU. The low-resolution arm is a synthetic resize,
not a measurement of an actual webcam. Every session created here is deleted.
"""
import argparse
import base64
import io
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]


def detect_in_browser():
    from playwright.sync_api import sync_playwright
    payload = json.load(sys.stdin)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path='/tmp/face-age-chrome/chrome',
            args=['--no-sandbox', '--disable-dev-shm-usage'])
        page = browser.new_page()
        page.goto(payload['url'], wait_until='domcontentloaded')
        face = page.evaluate('''async (url) => {
            await initDetector();
            const image = new Image(); image.src = url; await image.decode();
            const c = document.createElement('canvas'); c.width=image.width; c.height=image.height;
            c.getContext('2d').drawImage(image,0,0);
            const faces=detectFaces(c).sort((a,b)=>b.w*b.h-a.w*a.h);
            if(!faces.length)throw new Error('No browser face');
            const {x,y,w,h}=faces[0]; return {x,y,w,h,id:'public-diagnostic'};
        }''', payload['image'])
        browser.close()
    print(json.dumps(face))


def main():
    import numpy as np
    from PIL import Image
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:65512')
    p.add_argument('--photo', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--target', type=int, help='Optional real inference for every arm')
    p.add_argument('--arms', nargs='+', choices=['full','legacy_crop','context_crop','synthetic_640'])
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    original = Image.open(args.photo).convert('RGB')

    def encode(im):
        b = io.BytesIO(); im.save(b, format='PNG')
        return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()

    def api(path, payload=None, method=None):
        body = json.dumps(payload).encode() if payload is not None else None
        req = Request(args.url+path, data=body, method=method,
            headers={'Content-Type':'application/json'})
        with urlopen(req, timeout=120) as response:
            return json.load(response)

    def save(url, path):
        path.write_bytes(base64.b64decode(url.split(',',1)[1]))
        with Image.open(path) as im:
            im.load(); assert im.format == 'PNG'
            return np.asarray(im.convert('RGB')).astype(float)

    detected = subprocess.run([str(ROOT/'deployment/control-venv/bin/python'), __file__, '--detect'],
        input=json.dumps({'url':args.url, 'image':encode(original)}), text=True,
        capture_output=True, check=True, timeout=120)
    face = json.loads(detected.stdout)
    w,h = original.size
    x,y,fw,fh = face['x']*w,face['y']*h,face['w']*w,face['h']*h
    import math
    crop = (max(0,math.floor(x-.22*fw)), max(0,math.floor(y-.36*fh)),
        min(w,math.ceil(x+1.22*fw)), min(h,math.ceil(y+1.18*fh)))
    left,top,right,bottom = crop
    box = dict(face, x=(x-left)/(right-left), y=(y-top)/(bottom-top),
        w=fw/(right-left), h=fh/(bottom-top))
    low = original.resize((640,round(h*640/w)), Image.Resampling.LANCZOS)
    side,cx,cy = 2.2*max(fw,fh),x+fw/2,y+.4*fh
    context = (max(0,math.floor(cx-side/2)),max(0,math.floor(cy-side/2)),
        min(w,math.ceil(cx+side/2)),min(h,math.ceil(cy+side/2)))
    cl,ct,cr,cb = context
    context_box = dict(face,x=(x-cl)/(cr-cl),y=(y-ct)/(cb-ct),w=fw/(cr-cl),h=fh/(cb-ct))
    arms = [('full', original, face, (0,0)),
        ('legacy_crop', original.crop(crop), box, (left,top)),
        ('context_crop', original.crop(context), context_box, (cl,ct)),
        ('synthetic_640', low, face, (0,0))]
    if args.arms: arms = [arm for arm in arms if arm[0] in args.arms]
    report = dict(photo=str(args.photo.resolve()), face=face, legacy_crop=crop,
        health=api('/api/health'), target=args.target, context_crop=context, arms=[],
        note='Public same-photo controlled comparison; synthetic resize is not an actual webcam.')
    refs = {}
    for name,image,selected,offset in arms:
        session = None
        try:
            started = time.monotonic()
            data = api('/api/prepare', {'image':encode(image), 'face':selected})
            session = data['session_id']
            row = dict(name=name, source_size=image.size, age=data['age']['age'],
                prepare_seconds=time.monotonic()-started, transform=data['transform'], meta=data['meta'])
            save(data['aligned_image'],args.out/(name+'-aligned.png'))
            refs[name] = save(data['reference_image'],args.out/(name+'-reference.png'))
            a,b,c,d,e,f = data['transform']['inverse']
            yy,xx = np.mgrid[0:256,0:256]
            sx,sy = a*xx+c*yy+e,b*xx+d*yy+f
            row['reflected_fraction'] = float(((sx<0)|(sy<0)|(sx>=image.width)|(sy>=image.height)).mean())
            if args.target is not None:
                started = time.monotonic()
                result = api('/api/transform',dict(session_id=session,target_age=args.target,
                    request_id='public-input-diagnostic-'+name,preserve_color=True,enhance=False))
                row['request_seconds']=time.monotonic()-started;row['generation']=result['meta']
                for field,suffix in [('raw_image','raw'),('image','color')]:
                    pixels=save(result[field],args.out/(name+'-'+suffix+'.png'))
                    assert pixels.shape==(1024,1024,3)
            report['arms'].append(row)
            print(json.dumps(row),flush=True)
        finally:
            if session is not None: api('/api/sessions/'+session,method='DELETE')
    if 'full' in refs:
        for row in report['arms']:
            row['reference_mae_vs_full'] = float(np.abs(refs[row['name']]-refs['full']).mean())
    (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    if '--detect' in sys.argv: detect_in_browser()
    else: main()

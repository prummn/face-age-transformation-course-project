"""Real API acceptance using the project's public MediaPipe portrait fixture."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import time

import requests
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:65512')
    parser.add_argument('--output', type=Path, default=ROOT / 'deployment/validation')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fixture = ROOT / 'tmp/qa_portrait.jpg'
    source = fixture.read_bytes()
    payload = dict(image='data:image/jpeg;base64,' + base64.b64encode(source).decode(),
                   face=dict(x=.28, y=.02, w=.40, h=.36, id='public-test-portrait'))
    report = dict(fixture=dict(url='https://storage.googleapis.com/mediapipe-assets/portrait.jpg',
                               sha256=hashlib.sha256(source).hexdigest()), base=args.base,
                  scope='Real SAM inference; public test fixture outputs explicitly saved for deployment acceptance',
                  health=requests.get(args.base+'/api/health', timeout=10).json(), outputs=[])
    assert report['health']['ready'] is True

    def prepare():
        started = time.monotonic()
        r = requests.post(args.base+'/api/prepare', json=payload, timeout=60)
        r.raise_for_status()
        return r.json(), time.monotonic()-started

    def save_image(value, name, expected):
        raw = base64.b64decode(value.split(',', 1)[1], validate=True)
        with Image.open(io.BytesIO(raw)) as img:
            img.load()
            assert img.format == 'PNG' and img.size == expected and img.mode == 'RGB'
        (args.output / name).write_bytes(raw)
        return dict(file=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), size=expected)

    prepared, elapsed = prepare()
    sid = prepared['session_id']
    try:
        report['prepare'] = dict(reference_age=prepared['age'], elapsed_seconds=elapsed,
                                  meta=prepared['meta'], transform=prepared['transform'])
        save_image(prepared['aligned_image'], 'aligned256.png', (256, 256))
        save_image(prepared['reference_image'], 'reference1024.png', (1024, 1024))
        for age in dict.fromkeys([prepared['age']['age'], 20, 40, 60, 75]):
            started = time.monotonic()
            r = requests.post(args.base+'/api/transform', json=dict(session_id=sid, target_age=age), timeout=150)
            r.raise_for_status()
            full_elapsed = time.monotonic()-started
            data = r.json()
            assert data['width'] == data['height'] == 1024
            raw = save_image(data['raw_image'], f'age{age}-raw.png', (1024, 1024))
            color = save_image(data['image'], f'age{age}-color.png', (1024, 1024))
            report['outputs'].append(dict(target_age=age, request_seconds=full_elapsed,
                meta=data['meta'], raw=raw, color=color))
            print(age, 'model', data['meta']['elapsed_seconds'], 'request', round(full_elapsed,3), flush=True)
        # The off switch must return the unmodified native result.
        r = requests.post(args.base+'/api/transform', json=dict(session_id=sid,target_age=40,preserve_color=False),timeout=150)
        r.raise_for_status()
        off = r.json()
        assert off['image'] == off['raw_image']
        report['preserve_color_false'] = dict(passed=True, meta=off['meta'])
    finally:
        requests.delete(args.base+'/api/sessions/'+sid, timeout=10).raise_for_status()
    r = requests.post(args.base+'/api/transform', json=dict(session_id=sid,target_age=40),timeout=20)
    assert r.status_code == 410
    report['post_delete_status'] = r.status_code
    # Cancel while a real request is in progress; no fake engine or replay.
    prepared, _ = prepare()
    sid = prepared['session_id']
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(requests.post, args.base+'/api/transform',
                             json=dict(session_id=sid,target_age=75), timeout=150)
        time.sleep(.2)
        deletion = requests.delete(args.base+'/api/sessions/'+sid, timeout=10)
        deletion.raise_for_status()
        cancelled = future.result()
    assert cancelled.status_code == 410, cancelled.status_code
    report['inflight_delete'] = dict(delete_status=deletion.status_code, transform_status=cancelled.status_code)
    # Compact overview only; original native PNGs remain available separately.
    rows = report['outputs']
    sheet = Image.new('RGB', (256 * (len(rows)+1), 2*284), 'white')
    draw = ImageDraw.Draw(sheet)
    for row_index, kind in enumerate(['raw', 'color']):
        sheet.paste(Image.open(args.output/'reference1024.png').resize((256,256)), (0,row_index*284+28))
        draw.text((8,row_index*284+8), 'Uploaded original (resampled)', fill='black')
        for col, row in enumerate(rows,1):
            name = row[kind]['file']
            sheet.paste(Image.open(args.output/name).resize((256,256)), (col*256,row_index*284+28))
            draw.text((col*256+8,row_index*284+8), f"{row['target_age']}y / {kind}", fill='black')
    sheet.save(args.output/'overview.jpg',quality=92)
    report['passed'] = True
    (args.output/'api-report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print('REAL API acceptance passed:', args.output)


if __name__ == '__main__':
    main()

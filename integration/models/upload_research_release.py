"""Resume uploads into an existing draft Release; verify GitHub SHA256 digests.

Does not publish the draft, edit previous releases, or upload arbitrary files.
Requires gh authentication. Bounded independent uploads, no GPU involved.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time

REPO='prummn/face-age-transformation-course-project'
TAG='research-mobei-2026-11-17'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def api(path):return json.loads(subprocess.check_output(['gh','api',path],text=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--manifest-name',default='research-20261117.json')
    a=p.parse_args()
    if not 1<=a.workers<=4:p.error('workers must be1..4')
    stage=a.stage.resolve();folder=stage/'assets'
    if Path(a.manifest_name).name!=a.manifest_name:raise ValueError('Manifest name must be a basename')
    manifest=json.loads((folder/a.manifest_name).read_text());tag=manifest['release_tag']
    if manifest.get('repository')!=REPO or not tag or '/' in tag or '\\' in tag:raise ValueError('Unexpected repository or release tag')
    expected={}
    for item in manifest['items']:
        for spec in item['assets']:
            if spec['tag']==tag:expected[spec['name']]=spec
    for name in [a.manifest_name,'SHA256SUMS']:
        path=folder/name;expected[name]=dict(name=name,bytes=path.stat().st_size,sha256=sha(path))
    for spec in expected.values():
        path=folder/spec['name'];assert path.is_file() and path.stat().st_size==spec['bytes'] and sha(path)==spec['sha256']
    releases=api(f'repos/{REPO}/releases?per_page=100')
    release=next(x for x in releases if x['tag_name']==tag)
    if not release['draft']:raise RuntimeError('Only this unpublished draft may be populated')
    existing={x['name']:x for x in release['assets']}
    def matches(asset,spec):return asset.get('state')=='uploaded' and asset['size']==spec['bytes'] and asset.get('digest')=='sha256:'+spec['sha256']
    for name,asset in existing.items():
        if name not in expected:raise RuntimeError('Unexpected asset in draft: '+name)
        if not matches(asset,expected[name]):raise RuntimeError('Existing asset differs; inspect before replacing: '+name)
    pending=[x for n,x in expected.items() if n not in existing];lock=threading.Lock();receipts=[]
    print('UPLOAD_START',len(pending),'pending',len(existing),'already_verified',flush=True)
    def upload(spec):
        tick=time.monotonic();path=folder/spec['name']
        for attempt in range(3):
            r=subprocess.run(['gh','release','upload',tag,str(path),'--repo',REPO],capture_output=True,text=True,timeout=600)
            assets=api(f'repos/{REPO}/releases/{release["id"]}/assets?per_page=100')
            asset=next((x for x in assets if x['name']==spec['name']),None)
            if asset and matches(asset,spec):break
            if asset:
                # Only remove the named incomplete asset created by this draft
                # upload attempt; never replace a completed different artifact.
                if asset.get('state')=='uploaded' and asset['size']>0:raise RuntimeError('Server digest mismatch: '+spec['name'])
                subprocess.run(['gh','api',f'repos/{REPO}/releases/assets/{asset["id"]}','--method','DELETE'],check=True,capture_output=True)
            if attempt==2:raise RuntimeError('Upload failed: '+spec['name']+' '+r.stderr[-400:])
            time.sleep(2)
        receipt=dict(name=spec['name'],size=asset['size'],digest=asset['digest'],asset_id=asset['id'],seconds=time.monotonic()-tick)
        with lock:
            receipts.append(receipt)
            with (stage/'upload.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')
            print('UPLOADED',len(receipts),'/',len(pending),spec['name'],round(receipt['seconds'],1),'s',flush=True)
        return receipt
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(upload,x) for x in pending]):future.result()
    actual=api(f'repos/{REPO}/releases/{release["id"]}/assets?per_page=100')
    assert {x['name'] for x in actual}==set(expected)
    assert all(matches(x,expected[x['name']]) for x in actual)
    report=dict(release_id=release['id'],tag=tag,draft=True,assets=len(actual),bytes=sum(x['size'] for x in actual),
                all_github_sha256_match=True,checks=[dict(name=x['name'],size=x['size'],digest=x['digest'],asset_id=x['id']) for x in actual])
    (stage/'upload-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('UPLOAD_VERIFIED',len(actual),'assets',report['bytes'],'bytes; draft remains unpublished',flush=True)


if __name__=='__main__':main()

"""Restore selected, hash-verified public research Release bundles (Python3.10+).

Default: saved HD demo only, no GPU or model environment required. Archives are
validated against an explicit file inventory before extraction. Existing files
with different contents are never overwritten. No GitHub login is required.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO='prummn/face-age-transformation-course-project'
TAG='research-mobei-2026-11-17'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def target(root,name):
    p=PurePosixPath(name)
    if p.is_absolute() or not p.parts or any(s in ['..','.'] for s in p.parts) or '\\' in name:
        raise ValueError('Unsafe destination: '+name)
    dest=root.joinpath(*p.parts)
    if not dest.resolve().is_relative_to(root.resolve()):raise ValueError('Destination escapes root: '+name)
    return dest


def download(spec,cache):
    path=cache/spec['name']
    if path.exists() and path.stat().st_size==spec['bytes'] and digest(path)==spec['sha256']:return path
    if path.exists():raise ValueError('Different cached asset; preserve or move it before retrying: '+str(path))
    temp=path.with_name(path.name+'.partial');url=spec.get('url') or f"https://github.com/{REPO}/releases/download/{spec.get('tag',TAG)}/{spec['name']}"
    for attempt in range(4):
        try:
            offset=temp.stat().st_size if temp.exists() else 0
            if offset>spec['bytes']:temp.unlink();offset=0
            req=Request(url,headers={'User-Agent':'face-age-research-restore','Range':f'bytes={offset}-'} if offset else {'User-Agent':'face-age-research-restore'})
            with urlopen(req,timeout=45) as r:
                resumed=r.status==206 and offset>0
                if resumed and not r.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                    raise ValueError('Wrong download range')
                with temp.open('ab' if resumed else 'wb') as f:
                    while block:=r.read(1024**2):
                        f.write(block)
                        if f.tell()>spec['bytes']:raise ValueError('Oversized release asset')
            if temp.stat().st_size!=spec['bytes'] or digest(temp)!=spec['sha256']:
                temp.unlink(missing_ok=True);raise ValueError('Release asset hash/size mismatch')
            temp.replace(path);return path
        except (OSError,URLError,HTTPError,ValueError):
            if attempt==3:raise
            if temp.exists() and temp.stat().st_size==spec['bytes']:
                if digest(temp)==spec['sha256']:temp.replace(path);return path
                temp.unlink()
            time.sleep(1+attempt)


def assemble(item,cache):
    path=cache/item['filename']
    if path.exists():
        if path.stat().st_size==item['bytes'] and digest(path)==item['sha256']:return path
        raise ValueError('Existing assembled file differs: '+str(path))
    partial=path.with_name(path.name+'.assembling')
    with partial.open('wb') as out:
        for part in item['assets']:
            source=download(part,cache)
            with source.open('rb') as f:shutil.copyfileobj(f,out,8*1024**2)
    if partial.stat().st_size!=item['bytes'] or digest(partial)!=item['sha256']:
        partial.unlink(missing_ok=True);raise ValueError('Complete file hash/size mismatch')
    partial.replace(path);return path


def installed(root,item):
    for spec in item['files']:
        dest=target(root,spec['path'])
        if dest.exists() and (not dest.is_file() or dest.stat().st_size!=spec['bytes'] or digest(dest)!=spec['sha256']):
            raise ValueError('Preserving different existing file: '+str(dest))
    return all(target(root,s['path']).is_file() for s in item['files'])


def install(root,item,blob):
    if item['format']=='file':
        if len(item['files'])!=1:raise ValueError('Invalid binary inventory')
        spec=item['files'][0];dest=target(root,spec['path']);dest.parent.mkdir(parents=True,exist_ok=True)
        if digest(blob)!=spec['sha256']:raise ValueError('Binary digest mismatch')
        with tempfile.NamedTemporaryFile(dir=dest.parent,delete=False) as temp:
            name=Path(temp.name)
            with blob.open('rb') as f:shutil.copyfileobj(f,temp,8*1024**2)
        os.chmod(name,0o644);name.replace(dest)
        return
    if item['format']!='tar.gz':raise ValueError('Unsupported archive format')
    inventory={s['path']:s for s in item['files']}
    # Validate all entries (including content) before writing any restored file.
    with tarfile.open(blob,'r:gz') as archive:
        members=archive.getmembers();seen=set()
        for member in members:
            target(root,member.name)
            if not member.isfile() or member.name not in inventory or member.name in seen:
                raise ValueError('Unexpected archive entry: '+member.name)
            seen.add(member.name);spec=inventory[member.name]
            h=hashlib.sha256()
            with archive.extractfile(member) as f:
                for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
            if member.size!=spec['bytes'] or h.hexdigest()!=spec['sha256']:raise ValueError('Archive entry hash/size mismatch')
        if seen!=set(inventory):raise ValueError('Incomplete archive inventory')
        for member in members:
            dest=target(root,member.name)
            if dest.exists():continue
            dest.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=dest.parent,delete=False) as temp:
                name=Path(temp.name)
                with archive.extractfile(member) as f:shutil.copyfileobj(f,temp,8*1024**2)
            os.chmod(name,0o644);name.replace(dest)


def configure(root,groups):
    if 'demo' in groups:
        source=root/'deployment/demo-hd/20261117-110722/results';link=root/'原型/hd-demo-assets'
        if not source.is_dir():raise ValueError('Missing restored gallery')
        if link.is_symlink() and link.resolve()==source.resolve():pass
        elif link.exists() or link.is_symlink():raise ValueError('Preserving existing gallery link: '+str(link))
        else:link.parent.mkdir(parents=True,exist_ok=True);link.symlink_to(os.path.relpath(source,link.parent),target_is_directory=True)
    if 'data' in groups:
        folder=root/'deployment/shared-research/manifests'
        for split in ['train','eval']:
            source=folder/(split+'.portable.jsonl');dest=folder/(split+'.jsonl')
            rows=[json.loads(s) for s in source.read_text().splitlines()]
            for row in rows:
                for field in ['image_path','reference_path','reference512_path']:
                    row[field]=str(target(root,row[field]))
                for field,hashfield in [('image_path','sha256'),('reference_path','reference_sha256'),('reference512_path','reference512_sha256')]:
                    if digest(row[field])!=row[hashfield]:raise ValueError('Restored training input hash mismatch')
            text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
            if dest.exists() and dest.read_text()!=text:raise ValueError('Preserving existing derived manifest: '+str(dest))
            dest.write_text(text)
            receipt=dict(portable_sha256=digest(source),local_sha256=digest(dest),root=str(root),path_change_only=True)
            (folder/(split+'.relocation.json')).write_text(json.dumps(receipt,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group',action='append',help='demo (default), baseline, data, evidence, weights100, weights500, weights800')
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--manifest',type=Path,default=Path(__file__).with_name('research-20261117.json'))
    p.add_argument('--list',action='store_true');p.add_argument('--verify-only',action='store_true')
    a=p.parse_args();manifest=json.loads(a.manifest.read_text());groups=a.group or ['demo']
    if a.list:
        for key,spec in manifest['groups'].items():
            size=sum(x['bytes'] for x in manifest['items'] if x['group']==key)
            print(f'{key}: {size/1024**3:.3f} GiB — {spec}')
        return
    unknown=set(groups)-set(manifest['groups'])
    if unknown:p.error('Unknown groups: '+', '.join(sorted(unknown)))
    items=[x for x in manifest['items'] if x['group'] in groups]
    tag=manifest.get('release_tag',TAG)
    if not isinstance(tag,str) or not tag or tag in ('.','..') or '/' in tag or '\\' in tag:
        raise ValueError('Invalid research release cache name')
    root=a.root.resolve();cache=root/'deployment/release-cache'/tag;cache.mkdir(parents=True,exist_ok=True)
    for item in items:
        if installed(root,item):print('VERIFIED existing',item['id'],flush=True);continue
        if a.verify_only:raise SystemExit('Missing files: '+item['id'])
        print('RESTORE',item['id'],flush=True);blob=assemble(item,cache);install(root,item,blob)
        if not installed(root,item):raise ValueError('Post-install verification failed')
    if not a.verify_only:configure(root,groups)
    print('READY '+','.join(groups)+'; all files SHA256 verified; different existing files preserved')
    if 'demo' in groups:print('python3 integration/serve_hd_demo.py   # /hd-demo.html, no GPU needed')


if __name__=='__main__':main()

"""Fetch a size/SHA256-pinned model with bounded, resumable HTTP ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path); p.add_argument('directory', type=Path)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--hf-mirror', action='store_true', help='Use hf-mirror.com transport; original pinned SHA256 still required')
    args = p.parse_args()
    if not 1 <= args.workers <= 8: p.error('workers must be1..8')
    spec = json.loads(args.manifest.read_text()); name = spec['name']
    url = spec['url']
    if args.hf_mirror:
        if not url.startswith('https://huggingface.co/'): p.error('Mirror option requires a Hugging Face source')
        url = url.replace('https://huggingface.co/', 'https://hf-mirror.com/', 1)
    if Path(name).name != name: raise ValueError('Expected a plain filename')
    args.directory.mkdir(parents=True, exist_ok=True); target = args.directory/name
    if target.exists():
        if target.stat().st_size == spec['bytes'] and digest(target) == spec['sha256']:
            print('MODEL_ALREADY_VERIFIED', target); return
        raise ValueError('Preserving different existing model: '+str(target))
    parts = args.directory/(name+'.ranges'); parts.mkdir(exist_ok=True)
    chunk = 16 * 1024**2; count = (spec['bytes']+chunk-1)//chunk
    # Reuse completed ranges from our interrupted sequential download.
    prefix = args.directory/(name+'.partial')
    if prefix.exists():
        with prefix.open('rb') as stream:
            for index in range(min(count, prefix.stat().st_size//chunk)):
                part = parts/str(index)
                block = stream.read(chunk)
                if not part.exists(): part.write_bytes(block)

    def fetch(index):
        start = index*chunk; size = min(chunk, spec['bytes']-start)
        path = parts/str(index)
        for attempt in range(4):
            offset = path.stat().st_size if path.exists() else 0
            if offset == size: return index
            if offset > size: raise ValueError('Oversized cached range')
            first, last = start+offset, start+size-1
            try:
                req = Request(url, headers={'Range': f'bytes={first}-{last}',
                    'User-Agent': 'face-age-verified-model-fetch'})
                with urlopen(req, timeout=40) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {first}-{last}/{spec["bytes"]}':
                        raise ValueError('Server returned a different range')
                    with path.open('ab') as stream:
                        remaining = size-offset
                        while remaining:
                            block = response.read(min(256*1024, remaining))
                            if not block: raise OSError('Truncated HTTP range')
                            stream.write(block); remaining -= len(block)
                return index
            except (OSError, ValueError):
                if attempt == 3: raise
                time.sleep(attempt+1)
        raise RuntimeError('Unreachable')

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, index) for index in range(count)]
        for future in as_completed(futures):
            future.result(); done += 1
            print('MODEL_RANGE_RECEIVED', done, '/', count, flush=True)
    assembled = args.directory/(name+'.assembling')
    h = hashlib.sha256()
    with assembled.open('wb') as stream:
        for index in range(count):
            with (parts/str(index)).open('rb') as source:
                for block in iter(lambda: source.read(8*1024**2), b''):
                    stream.write(block); h.update(block)
    if assembled.stat().st_size != spec['bytes'] or h.hexdigest() != spec['sha256']:
        raise ValueError('Final model SHA256/size mismatch; preserving ranges for diagnosis')
    assembled.replace(target)
    receipt = dict(path=str(target), bytes=spec['bytes'], sha256=h.hexdigest(),
        source_commit=spec['source_commit'], manifest_sha256=digest(args.manifest), workers=args.workers,
        transport_url=url)
    (args.directory/(name+'.verified.json')).write_text(json.dumps(receipt, indent=2)+'\n')
    print('MODEL_SHA256_VERIFIED', target, flush=True)


if __name__ == '__main__': main()

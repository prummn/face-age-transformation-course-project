"""Pick a working replica of YiCloud's documented headless egress proxy."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import socket
import ssl
import tempfile
import time

HOST='httpproxy-headless.kubebrain.svc.lg.shzhisuan.local'
CACHE=Path(__file__).resolve().parents[1]/'deployment/network-proxy.json'


def probe(ip):
    started=time.monotonic()
    with socket.create_connection((ip,3128),timeout=3) as stream:
        stream.settimeout(3)
        stream.sendall(b'CONNECT raw.githubusercontent.com:443 HTTP/1.1\r\nHost: raw.githubusercontent.com:443\r\n\r\n')
        header=b''
        while b'\r\n\r\n' not in header:
            block=stream.recv(4096)
            if not block or len(header)>65536:raise OSError('Invalid proxy response')
            header+=block
        if header.split(b' ',2)[1]!=b'200':raise OSError('Proxy refused CONNECT')
        with ssl.create_default_context().wrap_socket(stream,server_hostname='raw.githubusercontent.com'):
            pass
    return time.monotonic()-started


def main():
    saved=None
    if CACHE.exists():
        try:
            saved=json.loads(CACHE.read_text())
            if time.time()-saved['checked_at']<300:
                print(f"http://{saved['ip']}:3128");return
        except (OSError,ValueError,KeyError):pass
    ips=sorted({entry[4][0] for entry in socket.getaddrinfo(HOST,3128,type=socket.SOCK_STREAM)})
    results=[];chosen=None
    with ThreadPoolExecutor(max_workers=min(4,len(ips))) as pool:
        pending={pool.submit(probe,ip):ip for ip in ips}
        for future in as_completed(pending):
            ip=pending[future]
            try:
                seconds=future.result();results.append(dict(ip=ip,ready=True,seconds=seconds))
                if chosen is None:chosen=ip
            except Exception as e:results.append(dict(ip=ip,ready=False,error=type(e).__name__))
    # A target outage must not prevent commands for other sites from running.
    # Reuse a known replica still in DNS, otherwise use the documented service.
    fallback=chosen is None
    if fallback:chosen=saved['ip'] if saved and saved.get('ip') in ips else HOST
    CACHE.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=CACHE.parent,prefix='network-proxy-',suffix='.tmp',delete=False) as stream:
        stream.write(json.dumps(dict(ip=chosen,checked_at=time.time(),replicas=results,
                                     probe_fallback=fallback),indent=2)+'\n')
        temp=Path(stream.name)
    temp.replace(CACHE)
    print(f'http://{chosen}:3128')


if __name__=='__main__':main()

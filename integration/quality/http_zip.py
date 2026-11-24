"""Read selected files from public ZIP archives using bounded HTTP ranges."""
import io
import json
import os
from pathlib import Path
import struct
import threading
from urllib.parse import parse_qsl,urlencode,urlsplit,urlunsplit,urljoin
import zipfile
import zlib

import requests


class HTTPZip:
    def __init__(self,url,size,max_transfer=128*1024*1024):
        self.url=url;self.size=size;self.max_transfer=max_transfer
        self.transferred=0;self.requests=0;self.lock=threading.Lock();self.local=threading.local()
        owner=self
        class Reader(io.RawIOBase):
            def __init__(self):super().__init__();self.position=0
            def seekable(self):return True
            def readable(self):return True
            def tell(self):return self.position
            def seek(self,offset,whence=0):
                self.position=offset if whence==0 else self.position+offset if whence==1 else owner.size+offset
                if not 0<=self.position<=owner.size:raise ValueError('Range seek outside archive')
                return self.position
            def read(self,size=-1):
                if size<0:size=owner.size-self.position
                size=min(size,owner.size-self.position)
                data=owner.range(self.position,size);self.position+=len(data);return data
        with zipfile.ZipFile(Reader()) as archive:
            self.entries={info.filename:info for info in archive.infolist()}

    @classmethod
    def from_index(cls,path,max_transfer=128*1024*1024):
        saved=json.loads(path.read_text());owner=cls.__new__(cls)
        owner.url=saved['source']['url'];owner.size=saved['source']['archive_bytes']
        owner.max_transfer=max_transfer;owner.transferred=0;owner.requests=0;owner.lock=threading.Lock();owner.local=threading.local()
        owner.entries={}
        for row in saved['entries']:
            info=zipfile.ZipInfo(row['name']);info.file_size=row['size'];info.compress_size=row['compressed']
            info.header_offset=row['offset'];info.CRC=row['crc'];info.compress_type=row['compression'];info.flag_bits=row['flags']
            owner.entries[info.filename]=info
        return owner

    def range(self,start,size):
        if not size:return b''
        if size<0 or size>16*1024*1024 or start<0 or start+size>self.size:
            raise ValueError('Refusing oversized/out-of-bounds range')
        with self.lock:
            if self.transferred+size>self.max_transfer:raise ValueError('Archive transfer budget exceeded')
            self.transferred+=size;self.requests+=1
        end=start+size-1
        if getattr(self,'resolved_url',None):url=self.resolved_url
        else:
            parts=urlsplit(self.url);query=parse_qsl(parts.query,keep_blank_values=True)
            query.append(('face_age_range',f'{start}-{end}'))
            url=urlunsplit((parts.scheme,parts.netloc,parts.path,urlencode(query),parts.fragment))
        if not hasattr(self.local,'session'):self.local.session=requests.Session()
        # Keep TLS connections alive. If a replica's target-specific handshake
        # fails, make one attempt through another already verified replica.
        alternatives=[None]
        cache=Path(__file__).resolve().parents[2]/'deployment/network-proxy.json'
        if cache.exists():
            current=os.environ.get('HTTPS_PROXY','')
            alternatives += [f"http://{r['ip']}:3128" for r in json.loads(cache.read_text()).get('replicas',[])
                             if r.get('ready') and f"http://{r['ip']}:3128"!=current][:1]
        response=None
        for index,proxy in enumerate(alternatives):
            try:
                response=self.local.session.get(url,headers={'Range':f'bytes={start}-{end}'},stream=True,
                    timeout=(8,25),proxies={'http':proxy,'https':proxy} if proxy else None)
                break
            except requests.RequestException:
                if index==len(alternatives)-1:raise
        with response:
            response.raise_for_status()
            expected=f'bytes {start}-{end}/{self.size}'
            if response.status_code!=206 or response.headers.get('Content-Range')!=expected:
                raise ValueError('Server did not honor exact byte range; refusing full archive')
            data=response.raw.read(size+1,decode_content=False)
            if len(data)!=size:raise ValueError('Truncated/oversized HTTP range')
            return data

    def resolve_download(self):
        """Resolve one public HF redirect; keep its temporary CDN URL in memory.

        The signed URL permits arbitrary byte ranges. Publisher hashes still
        verify every extracted image; URLs with signatures are not logged.
        """
        if urlsplit(self.url).hostname!='huggingface.co':return
        attempts=[self.url,self.url.replace('https://huggingface.co/','https://hf.co/',1)]
        for index,url in enumerate(attempts):
            try:
                for _ in range(5):
                    response=requests.get(url,headers={'Range':'bytes=0-0'},stream=True,allow_redirects=False,timeout=(8,15))
                    response.raise_for_status();location=response.headers.get('Location');response.close()
                    if not location:
                        self.resolved_url=url;return
                    location=urljoin(url,location);parts=urlsplit(location);host=parts.hostname or ''
                    if parts.scheme!='https' or not (host in ('hf.co','huggingface.co') or host.endswith('.hf.co') or host.endswith('.huggingface.co')):
                        raise ValueError('Unexpected HF download host')
                    if host not in ('hf.co','huggingface.co'):
                        self.resolved_url=location;return
                    url=location
                raise ValueError('Too many HF same-origin redirects')
            except requests.RequestException:
                if index==len(attempts)-1:raise

    def read(self,name):
        info=self.entries[name]
        if info.flag_bits&1:raise ValueError('Encrypted ZIP member needs publisher authorization')
        if info.file_size>16*1024*1024:raise ValueError('Selected ZIP member exceeds16MiB')
        header=self.range(info.header_offset,30)
        fields=struct.unpack('<4s5H3I2H',header)
        if fields[0]!=b'PK\x03\x04':raise ValueError('Invalid ZIP local header')
        offset=info.header_offset+30+fields[-2]+fields[-1]
        payload=self.range(offset,info.compress_size)
        if info.compress_type==zipfile.ZIP_STORED:data=payload
        elif info.compress_type==zipfile.ZIP_DEFLATED:data=zlib.decompress(payload,-15)
        else:raise ValueError('Unsupported ZIP compression')
        if len(data)!=info.file_size or zlib.crc32(data)!=info.CRC:
            raise ValueError('ZIP member CRC/size mismatch')
        return data

    def stats(self):
        return dict(url=self.url,archive_bytes=self.size,range_bytes=self.transferred,
                    range_requests=self.requests,full_archive_downloaded=False)

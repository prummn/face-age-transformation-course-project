"""Download a linked MCST public-release photo ZIP and preserve source attribution.

No login, no upload, no automatic training admission. Original image sizes and
ZIP CRC/SHA256 are recorded; size alone is not a face-quality or age guarantee.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.parse import urlsplit
from urllib.request import Request,urlopen
import zipfile


def fetch(url,max_bytes):
    request=Request(url,headers={'User-Agent':'FaceAgeCourseResearch/1.0 (canonical source audit)'})
    with urlopen(request,timeout=30) as response:
        result=bytearray()
        while block:=response.read(1024**2):
            result.extend(block)
            if len(result)>max_bytes:raise ValueError('Transfer budget exceeded')
    return bytes(result)


def preserve(path,data):
    if path.exists():
        if path.read_bytes()!=data:raise ValueError('Preserving different existing file: '+str(path))
    else:path.write_bytes(data)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--page',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();parts=urlsplit(a.page)
    if parts.scheme!='https' or parts.netloc!='www.mcst.go.kr' or parts.path!='/site/s_notice/press/pressView.jsp':raise ValueError('Expected canonical MCST press page')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    html_path=out/'source.html';html=html_path.read_bytes() if html_path.exists() else fetch(a.page,2*1024**2)
    text=html.decode('utf-8')
    if '공공누리 1유형' not in text or 'openright_01.svg' not in text:raise ValueError('Page does not explicitly advertise KOGL Type1')
    options=[m for m in re.finditer(r"file_download\('([^']+)', '([^']+)', '([^']+)'",text) if m[2].endswith('.zip')]
    if len(options)!=1:raise ValueError('Need exactly one photo archive; inspect other pages manually')
    m=options[0];url='https://www.mcst.go.kr/servlets/eduport/front/upload/UplDownloadFile?pFileName='+m[1]+'&pRealName='+m[2]+'&pPath='+m[3]+'&pFlag='
    preserve(html_path,html);archive=out/'originals.zip';data=archive.read_bytes() if archive.exists() else fetch(url,50*1024**2)
    from PIL import Image
    records=[]
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for index,entry in enumerate(z.infolist()):
            if entry.is_dir():continue
            if not entry.filename.lower().endswith(('.jpg','.jpeg','.png')) or entry.file_size>30*1024**2:raise ValueError('Unexpected archive member')
            image=z.read(entry)  # zipfile verifies the stored CRC.
            with Image.open(io.BytesIO(image)) as im:im.load();size=list(im.size);fmt=im.format
            name='image-'+str(index).zfill(3)+('.png' if fmt=='PNG' else '.jpg');preserve(out/name,image)
            records.append(dict(file=name,member=entry.filename,bytes=len(image),sha256=hashlib.sha256(image).hexdigest(),zip_crc32=entry.CRC,size=size))
    preserve(archive,data)
    receipt=dict(source_page=a.page,download_url=url,licence='KOGL Type1 as explicitly stated by source page',attribution='Ministry of Culture, Sports and Tourism, Republic of Korea',
        archive_sha256=hashlib.sha256(data).hexdigest(),archive_bytes=len(data),images=records,training_admitted=False,
        warning='Verify face quality, depicted identity, photo date, age evidence and common groups before admitting to any training split.')
    preserve(out/'provenance.json',(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n').encode())
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

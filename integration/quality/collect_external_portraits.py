"""Collect 32 new official APPA test portraits using CRC-checked HTTP ranges."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import io
import json
from pathlib import Path
import random
import time
from PIL import Image
from http_zip import HTTPZip
from probe_inversion import sha256,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();source=a.source.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    ages={r['file_name']:r for r in csv.DictReader((source/'appa-metadata/gt_avg_test.csv').open())}
    extra=list(csv.DictReader((source/'appa-extended/allcategories_test.csv').open()))
    used={r['file'] for r in json.loads((source/'appa-selected.json').read_text())}
    candidates=[dict(ages[r['file']],**r) for r in extra if r['file'] not in used and float(ages[r['file']]['real_age'])>=20]
    rng=random.Random(20261117);selected=[]
    # Exhaust the scarce adult Asian40+ cases before selecting young adults.
    older=sorted([r for r in candidates if r['race']=='asian' and float(r['real_age'])>=40],key=lambda r:r['file'])
    selected.extend(older)
    young=[r for r in candidates if r['race']=='asian' and float(r['real_age'])<40]
    used_new={r['file'] for r in selected}
    for gender in ['female','male']:
        pool=sorted([r for r in young if r['gender']==gender],key=lambda r:r['file']);rng.shuffle(pool)
        needed=(24-len(older))//2 + (1 if gender=='female' and (24-len(older))%2 else 0)
        selected.extend(pool[:needed]);used_new.update(r['file'] for r in pool[:needed])
    assert len(selected)==24
    for low,high in [(20,39),(40,59),(60,74),(75,100)]:
        for gender in ['female','male']:
            pool=sorted([r for r in candidates if r['race']!='asian' and r['gender']==gender and low<=float(r['real_age'])<=high],key=lambda r:r['file'])
            rng.shuffle(pool)
            # Include an old-photo stress example when the stratum offers one.
            pool.sort(key=lambda r:r['time']!='oldphoto')
            selected.append(pool[0])
    assert len(selected)==32 and len({r['file'] for r in selected})==32
    plan=dict(seed=20261117,source='APPA-REAL official test split only',
        source_archive=json.loads((source/'appa_zip-index.json').read_text())['source']['url'],
        selection='All11 remaining Asian40+ +13 Asian20..39 gender balanced as possible +8 other-group age/gender strata; prefer old photos for other-group stress cases.',
        samples=selected,excluded_previous_files=sorted(used),training_authorized=False,
        use='Local model evaluation only; no network-weight training or image redistribution to Git',
        labels_sha256={str(p.relative_to(source)):sha256(p) for p in [source/'appa-metadata/gt_avg_test.csv',source/'appa-extended/allcategories_test.csv',source/'appa_zip-index.json']})
    dest=out/'collection-plan.json'
    if dest.exists():assert json.loads(dest.read_text())==plan,'Frozen selection changed'
    else:write_json(dest,plan)
    archive=HTTPZip.from_index(source/'appa_zip-index.json',max_transfer=64*1024*1024)
    started=time.monotonic()
    def fetch(row):
        folder=out/'raw'/Path(row['file']).stem;folder.mkdir(parents=True,exist_ok=True);files=[]
        result=dict(name='appa_new_'+Path(row['file']).stem,annotation=row,source_archive=archive.url,files=files)
        try:
            for suffix in ['', '_face.jpg']:
                member='appa-real-release/test/'+row['file']+suffix
                path=folder/(row['file']+suffix)
                info=archive.entries[member]
                if path.exists():
                    import zlib
                    data=path.read_bytes();assert len(data)==info.file_size and zlib.crc32(data)==info.CRC
                else:
                    data=archive.read(member);temp=path.with_suffix('.tmp');temp.write_bytes(data);temp.replace(path)
                with Image.open(io.BytesIO(data)) as image:
                    image.load();size=list(image.size);mode=image.mode
                files.append(dict(path=str(path),member=member,sha256=sha256(path),bytes=len(data),size=size,mode=mode,zip_crc=info.CRC))
            result['status']='verified'
        except Exception as e:result.update(status='failed',error=type(e).__name__,detail=str(e)[:200])
        print('PORTRAIT',result['name'],result['status'],flush=True)
        return result
    with ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(fetch,selected))
    (out/'collection-manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    report=dict(selected=32,verified=sum(r['status']=='verified' for r in records),
        counts=dict(Counter(r['race'] for r in selected)),gender=dict(Counter(r['gender'] for r in selected)),
        age_bins=dict(Counter('20-39' if float(r['real_age'])<40 else '40-59' if float(r['real_age'])<60 else '60-74' if float(r['real_age'])<75 else '75+' for r in selected)),
        seconds=time.monotonic()-started,transfer=archive.stats(),training=False)
    write_json(out/'collection-report.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)
    if report['verified']!=32:raise SystemExit(1)


if __name__=='__main__':main()

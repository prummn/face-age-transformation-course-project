"""CPU validation and a few explicitly selected contact sheets after GPU release."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont


def sha256(path):
    result=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    run=p.parse_args().run
    if not (run/'DONE').exists():
        raise RuntimeError('Wait for completion')
    report=json.loads((run/'evaluation/report.json').read_text());outputs=report['outputs']
    paths={};native=0
    for row in outputs:
        assert row['native_size']==[1024,1024]
        paths[row['preview']]=(256,256)
        if row['native']:
            paths[row['native']]=(1024,1024);native+=1
    for row in report['records']:
        paths[row['name']+'/reference.png']=None
    for name,size in paths.items():
        with Image.open(run/'evaluation'/name) as image:
            image.load();assert image.format=='PNG' and image.mode=='RGB'
            if size:assert image.size==size
    assert len(outputs)==1560 and native==390
    candidates=[]
    for arm in ['id_control','id_strong']:
        path=run/arm/'encoder.pt';digest=sha256(path)
        assert digest==json.loads((run/arm/'completed.json').read_text())['encoder_sha256']
        candidates.append(dict(arm=arm,path=str(path),bytes=path.stat().st_size,sha256=digest))
    (run/'candidate-manifest.json').write_text(json.dumps(candidates,indent=2)+'\n')
    lookup={(r['name'],r['route'],r['condition']):r for r in outputs}
    names=[r['name'] for r in report['records'] if r['cohort']=='appa_asian_test']
    def gain(name):
        return sum(lookup[(name,'id_strong',age)]['sface_cosine']-lookup[(name,'id_control',age)]['sface_cosine'] for age in ['20','40','60','75'])/4
    ordered=sorted(names,key=gain)
    selected=[('uploaded','user diagnostic'),(ordered[len(ordered)//2],'APPA median identity gain'),(ordered[0],'APPA worst identity gain')]
    folder=run/'representatives';folder.mkdir(exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    for name,reason in selected:
        cell=224;top=42;height=264
        board=Image.new('RGB',(cell*6,height*3+top),'#eff2ee');draw=ImageDraw.Draw(board)
        draw.text((8,8),name+' / '+reason+' / raw outputs',font=font,fill='black')
        for i,route in enumerate(report['routes']):
            y=top+i*height
            with Image.open(run/'evaluation'/name/'reference.png') as ref:
                board.paste(ref.resize((cell,cell)),(0,y+36))
            draw.text((4,y+3),route+' / source',font=font,fill='black')
            for j,age in enumerate(['same','20','40','60','75'],1):
                row=lookup[(name,route,age)]
                with Image.open(run/'evaluation'/(row['native'] or row['preview'])) as image:
                    board.paste(image.resize((cell,cell)),(j*cell,y+36))
                draw.text((j*cell+4,y+3),f"{age} ID:{row['sface_cosine']:.3f} age:{row['shared_output_age']:.1f}",font=font,fill='black')
        board.save(folder/(name+'.jpg'),quality=93)
    result=dict(pngs_decoded=len(paths),native_outputs=native,previews=1560,native_tensor_checks=1560,
                paired_sampling=(run/'PAIRED_SAMPLING_VERIFIED').read_text().strip(),
                representatives=[dict(name=name,reason=reason,path='representatives/'+name+'.jpg') for name,reason in selected])
    (run/'artifact-validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

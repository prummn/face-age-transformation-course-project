"""Publish only two source-attributed, derivative-permitted FFHQ review panels."""
import argparse
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from probe_inversion import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    run=a.run.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    rows={r['name']:r for r in json.loads((run/'selection.json').read_text())}
    fonts=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font=ImageFont.truetype(str(fonts),16) if fonts.exists() else ImageFont.load_default()
    credits=[]
    for name in ('ffhq_64376','ffhq_64919'):
        r=rows[name];meta=r['source_metadata'];licence=meta['license_url']
        if not licence.startswith(('https://creativecommons.org/licenses/by/','https://creativecommons.org/publicdomain/')):
            raise ValueError('Example not on explicit derivative-permitted source allowlist')
        canvas=Image.new('RGB',(1280,650),'white');d=ImageDraw.Draw(canvas)
        routes=[('reference.png','Source'),('wplus_base_75.png','W+ baseline'),('highres_control_75.png','Reconstruction adapter'),('highres_patch_gan_75.png','Reconstruction + local GAN')]
        for i,(file,label) in enumerate(routes):
            with Image.open(run/'evaluation'/name/file) as im:
                image=im.convert('RGB');assert image.size==(1024,1024)
                canvas.paste(image.resize((300,300),Image.Resampling.LANCZOS),(i*320,35))
                canvas.paste(image.crop((220,310,520,490)),(i*320,365))
                d.text((i*320+4,7),label,fill='black',font=font)
        author=meta['author']
        try:d.text((4,565),'Source: '+author+' | '+meta['photo_url'],fill='black',font=font)
        except UnicodeEncodeError:d.text((4,565),'Source attribution: public-panel-credits.json | '+meta['photo_url'],fill='black',font=font)
        d.text((4,590),'Source licence: '+licence,fill='black',font=font)
        d.text((4,615),'Synthetic target 75; not a real aged photo. Lower row: unscaled 300x180 native eye/forehead crops.',fill='black',font=font)
        path=out/(name+'-native75.png');canvas.save(path)
        credits.append(dict(name=name,file=path.name,author=author,photo_url=meta['photo_url'],source_licence=licence,
            source_title=meta['photo_title'],processing='FFHQ aligned crop; model age edit; top preview resized, lower native crop unchanged',
            ffhq_source='https://github.com/NVlabs/ffhq-dataset',ffhq_terms='https://creativecommons.org/licenses/by-nc-sa/4.0/',
            no_real_future_appearance_claim=True))
    write_json(out/'public-panel-credits.json',credits);print('PUBLIC_PANELS_READY',len(credits),out)

if __name__=='__main__':main()

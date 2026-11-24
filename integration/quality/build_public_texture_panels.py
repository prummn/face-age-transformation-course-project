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
    routes=json.loads((run/'evaluation/report.json').read_text())['routes']
    if len(routes)!=3 or routes[0]!='wplus_base':raise ValueError('Expected baseline and two paired candidates')
    labels=dict(highres_control='Reconstruction adapter',highres_patch_gan='Reconstruction + local GAN',
        direct_latent='Direct latent training',matched_wplus='Matched W+ training',
        conditional_control='Age-conditioned + age guard',conditional_gan='Age-conditioned + age guard + GAN',
        native_control='Full native reconstruction',native_gan='Full native + RGB local GAN',
        prior_control='Fine W+ control',pretrained_D='Fine W+ + pretrained D',
        original_delta='Original SAM delta',original_delta_constrained='Original delta + refinement')
    fonts=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font=ImageFont.truetype(str(fonts),16) if fonts.exists() else ImageFont.load_default()
    credits=[]
    for name in ('ffhq_64376','ffhq_64919'):
        r=rows[name];meta=r['source_metadata'];licence=meta['license_url']
        if not licence.startswith(('https://creativecommons.org/licenses/by/','https://creativecommons.org/publicdomain/')):
            raise ValueError('Example not on explicit derivative-permitted source allowlist')
        canvas=Image.new('RGB',(1440,870),'white');d=ImageDraw.Draw(canvas)
        columns=[('reference.png','Source'),('wplus_base_75.png','W+ baseline'),*[(route+'_75.png',labels[route]) for route in routes[1:]]]
        for i,(file,label) in enumerate(columns):
            with Image.open(run/'evaluation'/name/file) as im:
                image=im.convert('RGB');assert image.size==(1024,1024)
                canvas.paste(image.resize((300,300),Image.Resampling.LANCZOS),(i*360,35))
                canvas.paste(image.crop((220,310,520,490)),(i*360,365))
                canvas.paste(image.crop((220,490,520,670)),(i*360,560))
                d.text((i*360+4,7),label,fill='black',font=font)
        author=meta['author']
        try:d.text((4,765),'Source: '+author+' | '+meta['photo_url'],fill='black',font=font)
        except UnicodeEncodeError:d.text((4,765),'Source attribution: public-panel-credits.json | '+meta['photo_url'],fill='black',font=font)
        d.text((4,790),'Source licence: '+licence,fill='black',font=font)
        d.text((4,815),'Synthetic target 75; not a real aged photo. Two lower rows: unscaled 300x180 eye and cheek crops.',fill='black',font=font)
        d.text((4,840),'FFHQ derivatives: CC BY-NC-SA 4.0 | Exploratory development comparison; raw model outputs shown.',fill='black',font=font)
        path=out/(name+'-native75.png');canvas.save(path)
        credits.append(dict(name=name,file=path.name,author=author,photo_url=meta['photo_url'],source_licence=licence,
            source_title=meta['photo_title'],routes=routes,processing='FFHQ aligned crop; model age edit; top preview resized; two lower300x180 native eye/cheek crops unchanged',
            ffhq_source='https://github.com/NVlabs/ffhq-dataset',ffhq_terms='https://creativecommons.org/licenses/by-nc-sa/4.0/',
            no_real_future_appearance_claim=True))
    write_json(out/'public-panel-credits.json',credits);print('PUBLIC_PANELS_READY',len(credits),out)

if __name__=='__main__':main()

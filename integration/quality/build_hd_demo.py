"""Build a source-attributed HD gallery without copying private images into Git."""
import argparse
import json
from pathlib import Path
from PIL import Image
from probe_inversion import sha256,write_json

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);run=p.parse_args().run.resolve()
    assert (run/'DONE').exists()
    report=json.loads((run/'report.json').read_text());out=run/'results';people=[];checks=[]
    for row in report['records']:
        folder=out/row['name'];path=folder/'reference.png'
        with Image.open(path) as im:
            im.load();assert im.size==(1024,1024);im.resize((512,512),Image.Resampling.LANCZOS).save(folder/'thumbnail.webp',quality=92)
        outputs=[o for o in report['outputs'] if o['name']==row['name']]
        assert [o['condition'] for o in outputs]==['same','20','40','60','75']
        for output in outputs:
            for variant in ['raw','preserved']:
                spec=output[variant];image_path=out/spec['path'];assert sha256(image_path)==spec['sha256']
                with Image.open(image_path) as im:im.load();assert im.format=='PNG' and im.size==(1024,1024) and im.mode=='RGB'
                checks.append(dict(path=spec['path'],sha256=spec['sha256'],size=[1024,1024]))
        people.append(dict(id=row['name'],ageGroup=row['age_group'],gender=row['publisher_gender'],
            reference=f"{row['name']}/reference.png",thumbnail=f"{row['name']}/thumbnail.webp",
            referenceAge=row['reference_age'],anchorSeconds=row['anchor_seconds'],credit=row['source_metadata'],
            sourceQuality=row['quality'],sourceFaceShortEdge=row['source_face_short_edge'],
            originalFileMd5=row['image_spec']['file_md5'],originalPixelMd5=row['image_spec']['pixel_md5'],
            outputs=[dict(condition=o['condition'],target=o['target'],modelSeconds=o['model_seconds'],
                          raw=o['raw'],preserved=o['preserved'],color=o['color']) for o in outputs]))
    write_json(out/'demo.json',dict(profile=report['profile'],people=people,
        generatedBy='Frozen Release SAM/StyleGAN2 + accepted constrained per-photo W+ baseline; no network training.',
        modelReportSha256=sha256(run/'report.json'),selection=json.loads((run/'selection.json').read_text()),
        coldSetupSeconds=report['cold_setup_seconds'],environment=report['environment'],
        license='FFHQ collection and derived demo: CC BY-NC-SA 4.0; individual photo licenses retained in credits.'))
    write_json(run/'native-validation.json',dict(samples=len(people),generated_pngs=len(checks),all_decodable=True,outputs=checks))
    link=ROOT/'原型/hd-demo-assets'
    if link.is_symlink():link.unlink()
    elif link.exists():raise RuntimeError('Refusing to replace existing non-symlink assets')
    link.symlink_to(out,target_is_directory=True)
    print(json.dumps(dict(samples=len(people),generated_pngs=len(checks),url='/hd-demo.html',assets=str(out)),ensure_ascii=False))


if __name__=='__main__':main()

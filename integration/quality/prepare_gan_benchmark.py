"""Freeze small, traceable diagnostic fixtures; never claim population coverage."""
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[2]
    source = root / 'deployment/CUSP'
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    images = [('uploaded', root/'tmp/quality-review/origin.png',
               'User uploaded this portrait and authorized quality diagnosis; no network-weight training')]
    images += [('cusp_'+p.stem,p,
                f'https://github.com/guillermogotre/CUSP/blob/{commit}/sample_images/{p.name}')
               for p in sorted((source/'sample_images').glob('*.png'))]
    from PIL import Image
    items = []
    for name,path,provenance in images:
        with Image.open(path) as im:
            im.load()
            if im.width != im.height or im.width < 256:
                raise ValueError('Expected an aligned square reference')
            size = list(im.size)
        items.append(dict(name=name,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          size=size,source=provenance,already_aligned=True,
                          allowed_use='offline diagnostic inference and per-photo latent fitting only',
                          true_age=None,identity_group=None))
    record=dict(version=1,cusp_commit=commit,targets=[20,40,60,75],samples=items,
                scope='Small author-provided and user-authorized diagnostic fixtures; not blind/independent population validation',
                caveats=['Author examples may overlap model training; unknown.',
                         'No verified ages, identities or demographic annotations.',
                         'These images are not a general encoder/generator training set.',
                         'Every method receives the same256 area-resampled input; no second alignment.',
                         'CUSP FFHQ-LS targets are age bins, not exact-age equivalents to SAM.'])
    out=root/'deployment/gan-research/diagnostic-manifest.json'
    if out.exists() and json.loads(out.read_text())!=record:
        raise RuntimeError('Existing frozen manifest differs')
    out.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(out, len(items), 'diagnostic inputs')


if __name__=='__main__':main()

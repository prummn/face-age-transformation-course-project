"""Freeze numeric SAM source conditions as explicitly shared FairFace pseudo ages."""
import argparse,json,os,sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from probe_inversion import sha256,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(Path(os.environ['FACE_AGE_BASE'])/'src'))
    from sam_losses import AgeEst
    torch.set_num_threads(4);net=AgeEst().cuda().eval().requires_grad_(False)
    for split in ['train','eval','external']:
        source=a.data/(split+'.pre-age.jsonl' if split!='external' else 'external.jsonl')
        records=[json.loads(x) for x in source.read_text().splitlines()]
        for k in range(0,len(records),16):
            batch=records[k:k+16];images=[]
            for r in batch:
                assert sha256(r['image_path'])==r['sha256']
                rgb=np.asarray(Image.open(r['image_path']).convert('RGB'))
                images.append(cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA))
            x=torch.from_numpy(np.stack(images)).permute(0,3,1,2).cuda().float()/127.5-1
            with torch.no_grad():ages=net(x)[0].cpu().numpy()*100
            for r,age in zip(batch,ages):
                r.update(source_age=round(float(age)),shared_reference_age=float(age),age_kind='pseudo',
                         source_age_method='Frozen shared ConvNeXt-T FairFace expected age, rounded SAM condition; not chronological age')
        (a.output/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
        print('ANNOTATED',split,len(records),flush=True)
    write_json(a.output/'receipt.json',dict(model_sha256=sha256(Path(os.environ['FACE_AGE_BASE'])/'models/faceattr/convnext_t_fairface.pt'),
        files={p.name:sha256(p) for p in a.output.glob('*.jsonl')},note='Human intervals preserved separately; no midpoint labels; pseudo source ages retain bias.'))
if __name__=='__main__':main()

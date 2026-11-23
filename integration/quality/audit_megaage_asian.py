"""Audit the official MegaAge-Asian archive locally without exporting portraits."""
import argparse
from collections import Counter
from hashlib import sha256 as bytes_hash
import io
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image

from probe_inversion import sha256, write_json


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_PAGE = 'https://mmlab.ie.cuhk.edu.hk/projects/MegaAge/'
EXPECTED_SIZE = 290421152
EXPECTED_SHA256 = 'bb2abe5efa0572ffc1bdb5810eabd6177b95b5d8972b5c3aa221bea49adddf2c'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    if args.archive.stat().st_size != EXPECTED_SIZE or sha256(args.archive) != EXPECTED_SHA256:
        raise ValueError('Archive differs from the recorded official download')
    cv2.setNumThreads(2)
    model = ROOT/'deployment/model-files/face-age/models/yunet/face_detection_yunet_2023mar.onnx'
    det = cv2.FaceDetectorYN.create(str(model),'',(178,218),.65,.3,100)
    stats,exact = {},{}
    with zipfile.ZipFile(args.archive) as archive:
        for split,expected in [('train',40000),('test',3945)]:
            columns=[archive.read(f'megaage_asian/list/{split}_{col}.txt').decode().splitlines() for col in ['name','age','dis']]
            if any(len(c)!=expected for c in columns):raise ValueError('Publisher list length mismatch')
            sizes = Counter()
            ages = Counter()
            hashes = set()
            face_edges = []
            sample_count = 0
            unique_names = set()
            for index,(name,age,label) in enumerate(zip(*columns)):
                name = name.strip()
                if Path(name).name!=name or name in unique_names:raise ValueError('Unexpected sample path or duplicate name')
                unique_names.add(name)
                posterior = np.fromstring(label,sep=' ')
                if len(posterior)!=70 or not np.isfinite(posterior).all() or (posterior<0).any() or not .98<=posterior.sum()<=1.02:
                    raise ValueError('Invalid publisher posterior')
                age=int(age)
                if not 0<=age<=70:raise ValueError('Invalid publisher reference estimate')
                ages[age]+=1
                data=archive.read(f'megaage_asian/{split}/{name}')
                hashes.add(bytes_hash(data).digest())
                with Image.open(io.BytesIO(data)) as im:
                    w,h=im.size
                    sizes[(w,h)]+=1
                    if index%max(1,expected//256)==0:
                        rgb=np.asarray(im.convert('RGB'))
                        det.setInputSize((w,h))
                        _,faces=det.detect(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                        sample_count+=1
                        if faces is not None:face_edges.append(float(min(max(faces,key=lambda f:f[2]*f[3])[2:4])))
            exact[split]=hashes
            stats[split]=dict(records=expected,image_sizes={f'{w}x{h}':n for (w,h),n in sizes.items()},
                publisher_age_counts=dict(sorted(ages.items())),exact_duplicate_file_rows=expected-len(hashes),
                sample_face_detection=dict(sampled=sample_count,detected=len(face_edges),
                    face_edge_percentiles_px=dict(zip(['p0','p10','p50','p90','p100'],np.percentile(face_edges,[0,10,50,90,100]).tolist()))))
    result=dict(official_page=OFFICIAL_PAGE,archive_bytes=EXPECTED_SIZE,archive_sha256=EXPECTED_SHA256,
        age_kind='Publisher apparent-age posterior, 70 bins; no verified chronological age',
        permitted_scope='Noncommercial single-site internal research under publisher terms',
        splits=stats,exact_train_test_shared_files=len(exact['train']&exact['test']),
        decision='Candidate for age supervision calibration. Image dimensions prevent native 512 detail supervision.')
    write_json(args.output,result)
    print('MEGAAGE_AUDITED',*(f'{s}={v["records"]} {v["image_sizes"]}' for s,v in stats.items()),flush=True)


if __name__ == '__main__':
    main()

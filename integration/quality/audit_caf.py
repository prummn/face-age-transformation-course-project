"""Audit the CAF archive as an identity-labelled cross-age evaluation source."""
import argparse
from collections import Counter, defaultdict
import io
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageOps

from probe_inversion import sha256, write_json


ROOT = Path(__file__).resolve().parents[2]
EXPECTED = 'af59f6448b62a7119548c82376a4c37c4a5a4a49c032acf17eeab2d3cfbe2342'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    if sha256(args.archive) != EXPECTED:
        raise ValueError('CAF archive differs from recorded publisher download')
    cv2.setNumThreads(2)
    model = ROOT/'deployment/model-files/face-age/models/yunet/face_detection_yunet_2023mar.onnx'
    detector = cv2.FaceDetectorYN.create(str(model),'',(640,640),.8,.3,100)
    identity_ages = defaultdict(list)
    qualities = Counter()
    age_counts = Counter()
    edges = []
    pixels = []
    other_members = 0
    with ZipFile(args.archive) as archive:
        for info in archive.infolist():
            if info.is_dir():continue
            p = PurePosixPath(info.filename)
            match = re.fullmatch(r'(\d{3})_(\d{1,3})\.(?:jpe?g|png)',p.name,re.I)
            if len(p.parts)!=3 or p.parts[0]!='Cross_Age_Face' or not match:
                other_members+=1;continue
            identity,age = match.group(1),int(match.group(2))
            if not p.parts[1].startswith(identity+'_'):raise ValueError('CAF filename/identity conflict')
            if info.file_size>32*1024**2:raise ValueError('Unexpectedly large CAF member')
            age_counts[str(age//10*10)]+=1
            identity_ages[identity]
            try:
                with Image.open(io.BytesIO(archive.read(info))) as image:
                    rgb = ImageOps.exif_transpose(image).convert('RGB')
                    w,h=rgb.size
                    pixels.append(min(w,h))
                    # Keep the native dimension for quality accounting. Detect in a bounded preview.
                    factor=min(1.,768/max(w,h))
                    size=(max(1,round(w*factor)),max(1,round(h*factor)))
                    small=np.asarray(rgb.resize(size,Image.Resampling.LANCZOS) if factor<1 else rgb)
            except (OSError,ValueError):
                qualities['decode_error']+=1;continue
            detector.setInputSize(size)
            _,faces=detector.detect(cv2.cvtColor(small,cv2.COLOR_RGB2BGR))
            if faces is None:
                qualities['no_face']+=1;continue
            ordered=sorted(faces,key=lambda f:f[2]*f[3],reverse=True)
            if len(ordered)>1 and ordered[1][2]*ordered[1][3]>.25*ordered[0][2]*ordered[0][3]:
                qualities['multiple_faces']+=1;continue
            edge=float(min(ordered[0][2]/size[0]*w,ordered[0][3]/size[1]*h))
            edges.append(edge)
            if edge<512:
                qualities['face_short_edge_below512']+=1;continue
            qualities['single_face_ge512']+=1
            identity_ages[identity].append(age)
    if not edges:raise RuntimeError('No detected CAF faces')
    report=dict(publisher='https://github.com/AvLab-CV/AgeTransGAN#cross-age-face-dataset',
        archive_bytes=args.archive.stat().st_size,archive_sha256=EXPECTED,
        identity_folders=len(identity_ages),labeled_files=sum(age_counts.values()),unmatched_files=other_members,
        photos_by_age_decade=dict(sorted(age_counts.items(),key=lambda x:int(x[0]))),quality=dict(qualities),
        face_short_edge_percentiles=dict(zip(['p10','p50','p90'],np.percentile(edges,[10,50,90]).tolist())),
        photo_short_edge_percentiles=dict(zip(['p10','p50','p90'],np.percentile(pixels,[10,50,90]).tolist())),
        potential_hd_pair_subjects=dict(at_least_10_years=sum(max(v)-min(v)>=10 for v in identity_ages.values() if v),
            at_least_20_years=sum(max(v)-min(v)>=20 for v in identity_ages.values() if v),
            at_least_20_years_both_adult=sum(max(a)-min(a)>=20 for v in identity_ages.values() if (a:=[x for x in v if x>=20]))),
        limitations=['Bounding-box pixels estimate usable detail and do not verify sharpness or identity.',
            'Subject and age given by publisher; no ethnicity labels. Independent evaluation requires cross-corpus duplicate audit and manual pair verification.',
            'Archive has no per-image license manifest. Keep pictures local to this research audit pending individual attribution.'],
        included_in_training=False)
    write_json(args.output,report)
    print('CAF_AUDITED',report['identity_folders'],report['labeled_files'],report['quality'],report['potential_hd_pair_subjects'])


if __name__=='__main__':
    main()

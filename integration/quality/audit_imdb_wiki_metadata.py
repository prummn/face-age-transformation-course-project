"""Audit publisher age/face-box metadata only, without inferring demographic labels.

Requires SciPy. Inputs are local author-provided .mat files; no network, image
processing, training admission or redistribution is performed by this script.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path


EXPECTED = {
    'imdb': 'be67a7d84df5f90893884cd7f206c8c935ceeaa2afc09b6a96076fd7f982a035',
    'wiki': 'bd6adc5eb6a31c15d36d93a1a9c63940f27b62ca51b9028606b3a57d6a33993c',
}


def audit(path, kind):
    import numpy as np
    from scipy.io import loadmat
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED[kind]:
        raise ValueError('Different metadata version; audit the source before accepting it')
    data = loadmat(path, squeeze_me=True, struct_as_record=False)[kind]
    valid, invalid = [], 0
    for i in range(len(data.dob)):
        try:
            birth = datetime.fromordinal(int(data.dob[i]) - 366)
            photo_year = int(data.photo_taken[i])
            age = photo_year - birth.year
            box = np.asarray(data.face_location[i], dtype=float).ravel()
            if len(box) != 4 or not np.isfinite(box).all() or not 0 <= age <= 120:
                invalid += 1
                continue
            short = float(min(box[2] - box[0] + 1, box[3] - box[1] + 1))
            if short <= 0:
                invalid += 1
                continue
            score, second = float(data.face_score[i]), float(data.second_face_score[i])
            single = np.isfinite(score) and score > 1 and (np.isnan(second) or second <= 0)
            identity = int(data.celeb_id[i]) if kind == 'imdb' else str(data.name[i])
            valid.append(dict(index=i, path=str(data.full_path[i]), publisher_nominal_age=age,
                face_short_edge=short, conservative_single_face=bool(single), identity=identity,
                training_admitted=False))
        except (OverflowError, ValueError, IndexError):
            invalid += 1
    senior = [r for r in valid if 65 <= r['publisher_nominal_age'] <= 85]
    single = [r for r in senior if r['conservative_single_face']]
    hd = [r for r in single if r['face_short_edge'] >= 512]
    bins = Counter('65-69' if r['publisher_nominal_age'] < 70 else '70-74' if r['publisher_nominal_age'] < 75
                   else '75-84' if r['publisher_nominal_age'] < 85 else '85' for r in hd)
    result = dict(dataset=kind, metadata_sha256=digest, source_rows=len(data.dob), valid_age_and_bbox=len(valid),
        invalid=invalid, age65to85=len(senior), conservative_single_face_age65to85=len(single),
        single_face_bbox_ge512=len(hd), single_face_bbox_ge700=sum(r['face_short_edge'] >= 700 for r in single),
        candidate_age_bins=dict(bins), single_face_publisher_ids=len({r['identity'] for r in single}),
        hd_candidate_publisher_ids=len({r['identity'] for r in hd}),
        face_short_edge_percentiles=np.percentile([r['face_short_edge'] for r in single], [0, 25, 50, 75, 100]).tolist() if single else [],
        limitations=['Publisher photo year minus birth year is approximate and may be mislabeled.',
            'Face boxes are publisher metadata, not verified native pixels or texture quality.',
            'No ethnicity labels; no Asian coverage claim.',
            'No images admitted; any later image retrieval must preserve rights, provenance and protected identity groups.'])
    return result, hd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=EXPECTED)
    parser.add_argument('mat', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--candidates', type=Path, help='Optional local-only row indices/paths, never an admitted training list')
    args = parser.parse_args()
    if args.output.exists() or (args.candidates and args.candidates.exists()):
        raise FileExistsError('Preserve previous audits; use new output paths')
    result, rows = audit(args.mat, args.kind)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    if args.candidates:
        args.candidates.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

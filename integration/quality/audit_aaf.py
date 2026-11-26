"""Audit the author-linked AAF archive locally without admitting training data.

Checks every ZIP member CRC, paired inventories, published splits and labels,
decoded dimensions, exact duplicates, and native face sizes on all age65+ images.
Dataset filenames are publisher age/individual claims, not verified identities or
chronological ages. No GPU or training dependencies are used.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import time
import zipfile

from probe_inversion import sha256, write_json

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_BYTES = 365164027
UPLOAD_SHA256 = '0dcebf92031e43b987d78fb3621ab7f0bcb3af62aaf8871de060681c1f08db77'
PREFIX = 'All-Age-Faces Dataset/'
FOLDERS = {'original': 'original images', 'aligned': 'aglined faces'}


def percentiles(values):
    import numpy as np
    return {str(q): float(v) for q, v in zip([0, 10, 50, 90, 100],
                                            np.percentile(values, [0, 10, 50, 90, 100]))} if values else {}


def screen_age_candidates(rows, output):
    """Exact-byte/pixel grouping only; preserve publisher validation membership.

    These are quarantine candidates, never a claim of identity-separated splits.
    Missing younger-face QC and all approximate-identity checks remain explicit.
    """
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def join(i,j):
        a,b = sorted([find(i),find(j)])
        parent[b] = a
    seen = {}
    for i,row in enumerate(rows):
        for label in FOLDERS:
            key = (label,row[label]['pixel_sha256'])
            if key in seen:join(i,seen[key])
            else:seen[key] = i
    groups = defaultdict(list)
    for i,row in enumerate(rows):groups[find(i)].append(row)
    candidates = {'train': [], 'val': []}
    rejected = []
    conflict_age_groups = conflict_gender_groups = cross_split_groups = 0
    for group in groups.values():
        age_conflict = len({r['publisher_age'] for r in group})>1
        gender_conflict = len({r['publisher_gender'] for r in group})>1
        cross_split = len({r['split'] for r in group})>1
        conflict_age_groups += age_conflict
        conflict_gender_groups += gender_conflict
        cross_split_groups += cross_split
        if age_conflict or gender_conflict:
            for row in group:
                rejected.append(dict(name=row['name'],publisher_age=row['publisher_age'],split=row['split'],
                    reason='exact_group_label_conflict',age_conflict=age_conflict,gender_conflict=gender_conflict))
            continue
        # Keep the official validation split protected; never move its photo to train.
        split = 'val' if any(r['split']=='val' for r in group) else 'train'
        kept = False
        group_id = 'aaf_exact_' + hashlib.sha256('|'.join(sorted(r['name'] for r in group)).encode()).hexdigest()[:16]
        for row in sorted(group,key=lambda r:r['name']):
            reason = None
            if row['split'] != split:reason = 'exact_group_touches_publisher_validation'
            elif row['publisher_age']<18:reason = 'outside_adult_course_scope'
            elif row.get('face_qc',{}).get('multiple_prominent_faces'):reason = 'multiple_prominent_faces'
            elif kept:reason = 'exact_duplicate_within_split'
            if reason:
                rejected.append(dict(name=row['name'],publisher_age=row['publisher_age'],split=row['split'],reason=reason))
            else:
                candidates[split].append(dict(row,exact_group=group_id,
                    allowed_candidate_purpose='Age-head audit/calibration only; not texture supervision',
                    texture_training_eligible=False,training_admitted=False,
                    pending_checks=['Approximate duplicate and source/identity grouping across splits and all prior datasets',
                                    'Age provenance and label-noise handling',
                                    'Face/pose/quality checks for ages below65']))
                kept = True
    def summary(items):
        return dict(records=len(items),age65plus=sum(r['publisher_age']>=65 for r in items),
                    age_counts=dict(sorted(Counter(str(r['publisher_age']) for r in items).items(),key=lambda p:int(p[0]))))
    for split,records in candidates.items():
        (output/f'age-{split}.candidates.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    source_hashes = {s:{r[k]['pixel_sha256'] for r in rr for k in FOLDERS} for s,rr in candidates.items()}
    assert not source_hashes['train'] & source_hashes['val']
    assert sum(map(len,candidates.values()))+len(rejected) == len(rows)
    result = dict(stage='Exact-duplicate cleaned adult quarantine candidates; not approved training splits',
        source_records=len(rows),grouping='Transitive union of identical decoded original or aligned pixels',
        exact_groups=len(groups),conflicting_age_groups=conflict_age_groups,
        conflicting_gender_groups=conflict_gender_groups,cross_split_exact_groups=cross_split_groups,
        candidates={s:summary(r) for s,r in candidates.items()},
        rejections_by_reason=dict(Counter(r['reason'] for r in rejected)),rejected_records=len(rejected),
        rejected_age65plus=sum(r['publisher_age']>=65 for r in rejected),
        exact_train_val_overlap_after_screen=0,
        approximate_identity_isolation_done=False,all_age_labels_verified=False,
        training_admitted=0,texture_training_admitted=0)
    write_json(output/'exact-age-screen.json',result)
    write_json(output/'exact-age-rejections.json',rejected)
    return result


def audit(archive_path, output):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(2)
    digest = sha256(archive_path)
    if archive_path.stat().st_size != EXPECTED_BYTES or digest != UPLOAD_SHA256:
        raise ValueError('Archive differs from the recorded upload; retain it and audit its provenance separately')
    detector_path = ROOT / 'deployment/model-files/face-age/models/yunet/face_detection_yunet_2023mar.onnx'
    detector = cv2.FaceDetectorYN.create(str(detector_path), '', (320, 320), .65, .3, 100)
    rows, membership, annotation_sha = [], {}, {}
    keypoint_counts = Counter()
    split_members = defaultdict(set)
    member_errors = []
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [i.filename for i in infos]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP member names')
        for info in infos:
            p = PurePosixPath(info.filename)
            if p.is_absolute() or '..' in p.parts or info.flag_bits & 1:
                raise ValueError('Unsafe or encrypted archive member: ' + info.filename)
        bad = archive.testzip()
        if bad:
            raise ValueError('ZIP CRC failure: ' + bad)
        print('AAF_CRC_VERIFIED', sum(not i.is_dir() for i in infos), flush=True)
        for split in ['train', 'val']:
            path = PREFIX + 'image sets/' + split + '.txt'
            data = archive.read(path)
            annotation_sha[split] = hashlib.sha256(data).hexdigest()
            for line in data.decode('utf-8').splitlines():
                name, sex = line.split()
                if name in membership or sex not in {'0', '1'}:
                    raise ValueError('Duplicate split membership or invalid published gender')
                membership[name] = dict(split=split, publisher_gender='female' if sex == '0' else 'male')
                split_members[split].add(name)
        inventories = {}
        for label, folder in FOLDERS.items():
            inventories[label] = {PurePosixPath(i.filename).name for i in infos
                                  if i.filename.startswith(PREFIX+folder+'/') and not i.is_dir()}
        keys = {PurePosixPath(i.filename).stem for i in infos
                if i.filename.startswith(PREFIX+'key points/') and not i.is_dir()}
        if inventories['original'] != inventories['aligned'] or inventories['original'] != set(membership):
            raise ValueError('Original/aligned/split inventories differ')
        if keys != {Path(n).stem for n in membership} or len(membership) != 13322:
            raise ValueError('Unexpected landmark inventory or record count')
        for index, name in enumerate(sorted(membership)):
            match = re.fullmatch(r'(\d{5})A(\d{2})\.jpg', name)
            if not match:
                raise ValueError('Unexpected publisher filename: ' + name)
            subject, age = map(int, match.groups())
            if not 2 <= age <= 80:
                raise ValueError('Published age outside stated range')
            row = dict(name=name, publisher_individual_id=subject, publisher_age=age,
                       age_kind='Publisher filename age; exact age provenance not established',
                       subject_kind='Publisher says each image is a different individual; not verified',
                       **membership[name], training_admitted=False)
            if row['publisher_gender'] != ('female' if subject <= 7380 else 'male'):
                raise ValueError('Publisher filename/gender range conflict')
            images = {}
            for label, folder in FOLDERS.items():
                path = PREFIX + folder + '/' + name
                data = archive.read(path)
                with Image.open(BytesIO(data)) as image:
                    image.load()
                    if image.format != 'JPEG':
                        raise ValueError('Unexpected image encoding')
                    rgb = np.array(image.convert('RGB'))
                    images[label] = rgb
                    row[label] = dict(member=path, bytes=len(data),
                                      sha256=hashlib.sha256(data).hexdigest(),
                                      pixel_sha256=hashlib.sha256(str(image.size).encode()+rgb.tobytes()).hexdigest(),
                                      size=list(image.size), short_edge=min(image.size))
            landmark_path = PREFIX + 'key points/' + Path(name).stem + '.txt'
            points = np.loadtxt(BytesIO(archive.read(landmark_path)))
            if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
                raise ValueError('Invalid landmarks: ' + name)
            keypoint_counts[len(points)] += 1
            # The dimension bound is rigorous even when face detection fails.
            row['native512_possible_by_canvas'] = row['original']['short_edge'] >= 512
            if age >= 65 or row['native512_possible_by_canvas']:
                rgb = images['original']
                h, w = rgb.shape[:2]
                scale = min(1, 1400/max(h,w))
                small = cv2.resize(rgb, (round(w*scale),round(h*scale)), interpolation=cv2.INTER_AREA)
                detector.setInputSize((small.shape[1],small.shape[0]))
                _, faces = detector.detect(cv2.cvtColor(small,cv2.COLOR_RGB2BGR))
                if faces is None:
                    row['face_qc'] = dict(detected=False, purpose='Age65+ native-pixel audit')
                else:
                    faces = sorted(faces,key=lambda f:f[2]*f[3],reverse=True)
                    face = faces[0].copy();face[:14] /= scale
                    x,y,bw,bh = map(float,face[:4])
                    left,top = max(0,int(x)),max(0,int(y))
                    right,bottom = min(w,int(np.ceil(x+bw))),min(h,int(np.ceil(y+bh)))
                    native = rgb[top:bottom,left:right]
                    multiple = bool(len(faces)>1 and faces[1][2]*faces[1][3] > faces[0][2]*faces[0][3]/6)
                    sharp = float(cv2.Laplacian(cv2.cvtColor(native,cv2.COLOR_RGB2GRAY),cv2.CV_64F).var()) if native.size else None
                    row['face_qc'] = dict(detected=True, significant_faces=int(1+sum(
                        f[2]*f[3]>faces[0][2]*faces[0][3]/6 for f in faces[1:])),
                        native_box_short_edge=float(min(bw,bh)), visible_crop_short_edge=min(native.shape[:2]),
                        face_rect=[x,y,bw,bh], face_confidence=float(face[-1]),
                        native_laplacian_variance=sharp, multiple_prominent_faces=multiple,
                        single_face512_candidate=not multiple and min(bw,bh)>=512 and min(native.shape[:2])>=512)
            rows.append(row)
            if (index+1)%1000 == 0:
                print('AAF_AUDIT',index+1,'/',len(membership),flush=True)
        # Read any accompanying terms, without mistaking the demo landmarks for a licence.
        terms_members = [i.filename for i in infos if not i.is_dir() and
                         re.search(r'(?:readme|licen[cs]e|terms|agreement)',PurePosixPath(i.filename).name,re.I)]
    manifests = output/'records.jsonl'
    manifests.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    duplicate_summary = {}
    for label in FOLDERS:
        duplicate_summary[label] = {}
        for key in ['sha256','pixel_sha256']:
            groups = defaultdict(list)
            for row in rows:groups[row[label][key]].append(row)
            dup = [g for g in groups.values() if len(g)>1]
            duplicate_summary[label][key] = dict(
                redundant_rows=sum(len(g)-1 for g in dup), duplicate_groups=len(dup),
                shared_train_val_groups=sum(len({r['split'] for r in g})>1 for g in dup),
                conflicting_age_groups=sum(len({r['publisher_age'] for r in g})>1 for g in dup))
    def summarize(selection):
        stats = dict(records=len(selection),by_published_gender=dict(Counter(r['publisher_gender'] for r in selection)),
                     by_split=dict(Counter(r['split'] for r in selection)))
        stats['by_published_age'] = dict(sorted(Counter(str(r['publisher_age']) for r in selection).items(),key=lambda p:int(p[0])))
        for label in FOLDERS:
            sizes = Counter(tuple(r[label]['size']) for r in selection)
            stats[label] = dict(short_edge_percentiles=percentiles([r[label]['short_edge'] for r in selection]),
                                short_edge512_count=sum(r[label]['short_edge']>=512 for r in selection),
                                common_sizes=[dict(size=list(size),count=n) for size,n in sizes.most_common(10)])
        tested = [r['face_qc'] for r in selection if 'face_qc' in r]
        if tested:
            valid = [r for r in tested if r['detected']]
            stats['face_qc'] = dict(tested=len(tested),detected=len(valid),
                no_face=len(tested)-len(valid),multiple_prominent_faces=sum(r['multiple_prominent_faces'] for r in valid),
                detected_box_short_edge_percentiles=percentiles([r['native_box_short_edge'] for r in valid]),
                visible_crop_short_edge_percentiles=percentiles([r['visible_crop_short_edge'] for r in valid]),
                single_face512_candidates=sum(r['single_face512_candidate'] for r in valid))
        return stats
    age_ranges = [('55-64',55,64),('65-69',65,69),('70-74',70,74),('75-80',75,80),('65-80',65,80)]
    result = dict(schema=1,archive_name=archive_path.name,archive_bytes=archive_path.stat().st_size,
        archive_sha256=digest,archive_hash_kind='Local upload fingerprint; no publisher independent checksum provided',
        publisher_advertised_bytes_match=True,zip_crc_all_members=True,
        non_directory_members=sum(not i.is_dir() for i in infos),encrypted_members=0,
        paired_inventories_equal=True,unique_publisher_ids=len({r['publisher_individual_id'] for r in rows}),
        published_gender_counts=dict(Counter(r['publisher_gender'] for r in rows)),landmark_point_counts=dict(keypoint_counts),
        annotation_sha256=annotation_sha,internal_terms_members=terms_members,
        all=summarize(rows),age_groups={label:summarize([r for r in rows if lo<=r['publisher_age']<=hi]) for label,lo,hi in age_ranges},
        exact_duplicates=duplicate_summary,manifest_sha256=sha256(manifests),detector_sha256=sha256(detector_path),
        training_admitted=0,gpu_used=False,elapsed_seconds=round(time.monotonic()-started,3),
        decision='No original image canvas reaches512 short edge, so none meets the native512 texture criterion. Age-supervision candidate only; no training started.',
        limitations=['Publisher original images are delivered images, not proof of unresized camera originals.',
                     'Publisher mostly-Asian is a dataset-level description; no individual ethnicity inferred.',
                     'Published filename age provenance is not established as chronological age.',
                     'Exact duplicate audit is not approximate identity isolation; no new training split made.',
                     'Public README names age prediction/gender classification; generative-use terms remain unclear.'])
    write_json(output/'audit.json',result)
    # A few local-only examples selected before any model output. No public route.
    examples = []
    for lo,hi in [(65,69),(70,74),(75,80)]:
        for gender in ['female','male']:
            subset = [r for r in rows if lo<=r['publisher_age']<=hi and r['publisher_gender']==gender and r.get('face_qc',{}).get('detected') and not r['face_qc']['multiple_prominent_faces']]
            subset.sort(key=lambda r:(r['face_qc']['visible_crop_short_edge'],r['name']))
            if subset:examples.append(subset[len(subset)//2])
    canvas = Image.new('RGB',(3*410,2*455),'white')
    draw = ImageDraw.Draw(canvas)
    with zipfile.ZipFile(archive_path) as archive:
        for index,row in enumerate(examples):
            with Image.open(BytesIO(archive.read(row['original']['member']))) as image:
                image=image.convert('RGB');image.thumbnail((393,393))
                x=(index%3)*410;y=(index//3)*455
                canvas.paste(image,(x,y+48))
                draw.text((x+5,y+3),f"{row['name']} {row['publisher_gender']} label={row['publisher_age']}",fill='black')
                draw.text((x+5,y+22),f"canvas={row['original']['size']} face={row['face_qc']['visible_crop_short_edge']}px",fill='black')
    canvas.save(output/'source-quality-six.jpg',quality=95)
    write_json(output/'source-quality-six.json',[dict(name=r['name'],original=r['original'],face_qc=r['face_qc']) for r in examples])
    screen_age_candidates(rows,output)
    (output/'DONE').write_text('CPU source and resolution audit complete; no training admission\n')
    print('AAF_AUDIT_COMPLETE',json.dumps(dict(rows=len(rows),age65plus=result['age_groups']['65-80']['records'],
        native512=result['all']['original']['short_edge512_count'],seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();audit(a.archive.resolve(),a.output.resolve())

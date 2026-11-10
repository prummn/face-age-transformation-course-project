#!/usr/bin/env python3
"""Read-only, CPU-only audit of existing face-age data/checkpoints/evaluation PNGs.

No model is loaded. Existing demographic values are classifier metadata, not
verified personal attributes. Fixed cheek-region color is a geometric proxy,
not a skin segmentation or ethnicity determination.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit

import cv2
import numpy as np
import pandas as pd

cv2.setNumThreads(1)


def finite_stats(values):
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {"n": 0}
    return {"n": len(a), "mean": round(float(a.mean()), 5), "std": round(float(a.std()), 5), "min": round(float(a.min()), 5), "p10": round(float(np.quantile(a, .1)), 5), "p25": round(float(np.quantile(a, .25)), 5), "median": round(float(np.median(a)), 5), "p75": round(float(np.quantile(a, .75)), 5), "p90": round(float(np.quantile(a, .9)), 5), "max": round(float(a.max()), 5)}


def corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 3 or np.std(a[good]) == 0 or np.std(b[good]) == 0:
        return None
    return round(float(np.corrcoef(a[good], b[good])[0, 1]), 5)


def photograph_id(url):
    parts = urlsplit(str(url))
    if "staticflickr.com" in (parts.hostname or "") or "static.flickr.com" in (parts.hostname or ""):
        name = Path(parts.path).name
        candidate = name.split("_")[0]
        return candidate if candidate.isdigit() else None
    return None


def image_features(path):
    if not Path(path).is_file():
        return None
    image = cv2.imread(str(path))
    if image is None:
        return None
    h, w = image.shape[:2]
    small = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    # Bilateral cheek windows in FFHQ coordinates. Pose/hair can contaminate this
    # proxy; aggregate statistics are paired, not a true skin-color metric.
    pixels = np.concatenate([lab[123:159, 65:105].reshape(-1, 3), lab[123:159, 151:191].reshape(-1, 3)])
    cheek = np.median(pixels, axis=0)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    coeffs = cv2.dct(cv2.resize(gray.astype(np.float32), (32, 32)))[:8, :8]
    threshold = np.median(coeffs.flatten()[1:])
    phash = np.packbits((coeffs > threshold).flatten()).tobytes().hex()
    # A true channel swap dramatically increases deviation; report raw BGR order
    # explicitly and do not silently relabel it as RGB.
    rgb_mean = small[:, :, ::-1].mean(axis=(0, 1))
    return {"width": w, "height": h, "cheek_lab": [float(v) for v in cheek], "rgb_mean": [float(v) for v in rgb_mean], "blur_laplacian": float(cv2.Laplacian(gray, cv2.CV_64F).var()), "phash64": phash}


def near_cross_split(train, evaluation, hashes, threshold=6):
    if not train or not evaluation:
        return {"pairs": 0}
    byte_train = np.stack([np.frombuffer(bytes.fromhex(hashes[k]), dtype=np.uint8) for k in train])
    bits = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    matched, pairs, bestdist, examples = set(), 0, [], []
    for key in evaluation:
        byte_eval = np.frombuffer(bytes.fromhex(hashes[key]), dtype=np.uint8)
        distances = bits[np.bitwise_xor(byte_train, byte_eval)].sum(axis=1)
        bestdist.append(int(distances.min()))
        where = np.flatnonzero(distances <= threshold)
        if len(where):
            matched.add(key)
        pairs += len(where)
        for j in where[:3]:
            if len(examples) < 30:
                examples.append({"eval_key": key, "train_key": train[j], "hamming": int(distances[j])})
    return {"threshold": threshold, "pairs": pairs, "eval_images_with_candidate": len(matched), "nearest_distance": finite_stats(bestdist), "examples": examples, "interpretation": "Candidate near-duplicate photographs only; visually review before declaring leakage. This does not identify distinct images of the same person."}


def split_summary(frame):
    result = {"n": len(frame)}
    for column in ["age_exp", "p_asian", "p_ea", "p_sea", "p_female", "p_minor", "fw", "blur", "sat", "face_frac", "W", "H"]:
        result[column] = finite_stats(frame[column])
    ages = pd.cut(frame.age_exp, [-float('inf'),20,30,40,50,60,70,float('inf')], right=False)
    result["pseudo_age_bins"] = {str(k): int(v) for k, v in ages.value_counts(sort=False).items()}
    result["source_face_short_edge"] = {f"below_{n}": int((frame.fw < n).sum()) for n in [128,192,256,384,512]}
    result["pseudo_gender"] = {"female_probability_ge_0_5": int((frame.p_female >= .5).sum()), "female_probability_0_2_to_0_8": int(frame.p_female.between(.2,.8).sum())}
    result["regional_classifier_higher_probability"] = {"p_ea_gt_p_sea": int((frame.p_ea > frame.p_sea).sum()), "p_sea_ge_p_ea": int((frame.p_sea >= frame.p_ea).sum())}
    return result


def run(args):
    base, output = Path(args.base).resolve(), Path(args.output).resolve()
    if output == base or base in output.parents:
        raise ValueError("Audit output must be outside the original read-only face-age project.")
    output.mkdir(parents=True, exist_ok=True)
    index = pd.read_parquet(base / "data/index.parquet")
    stage1 = pd.read_parquet(base / "data/stage1_tierA.parquet")
    stage3 = pd.read_parquet(base / "data/stage3a.parquet")
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "Existing artifacts only; CPU statistics; no inference, training, downloads, or writes in face-age.", "dataset": {"stage1": len(stage1), "stage3": len(stage3), "stage3_reasons": {str(k): int(v) for k,v in stage3.why.value_counts().items()}, "index": len(index), "columns": list(index.columns), "license_values": dict(Counter(index.license)), "identity_fields": [c for c in index.columns if any(t in c.lower() for t in ['identity','subject','person','author'])]}}
    report["dataset"]["splits"] = {name: split_summary(group) for name,group in index.groupby('split')}
    report["dataset"]["exact_duplicate_urls"] = int(index.url.duplicated().sum())
    report["dataset"]["exact_duplicate_raw_dhash"] = int(index.dhash.duplicated().sum())
    photo_ids = index.url.map(photograph_id)
    report["dataset"]["flickr_photo_id_available"] = int(photo_ids.notna().sum())
    report["dataset"]["duplicate_flickr_photo_ids"] = int(photo_ids.dropna().duplicated().sum())
    cross_groups = []
    for photo, group in index.assign(photo_id=photo_ids).dropna(subset=['photo_id']).groupby('photo_id'):
        if group.split.nunique() > 1:
            cross_groups.append(group.key.tolist())
    report["dataset"]["cross_split_same_flickr_photo"] = {"n_groups": len(cross_groups), "keys": cross_groups[:30]}
    paths = sorted((base / "data/aligned").glob('*/*.png'))
    print(f"Analyzing {len(paths)} aligned photos (CPU, {args.workers} I/O workers)", flush=True)
    cache_path = output / "aligned_features.json"
    # Recompute from current images on every audit; never trust a stale feature
    # file after a dataset update. The exported features feed manifest creation.
    features = {}
    with ThreadPoolExecutor(args.workers) as pool:
        for p, result in zip(paths, pool.map(image_features, paths)):
            if result:
                features[p.stem] = {**result, "split": p.parent.name}
    cache_path.write_text(json.dumps(features))
    report["dataset"]["aligned_count"] = len(features)
    report["dataset"]["missing_aligned_keys"] = sorted(set(index.key) - set(features))
    report["dataset"]["aligned_dimensions"] = dict(Counter(f"{v['width']}x{v['height']}" for v in features.values()))
    report["dataset"]["aligned_blur"] = {split: finite_stats(v['blur_laplacian'] for v in features.values() if v['split']==split) for split in ['train','eval']}
    hashes = {k:v['phash64'] for k,v in features.items()}
    tr = [k for k,v in features.items() if v['split']=='train']
    ev = [k for k,v in features.items() if v['split']=='eval']
    report["dataset"]["aligned_phash_cross_split"] = near_cross_split(tr,ev,hashes)
    raw_hashes = dict(zip(index.key,index.dhash))
    report["dataset"]["raw_dhash_cross_split"] = near_cross_split(list(index[index.split=='train'].key), list(index[index.split=='eval'].key),raw_hashes,12)
    rows=[]
    for row in index.to_dict('records'):
        if row['key'] in features:
            L,a,b = features[row['key']]['cheek_lab']
            rows.append({'key':row['key'],'split':row['split'],'pseudo_age':row['age_exp'],'p_female':row['p_female'],'p_ea':row['p_ea'],'p_sea':row['p_sea'],'L':L,'a':a,'b':b})
    color_frame=pd.DataFrame(rows)
    report['dataset']['pseudo_age_cheek_color_correlations']={key:corr(color_frame.pseudo_age,color_frame[key]) for key in ['L','a','b']}
    report['dataset']['pseudo_age_cheek_color_bins']={str(name):{'n':len(group), **{key:finite_stats(group[key]) for key in ['L','a','b']}} for name,group in color_frame.groupby(pd.cut(color_frame.pseudo_age,[20,30,40,50,60,70,80],right=False),observed=False)}
    report['training']={}
    for name in ['ft_l1','ft_l5']:
        folder=base/'out'/name
        log=[json.loads(line) for line in (folder/'train_log.jsonl').read_text().splitlines() if line.strip()]
        rec={'config':json.loads((folder/'config.json').read_text()),'log_rows':len(log),'skipped':sum(r['skipped'] for r in log),'last_step':log[-1]['step'],'successful_updates':sum(not r['skipped'] for r in log),'first25':{},'last25':{}}
        for metric in ['loss','aging','id','rec_l2','lpips','id_cycle','grad_norm_preclip','p_asian_in_ff','p_asian_out_ff']:
            rec['first25'][metric]=finite_stats(float(row[metric]) for row in log[:25])
            rec['last25'][metric]=finite_stats(float(row[metric]) for row in log[-25:])
        report['training'][name]=rec
    report['existing_summary']=json.loads((base/'out/summary.json').read_text())
    (output / 'dataset_summary_partial.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    report['paired_color_audit']={'method':'Fixed bilateral cheek ROI medians in OpenCV float CIELAB (L in 0..100; a/b signed). Delta = generated - same input. Not a segmentation, medical skin measure, ethnic label, or perceptual quality score. All old eval outputs are 256px; native1024 needs separate evaluation.','models':{}}
    for folder in sorted((base/'out').glob('eval_*')):
        if not (folder/'records.json').exists():
            continue
        records=json.loads((folder/'records.json').read_text())
        print(f"Analyzing paired colors: {folder.name} ({len(records)} outputs)",flush=True)
        eval_paths=[Path(r['file']) if Path(r['file']).is_absolute() else base/Path(r['file']) for r in records]
        paired=[]
        with ThreadPoolExecutor(args.workers) as pool:
            for record,result in zip(records,pool.map(image_features,eval_paths)):
                key=Path(record['input']).stem
                if not result or key not in features:
                    continue
                original=features[key]
                delta=np.array(result['cheek_lab'])-np.array(original['cheek_lab'])
                paired.append({'key':key,'tag':record['tag'],'delta_L':float(delta[0]),'delta_a':float(delta[1]),'delta_b':float(delta[2]),'delta_E76':float(np.linalg.norm(delta)),'chroma_delta':float(np.linalg.norm(delta[1:])),'source_L':original['cheek_lab'][0],'output_L':result['cheek_lab'][0],'blur_ratio':result['blur_laplacian']/max(original['blur_laplacian'],1e-6),'id_cos':record['id_cos'],'age_est':record['age_est'],'target_age':record['target_age']})
        frame=pd.DataFrame(paired)
        if len(frame) != len(records):
            raise RuntimeError(f'{folder}: expected {len(records)} images, read {len(frame)}')
        model={}
        for tag,group in frame.groupby('tag'):
            model[tag]={'n':len(group),**{metric:finite_stats(group[metric]) for metric in ['delta_L','delta_a','delta_b','delta_E76','chroma_delta','blur_ratio','id_cos','age_est']},'source_lightness_vs_change':corr(group.source_L,group.delta_L),'examples_largest_chroma_shift':group.nlargest(5,'chroma_delta')[['key','delta_L','delta_a','delta_b','chroma_delta']].to_dict('records')}
        report['paired_color_audit']['models'][folder.name]=model
    output_path=output/'数据模型审计.json'
    output_path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('AUDIT_COMPLETE '+str(output_path),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',default='/data/home/1120231322/scow/face-age')
    parser.add_argument('--output',required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    run(args)

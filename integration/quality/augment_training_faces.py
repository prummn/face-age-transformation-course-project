#!/usr/bin/env python3
"""Build a frozen CPU-only augmentation bank from the quality-filtered train parents.

Each parent gets one deterministic Lab-L illumination attempt and one JPEG
attempt. No geometry, a/b edit, sharpening, smoothing, age edit or super-resolution.
RGB conversion/JPEG can change decoded chroma; gates measure and reject this.
Rejected attempts are recorded, never regenerated. No image/model is uploaded.
Train sampling must remain parent-based; accepted rows are not new identities.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import platform
import textwrap
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, __version__ as PIL_VERSION

VERSION = 'controlled_training_bank_v1'
KINDS = ('illumination', 'jpeg')
GATES = {'max_cheek_ab_shift': .8, 'max_global_mean_ab_shift': 1.,
         'max_L_abs_p99': 5.5, 'max_L_abs_mean': 3.5,
         'max_new_clipped_fraction': .005,
         'min_laplacian_ratio_illumination': .85,
         'min_laplacian_ratio_jpeg': .75}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def read_manifest(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def stable_seed(seed, parent_key, kind):
    raw = json.dumps([VERSION, int(seed), str(parent_key), kind], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big')


def rgb(path):
    with Image.open(path) as image:
        if image.mode != 'RGB' or image.size != (256, 256):
            raise ValueError('Expected existing aligned 256x256 RGB input: ' + str(path))
        return np.asarray(image).copy()


def lab(image):
    return cv2.cvtColor(image.astype(np.float32) / 255., cv2.COLOR_RGB2LAB)


def candidate(source, kind, seed):
    """Return encoded bytes and recipe. Seed is independent of traversal order."""
    rng = np.random.Generator(np.random.PCG64(seed))
    recipe = {'kind': kind, 'version': VERSION, 'geometry': 'unchanged',
              'rng': 'numpy.PCG64', 'nominal_ab_edit': False}
    buffer = io.BytesIO()
    if kind == 'illumination':
        offset = float(rng.choice([-1., 1.]) * rng.uniform(1., 3.))
        amplitude = float(rng.uniform(.5, 1.5))
        angle = float(rng.uniform(0., 2 * np.pi))
        yy, xx = np.mgrid[-1:1:complex(source.shape[0]), -1:1:complex(source.shape[1])]
        projection = (xx * np.cos(angle) + yy * np.sin(angle)) / (abs(np.cos(angle)) + abs(np.sin(angle)))
        value = lab(source)
        value[:, :, 0] = np.clip(value[:, :, 0] + offset + amplitude * projection, 0., 100.)
        generated = cv2.cvtColor(value, cv2.COLOR_LAB2RGB)
        generated = np.rint(np.clip(generated, 0., 1.) * 255).astype(np.uint8)
        Image.fromarray(generated).save(buffer, format='PNG', compress_level=6)
        recipe.update({'L_offset': offset, 'L_gradient_amplitude': amplitude,
                       'gradient_angle_radians': angle, 'format': 'PNG'})
    elif kind == 'jpeg':
        quality = int(rng.integers(96, 99))
        Image.fromarray(source).save(buffer, format='JPEG', quality=quality,
                                     subsampling=0, optimize=False, progressive=False)
        recipe.update({'quality': quality, 'subsampling': 0, 'format': 'JPEG',
                       'optimize': False, 'progressive': False})
    else:
        raise ValueError('Unknown augmentation kind: ' + kind)
    return buffer.getvalue(), recipe


def decode(encoded):
    with Image.open(io.BytesIO(encoded)) as image:
        if image.mode != 'RGB' or image.size != (256, 256):
            raise ValueError('Candidate changed RGB mode or geometry')
        return np.asarray(image).copy()


def measure(source, generated):
    if source.shape != generated.shape:
        raise ValueError('Geometry changed')
    a, b = lab(source), lab(generated)
    delta = b - a
    cheek_shifts = []
    # Same fixed FFHQ cheek windows as the existing training audit; not skin truth.
    for x0, x1 in ((65, 105), (151, 191)):
        before = np.median(a[123:159, x0:x1].reshape(-1, 3), axis=0)
        after = np.median(b[123:159, x0:x1].reshape(-1, 3), axis=0)
        cheek_shifts.append(float(np.linalg.norm((after - before)[1:])))
    L_abs = abs(delta[:, :, 0])
    # A fixed subpixel prefilter suppresses uint8/Lab quantization edges, which
    # can otherwise make a visibly blurred image look sharper numerically.
    source_lap = float(cv2.Laplacian(cv2.GaussianBlur(a[:, :, 0], (0, 0), .7), cv2.CV_32F).var())
    output_lap = float(cv2.Laplacian(cv2.GaussianBlur(b[:, :, 0], (0, 0), .7), cv2.CV_32F).var())
    src_clip = (a[:, :, 0] < 1.5) | (a[:, :, 0] > 98.5)
    out_clip = (b[:, :, 0] < 1.5) | (b[:, :, 0] > 98.5)
    src_channel_clip = ((source <= 1) | (source >= 254)).any(axis=2)
    out_channel_clip = ((generated <= 1) | (generated >= 254)).any(axis=2)
    return {'left_cheek_ab_shift': cheek_shifts[0], 'right_cheek_ab_shift': cheek_shifts[1],
            'max_cheek_ab_shift': max(cheek_shifts),
            'global_mean_ab_shift': float(np.linalg.norm(delta[:, :, 1:], axis=2).mean()),
            'L_abs_mean': float(L_abs.mean()), 'L_abs_p99': float(np.percentile(L_abs, 99)),
            'L_abs_max': float(L_abs.max()), 'L_mean_signed': float(delta[:, :, 0].mean()),
            'new_luminance_clipped_fraction': float((out_clip & ~src_clip).mean()),
            'new_channel_clipped_fraction': float((out_channel_clip & ~src_channel_clip).mean()),
            'source_L_laplacian_variance': source_lap, 'output_L_laplacian_variance': output_lap,
            'L_laplacian_prefilter_sigma': .7,
            'L_laplacian_ratio': output_lap / max(source_lap, 1e-9),
            'geometry_unchanged': True}


def reject_reasons(metrics, kind):
    if not all(np.isfinite(v) for v in metrics.values() if isinstance(v, (int, float))):
        return ['nonfinite_metric']
    reasons = []
    for name, limit in [('max_cheek_ab_shift', GATES['max_cheek_ab_shift']),
                        ('global_mean_ab_shift', GATES['max_global_mean_ab_shift']),
                        ('L_abs_mean', GATES['max_L_abs_mean']),
                        ('L_abs_p99', GATES['max_L_abs_p99']),
                        ('new_luminance_clipped_fraction', GATES['max_new_clipped_fraction']),
                        ('new_channel_clipped_fraction', GATES['max_new_clipped_fraction'])]:
        if metrics[name] > limit:
            reasons.append(name + '_above_limit')
    if metrics['source_L_laplacian_variance'] <= 1e-9:
        reasons.append('unmeasurable_source_texture')
    if metrics['L_laplacian_ratio'] < GATES['min_laplacian_ratio_' + kind]:
        reasons.append('L_laplacian_ratio_below_limit')
    return reasons


def validate_manifests(training, evaluation, expected_count):
    if len(training) != expected_count or not evaluation:
        raise ValueError('Unexpected train parent count or empty eval manifest')
    keys = [r['key'] for r in training]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate train parent key')
    groups = [r['duplicate_candidate_group'] for r in training]
    if len(groups) != len(set(groups)):
        raise ValueError('Expected one quality-selected train parent per candidate group')
    if set(keys) & {r['key'] for r in evaluation}:
        raise ValueError('Train/eval key overlap')
    if set(groups) & {r['duplicate_candidate_group'] for r in evaluation}:
        raise ValueError('Train/eval candidate group overlap')
    required = {'key', 'image_path', 'duplicate_candidate_group', 'pseudo_age', 'pseudo_age_source'}
    for row in training:
        if not required <= set(row) or row.get('split') != 'train':
            raise ValueError('Parent is not an original train-manifest row')
        if not np.isfinite(row['pseudo_age']) or not row['pseudo_age_source']:
            raise ValueError('Invalid pseudo-age metadata')
        if not Path(row['image_path']).is_file():
            raise ValueError('Missing train source: ' + row['key'])
    for row in evaluation:
        if sha256(row['image_path']) != row['sha256']:
            raise ValueError('Frozen eval image changed: ' + row['key'])
    eval_hashes = {row['sha256'] for row in evaluation}
    for row in training:
        if sha256(row['image_path']) in eval_hashes:
            raise ValueError('Train/eval source content hash overlap: ' + row['key'])


def preview_selection(rows):
    """Select source-brightness quantiles before looking at augmentation results."""
    measured = []
    for row in rows:
        value = lab(rgb(row['image_path']))
        cheeks = np.concatenate([value[123:159, 65:105, 0].ravel(), value[123:159, 151:191, 0].ravel()])
        measured.append((float(np.median(cheeks)), row['key']))
    measured.sort()
    indices = np.linspace(0, len(measured) - 1, min(8, len(measured))).round().astype(int)
    return {measured[i][1]: measured[i][0] for i in indices}


def preview_panel(path, cells):
    panel = Image.new('RGB', (768, 336), 'white')
    draw = ImageDraw.Draw(panel)
    for index, (image, title, details) in enumerate(cells):
        x = index * 256
        draw.text((x + 4, 3), title, fill='black')
        for line, text in enumerate(textwrap.wrap(details, width=35)[:4]):
            draw.text((x + 4, 17 + line * 13), text, fill='black')
        panel.paste(Image.fromarray(image), (x, 80))
    panel.save(path)


def build(args):
    start = time.monotonic()
    out = Path(args.out).resolve()
    for root in args.protected_root:
        root = Path(root).resolve()
        if out == root or root in out.parents:
            raise ValueError('Output is under a protected source/code root')
    if out.exists():
        raise ValueError('Output already exists; choose a new directory')
    rows = read_manifest(args.train_manifest)
    evaluation = read_manifest(args.eval_manifest)
    validate_manifests(rows, evaluation, args.expected_parent_count)
    eval_hashes = {row['sha256'] for row in evaluation}
    rows = sorted(rows, key=lambda r: r['key'])
    if args.limit:
        rows = rows[:args.limit]
    previews = preview_selection(rows)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'images').mkdir()
    (out / 'previews').mkdir()
    accepted, rejected, parent_counts, reasons = Counter(), Counter(), Counter(), Counter()
    coverage = {}
    source_records = []
    with (out / 'accepted_variants.jsonl').open('x') as good, (out / 'rejected_variants.jsonl').open('x') as bad:
        for index, row in enumerate(rows):
            source_hash = sha256(row['image_path'])
            source = rgb(row['image_path'])
            if sha256(row['image_path']) != source_hash:
                raise RuntimeError('Source changed while being read: ' + row['key'])
            if source_hash in eval_hashes:
                raise RuntimeError('Train/eval source content hash overlap during generation: ' + row['key'])
            source_records.append({'parent_key': row['key'], 'image_path': row['image_path'], 'sha256': source_hash})
            decade = int(row['pseudo_age'] // 10) * 10
            bucket = coverage.setdefault(f'{decade}-{decade+9}', {'parents': 0, 'accepted': Counter(), 'rejected': Counter()})
            bucket['parents'] += 1
            cells = [(source, row['key'] + ' ORIGINAL', 'source cheek L %.2f; pseudo-age %.2f' % (previews[row['key']], row['pseudo_age']))] if row['key'] in previews else None
            for kind in KINDS:
                seed = stable_seed(args.seed, row['key'], kind)
                encoded, recipe = candidate(source, kind, seed)
                metrics = measure(source, decode(encoded))
                failures = reject_reasons(metrics, kind)
                candidate_hash = hashlib.sha256(encoded).hexdigest()
                if candidate_hash in eval_hashes:
                    failures.append('candidate_content_hash_overlaps_eval')
                variant_key = row['key'] + '__' + kind + '_v1'
                record = {'key': variant_key, 'parent_key': row['key'],
                          'source_image_path': row['image_path'], 'source_sha256': source_hash,
                          'sha256': candidate_hash,
                          'duplicate_candidate_group': row['duplicate_candidate_group'],
                          'pseudo_age': row['pseudo_age'], 'pseudo_age_source': row['pseudo_age_source'],
                          'split': 'train_augmentation', 'recipe': recipe, 'seed': seed, 'metrics': metrics}
                if failures:
                    rejected[kind] += 1; reasons.update(failures)
                    bucket['rejected'][kind] += 1
                    record['reject_reasons'] = failures
                    bad.write(json.dumps(record, ensure_ascii=False) + '\n')
                else:
                    destination = out / 'images' / (variant_key + ('.png' if kind == 'illumination' else '.jpg'))
                    destination.write_bytes(encoded)
                    record['image_path'] = str(destination)
                    good.write(json.dumps(record, ensure_ascii=False) + '\n')
                    accepted[kind] += 1; parent_counts[row['key']] += 1
                    bucket['accepted'][kind] += 1
                if cells is not None:
                    detail = ', '.join(failures) if failures else ('L offset %.2f; gradient %.2f' % (recipe['L_offset'], recipe['L_gradient_amplitude']) if kind == 'illumination' else 'JPEG quality %d, subsampling 0' % recipe['quality'])
                    cells.append((decode(encoded), kind.upper() + (' REJECT' if failures else ' ACCEPT'), detail))
            if cells is not None:
                preview_panel(out / 'previews' / (row['key'] + '_original_illumination_jpeg.png'), cells)
            if (index + 1) % 200 == 0:
                print(json.dumps({'parents_done': index + 1, 'accepted': dict(accepted), 'rejected': dict(rejected)}), flush=True)
    # Detect concurrent source mutations, without writing to any source or eval.
    for row in source_records:
        if sha256(row['image_path']) != row['sha256']:
            raise RuntimeError('Source changed during bank generation: ' + row['parent_key'])
    for row in evaluation:
        if sha256(row['image_path']) != row['sha256']:
            raise RuntimeError('Frozen eval changed during bank generation: ' + row['key'])
    dump(out / 'source_hashes.json', source_records)
    summary = {'version': VERSION, 'seed': args.seed, 'parents_attempted': len(rows),
               'attempts': len(rows) * 2, 'accepted': dict(accepted), 'rejected': dict(rejected),
               'reject_reasons': dict(reasons),
               'pseudo_age_bin_coverage': coverage,
               'preview_sources': previews,
               'preview_selection': '8 evenly spaced source-cheek-L rank quantiles, chosen before seeing variant quality; rejected variants are labeled and never enter accepted bank',
               'parents_with_0_1_2_variants': {str(n): sum(parent_counts[r['key']] == n for r in rows) for n in range(3)},
               'gates': GATES, 'train_manifest_sha256': sha256(args.train_manifest),
               'eval_manifest_sha256': sha256(args.eval_manifest), 'eval_unchanged': True,
               'script_sha256': sha256(__file__), 'accepted_manifest_sha256': sha256(out / 'accepted_variants.jsonl'),
               'rejected_manifest_sha256': sha256(out / 'rejected_variants.jsonl'),
               'runtime': {'python': platform.python_version(), 'opencv': cv2.__version__, 'numpy': np.__version__, 'pillow': PIL_VERSION},
               'seconds': time.monotonic() - start,
               'disclosure': 'Deterministic augmentations of train photos, not new identities or true ages. Fixed cheek windows are proxies, not semantic skin ground truth. Sampling must choose original parents first; do not treat accepted rows as a larger independent population.'}
    dump(out / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-manifest', required=True)
    parser.add_argument('--eval-manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--protected-root', action='append', required=True, help='Source project/code root; repeat as needed')
    parser.add_argument('--seed', type=int, default=20261110)
    parser.add_argument('--expected-parent-count', type=int, default=2293)
    parser.add_argument('--limit', type=int, help='Pilot only: first N sorted train parents, after validating full manifests')
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    cv2.setNumThreads(2)
    build(args)


if __name__ == '__main__':
    main()

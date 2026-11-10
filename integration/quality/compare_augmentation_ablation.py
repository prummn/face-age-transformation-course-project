#!/usr/bin/env python3
"""Compare fresh, paired 100-step runs: same colour loss, p(augmentation)=0/.25.

CPU only. Generated photos/logs belong outside the source repository. This is
an exploratory frozen regression check, not independent population validation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ARMS = ('control', 'augmented')
TAGS = ('same_age', 'a20', 'a40', 'a60', 'a75')
METRICS = ('cheek_delta_L', 'cheek_delta_a', 'cheek_delta_b',
           'cheek_chroma_shift', 'diagnostic_id_cos_shared_training_model',
           'diagnostic_age_shared_training_model')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def stats(values):
    values = np.asarray(list(values), dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Empty or non-finite metric values')
    return {'n': len(values), 'mean': float(values.mean()),
            'median': float(np.median(values)), 'p90': float(np.quantile(values, .9))}


def index(blob):
    result = {(row['key'], row['tag']): row for row in blob['metrics']}
    if len(result) != len(blob['metrics']):
        raise ValueError('Repeated evaluation key/tag')
    return result


def read_json(path):
    return json.loads(Path(path).read_text())


def compare(experiments, output, train_manifest=None):
    base = Path(experiments).resolve()
    plans = {arm: read_json(base / arm / 'plan.json') for arm in ARMS}
    original = Path(plans['control']['base']).resolve()
    output = Path(output).resolve()
    if output == original or original in output.parents:
        raise ValueError('Output must be outside the original project')
    for arm in ARMS:
        if plans[arm]['arm'] != 'colour_only' or plans[arm]['colour_weight'] != 1:
            raise ValueError('Both fresh runs must use colour_only, weight 1')
        if plans[arm]['unique_parent_count'] != 2293:
            raise ValueError('Expected exactly 2293 unique training parents')
    if [plans[arm]['augmentation_probability'] for arm in ARMS] != [0, .25]:
        raise ValueError('Expected augmentation probabilities 0 and .25')
    allowed = {'out', 'physical_device', 'ascend_rt_visible_devices', 'augmentation_probability'}
    differences = {key: [plans[arm].get(key) for arm in ARMS]
                   for key in set(plans['control']) | set(plans['augmented'])
                   if plans['control'].get(key) != plans['augmented'].get(key)}
    if set(differences) - allowed:
        raise ValueError('Unexpected plan differences: ' + str(set(differences) - allowed))
    blobs = {(arm, step): read_json(base / arm / f'validation_step{step}' / 'metrics.json')
             for arm in ARMS for step in (0, 100)}
    tables = {name: index(blob) for name, blob in blobs.items()}
    keys = blobs[('control', 0)]['keys']
    expected = {(key, tag) for key in keys for tag in TAGS}
    if len(keys) != 24 or any(blob['keys'] != keys for blob in blobs.values()):
        raise ValueError('All evaluations must use the same ordered 24 photos')
    if any(set(table) != expected for table in tables.values()):
        raise ValueError('Every evaluation must contain all 24 x 5 outputs')
    logs = {arm: [json.loads(line) for line in (base / arm / 'train_log.jsonl').read_text().splitlines() if line]
            for arm in ARMS}
    for arm, rows in logs.items():
        if [row['step'] for row in rows] != list(range(1, 101)):
            raise ValueError(arm + ' did not finish exactly 100 ordered steps')
        if any(not np.isfinite(row['loss']) or not np.isfinite(row['gradient_norm']) for row in rows):
            raise ValueError(arm + ' contains non-finite training values')
        if any(len(row['parent_keys']) != 2 or len(row['variant_keys']) != 2 or
               row['augmented_inputs'] != sum(key is not None for key in row['variant_keys']) for row in rows):
            raise ValueError(arm + ' has inconsistent exposure counts')
    trace = {}
    for field in ('parent_keys', 'flip_flags', 'source_age_conditions', 'target_age_conditions'):
        numeric = field.endswith('_conditions')
        mismatches = [a['step'] for a, b in zip(logs['control'], logs['augmented'])
                      if not (np.allclose(a[field], b[field], rtol=0, atol=1e-7) if numeric else a[field] == b[field])]
        trace[field] = {'paired': not mismatches, 'absolute_tolerance': 1e-7 if numeric else 0,
                        'mismatched_steps': mismatches}
    if not all(item['paired'] for item in trace.values()):
        raise ValueError('Source/flip/age/target sequences differ; paired comparison is invalid: ' + str(trace))
    if any(row['augmented_inputs'] for row in logs['control']):
        raise ValueError('Control unexpectedly used augmented input')
    replay = None
    if train_manifest:
        if sha256(train_manifest) != plans['control']['train_manifest_sha256']:
            raise ValueError('Training manifest hash differs from the recorded plan')
        bank_path = plans['control']['augmentation_manifest']
        if sha256(bank_path) != plans['control']['augmentation_manifest_sha256']:
            raise ValueError('Accepted augmentation bank changed after training')
        parents = [json.loads(line) for line in Path(train_manifest).read_text().splitlines() if line]
        bank = {}
        for line in Path(bank_path).read_text().splitlines():
            if line:
                row = json.loads(line)
                bank.setdefault(row['parent_key'], []).append(row)
        for variants in bank.values():
            variants.sort(key=lambda row: row['key'])
        source_rng = random.Random(plans['control']['seed'])
        aug_rng = random.Random(plans['control']['seed'] + 1000003)
        replay = {'augmentation_requested': 0, 'no_variant_fallback': 0, 'by_recipe': {}, 'verified_against_all_logged_parent_and_variant_keys': True}
        recipes = Counter()
        for logged in logs['augmented']:
            selected = source_rng.choices(parents, weights=[row['sampling_weight'] for row in parents], k=2)
            picked = []
            for parent in selected:
                candidates = bank.get(parent['key'], [])
                use = aug_rng.random() < .25
                variant = aug_rng.choice(candidates) if use and candidates else None
                replay['augmentation_requested'] += int(use)
                replay['no_variant_fallback'] += int(use and not candidates)
                if variant:
                    recipes[variant['recipe']['kind']] += 1
                picked.append(variant['key'] if variant else None)
            if [row['key'] for row in selected] != logged['parent_keys'] or picked != logged['variant_keys']:
                raise ValueError('Deterministic augmentation replay differs from log at step ' + str(logged['step']))
        replay['by_recipe'] = dict(recipes)
    baseline = {metric: max(abs(tables[('control', 0)][item][metric] - tables[('augmented', 0)][item][metric])
                            for item in expected) for metric in METRICS}
    if max(baseline.values()) > 1e-5:
        raise ValueError('Initial output metrics differ beyond 1e-5: ' + str(baseline))
    report = {
        'scope': 'Fresh 100-step control and augmented runs; identical colour loss, seed, parents, flips and target ages. Only input augmentation probability changes 0 to .25. Original ft_l1 remains deployed.',
        'limitations': ['24 previously used frozen regression photos, not a new blind test.',
                        'No subject IDs: identity independence is unverified; multiple photos may show the same person.',
                        'Age and identity metrics use training networks and are secondary diagnostics.',
                        'Fixed cheek ROIs are not skin segmentation; texture metrics are not perceived clarity.',
                        'Augmentations are not new people, true age labels or recovered high-resolution details.'],
        'keys': keys, 'plan_differences': differences, 'sampling_trace_check': trace,
        'augmentation_decision_replay': replay,
        'initial_metric_max_absolute_difference': baseline, 'per_arm': {},
        'paired_augmented_minus_control': {}, 'training': {}, 'native_output_check': {}}
    for (arm, step), table in tables.items():
        summary = {}
        for tag in TAGS:
            rows = [table[(key, tag)] for key in keys]
            summary[tag] = {metric: stats(row[metric] for row in rows) for metric in METRICS}
            if tag == 'same_age':
                summary[tag]['same_age_mae_rgb'] = stats(row['same_age_mae_rgb'] for row in rows)
            else:
                summary[tag]['shared_age_target_mae'] = stats(abs(row['diagnostic_age_shared_training_model'] - float(tag[1:])) for row in rows)
        aged = [table[(key, tag)] for key in keys for tag in TAGS[1:]]
        summary['aged_aggregate'] = {metric: stats(row[metric] for row in aged) for metric in METRICS}
        summary['aged_aggregate']['shared_age_target_mae'] = stats(abs(table[(key, tag)]['diagnostic_age_shared_training_model'] - float(tag[1:])) for key in keys for tag in TAGS[1:])
        report['per_arm'][f'{arm}_step{step}'] = summary
    before, after = tables[('control', 100)], tables[('augmented', 100)]
    for tag in TAGS:
        deltas = [after[(key, tag)]['cheek_chroma_shift'] - before[(key, tag)]['cheek_chroma_shift'] for key in keys]
        report['paired_augmented_minus_control'][tag] = {
            'chroma_delta': stats(deltas), 'n_inputs_improved': sum(value < 0 for value in deltas),
            'n_inputs_worse': sum(value > 0 for value in deltas),
            'shared_id_delta': stats(after[(key, tag)]['diagnostic_id_cos_shared_training_model'] - before[(key, tag)]['diagnostic_id_cos_shared_training_model'] for key in keys)}
    # Resample photos, not age outputs; this does not establish subject independence.
    per_input = np.array([np.mean([after[(key, tag)]['cheek_chroma_shift'] - before[(key, tag)]['cheek_chroma_shift'] for tag in TAGS[1:]]) for key in keys])
    rng = np.random.default_rng(20261110)
    boots = np.array([rng.choice(per_input, len(per_input), replace=True).mean() for _ in range(4000)])
    report['photo_bootstrap_chroma'] = {'n_inputs': len(keys), 'mean_augmented_minus_control': float(per_input.mean()),
                                      'ci95': np.quantile(boots, [.025, .975]).tolist(),
                                      'n_inputs_improved': int((per_input < 0).sum()),
                                      'unit': 'input photo; unverified identity independence'}
    for arm, rows in logs.items():
        exposures = sum(len(row['parent_keys']) for row in rows)
        augmented_exposures = sum(row['augmented_inputs'] for row in rows)
        variant_counts = Counter(key for row in rows for key in row['variant_keys'] if key is not None)
        report['training'][arm] = {'steps': len(rows), 'skipped': sum(row['skipped'] for row in rows),
                                  'source_exposures': exposures, 'unique_source_photos': len({key for row in rows for key in row['parent_keys']}),
                                  'augmented_exposures': augmented_exposures, 'actual_augmented_fraction': augmented_exposures / exposures,
                                  'unique_variant_photos_seen': len(variant_counts), 'variant_exposure_counts': dict(variant_counts),
                                  'first10_loss': stats(row['loss'] for row in rows[:10]), 'last10_loss': stats(row['loss'] for row in rows[-10:]),
                                  'step_seconds': stats(row['seconds'] for row in rows),
                                  'checkpoint_path': str(base / arm / 'encoder_step100.pt'),
                                  'checkpoint_sha256': sha256(base / arm / 'encoder_step100.pt')}
        native_files = sorted((base / arm / 'validation_step100').glob('*_native1024.png'))
        if len(native_files) != 20:
            raise ValueError('Expected 4 photos x 5 native outputs for ' + arm)
        for path in native_files:
            with Image.open(path) as image:
                image.load()
                if image.size != (1024, 1024) or image.mode != 'RGB':
                    raise ValueError('Invalid native image: ' + str(path))
        report['native_output_check'][arm] = {'files': len(native_files), 'size': [1024, 1024], 'mode': 'RGB', 'decoded': True}
    output.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    except OSError:
        font = ImageFont.load_default()
    # All 24 photos and all five output targets are retained, regardless of outcome.
    for tag_index, tag in enumerate(TAGS, 1):
        for page in range(3):
            sheet = Image.new('RGB', (768, 8 * 282 + 30), '#f4f4f4')
            draw = ImageDraw.Draw(sheet)
            for column, title in enumerate(('Input', 'Control ' + tag, 'Augmented ' + tag)):
                draw.text((column * 256 + 8, 5), title, fill='#111', font=font)
            for position, key in enumerate(keys[page * 8:(page + 1) * 8]):
                images = [Image.open(base / arm / 'validation_step100' / f'{key}_input_sameage_20_40_60_75.png').convert('RGB') for arm in ARMS]
                tiles = [images[0].crop((0, 0, 256, 256))] + [image.crop((tag_index * 256, 0, (tag_index + 1) * 256, 256)) for image in images]
                y = 30 + position * 282
                for column, tile in enumerate(tiles):
                    sheet.paste(tile, (column * 256, y))
                draw.text((8, y + 258), f"{key} chroma {before[(key, tag)]['cheek_chroma_shift']:.2f} -> {after[(key, tag)]['cheek_chroma_shift']:.2f}", fill='#111', font=font)
            sheet.save(output / f'{tag}_all24_page{page + 1}.jpg', quality=94)
    (output / 'augmentation_comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'trace': trace, 'chroma': report['photo_bootstrap_chroma'],
                      'training': report['training']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--train-manifest', required=True, help='Verify exact bank RNG replay and count no-variant fallbacks')
    arguments = parser.parse_args()
    compare(arguments.experiments, arguments.output, arguments.train_manifest)

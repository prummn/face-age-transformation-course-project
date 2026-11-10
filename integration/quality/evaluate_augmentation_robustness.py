#!/usr/bin/env python3
"""Optional fixed eight-photo input stress probe, after the clean 24x5 check.

Uses frozen eval photos only. Combined Lab L +/-6 and JPEG86 is deliberately
stronger than the training bank; this is a separate stress test. No training,
new labels, online service changes or model downloads. Defaults to dry run.
"""
import argparse
import json
from pathlib import Path
import sys

from train_colour_ablation import physical_device, ensure_device_idle, sha256, read_manifest


def run(args):
    base = Path(args.base).resolve()
    experiments = Path(args.experiments).resolve()
    output = Path(args.output).resolve()
    if output == base or base in output.parents:
        raise ValueError('Output must be outside original project')
    if (output / 'probe_metrics.json').exists():
        raise ValueError('Do not overwrite an earlier probe')
    plans = {arm: json.loads((experiments / arm / 'plan.json').read_text()) for arm in ('control', 'augmented')}
    clean = {arm: json.loads((experiments / arm / 'validation_step100' / 'metrics.json').read_text()) for arm in plans}
    if clean['control']['keys'] != clean['augmented']['keys'] or len(clean['control']['keys']) != 24:
        raise ValueError('Clean eval keys must match, 24 photos')
    keys = clean['control']['keys'][:8]
    manifest = read_manifest(args.eval_manifest)
    if any(sha256(args.eval_manifest) != plan['eval_manifest_sha256'] for plan in plans.values()):
        raise ValueError('Frozen manifest differs from training evaluation')
    by_key = {row['key']: row for row in manifest}
    for key in keys:
        if sha256(by_key[key]['image_path']) != by_key[key]['sha256']:
            raise ValueError('Frozen source image changed')
    checkpoints = {arm: experiments / arm / 'encoder_step100.pt' for arm in plans}
    for path in checkpoints.values():
        if not path.is_file():
            raise ValueError('Checkpoint missing: ' + str(path))
    physical, visible = physical_device(args.device)
    if physical not in {1, 5}:
        raise ValueError('This optional probe is limited to free physical NPU 1 or 5')
    plan = {'keys': keys, 'target_ages': [20, 60, 75], 'n_inputs': 8,
            'perturbation': {'Lab_L_offset': 'alternating +6/-6 by fixed key position', 'JPEG_quality': 86, 'JPEG_subsampling': 0},
            'scope': 'Combined perturbation outside training range; secondary stress test, separate from clean regression.',
            'physical_device': physical, 'visible_devices': visible,
            'checkpoint_sha256': {arm: sha256(path) for arm, path in checkpoints.items()},
            'eval_manifest_sha256': sha256(args.eval_manifest), 'script_sha256': sha256(__file__)}
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        print('DRY_RUN_ONLY: no NPU allocation, inference or training.')
        return
    ensure_device_idle(physical)
    import io
    import numpy as np
    import cv2
    from PIL import Image
    import torch
    import torch_npu  # noqa: F401
    sys.path.insert(0, str(base / 'src'))
    from sam_npu import load_sam
    from sam_losses import AgeEst, IDLoss
    torch.npu.set_device(args.device)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'probe_plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    net = load_sam(args.device).eval()
    ages = AgeEst().to(args.device).eval()
    identity = IDLoss().to(args.device).eval()
    def tensor(rgb):
        return torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(127.5).sub(1)[None].to(args.device)
    def rgb(tensor_value):
        return ((tensor_value[0].clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).byte().cpu().numpy()
    def lab(rgb_value):
        converted = cv2.cvtColor(rgb_value.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
        return np.median(np.concatenate((converted[123:159, 65:105].reshape(-1, 3), converted[123:159, 151:191].reshape(-1, 3))), axis=0)
    def blur_proxy(rgb_value):
        return float(cv2.Laplacian(cv2.cvtColor(rgb_value, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
    originals = {}; perturbed = {}; inputs = {}
    for position, key in enumerate(keys):
        source = cv2.cvtColor(cv2.imread(by_key[key]['image_path']), cv2.COLOR_BGR2RGB)
        if source.shape != (256, 256, 3):
            raise ValueError('Expected frozen aligned RGB256 source')
        altered = cv2.cvtColor(source.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
        altered[:, :, 0] = np.clip(altered[:, :, 0] + (6 if position % 2 == 0 else -6), 0, 100)
        altered = np.rint(np.clip(cv2.cvtColor(altered, cv2.COLOR_LAB2RGB), 0, 1) * 255).astype(np.uint8)
        stream = io.BytesIO()
        Image.fromarray(altered).save(stream, format='JPEG', quality=86, subsampling=0)
        stream.seek(0)
        altered = np.array(Image.open(stream).convert('RGB'))
        originals[key] = source; perturbed[key] = altered
        inputs[key] = {'cheek_Lab_delta': (lab(altered) - lab(source)).tolist(),
                       'pixel_mae': float(np.abs(altered.astype(float) - source).mean()),
                       'pseudo_age': by_key[key]['pseudo_age'], 'pseudo_age_source': by_key[key]['pseudo_age_source']}
        Image.fromarray(altered).save(output / f'{key}_stress_input.png')
    records = []; generated = {}
    with torch.no_grad():
        for arm in plans:
            net.encoder.load_state_dict(torch.load(checkpoints[arm], map_location='cpu'))
            net.eval()
            for position, key in enumerate(keys):
                source = tensor(originals[key]); inp = tensor(perturbed[key])
                # Canonical clean source remains the identity/colour reference.
                source_lab = lab(originals[key])
                for age in (20, 60, 75):
                    condition = torch.full((1, 1, 256, 256), age / 100, device=args.device)
                    native = net(torch.cat((inp, condition), 1), randomize_noise=False, resize=False)
                    pooled = net.face_pool(native)
                    result = rgb(pooled)
                    generated[(arm, key, age)] = result
                    delta = lab(result) - source_lab
                    estimated, _ = ages(pooled)
                    clean_row = next(row for row in clean[arm]['metrics'] if row['key'] == key and row['tag'] == 'a' + str(age))
                    clean_strip = np.array(Image.open(experiments / arm / 'validation_step100' / f'{key}_input_sameage_20_40_60_75.png').convert('RGB'))
                    column = {20: 2, 60: 4, 75: 5}[age]
                    clean_output = clean_strip[:, column * 256:(column + 1) * 256]
                    drift = lab(result) - lab(clean_output)
                    record = {'arm': arm, 'key': key, 'target_age': age,
                              'cheek_delta_L': float(delta[0]), 'cheek_delta_a': float(delta[1]), 'cheek_delta_b': float(delta[2]),
                              'cheek_chroma_shift_from_clean_source': float(np.linalg.norm(delta[1:])),
                              'output_chroma_change_from_clean_run': float(np.linalg.norm(drift[1:])),
                              'output_L_change_from_clean_run': float(drift[0]),
                              'output_pixel_mae_from_clean_run': float(np.abs(result.astype(float) - clean_output).mean()),
                              'output_laplacian_variance_proxy': blur_proxy(result),
                              'diagnostic_id_cos_shared_training_model': float(identity.cos(pooled, source).item()),
                              'diagnostic_age_shared_training_model': float(estimated.item() * 100),
                              'clean_run_chroma': clean_row['cheek_chroma_shift'],
                              'clean_run_shared_id': clean_row['diagnostic_id_cos_shared_training_model']}
                    records.append(record)
                    if position < 2:
                        Image.fromarray(rgb(native)).save(output / f'{arm}_{key}_a{age}_native1024.png')
                print(f'PROBE_PROGRESS arm={arm} photo={position + 1}/8', flush=True)
    for key in keys:
        strips = []
        for age in (20, 60, 75):
            column = {20: 2, 60: 4, 75: 5}[age]
            tiles = [originals[key], perturbed[key]]
            for arm in plans:
                strip = np.array(Image.open(experiments / arm / 'validation_step100' / f'{key}_input_sameage_20_40_60_75.png').convert('RGB'))
                tiles.extend((strip[:, column * 256:(column + 1) * 256], generated[(arm, key, age)]))
            strips.append(np.concatenate(tiles, axis=1))
        Image.fromarray(np.concatenate(strips, axis=0)).save(output / f'{key}_clean_stress_control_augmented.jpg', quality=94)
    summary = {}
    for arm in plans:
        selected = [row for row in records if row['arm'] == arm]
        numeric = [name for name in selected[0] if name not in {'arm', 'key', 'target_age'}]
        summary[arm] = {name: float(np.mean([row[name] for row in selected])) for name in numeric}
        summary[arm]['shared_age_target_mae'] = float(np.mean([abs(row['diagnostic_age_shared_training_model'] - row['target_age']) for row in selected]))
    result = {'plan': plan, 'inputs': inputs, 'metrics': records, 'summary': summary,
              'sheet_legend': 'Rows 20,60,75. Columns clean input, stress input, control clean, control stress, augmented clean, augmented stress.',
              'limitations': 'No subject IDs; eight photos may not be independent people. Shared training age/ID networks are not independent evaluators. Laplacian variance is a texture proxy, not clarity. Stronger-than-training combined perturbation is a stress test, not clean validation.'}
    (output / 'probe_metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print('STRESS_PROBE_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='/data/home/1120231322/scow/face-age')
    parser.add_argument('--experiments', required=True)
    parser.add_argument('--eval-manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--execute', action='store_true')
    run(parser.parse_args())

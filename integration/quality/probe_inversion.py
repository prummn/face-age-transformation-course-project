"""Offline, frozen-weight reconstruction/age-edit ablation on aligned portraits.

Never imported by the serving backend. Inputs are already aligned; no second
detection/alignment is performed. Per-photo W+ fitting is inversion, not a new
trained encoder/generator, and must be evaluated separately from feed-forward.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', action='append', required=True, help='name=/absolute/aligned.png')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ages', type=int, nargs='+', default=[5, 20, 30, 40, 45, 60, 75])
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--lr', type=float, default=.03)
    args = parser.parse_args()
    if not 0 <= args.steps <= 300:
        parser.error('steps must be between 0 and 300')
    if any(age < 0 or age > 100 for age in args.ages):
        parser.error('age outside 0..100')
    repo = Path(__file__).resolve().parents[2]
    base = Path(os.environ['FACE_AGE_BASE'])
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = json.loads((repo / 'deployment/assets-ready.json').read_text())
    checked = []
    for item in manifest['weights']:
        path = base.parent / item['path']
        actual = sha256(path)
        if actual != item['sha256']:
            raise RuntimeError(f'Weight hash mismatch: {path}')
        checked.append(dict(path=str(path), sha256=actual, bytes=path.stat().st_size))

    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, str(base / 'src'))
    sys.path.insert(0, str(repo / 'integration'))
    from sam_npu import load_sam
    from sam_losses import AgeEst
    from source_color import preserve_source_chroma

    torch.manual_seed(20261112)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = os.environ.get('FACE_AGE_DEVICE', 'cuda:0')
    if not device.startswith('cuda:') or not torch.cuda.is_available():
        raise RuntimeError('This bounded experiment requires a dedicated CUDA allocation')
    torch.cuda.set_device(device)
    net = load_sam(device)
    fine = deepcopy(net.encoder)
    fine.load_state_dict(torch.load(base / 'out/ft_l1/encoder_step1000.pt', map_location='cpu', weights_only=True))
    age_net = AgeEst().to(device)
    for model in (net, fine, age_net):
        model.eval()
        model.requires_grad_(False)

    def sync():
        torch.cuda.synchronize()

    def tensor(rgb):
        return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].to(device).float() / 127.5 - 1

    def decode(code):
        return net(code, input_code=True, input_is_full=True, randomize_noise=False, resize=False)

    with torch.no_grad():
        net(torch.zeros(1, 4, 256, 256, device=device), randomize_noise=False, resize=False)
    sync()
    report = dict(
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        script_sha256=sha256(__file__), weights=checked,
        environment=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(), device=device),
        seed=20261112, model_weights_frozen=True, noise='stored fixed StyleGAN2 noise',
        cold_setup_seconds=time.monotonic() - started,
        input_protocol='Already-aligned RGB reference, OpenCV INTER_AREA to256; no re-alignment. '
                       'User reference is resampled1024 and cannot exactly recover original encoder input.',
        caveats=['Reference condition is shared FairFace expected age, not true age.',
                 'Two selected photos are diagnostic examples, not independent population validation.',
                 'Pixel errors measure reconstruction only, not identity accuracy or aging quality.',
                 'No image compositing, geometry correction, generator or encoder training.',
                 'Layer masks and reduced gain can weaken age changes; inspect this tradeoff.'],
        fitting=dict(steps=args.steps, learning_rate=args.lr, parameters='one photo W+ only',
                     objective='weighted multiscale L1 + 0.2 face-edge L1 + 0.01 W+ trust MSE',
                     note='Exploratory settings, no held-out hyperparameter selection'),
        samples=[])
    print('MODEL_READY', json.dumps(report['environment']), flush=True)

    for item in args.input:
        name, filename = item.split('=', 1)
        if not name.replace('_', '').isalnum():
            raise ValueError('Sample name must be alphanumeric with optional underscores')
        folder = args.output / name
        folder.mkdir()
        reference = np.asarray(Image.open(filename).convert('RGB'))
        if reference.shape != (1024, 1024, 3):
            raise ValueError('Expected an already-aligned1024 RGB reference')
        Image.fromarray(reference).save(folder / 'reference.png')
        rgb = cv2.resize(reference, (256, 256), interpolation=cv2.INTER_AREA)
        Image.fromarray(rgb).save(folder / 'input256.png')
        x = tensor(rgb)
        with torch.no_grad():
            estimate = float(age_net(x)[0].item()) * 100
            source_age = round(estimate)
            targets = list(dict.fromkeys([source_age] + args.ages))
            def conditioned(age):
                return torch.cat([x, torch.full_like(x[:, :1], age / 100)], 1)
            source = conditioned(source_age)
            original_source = net.encoder(source)
            fine_source = fine(source)
            if net.opts.start_from_latent_avg:
                base_code = net.latent_avg
            elif net.opts.start_from_encoded_w_plus:
                base_code = net.pretrained_encoder(x) + net.latent_avg
            else:
                base_code = torch.zeros_like(original_source)
            anchor = fine_source + base_code
            original_codes = {age: net.encoder(conditioned(age)) for age in targets}
            fine_codes = {age: fine(conditioned(age)) for age in targets}
            deltas = {age: code - original_source for age, code in original_codes.items()}
            # This must agree with the live backend's offset formulation.
            error = max(float(((anchor + deltas[age]) -
                              (original_codes[age] + fine_source - original_source + base_code)).abs().max())
                        for age in targets)
            assert error < 1e-5
            assert torch.equal(anchor + deltas[source_age], fine_source + base_code)
        sample = dict(name=name, input_path=str(Path(filename).resolve()), input_sha256=sha256(filename),
                      reference_age_unrounded=estimate, source_condition=source_age,
                      targets=targets, latent_formula_max_abs_error=error,
                      options=dict(start_from_latent_avg=net.opts.start_from_latent_avg,
                                   start_from_encoded_w_plus=net.opts.start_from_encoded_w_plus),
                      outputs=[], fitting=[])

        def save_output(route, age, codes):
            tick = time.monotonic()
            with torch.no_grad():
                sync()
                decode_start = time.monotonic()
                image = decode(codes)
                sync()
                decode_ms = (time.monotonic() - decode_start) * 1000
                measured = float(age_net(F.interpolate(image, (256, 256), mode='area'))[0].item()) * 100
                native = ((image[0].clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).byte().cpu().numpy()
            assert native.shape == (1024, 1024, 3)
            stem = f'{route}_a{age:02d}'
            Image.fromarray(native).save(folder / f'{stem}_raw.png')
            preserved, metadata = preserve_source_chroma(rgb, native)
            Image.fromarray(preserved).save(folder / f'{stem}_color.png')
            with Image.open(folder / f'{stem}_raw.png') as check:
                check.load()
                assert check.size == (1024, 1024) and check.format == 'PNG'
            reduced = cv2.resize(native, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
            difference = np.abs(reduced - rgb.astype(np.float32))
            sample['outputs'].append(dict(route=route, condition=age, raw=f'{name}/{stem}_raw.png',
                                          color=f'{name}/{stem}_color.png', reference_age_diagnostic=measured,
                                          decode_ms=decode_ms, generation_and_save_seconds=time.monotonic()-tick,
                                          rgb_mae_0_255=float(difference.mean()),
                                          central_face_mae_0_255=float(difference[70:205, 60:196].mean()),
                                          reconstruction_metric_valid=age == source_age,
                                          color_metadata=metadata))

        with torch.no_grad():
            if net.opts.start_from_encoded_w_plus:
                save_output('psp_only', source_age, base_code)
            for age in targets:
                save_output('original', age, original_codes[age] + base_code)
                save_output('fine_direct', age, fine_codes[age] + base_code)
                save_output('mixed', age, anchor + deltas[age])
                if age != source_age:
                    for coarse_count in (3, 7):
                        mask = torch.ones_like(deltas[age])
                        mask[:, :coarse_count] = 0
                        save_output(f'freeze_first{coarse_count}', age, anchor + deltas[age] * mask)
                    save_output('half_delta', age, anchor + .5 * deltas[age])
        print('ABLATION_DONE', name, 'source_condition', source_age, flush=True)
        write_json(args.output / 'report.partial.json', dict(report, current_sample=sample))

        refined = anchor.detach().clone().requires_grad_(True)
        if args.steps:
            # Fixed smooth oval only weights the fitting objective. It never
            # pastes source pixels into generated outputs, and is not segmentation.
            grid = torch.linspace(0, 1, 256, device=device)
            yy, xx = torch.meshgrid(grid, grid, indexing='ij')
            oval = (((xx - .5) / .31) ** 2 + ((yy - .53) / .36) ** 2)
            face = ((1 - oval) / .25).clamp(0, 1)[None, None]
            weight = .2 + .8 * face
            def weighted_l1(a, b, support):
                return ((a - b).abs() * support).sum() / (support.sum() * 3)
            def loss_fn(image):
                y256 = F.interpolate(image, (256, 256), mode='area')
                pixel = weighted_l1(y256, x, weight)
                for scale in (128, 64):
                    pixel = pixel + .5 * weighted_l1(
                        F.interpolate(y256, (scale, scale), mode='area'),
                        F.interpolate(x, (scale, scale), mode='area'),
                        F.interpolate(weight, (scale, scale), mode='area'))
                dx = weighted_l1(y256[:, :, :, 1:] - y256[:, :, :, :-1],
                                 x[:, :, :, 1:] - x[:, :, :, :-1], face[:, :, :, 1:])
                dy = weighted_l1(y256[:, :, 1:] - y256[:, :, :-1],
                                 x[:, :, 1:] - x[:, :, :-1], face[:, :, 1:])
                trust = (refined - anchor).square().mean()
                return pixel + .2 * (dx + dy) + .01 * trust, pixel, trust
            optimizer = torch.optim.Adam([refined], lr=args.lr)
            fit_start = time.monotonic()
            for step in range(args.steps):
                optimizer.param_groups[0]['lr'] = args.lr * (.15 + .85 * (1 - step / args.steps))
                optimizer.zero_grad(set_to_none=True)
                loss, pixel, trust = loss_fn(decode(refined))
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite inversion loss')
                loss.backward()
                if not torch.isfinite(refined.grad).all():
                    raise RuntimeError('Nonfinite inversion gradient')
                optimizer.step()
                if step == 0 or (step + 1) % 20 == 0 or step + 1 == args.steps:
                    sync()
                    point = dict(step=step+1, objective_before_update=float(loss.detach()),
                                 weighted_pixel_loss=float(pixel.detach()), trust_mse=float(trust.detach()),
                                 elapsed_seconds=time.monotonic()-fit_start)
                    sample['fitting'].append(point)
                    print('FIT', name, json.dumps(point), flush=True)
            sample['fitting_seconds'] = time.monotonic() - fit_start
            assert all(p.grad is None and not p.requires_grad for model in (net, fine, age_net)
                       for p in model.parameters())
            for age in targets:
                save_output('refined', age, refined.detach() + deltas[age])
            for age in (20, 60, 75):
                if age in deltas:
                    save_output('refined_half', age, refined.detach() + .5 * deltas[age])
        torch.save(dict(anchor=anchor.detach().cpu(), refined=refined.detach().cpu(),
                        source_age=source_age, original_deltas={a: d.cpu() for a,d in deltas.items()}),
                   folder / 'latents.pt')
        report['samples'].append(sample)
        write_json(args.output / 'report.partial.json', report)

        # Compact raw-only comparisons: source repeated in each row for clarity.
        routes = ['original', 'fine_direct', 'mixed', 'freeze_first3', 'freeze_first7', 'half_delta']
        if args.steps:
            routes.append('refined')
        columns = list(dict.fromkeys([source_age, 20, 45, 60, 75]))
        columns = [age for age in columns if age in targets]
        cell, label, left = 224, 28, 148
        board = Image.new('RGB', (left + (len(columns)+1)*cell, 30+len(routes)*(cell+label)), '#e8e8e8')
        draw = ImageDraw.Draw(board)
        draw.text((8, 8), f'{name}: shared source condition={source_age}; RAW outputs', fill='black')
        for row, route in enumerate(routes):
            y = 30 + row*(cell+label)
            draw.text((5, y+100), route, fill='black')
            board.paste(Image.fromarray(reference).resize((cell, cell)), (left, y+label))
            draw.text((left+4, y+5), 'source', fill='black')
            for col, age in enumerate(columns, 1):
                path = folder / f'{route}_a{age:02d}_raw.png'
                if not path.exists() and age == source_age:
                    path = folder / f'mixed_a{age:02d}_raw.png'
                board.paste(Image.open(path).resize((cell, cell)), (left+col*cell, y+label))
                draw.text((left+col*cell+4, y+5), f'condition {age}', fill='black')
        board.save(folder / 'overview.jpg', quality=92)
        print('SAMPLE_DONE', name, flush=True)

    report['total_seconds'] = time.monotonic() - started
    write_json(args.output / 'report.json', report)
    (args.output / 'DONE').write_text('Completed frozen-weight experiment\n')
    print('DONE', args.output, 'seconds', report['total_seconds'], flush=True)


if __name__ == '__main__':
    main()

"""CPU-only independent scoring and artifact checks for the fixed pilot."""
import argparse
import json
from pathlib import Path
import statistics
import time

from probe_inversion import sha256, write_json

ROOT = Path(__file__).resolve().parents[2]


def main():
    import cv2
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    from check_independent_age import MODEL_SHA256, MODEL_SOURCE, MEAN, STD
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('run', type=Path)
    args = parser.parse_args(); run = args.run.resolve()
    experiment = json.loads((run / 'experiment.json').read_text())
    model = ROOT / 'deployment/asia-adaptation/20261123/mivolo_v2_age.onnx'
    if sha256(model) != MODEL_SHA256: raise ValueError('Independent model hash differs')
    cv2.setNumThreads(2); options = ort.SessionOptions()
    options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), options, providers=['CPUExecutionProvider'])
    detector = cv2.FaceDetectorYN.create(str(ROOT / 'deployment/model-files/face-age/models/yunet/face_detection_yunet_2023mar.onnx'), '', (256, 256), .65, .3, 100)
    body = (np.zeros((3, 384, 384), dtype=np.float32)-MEAN)/STD
    def predict(path):
        bgr = cv2.imread(str(path)); _, faces = detector.detect(cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_AREA))
        if faces is None: return None
        x, y, w, h = max(faces, key=lambda face: face[2]*face[3])[:4]
        sx, sy = bgr.shape[1]/256, bgr.shape[0]/256
        crop = bgr[max(0, round(y*sy)):round((y+h)*sy), max(0, round(x*sx)):round((x+w)*sx)]
        if not crop.size: return None
        h, w = crop.shape[:2]; ratio = 384/max(w, h); tw, th = round(w*ratio), round(h*ratio)
        image = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (tw, th), interpolation=cv2.INTER_LINEAR)
        frame = np.zeros((384, 384, 3), dtype=np.uint8); left, top = (384-tw)//2, (384-th)//2
        frame[top:top+th, left:left+tw] = image
        face = (frame.transpose(2, 0, 1).astype(np.float32)/255-MEAN)/STD
        value = float(session.run(['age'], {'faces_bodies': np.concatenate([face, body], axis=0)[None]})[0].item())
        if not np.isfinite(value): raise ValueError('Nonfinite age prediction')
        return value
    score_file = run / 'independent-age.json'
    scores = json.loads(score_file.read_text())['ages'] if score_file.exists() else {}
    deadline = time.monotonic()+60*60; count = -1
    while True:
        final = run / 'evaluation/report.json'; path = final if final.exists() else run / 'evaluation/report.partial.json'
        if not path.exists():
            if time.monotonic() >= deadline: raise TimeoutError('No outputs')
            time.sleep(4); continue
        try: report = json.loads(path.read_text())
        except json.JSONDecodeError: time.sleep(2); continue
        for output in report['outputs']:
            key = output['native']
            if key in scores: continue
            if sha256(run / 'evaluation' / key) != output['sha256']: raise ValueError('PNG hash differs')
            for field in ['native', 'preserved']:
                with Image.open(run / 'evaluation' / output[field]) as image:
                    image.load()
                    if image.format != 'PNG' or image.size != (1024, 1024): raise ValueError('Invalid1024 PNG')
            scores[key] = predict(run / 'evaluation' / key)
        write_json(score_file, dict(model_source=MODEL_SOURCE, model_sha256=MODEL_SHA256, ages=scores))
        if count != len(scores): print('IDENTITY_BUDGET_CPU_SCORED', len(scores), flush=True); count = len(scores)
        if final.exists() and (run / 'DONE').exists(): break
        if time.monotonic() >= deadline: raise TimeoutError('Study did not finish')
        time.sleep(4)
    expected = experiment['cases']*len(experiment['routes'])*5
    if len(report['outputs']) != expected or len(scores) != expected: raise ValueError('Incomplete design')
    frozen = json.loads((run / 'frozen-verification.json').read_text())
    repeat = json.loads((run / 'repeatability.json').read_text())
    if not frozen['equal'] or not repeat['target_codes_equal']: raise ValueError('Frozen/repeat check failed')
    baseline = {(o['name'], o['condition']): o for o in report['outputs'] if o['route'] == 'accepted_baseline'}
    protected = 0
    for o in report['outputs']:
        source = next(r['source_age'] for r in report['records'] if r['name'] == o['name'])
        if o['condition'] in ['same', '20', '40'] or o['target'] == source:
            ref = baseline[o['name'], o['condition']]
            if o['sha256'] != ref['sha256'] or sha256(run/'evaluation'/o['preserved']) != sha256(run/'evaluation'/ref['preserved']):
                raise ValueError('Protected output changed')
            protected += 1
    hard = [t for t in report['timings'] if t['arm'] == 'hard_identity_budget']
    for t in hard:
        for values, floors in [('final_arcface', 'float_floor'), ('final_png_arcface', 'png_floor')]:
            if any(value < floor for value, floor in zip(t[values], t[floors])): raise ValueError('Hard budget violated')
        if not all(row['hard_feasible'] for row in t['trace']):
            # PNG feasibility is checked at snapshots; only selected snapshots
            # need satisfy PNG budget, every accepted step satisfies float budget.
            if any(any(v < f for v, f in zip(row['arcface'], t['float_floor'])) for row in t['trace']):
                raise ValueError('Float budget violated at snapshot')
    def mean(values):
        valid = [value for value in values if value is not None]
        return statistics.mean(valid) if valid else None
    groups = {}
    for route in report['routes']:
        outputs = [o for o in report['outputs'] if o['route'] == route]
        old = [o for o in outputs if o['condition'] in ['60', '75']]
        by = {(o['name'], o['condition']): scores[o['native']] for o in outputs}
        names = {o['name'] for o in outputs}
        spans = [by[n, '75']-by[n, '60'] for n in names if by[n, '75'] is not None and by[n, '60'] is not None]
        pairs = []
        for o in old:
            ref = baseline[o['name'], o['condition']]; age = scores[o['native']]; ref_age = scores[ref['native']]
            pairs.append(dict(name=o['name'], condition=o['condition'],
                identity_delta=o['source_cosine']-ref['source_cosine'] if o['source_cosine'] is not None and ref['source_cosine'] is not None else None,
                age_error_delta=abs(age-o['target'])-abs(ref_age-o['target']) if age is not None and ref_age is not None else None))
        groups[route] = dict(cases=len(names), high_age_identity=mean(o['source_cosine'] for o in old),
            high_age_error=mean(abs(scores[o['native']]-o['target']) for o in old if scores[o['native']] is not None),
            mean_age75=mean(by[n, '75'] for n in names), age60to75_span=mean(spans), weak60to75=sum(v < 5 for v in spans),
            age20to75_span=mean(by[n, '75']-by[n, '20'] for n in names if by[n, '75'] is not None and by[n, '20'] is not None),
            missing_faces=sum(not o['detected'] for o in outputs), independent_missing=sum(scores[o['native']] is None for o in outputs),
            identity_drop_over02=sum(p['identity_delta'] is not None and p['identity_delta'] < -.02 for p in pairs),
            age_error_improved=sum(p['age_error_delta'] is not None and p['age_error_delta'] < 0 for p in pairs), pairs=pairs)
    control = groups['accepted_baseline']; candidate = groups['hard_identity_budget']
    checks = dict(high_age_identity=candidate['high_age_identity'] >= control['high_age_identity']-.02,
        high_age_error=candidate['high_age_error'] <= control['high_age_error']-1,
        high_age_span=candidate['age60to75_span'] >= control['age60to75_span'],
        relative_span=candidate['age20to75_span'] >= .9*control['age20to75_span'],
        detection=candidate['missing_faces'] <= control['missing_faces'],
        independent_detection=candidate['independent_missing'] <= control['independent_missing'])
    write_json(run / 'artifact-validation.json', dict(raw=expected, color=expected, all_decodable=True,
        native_size=[1024, 1024], raw_sha256_verified=True, protected_raw_and_color_conditions=protected,
        all_models_unchanged=True, repeatable=True, hard_float_and_png_budget=True))
    write_json(run / 'summary.json', dict(groups=groups, checks=checks, all_numeric_checks_pass=all(checks.values()),
        independent_ages=scores, visual_review_pending=True, promotion=False,
        note=f"{experiment['cases']} development cases; proxy budgets do not guarantee human identity/naturalness."))
    print('IDENTITY_BUDGET_CPU_COMPLETE', json.dumps(groups), flush=True)


if __name__ == '__main__': main()

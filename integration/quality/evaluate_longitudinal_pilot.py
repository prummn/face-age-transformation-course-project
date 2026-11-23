"""Render both directions on documented real same-person photographs."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from probe_inversion import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    run = parser.parse_args().run
    import cv2
    import numpy as np
    from PIL import Image
    import torch

    root = Path(os.environ['FACE_AGE_REPO'])
    base = Path(os.environ['FACE_AGE_BASE'])
    sys.path.insert(0, str(base/'src'))
    from sam_npu import load_sam
    from sam_losses import AgeEst
    from latent_quality import LatentQuality

    assert torch.cuda.device_count() == 1
    torch.manual_seed(20261123)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cv2.setNumThreads(2)
    protocol = json.loads((run/'protocol.json').read_text())
    for path,digest in protocol['sha256'].items():
        if sha256(path) != digest:
            raise ValueError(f'Changed experiment input: {path}')

    net = load_sam('cuda:0').eval().requires_grad_(False)
    release = deepcopy(net.encoder).eval().requires_grad_(False)
    release.load_state_dict(torch.load(base/'out/ft_l1/encoder_step1000.pt',map_location='cpu',weights_only=True))
    arms = {}
    for name, path in protocol['checkpoints'].items():
        encoder = deepcopy(release)
        encoder.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        arms[name] = encoder
    age = AgeEst().cuda().eval().requires_grad_(False)

    def tensor(rgb):
        return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1)[None].contiguous().cuda().float()/127.5-1

    engine = SimpleNamespace(torch=torch,dev='cuda:0',net=net,identity_encoder=release,age_net=age,tensor=tensor)
    wplus = LatentQuality(engine,'constrained')
    detection = cv2.FaceDetectorYN.create(str(base/'models/yunet/face_detection_yunet_2023mar.onnx'),'',(256,256),.6,.3,100)
    sface = root/'deployment/gan-research/face_recognition_sface_2021dec.onnx'
    recognizer = cv2.FaceRecognizerSF.create(str(sface),'')
    calib = json.loads(Path(protocol['calibration']).read_text())
    out = run/'results'
    out.mkdir(exist_ok=False)
    outputs = []
    cases = []

    def feature(rgb):
        bgr = cv2.cvtColor(cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2BGR)
        _, faces = detection.detect(bgr)
        if faces is None:
            return None
        face = max(faces,key=lambda r:r[2]*r[3])
        v = recognizer.feature(recognizer.alignCrop(bgr,face)).ravel()
        return v/(np.linalg.norm(v)+1e-12)

    def similarity(a,b):
        return float(a@b) if a is not None and b is not None else None

    for pair in protocol['pairs']:
        for direction,source,target in [('older',pair['early'],pair['late']),('younger',pair['late'],pair['early'])]:
            identifier = f'{pair["id"]}-{direction}'
            source_rgb = np.asarray(Image.open(source['aligned']).convert('RGB'))
            target_rgb = np.asarray(Image.open(target['aligned']).convert('RGB'))
            small = cv2.resize(source_rgb,(256,256),interpolation=cv2.INTER_AREA)
            source_id, target_id = feature(source_rgb), feature(target_rgb)
            assert tuple(source_rgb.shape) == tuple(target_rgb.shape) == (1024,1024,3)
            x = tensor(small)
            target_age = target['age']
            with torch.no_grad():
                raw = round(float(age(x)[0])*100)
                condition = torch.cat([x,torch.full_like(x[:,:1],target_age/100)],1)
                baseline = net.pretrained_encoder(x) + net.latent_avg
            fixed = round(float(np.interp(raw,calib['knots'],calib['fitted_knots']))) if 45 < raw < 75 else raw
            cases.append(dict(pair=pair['id'],direction=direction,tier=pair['tier'],
                source=source['name'],target=target['name'],source_age=source['age'],target_age=target_age,
                fairface_source=raw,calibrated_source=fixed,real_pair_sface=similarity(source_id,target_id)))
            routes = ['wplus',*arms]
            if fixed != raw:
                routes.append('wplus_calibrated_source')
            for route in routes:
                tick = time.monotonic()
                if route in arms:
                    with torch.no_grad():
                        code = baseline + arms[route](condition)
                        y = net(code,input_code=True,input_is_full=True,randomize_noise=False,resize=False)
                else:
                    item = SimpleNamespace(aligned=small,reference_rgb=source_rgb,
                        reference_age=fixed if route != 'wplus' else raw,quality_state=None)
                    y = wplus.run(item,target_age,lambda:None)
                torch.cuda.synchronize()
                seconds = time.monotonic()-tick
                assert tuple(y.shape) == (1,3,1024,1024) and bool(torch.isfinite(y).all())
                rgb = ((y[0].clamp(-1,1)+1)*127.5).permute(1,2,0).byte().cpu().numpy()
                generated_id = feature(rgb)
                path = out/(identifier+'-'+route+'.png')
                Image.fromarray(rgb).save(path)
                with Image.open(path) as png:
                    png.load()
                    assert png.size == (1024,1024) and png.format == 'PNG'
                outputs.append(dict(case=identifier,route=route,source_cosine=similarity(generated_id,source_id),
                    target_cosine=similarity(generated_id,target_id),seconds=seconds,path=path.name,sha256=sha256(path)))
            print('PAIR_DIRECTION_DONE',identifier,len(routes),flush=True)
    write_json(run/'report.json',dict(cases=cases,outputs=outputs,
        notes=['Input photographs are distinct real photos; expressions, cosmetics, pose and lighting vary.',
               'SFace similarity is a diagnostic for already documented pairs, not a measure of age accuracy.',
               'W+ helper is the frozen serving baseline; CUDA fitting is not bitwise deterministic.']))
    (run/'DONE').write_text('Longitudinal pilot completed\n')


if __name__ == '__main__':
    main()

"""Render frozen HD demo inputs with the actual serving quality helper."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace
from probe_inversion import sha256, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('run',type=Path)
    run=p.parse_args().run; started=time.monotonic()
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    base=Path(os.environ['FACE_AGE_BASE']); sys.path.insert(0,str(base/'src'))
    from sam_npu import load_sam
    from sam_losses import AgeEst
    from latent_quality import LatentQuality
    from source_color import preserve_source_chroma
    assert torch.cuda.device_count()==1
    torch.set_num_threads(4); cv2.setNumThreads(2); torch.manual_seed(20261117)
    torch.backends.cudnn.benchmark=False; torch.backends.cuda.matmul.allow_tf32=False
    net=load_sam('cuda:0').eval().requires_grad_(False)
    fine=deepcopy(net.encoder).eval().requires_grad_(False)
    fine.load_state_dict(torch.load(base/'out/ft_l1/encoder_step1000.pt',map_location='cpu',weights_only=True))
    age=AgeEst().cuda().eval().requires_grad_(False)
    def tensor(rgb):return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1)[None].contiguous().cuda().float()/127.5-1
    engine=SimpleNamespace(torch=torch,dev='cuda:0',net=net,identity_encoder=fine,age_net=age,tensor=tensor)
    refiner=LatentQuality(engine,'constrained')
    with torch.no_grad():net(torch.zeros(1,4,256,256,device='cuda:0'),randomize_noise=False,resize=False)
    torch.cuda.synchronize()
    records=json.loads((run/'inputs.json').read_text()); out=run/'results'; out.mkdir(exist_ok=False)
    report=dict(profile='constrained',helper_sha256=sha256(Path(__file__).parent/'latent_quality.py'),
        cold_setup_seconds=time.monotonic()-started,model_weights_frozen=True,
        environment=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()),
        input_protocol='Official already-aligned FFHQ1024; area downsample256 for encoder; no second alignment.',
        timing_scope='GPU-container offline serving-helper calls; includes per-photo fitting; not HTTP/network/browser latency.',
        reproducibility='Native CUDA optimization may vary slightly between runs; fixed seed is not a bitwise determinism guarantee.',
        records=[],outputs=[])
    print('HD_MODEL_READY',json.dumps(report['environment']),flush=True)
    processing=time.monotonic()
    for row in records:
        assert sha256(row['image_path'])==row['sha256']
        tick=time.monotonic(); rgb=np.asarray(Image.open(row['image_path']).convert('RGB'))
        assert rgb.shape==(1024,1024,3)
        small=cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA)
        with torch.no_grad(): estimate=float(age(tensor(small))[0])*100
        source=round(estimate); item=SimpleNamespace(aligned=small,reference_rgb=rgb,reference_age=source,quality_state=None)
        folder=out/row['name']; folder.mkdir(); Image.fromarray(rgb).save(folder/'reference.png')
        row.update(reference_age=source,reference_age_unrounded=estimate,
                   reference_age_kind='Frozen FairFace pseudo reference; human apparent-age range retained separately')
        for condition,target in [('same',source)]+[(str(t),t) for t in [20,40,60,75]]:
            infer=time.monotonic(); y=refiner.run(item,target,lambda:None); torch.cuda.synchronize(); seconds=time.monotonic()-infer
            assert tuple(y.shape)==(1,3,1024,1024) and torch.isfinite(y).all()
            native=((y[0].clamp(-1,1)+1)*127.5).permute(1,2,0).byte().cpu().numpy()
            preserved,color=preserve_source_chroma(rgb,native)
            output=dict(name=row['name'],condition=condition,target=target,model_seconds=seconds,color=color,native_size=[1024,1024])
            for variant,image in [('raw',native),('preserved',preserved)]:
                path=folder/f'{condition}_{variant}.png'; Image.fromarray(image).save(path)
                with Image.open(path) as im: im.load(); assert im.size==(1024,1024) and im.mode=='RGB' and im.format=='PNG'
                output[variant]=dict(path=str(path.relative_to(out)),sha256=sha256(path),bytes=path.stat().st_size)
            report['outputs'].append(output)
        row.update(anchor_seconds=item.quality_state['anchor_seconds'],photo_total_seconds=time.monotonic()-tick)
        report['records'].append(row); write_json(run/'report.partial.json',report)
        assert all(not q.requires_grad and q.grad is None for m in [net,fine,age,refiner.identity,refiner.perceptual] for q in m.parameters())
        print('HD_SAMPLE_DONE',row['name'],row['age_group'],row['publisher_gender'],round(row['photo_total_seconds'],2),flush=True)
    report.update(processing_seconds=time.monotonic()-processing,total_seconds=time.monotonic()-started,
                  peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated())
    assert len(report['outputs'])==len(records)*5
    write_json(run/'report.json',report); (run/'DONE').write_text('HD real-model rendering complete; worker exits and releases its one GPU.\n')
    print('HD_DEMO_DONE',len(report['records']),len(report['outputs']),flush=True)


if __name__=='__main__':main()

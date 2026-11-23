"""Compare fixed W+ arms with common anchors and full-resolution images."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from probe_inversion import sha256, write_json
from train_gan_reconstruction import rows

ROUTES = ('wplus_base','wplus_real_age','wplus_trained_editor')


def choose_cases(run):
    def rank(row):
        return (row.get('publisher_gender',''), -row.get('source_face_short_edge',0), row['name'])

    manifests = run/'manifests'
    validation = rows(manifests/'eval.jsonl' if manifests.exists() else run/'data/eval.pre-age.jsonl')
    external = rows(manifests/'external.jsonl' if manifests.exists() else run/'data/external.jsonl')
    chosen = []
    for low in (20,30,40,50,70):
        pool = [r for r in validation if r['cohort'] == 'new-hd-holdout' and r['age_interval'][0] == low]
        for gender in ('female','male'):
            group = sorted((r for r in pool if r.get('publisher_gender') == gender),key=rank)
            chosen.extend(group[:(2 if low>=50 else 1)])
    annotated_older = sorted((r for r in validation if r['cohort']=='annotated-ffhq' and r['age_interval'][0]>=50),key=rank)
    selected_bins = set()
    for record in annotated_older:
        key = (record['age_interval'][0],record.get('publisher_gender'))
        if key not in selected_bins:
            chosen.append(record)
            selected_bins.add(key)
    appa = sorted((r for r in external if r['cohort']=='new_appa_asian'),key=lambda r:r['name'])
    chosen.extend([r for r in appa if r['publisher_age']>=60])
    chosen.extend([r for r in appa if r['publisher_age']<40][:2])
    for pair in json.loads((run/'data/longitudinal.json').read_text()):
        if pair['tier'] != 'dated':
            continue
        for side,other in (('early','late'),('late','early')):
            a,b = pair[side],pair[other]
            chosen.append(dict(name=a['name'],cohort='dated-pairs',image_path=a['aligned'],sha256=a['aligned_sha256'],
                target_photo=b['aligned'],target_sha256=b['aligned_sha256'],actual_target=b['age'],
                publisher_age=a['age'],source_face_short_edge=a['nominal_face_edge']))
    if len(chosen) > 30 or len(chosen) < 19 or len({r['name'] for r in chosen}) != len(chosen):
        raise ValueError(f'Unexpected fixed W+ sample count: {len(chosen)}')
    return chosen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    args = p.parse_args()
    run = args.run
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from torch.nn import functional as F
    from latent_quality import LatentQuality
    from real_age_guide import load_age_guide, ordinal_loss
    from source_color import preserve_source_chroma
    from deterministic_quality_ops import replace_pools, resize, verify

    base = Path(os.environ['FACE_AGE_BASE'])
    root = Path(os.environ['FACE_AGE_REPO'])
    sys.path.insert(0,str(base/'src'))
    from sam_npu import load_sam
    from sam_losses import AgeEst
    assert torch.cuda.device_count() == 1
    torch.manual_seed(20261123)
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    verify()
    net = load_sam('cuda:0').eval().requires_grad_(False)
    release = deepcopy(net.encoder).eval().requires_grad_(False)
    release.load_state_dict(torch.load(base/'out/ft_l1/encoder_step1000.pt',map_location='cpu',weights_only=True))
    trained = deepcopy(release)
    trained.load_state_dict(torch.load(run/'target_encoder/encoder.pt',map_location='cpu',weights_only=True))
    age = AgeEst().cuda().eval().requires_grad_(False)
    guide = load_age_guide(age,run/'age-guide/age_head.pt')
    replace_pools(age)
    replace_pools(guide)

    def tensor(rgb):
        return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1)[None].contiguous().cuda().float()/127.5-1

    engine = SimpleNamespace(torch=torch,dev='cuda:0',net=net,identity_encoder=release,age_net=age,tensor=tensor)
    helper = LatentQuality(engine,'constrained',resize=resize)
    replace_pools(helper.identity)
    replace_pools(helper.perceptual)
    detector = cv2.FaceDetectorYN.create(str(base/'models/yunet/face_detection_yunet_2023mar.onnx'),'',(256,256),.6,.3,100)
    recognizer = cv2.FaceRecognizerSF.create(str(root/'deployment/gan-research/face_recognition_sface_2021dec.onnx'),'')
    out = run/'wplus-evaluation'
    out.mkdir(exist_ok=False)
    records = choose_cases(run)
    selection = [dict(name=r['name'],cohort=r['cohort'],sha256=r['sha256']) for r in records]
    if json.loads((run/'wplus-selection.json').read_text()) != selection:
        raise ValueError('W+ evaluation selection differs from frozen protocol')
    write_json(out/'selection.json',selection)
    report = dict(records=records,outputs=[],routes=ROUTES,timings=[],output_root='wplus-evaluation')
    started = time.monotonic()

    def read_rgb(path,digest):
        if sha256(path) != digest:
            raise ValueError('Evaluation input changed')
        with Image.open(path) as im:
            return np.asarray(im.convert('RGB'))

    def small(rgb):
        return cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA)

    @torch.no_grad()
    def identity(rgb):
        bgr = cv2.cvtColor(small(rgb),cv2.COLOR_RGB2BGR)
        _,faces = detector.detect(bgr)
        if faces is None:
            return None
        face = max(faces,key=lambda f:f[2]*f[3])
        output = recognizer.feature(recognizer.alignCrop(bgr,face)).ravel()
        return output/(np.linalg.norm(output)+1e-12)

    def decode(code):
        return net(code,input_code=True,input_is_full=True,randomize_noise=False,resize=False)

    def features(images):
        """Exact shared ConvNeXt input path with deterministic resize backward."""
        values = resize((images+1)/2,(224,224),kind='bilinear')
        return age.net.b((values-age.m)/age.sd)

    for row in records:
        path = row.get('reference_path',row['image_path'])
        digest = row.get('reference_sha256',row['sha256'])
        rgb = read_rgb(path,digest)
        reduced = small(rgb)
        x = tensor(reduced)
        src_id = identity(rgb)
        expected = read_rgb(row['target_photo'],row['target_sha256']) if 'target_photo' in row else None
        expected_id = identity(expected) if expected is not None else None
        folder = out/row['name']
        folder.mkdir()
        Image.fromarray(rgb).save(folder/'reference.png')
        if expected is not None:
            Image.fromarray(expected).save(folder/'actual.png')
        with torch.no_grad():
            source = row.get('source_age')
            if source is None:
                source = round(float(age(x)[0])*100)
            row['source_age'] = source
        item = SimpleNamespace(aligned=reduced,reference_rgb=rgb,reference_age=source,quality_state=None)
        tick = time.monotonic()
        reconstruction = helper.run(item,source,lambda: None)
        torch.cuda.synchronize()
        anchor_time = time.monotonic()-tick
        state = item.quality_state
        targets = [row['actual_target']] if expected is not None else [20,40,60,75]
        requested = [t for t in targets if t != source]
        condition = lambda n:torch.cat([x,torch.full_like(x[:,:1],n/100)],1)
        def relative(encoder):
            codes = torch.cat([encoder(condition(t)) for t in requested])
            return codes-encoder(condition(source))
        with torch.no_grad():
            if requested:
                release_delta = relative(release)
                trained_delta = relative(trained)
                release_original = state['source']+release_delta
                target_distribution = age.net.age(features(resize(decode(release_original),(256,256),kind='area'))).float().softmax(1)
                paired_ages = x.new_tensor(requested)/100
                color_reference = F.avg_pool2d(x,32).expand(len(requested),-1,-1,-1)
        for route in ROUTES:
            tick = time.monotonic()
            generated = {}
            if requested:
                delta = trained_delta if route=='wplus_trained_editor' else release_delta
                pivot = state['anchor']+delta
                with torch.no_grad():
                    prior = resize(decode(pivot),(256,256),kind='area')
                correction = torch.zeros_like(pivot[:,:8]).requires_grad_(True)
                opt = torch.optim.Adam([correction],lr=.012)
                for step in range(180):
                    opt.param_groups[0]['lr']=.012*(.2+.8*(1-step/180))
                    code = pivot+torch.cat([correction,torch.zeros_like(pivot[:,8:])],1)
                    edited = resize(decode(code),(256,256),kind='area')
                    current_features = features(edited)
                    logit = age.net.age(current_features).float()
                    identity_loss = (1-(helper.identity.feats(edited)*state['identity']).sum(1)).mean()
                    original_age = F.kl_div(logit.log_softmax(1),target_distribution,reduction='batchmean')
                    age_loss = .5*original_age if route=='wplus_base' else .25*original_age+.20*ordinal_loss(guide(current_features),paired_ages)
                    color = F.l1_loss(F.avg_pool2d(edited,32),color_reference)
                    loss = .8*identity_loss+age_loss+.6*helper.perceptual(edited,prior).mean()+.05*correction.square().mean()+.1*color
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    if step%30==0 and not bool(torch.isfinite(loss) & torch.isfinite(correction.grad).all()):
                        raise RuntimeError('Non-finite W+ objective')
                    opt.step()
                with torch.no_grad():
                    images = decode(pivot+torch.cat([correction,torch.zeros_like(pivot[:,8:])],1))
                    generated = dict(zip(requested,images))
            torch.cuda.synchronize()
            seconds=time.monotonic()-tick
            output_conditions = [('actual',targets[0])] if expected is not None else [('same',source),*[(str(t),t) for t in targets]]
            for label,target in output_conditions:
                image = reconstruction[0] if label=='same' or target==source else generated[target]
                if image.shape!=(3,1024,1024) or not bool(torch.isfinite(image).all()):
                    raise RuntimeError('Invalid native output')
                native = ((image.clamp(-1,1)+1)*127.5).permute(1,2,0).byte().cpu().numpy()
                own = identity(native)
                lab = cv2.cvtColor(small(native),cv2.COLOR_RGB2LAB).astype(float)
                orig = cv2.cvtColor(reduced,cv2.COLOR_RGB2LAB).astype(float)
                chroma = float(np.linalg.norm((lab-orig)[70:205,60:196,1:].mean((0,1))))
                filename = folder/f'{route}_{label}.png'
                Image.fromarray(native).save(filename)
                with Image.open(filename) as im:
                    im.load()
                    assert im.size == (1024,1024) and im.format == 'PNG'
                preserved,_ = preserve_source_chroma(rgb,native)
                color_file = folder/f'{route}_{label}_color.png'
                Image.fromarray(preserved).save(color_file)
                report['outputs'].append(dict(name=row['name'],route=route,cohort=row['cohort'],condition=label,target=target,
                    native=str(filename.relative_to(out)),preserved=str(color_file.relative_to(out)),sha256=sha256(filename),
                    source_cosine=float(own@src_id) if own is not None and src_id is not None else None,
                    target_cosine=float(own@expected_id) if own is not None and expected_id is not None else None,
                    chroma=chroma,detected=own is not None))
            report['timings'].append(dict(name=row['name'],route=route,anchor_seconds=anchor_time,edit_seconds=seconds))
            if requested:
                del code,images,generated,loss,edited,current_features,logit
        write_json(out/'report.partial.json',report)
        print('WPLUS_SAMPLE_DONE',row['name'],row['cohort'],round(anchor_time,2),flush=True)
    report['total_seconds'] = time.monotonic()-started
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    write_json(out/'report.json',report)
    (out/'DONE').write_text('All locked W+ comparisons complete\n')


if __name__=='__main__':
    main()

"""Learn adult apparent-age supervision from actual photo intervals."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

from probe_inversion import sha256, write_json

INTERVALS = ((20,29), (30,39), (40,49), (50,69), (70,120))
BOUNDARIES = (30,40,50,70)


class AgeHead(nn.Sequential):
    def __init__(self, features):
        super().__init__(nn.LayerNorm(features), nn.Linear(features,128), nn.GELU(),
                         nn.Dropout(.20), nn.Linear(128,len(INTERVALS)))


def load_age_guide(fairface, checkpoint):
    """Reuse the same frozen net. Only the new apparent-age head is loaded."""
    checked = torch.load(checkpoint,map_location='cpu',weights_only=True)
    if checked['schema'] != 1 or checked['age_intervals'] != [list(x) for x in INTERVALS]:
        raise ValueError('Incompatible apparent-age head')
    head = AgeHead(fairface.net.b.num_features)
    head.load_state_dict(checked['head'])
    return head.to(fairface.m.device).eval().requires_grad_(False)


def image_features(fairface, images):
    values = F.interpolate((images+1)/2,(224,224),mode='bilinear',align_corners=False)
    return fairface.net.b((values-fairface.m)/fairface.sd)


def ordinal_loss(logits, targets):
    """Soft threshold constraints at 30, 40, 50, and 70, with no invented exact ages."""
    probability = logits.float().softmax(1)
    older = (1 - probability[:,:-1].cumsum(1)).clamp(.0001,.9999)
    thresholds = targets.new_tensor(BOUNDARIES)
    target = ((targets[:,None]*100-thresholds[None,:])/3.5).sigmoid()
    return F.binary_cross_entropy(older,target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    if os.environ.get('FACE_AGE_EXPERIMENT_ALLOCATION') != '1' or torch.cuda.device_count() != 1:
        raise RuntimeError('A dedicated one-GPU allocation is required')
    import numpy as np
    from PIL import Image
    torch.manual_seed(20261123)
    torch.set_num_threads(4)
    sys.path.insert(0,str(Path(os.environ['FACE_AGE_BASE'])/'src'))
    from sam_losses import AgeEst

    start = time.monotonic()
    records = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    group_bucket = lambda group: int(hashlib.sha256(group.encode()).hexdigest()[:8],16) % 10
    training = [r for r in records if group_bucket(r['identity_group']) != 0]
    dev = [r for r in records if group_bucket(r['identity_group']) == 0]
    labels = {pair:i for i,pair in enumerate(INTERVALS)}
    if {tuple(r['age_interval']) for r in training} != set(labels) or {tuple(r['age_interval']) for r in dev} != set(labels):
        raise ValueError('Training and development subsets require all five age ranges')
    if {r['identity_group'] for r in training}&{r['identity_group'] for r in dev}:
        raise ValueError('Age head dev groups overlap')
    age = AgeEst().cuda().eval().requires_grad_(False)

    def extract(records, augment):
        features, classes, confidence, baseline_age = [], [], [], []
        with torch.no_grad():
            for i in range(0,len(records),32):
                batch = records[i:i+32]
                images = []
                for r in batch:
                    if not r['training_authorized'] or sha256(r['image_path']) != r['sha256']:
                        raise ValueError('Training authorization or image hash differs')
                    with Image.open(r['image_path']) as im:
                        images.append(np.asarray(im.convert('RGB')))
                x = torch.from_numpy(np.stack(images)).permute(0,3,1,2).cuda().float()/127.5-1
                base = image_features(age,x)
                views = [base] if not augment else [base,image_features(age,x.flip(-1)[:,:,4:-4,4:-4])]
                for view in views:
                    features.append(view)
                    classes.extend(labels[tuple(r['age_interval'])] for r in batch)
                    confidence.extend(r['age_confidence'] for r in batch)
                baseline_age.extend(((age.net.age(base).softmax(1)@age.mid)*100).tolist())
        return torch.cat(features),torch.tensor(classes,device='cuda'),torch.tensor(confidence,device='cuda'),baseline_age

    x,y,confidence,_ = extract(training,True)
    v,u,_,fair = extract(dev,False)
    print('AGE_FEATURES',len(training),len(dev),flush=True)
    class_weight = y.bincount(minlength=len(INTERVALS)).float().rsqrt()
    weight = class_weight[y]*confidence
    weight /= weight.mean()
    head = AgeHead(x.shape[1]).cuda()
    opt = torch.optim.AdamW(head.parameters(),lr=4e-4,weight_decay=.02)
    best = float('inf')
    best_epoch = None
    best_weight = None
    for epoch in range(1,61):
        head.train()
        for indices in torch.randperm(len(x),device='cuda').split(128):
            error = F.cross_entropy(head(x[indices]),y[indices], reduction='none',label_smoothing=.03)
            loss = (error*weight[indices]).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite age-head training loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            head.eval()
            loss = F.cross_entropy(head(v),u,reduction='none')
            balanced = torch.stack([loss[u==k].mean() for k in range(len(INTERVALS))]).mean().item()
        if balanced < best:
            best,best_epoch = balanced,epoch
            best_weight = {n:p.detach().cpu().clone() for n,p in head.state_dict().items()}
        if epoch%10 == 0:
            print('AGE_HEAD_EPOCH',epoch,round(balanced,4),flush=True)
    head.load_state_dict(best_weight)
    with torch.no_grad():
        predictions = head(v).softmax(1)
        hard = predictions.argmax(1)
        average = predictions@predictions.new_tensor([24.5,34.5,44.5,59.5,75.])
        ordinal = (hard-u).float().abs().mean().item()
        accuracy = (hard==u).float().mean().item()
    out = args.output
    out.mkdir(exist_ok=False)
    checkpoint = out/'age_head.pt'
    torch.save(dict(schema=1,age_intervals=[list(x) for x in INTERVALS],head=best_weight),checkpoint)
    baseline_age = np.array(fair)
    model_age = average.cpu().numpy()
    lo,hi = np.array([r['age_interval'] for r in dev]).T
    distance = lambda values: float(np.abs(values-np.clip(values,lo,hi)).mean())
    write_json(out/'report.json',dict(train=len(training),dev=len(dev),epochs=60,best_epoch=best_epoch,
        seconds=time.monotonic()-start,intervals=INTERVALS,publisher_intervals_only=True,
        train_counts=Counter(str(r['age_interval']) for r in training),dev_counts=Counter(str(r['age_interval']) for r in dev),
        dev_balanced_nll=best,dev_bin_accuracy=accuracy,dev_bin_absolute_error=ordinal,
        dev_fairface_interval_mae=distance(baseline_age),dev_head_interval_mae=distance(model_age),
        train_manifest_sha256=sha256(args.manifest),checkpoint_sha256=sha256(checkpoint)))
    print('AGE_HEAD_DONE',best_epoch,distance(baseline_age),distance(model_age),flush=True)


if __name__ == '__main__':
    main()

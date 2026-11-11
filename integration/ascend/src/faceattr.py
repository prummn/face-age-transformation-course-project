"""FairFace (CC-BY-4.0) multi-head classifier race7/gender2/age9 (ConvNeXt-T). Used for data filtering, interim age loss, secondary eval.
No device side effects at import."""
import io, os, glob, time, sys, numpy as np, torch, timm, pyarrow.parquet as pq
from PIL import Image
import torchvision.transforms as T
B = "/data/home/1120231322/scow/face-age"
RACE = ["East Asian", "Indian", "Black", "White", "Middle Eastern", "Latino_Hispanic", "Southeast Asian"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
te = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])
class Net(torch.nn.Module):
    def __init__(s, pretrained=False):
        super().__init__()
        s.b = timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=pretrained, num_classes=0,
              pretrained_cfg_overlay=dict(file=f"{B}/models/convnext_tiny/model.safetensors") if pretrained else None)
        d = s.b.num_features; s.race = torch.nn.Linear(d, 7); s.gender = torch.nn.Linear(d, 2); s.age = torch.nn.Linear(d, 9)
    def forward(s, x): f = s.b(x); return s.race(f), s.gender(f), s.age(f)
if __name__ == "__main__":
    import torch_npu
    from torch.utils.data import Dataset, DataLoader
    dev = "npu:0"; torch.npu.set_device(dev)
    def load(split):
        rows = []
        for f in sorted(glob.glob(f"{B}/data/fairface/1.25/{split}-*.parquet")):
            t = pq.read_table(f, columns=["image", "age", "gender", "race"]).to_pydict()
            rows += [(im["bytes"], a, g, r) for im, a, g, r in zip(t["image"], t["age"], t["gender"], t["race"])]
        return rows
    tr = T.Compose([T.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)), T.RandomHorizontalFlip(), T.ColorJitter(0.3, 0.3, 0.2, 0.02), T.RandomGrayscale(0.05), T.ToTensor(), T.Normalize(MEAN, STD)])
    class DS(Dataset):
        def __init__(s, rows, tf): s.rows, s.tf = rows, tf
        def __len__(s): return len(s.rows)
        def __getitem__(s, i): b, a, g, r = s.rows[i]; return s.tf(Image.open(io.BytesIO(b)).convert("RGB")), a, g, r
    trn, val = load("train"), load("validation"); print("train", len(trn), "val", len(val), flush=True)
    dl = DataLoader(DS(trn, tr), batch_size=256, shuffle=True, num_workers=32, drop_last=True, persistent_workers=True)
    vl = DataLoader(DS(val, te), batch_size=512, num_workers=16)
    m = Net(pretrained=True).to(dev); EP = 10
    opt = torch.optim.AdamW(m.parameters(), lr=4e-4, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-4, total_steps=EP * len(dl), pct_start=0.1)
    ce = torch.nn.CrossEntropyLoss(label_smoothing=0.1); best = 0; os.makedirs(f"{B}/models/faceattr", exist_ok=True)
    for ep in range(EP):
        m.train()
        for x, a, g, r in dl:
            x, a, g, r = x.to(dev), a.to(dev), g.to(dev), r.to(dev)
            with torch.autocast("npu", dtype=torch.bfloat16):
                pr, pg, pa = m(x); loss = ce(pr.float(), r) + 0.5 * ce(pg.float(), g) + ce(pa.float(), a)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        m.eval(); C = np.zeros((7, 7), int); ga = aa = aa1 = n = 0
        with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
            for x, a, g, r in vl:
                pr, pg, pa = m(x.to(dev)); pr, pg, pa = pr.argmax(1).cpu(), pg.argmax(1).cpu(), pa.argmax(1).cpu()
                for t_, p_ in zip(r, pr): C[t_, p_] += 1
                ga += (pg == g).sum().item(); aa += (pa == a).sum().item(); aa1 += ((pa - a).abs() <= 1).sum().item(); n += len(a)
        racc = np.trace(C) / C.sum(); asn = [0, 6]; tp = C[np.ix_(asn, asn)].sum()
        print(f"EVAL ep{ep} race7 {racc:.3f} asian P {tp/C[:,asn].sum():.3f} R {tp/C[asn,:].sum():.3f} gender {ga/n:.3f} age {aa/n:.3f} age+-1 {aa1/n:.3f}", flush=True)
        if racc > best: best = racc; torch.save(m.state_dict(), f"{B}/models/faceattr/convnext_t_fairface.pt")
    print("FACEATTR_DONE best", best)

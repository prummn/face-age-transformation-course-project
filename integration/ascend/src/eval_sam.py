"""Same-source eval: for each held-out aligned input, outputs at estimated own age (rec) and 20/40/60/80. Scores: FairFace (secondary), IR-SE50 ID cos."""
import os, sys, glob, json, argparse, cv2, torch, torch_npu
B = "/data/home/1120231322/scow/face-age"; sys.path.insert(0, f"{B}/src")
from sam_npu import load_sam
from sam_losses import IDLoss, AgeEst
ap = argparse.ArgumentParser(); ap.add_argument("--enc", default=""); ap.add_argument("--out", required=True); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
dev = "npu:0"; torch.npu.set_device(dev); net = load_sam(dev)
if a.enc: net.encoder.load_state_dict(torch.load(a.enc, map_location="cpu"))
idl, agee = IDLoss().to(dev), AgeEst().to(dev); AGES = [20, 40, 60, 80]; recs = []
with torch.no_grad():
    for p in sorted(glob.glob(f"{B}/data/aligned/eval/*.png")):
        x = torch.from_numpy(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(127.5).sub(1)[None].to(dev)
        src, pa_in = agee(x); ages = torch.tensor([src.item()] + [g / 100 for g in AGES], device=dev)
        y = net(torch.cat([x.expand(len(ages), -1, -1, -1), ages.view(-1, 1, 1, 1).expand(-1, 1, 256, 256)], 1), randomize_noise=False, resize=True)
        age_o, pa_o = agee(y); cos = idl.cos(y, x.expand(len(ages), -1, -1, -1)); name = os.path.basename(p)[:-4]
        cv2.imwrite(f"{a.out}/{name}_input.png", cv2.imread(p))
        for i, tag in enumerate(["rec"] + [f"a{g}" for g in AGES]):
            fn = f"{a.out}/{name}_{tag}.png"; cv2.imwrite(fn, ((y[i].clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).byte().cpu().numpy()[:, :, ::-1])
            recs.append(dict(file=fn, input=p, tag=tag, target_age=None if tag == "rec" else int(tag[1:]), src_age_est=round(src.item() * 100, 1),
                             age_est=round(age_o[i].item() * 100, 1), id_cos=round(cos[i].item(), 4), p_asian_ff_in=round(pa_in.item(), 3), p_asian_ff=round(pa_o[i].item(), 3)))
json.dump(recs, open(f"{a.out}/records.json", "w")); print("EVAL_GEN_DONE", len(recs) // 5, "inputs")

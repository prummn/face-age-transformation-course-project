"""SAM encoder fine-tuning on Asian faces (Ascend, DDP via torchrun). Generator + pretrained pSp encoder frozen.
Loss: aging_lambda*MSE(age(y_hat),tgt) + 0.1*w*(1-cos_id(y_hat,x)) + cycle[0.25*L2 + 0.1*LPIPS + 0.1*(1-cos_id)](y_rec,x).
Divergence guard: skip update if pre-clip grad norm non-finite or >GN_MAX, or any rank has non-finite loss / rec_l2>REC_MAX."""
import os, sys, glob, time, json, random, argparse, math, numpy as np, cv2, torch, torch_npu, torch.nn.functional as F, torch.distributed as dist
B = "/data/home/1120231322/scow/face-age"; sys.path.insert(0, f"{B}/src")
ap = argparse.ArgumentParser()
for k, t, v in [("steps", int, 1000), ("bs", int, 6), ("lr", float, 5e-5), ("clip", float, 1.0), ("aging_lambda", float, 5.0), ("save_every", int, 250),
                ("log_every", int, 25), ("seed", int, 0), ("gn_max", float, 50.0), ("rec_max", float, 1.0)]: ap.add_argument("--" + k, type=t, default=v)
ap.add_argument("--out", required=True); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
DDP = "WORLD_SIZE" in os.environ; RANK = int(os.environ.get("RANK", 0)); LR_ = int(os.environ.get("LOCAL_RANK", 0))
dev = f"npu:{LR_}"; torch.npu.set_device(dev); random.seed(a.seed + RANK); torch.manual_seed(a.seed + RANK)
if DDP: dist.init_process_group("hccl")
from sam_npu import load_sam
from sam_losses import IDLoss, AgeEst
import lpips
net = load_sam(dev).train()
for m in (net.decoder, net.pretrained_encoder): m.eval(); [p.requires_grad_(False) for p in m.parameters()]
idl, agee = IDLoss().to(dev), AgeEst().to(dev); lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
for p in lp.parameters(): p.requires_grad_(False)
params = [p for p in net.encoder.parameters() if p.requires_grad]
if DDP: net.encoder = torch.nn.parallel.DistributedDataParallel(net.encoder, device_ids=[LR_], broadcast_buffers=False)
opt = torch.optim.Adam(params, lr=a.lr)
files = sorted(glob.glob(f"{B}/data/aligned/train/*.png"))
def batch():
    xs = [cv2.cvtColor(cv2.imread(random.choice(files)), cv2.COLOR_BGR2RGB) for _ in range(a.bs)]
    x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).float().div(127.5).sub(1)
    return torch.where(torch.rand(a.bs, 1, 1, 1) < .5, x, x.flip(3)).to(dev)
def run(x, age): return net(torch.cat([x, age.view(-1, 1, 1, 1).expand(-1, 1, 256, 256)], 1), randomize_noise=False, resize=True)
def save(tag):
    if RANK == 0:
        e = net.encoder.module if DDP else net.encoder
        torch.save({k: v.cpu() for k, v in e.state_dict().items()}, f"{a.out}/encoder_{tag}.pt.tmp"); os.replace(f"{a.out}/encoder_{tag}.pt.tmp", f"{a.out}/encoder_{tag}.pt")
log = open(f"{a.out}/train_log.jsonl" if RANK == 0 else os.devnull, "a"); json.dump(vars(a), open(f"{a.out}/config.json", "w")) if RANK == 0 else None
t0 = time.time(); skipped = 0
for step in range(1, a.steps + 1):
    torch.npu.synchronize(); ts = time.time()
    x = batch()
    with torch.no_grad(): src, pa_in = agee(x)
    tgt = torch.empty(a.bs, device=dev).uniform_(0.05, 0.90); tgt = torch.where(torch.rand(a.bs, device=dev) < 1 / 3, src, tgt)
    y_hat = run(x, tgt); y_rec = run(y_hat, src); age_hat, pa_hat = agee(y_hat); w = 1 - (tgt - src).abs()
    L = {"aging": a.aging_lambda * F.mse_loss(age_hat, tgt), "id": 0.1 * (w * (1 - idl.cos(y_hat, x))).mean(),
         "rec_l2": 0.25 * F.mse_loss(y_rec, x), "lpips": 0.1 * lp(y_rec, x).mean(), "id_cycle": 0.1 * (1 - idl.cos(y_rec, x)).mean()}
    loss = sum(L.values()); opt.zero_grad(set_to_none=True)
    bad_local = torch.tensor(float((not torch.isfinite(loss)) or L["rec_l2"].item() > a.rec_max), device=dev)
    if DDP: dist.all_reduce(bad_local, op=dist.ReduceOp.MAX)
    if bad_local.item() == 0: loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(params, a.clip) if bad_local.item() == 0 else torch.tensor(float("nan"))   # returns PRE-clip total norm (synced by DDP allreduce)
    nonfinite = sum(int(not torch.isfinite(p.grad).all()) for p in params if p.grad is not None)
    skip = bad_local.item() > 0 or not math.isfinite(gn.item()) or gn.item() > a.gn_max
    if skip: opt.zero_grad(set_to_none=True); skipped += 1
    else: opt.step()
    torch.npu.synchronize(); dt = time.time() - ts
    rec = dict(step=step, sec=round(dt, 3), loss=round(loss.item(), 5) if math.isfinite(loss.item()) else str(loss.item()), **{k: round(v.item(), 5) for k, v in L.items()},
               grad_norm_preclip=round(gn.item(), 4) if math.isfinite(gn.item()) else str(gn.item()), nonfinite_grad_tensors=nonfinite, skipped=int(skip), skipped_total=skipped,
               p_asian_in_ff=round(pa_in.mean().item(), 3), p_asian_out_ff=round(pa_hat.mean().item(), 3))
    log.write(json.dumps(rec) + "\n"); log.flush()
    if RANK == 0 and (step % a.log_every == 0 or step <= 3 or skip): print(json.dumps(rec), flush=True)
    if a.save_every and step % a.save_every == 0: save(f"step{step}")
save(f"step{a.steps}")
if DDP: dist.barrier()
if RANK == 0: print(f"FT_DONE steps={a.steps} skipped={skipped} total={time.time()-t0:.1f}s", flush=True)

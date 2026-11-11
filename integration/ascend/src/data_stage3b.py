"""Stage 3b (NPU): FairFace classifier on crops -> Asian adult filter (p_asian>=.75, p_minor<.15, age_exp>=20) -> dHash dedup -> split.
Split by md5(key)%50: 0,1 -> eval (held out, ~4%), else train."""
import os, sys, glob, hashlib, numpy as np, torch, torch_npu, pandas as pd, pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool
sys.path.insert(0, "/data/home/1120231322/scow/face-age/src"); from faceattr import Net, te
B = "/data/home/1120231322/scow/face-age"; MID = np.array([1, 6, 15, 25, 35, 45, 55, 65, 75.])
class DS(Dataset):
    def __init__(s, p): s.p = p
    def __len__(s): return len(s.p)
    def __getitem__(s, i): return te(Image.open(s.p[i]).convert("RGB")), i
def dhash(p):
    a = np.asarray(Image.open(p).convert("L").resize((17, 16)), dtype=np.int16); return np.packbits((a[:, 1:] > a[:, :-1]).flatten()).tobytes().hex()
if __name__ == "__main__":
    dev = "npu:0"; torch.npu.set_device(dev)
    m = Net(); m.load_state_dict(torch.load(f"{B}/models/faceattr/convnext_t_fairface.pt", map_location="cpu")); m = m.to(dev).eval()
    s3 = pq.read_table(f"{B}/data/stage3a.parquet").to_pandas(); s3 = s3[s3.ok]
    paths = [f"{B}/data/faces/{k[:5]}/{k}.jpg" for k in s3.key]; R, G, A = [], [], []
    with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
        for x, _ in DataLoader(DS(paths), batch_size=512, num_workers=32):
            r, g, a = m(x.to(dev)); R.append(r.float().softmax(1).cpu()); G.append(g.float().softmax(1).cpu()); A.append(a.float().softmax(1).cpu())
    R, G, A = torch.cat(R).numpy(), torch.cat(G).numpy(), torch.cat(A).numpy()
    s3["p_asian"] = R[:, 0] + R[:, 6]; s3["p_ea"] = R[:, 0]; s3["p_sea"] = R[:, 6]; s3["p_female"] = G[:, 1]; s3["age_exp"] = A @ MID; s3["p_minor"] = A[:, :3].sum(1)
    print("faces", len(s3), "asian>.8", int((s3.p_asian > .8).sum()))
    d = s3[(s3.p_asian >= .75) & (s3.p_minor < .15) & (s3.age_exp >= 20)].copy(); print("asian adult", len(d))
    d["path"] = [f"{B}/data/raw/{k[:5]}/{k}.jpg" for k in d.key]
    with Pool(96) as p: d["dhash"] = p.map(dhash, d.path.tolist(), chunksize=128)
    n0 = len(d); d = d.sort_values("face_frac", ascending=False).drop_duplicates("dhash"); print("dedup", n0, "->", len(d))
    d["split"] = ["eval" if int(hashlib.md5(k.encode()).hexdigest(), 16) % 50 in (0, 1) else "train" for k in d.key]
    meta = pq.read_table(f"{B}/data/stage1_tierA.parquet").to_pandas()[["key", "url", "license"]]
    d = d.merge(meta, on="key"); d.to_parquet(f"{B}/data/index.parquet", index=False)
    print(d.split.value_counts().to_dict(), d[["p_asian", "age_exp", "p_female"]].describe().round(3).to_string()); print("INDEX_DONE", len(d))

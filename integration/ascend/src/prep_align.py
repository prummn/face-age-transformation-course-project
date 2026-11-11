"""FFHQ-style alignment (YuNet 5 landmarks) of index images to 256x256 -> data/aligned/{train,eval}."""
import os, sys, cv2, pyarrow.parquet as pq
from multiprocessing import Pool
sys.path.insert(0, "/data/home/1120231322/scow/face-age/src"); cv2.setNumThreads(1)
B = "/data/home/1120231322/scow/face-age"
def work(a):
    src, dst = a
    if os.path.exists(dst): return True
    import sam_npu
    img = cv2.imread(src); r = sam_npu.ffhq_align(img, 256) if img is not None else None
    if r is None: return False
    cv2.imwrite(dst, r[0]); return True
if __name__ == "__main__":
    d = pq.read_table(f"{B}/data/index.parquet").to_pandas()
    for s in ("train", "eval"): os.makedirs(f"{B}/data/aligned/{s}", exist_ok=True)
    jobs = [(p, f"{B}/data/aligned/{s}/{k}.png") for k, p, s in zip(d.key, d.path, d.split)]
    with Pool(64) as pool: ok = pool.map(work, jobs, chunksize=16)
    print("ALIGN_DONE", sum(ok), "/", len(jobs))

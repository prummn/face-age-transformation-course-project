"""Resumable downloader: resize long side<=1536, min short side 512. Throttled (16 procs x 16 threads) to leave bandwidth for other fetches."""
import os, io, sys, json, time, requests, pyarrow.parquet as pq
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
Image.MAX_IMAGE_PIXELS = 200_000_000
B = "/data/home/1120231322/scow/face-age"; OUT = f"{B}/data/raw"; UA = {"User-Agent": "face-age-dataset-builder/0.1 (research; public-domain images only)"}
def fetch(row):
    url, key = row["url"], row["key"]; p = f"{OUT}/{key[:5]}/{key}.jpg"
    if os.path.exists(p): return key, "exists"
    err = "fail"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=20, headers=UA)
            if r.status_code != 200: return key, f"http{r.status_code}"
            im = Image.open(io.BytesIO(r.content)).convert("RGB"); w, h = im.size
            if min(w, h) < 512: return key, "small"
            s = 1536 / max(w, h)
            if s < 1: im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
            os.makedirs(os.path.dirname(p), exist_ok=True); im.save(p + ".tmp", "JPEG", quality=95); os.replace(p + ".tmp", p)
            return key, "ok"
        except Exception as e: err = type(e).__name__; time.sleep(1 + attempt)
    return key, err
def shard(rows):
    with ThreadPoolExecutor(16) as ex: return list(ex.map(fetch, rows))
if __name__ == "__main__":
    rows = pq.read_table(sys.argv[1]).to_pylist(); N = 500
    chunks = [rows[i:i + N] for i in range(0, len(rows), N)]; stat = {}; t0 = time.time(); done = 0
    with Pool(32) as p:
        for res in p.imap_unordered(shard, chunks):
            for k, s in res: stat[s] = stat.get(s, 0) + 1
            done += len(res); print(f"{done}/{len(rows)} {time.time()-t0:.0f}s {stat}", flush=True)
    print("DOWNLOAD_DONE", stat)

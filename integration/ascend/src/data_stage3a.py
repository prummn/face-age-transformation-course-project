"""Stage 3a (CPU): YuNet (MIT) detection + gates (one dominant face, face >=96px, score>=.85, not gray, not blurry) + padded face crop."""
import os, glob, cv2, numpy as np, pyarrow as pa, pyarrow.parquet as pq
from multiprocessing import Pool
cv2.setNumThreads(1)
B = "/data/home/1120231322/scow/face-age"; DET = None
COLS = ["key", "ok", "why", "W", "H", "n_big", "fw", "fscore", "face_frac", "sat", "blur"]
def init():
    global DET; DET = cv2.FaceDetectorYN.create(f"{B}/models/yunet/face_detection_yunet_2023mar.onnx", "", (320, 320), 0.6, 0.3, 50)
def proc(path):
    key = os.path.basename(path)[:-4]; im = cv2.imread(path)
    if im is None: return dict(key=key, ok=False, why="read")
    H, W = im.shape[:2]; s = 640 / max(H, W); sm = cv2.resize(im, (round(W * s), round(H * s)))
    DET.setInputSize((sm.shape[1], sm.shape[0])); _, f = DET.detect(sm)
    if f is None or len(f) == 0: return dict(key=key, ok=False, why="noface", W=W, H=H)
    f = f[f[:, 14] >= 0.85]
    if len(f) == 0: return dict(key=key, ok=False, why="lowscore", W=W, H=H)
    area = f[:, 2] * f[:, 3]; o = np.argsort(-area); f = f[o]; area = area[o]; nf = int((area >= 0.3 * area[0]).sum())
    x, y, w, h = f[0, :4] / s
    rec = dict(key=key, W=W, H=H, n_big=nf, fw=float(min(w, h)), fscore=float(f[0, 14]), face_frac=float(w * h / (W * H)))
    rec["sat"] = float(cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 1].mean())
    cx, cy = x + w / 2, y + h / 2; half = max(w, h) * 1.25 / 2 * 1.25
    x0, y0, x1, y1 = int(cx - half), int(cy - half), int(cx + half), int(cy + half)
    crop = cv2.copyMakeBorder(im, max(0, -y0), max(0, y1 - H), max(0, -x0), max(0, x1 - W), cv2.BORDER_REFLECT)[max(0, y0):max(0, y0) + (y1 - y0), max(0, x0):max(0, x0) + (x1 - x0)]
    rec["blur"] = float(cv2.Laplacian(cv2.cvtColor(cv2.resize(crop, (256, 256)), cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    why = "multi" if nf != 1 else "smallface" if rec["fw"] < 96 else "gray" if rec["sat"] < 25 else "blur" if rec["blur"] < 40 else None
    rec["ok"] = why is None; rec["why"] = why or "ok"
    if rec["ok"]:
        os.makedirs(f"{B}/data/faces/{key[:5]}", exist_ok=True)
        cv2.imwrite(f"{B}/data/faces/{key[:5]}/{key}.jpg", cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return rec
if __name__ == "__main__":
    paths = sorted(glob.glob(f"{B}/data/raw/*/*.jpg"))
    with Pool(96, initializer=init) as p: rows = p.map(proc, paths, chunksize=64)
    rows = [{c: r.get(c) for c in COLS} for r in rows]
    pq.write_table(pa.Table.from_pylist(rows), f"{B}/data/stage3a.parquet")
    import collections; print(collections.Counter(r["why"] for r in rows)); print("STAGE3A_DONE", len(rows))

"""SAM (pSp encoder + StyleGAN2 generator) on Ascend: FFHQ alignment from YuNet 5 landmarks; same-source age batches."""
import os, sys, numpy as np, cv2, torch
from argparse import Namespace
B = "/data/home/1120231322/scow/face-age"; sys.path.insert(0, f"{B}/src/SAM")
YUNET = f"{B}/models/yunet/face_detection_yunet_2023mar.onnx"
def detect5(img):
    h, w = img.shape[:2]; det = cv2.FaceDetectorYN.create(YUNET, "", (w, h), 0.6, 0.3, 50); _, f = det.detect(img)
    if f is None or len(f) == 0: return None
    f = f[np.argmax(f[:, 2] * f[:, 3])]; pts = f[4:14].reshape(5, 2)
    eyes = sorted([pts[0], pts[1]], key=lambda p: p[0]); mouth = sorted([pts[3], pts[4]], key=lambda p: p[0])
    return np.array(eyes[0]), np.array(eyes[1]), np.array(mouth[0]), np.array(mouth[1])
def ffhq_align(img, size=256):
    lm = detect5(img)
    if lm is None: return None
    el, er, ml, mr = lm; eye_avg = (el + er) * .5; e2e = er - el; mouth_avg = (ml + mr) * .5; e2m = mouth_avg - eye_avg
    x = e2e - np.flipud(e2m) * [-1, 1]; x /= np.hypot(*x); x *= max(np.hypot(*e2e) * 2.0, np.hypot(*e2m) * 1.8)
    y = np.flipud(x) * [-1, 1]; c = eye_avg + e2m * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y]).astype(np.float32)
    M = cv2.getPerspectiveTransform(quad, np.array([[0, 0], [0, size], [size, size], [size, 0]], np.float32))
    return cv2.warpPerspective(img, M, (size, size), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT), M
def load_sam(dev, ckpt=f"{B}/models/sam/sam_ffhq_aging.pt"):
    from models.psp import pSp
    o = torch.load(ckpt, map_location="cpu")["opts"]; o["checkpoint_path"] = ckpt; o["device"] = dev
    return pSp(Namespace(**o)).to(dev).eval()

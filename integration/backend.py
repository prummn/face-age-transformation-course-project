"""Memory-only SAM inference API for the Age Mirror classroom prototype.

Run with the existing face-age-web environment. Training files are read-only.
Only 127.0.0.1 is supported; expose through an SSH tunnel / same-origin gateway.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import io
import logging
import math
import os
from pathlib import Path
import secrets
import sys
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

VERSION = "1.3.0"
BASE = Path(os.environ.get("FACE_AGE_BASE", "/data/home/1120231322/scow/face-age"))
WEB_BASE = Path(os.environ.get("FACE_AGE_WEB_BASE", "/data/home/1120231322/scow/face-age-web"))
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_REQUEST_BYTES = 21 * 1024 * 1024
MAX_PIXELS = 24_000_000
SESSION_TTL = 30 * 60
MAX_SESSIONS = 8
ENGINE_NAME = "SAM / StyleGAN2"
LOG = logging.getLogger("face_age_api")


class SelectedFace(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)
    id: Optional[str | int] = None
    manual: bool = False


class PrepareRequest(BaseModel):
    image: str
    face: SelectedFace


class TransformRequest(BaseModel):
    session_id: str = Field(min_length=16, max_length=128)
    target_age: int = Field(ge=5, le=90, strict=True)
    request_id: Optional[str | int] = None
    enhance: bool = False
    preserve_color: bool = True


@dataclass
class Session:
    aligned: np.ndarray
    created_at: float
    touched_at: float
    cancelled: bool = False


class SessionStore:
    """Deletion immediately invalidates references held by queued/running work."""

    def __init__(self, ttl=SESSION_TTL, capacity=MAX_SESSIONS, clock=time.monotonic):
        self.ttl, self.capacity, self.clock = ttl, capacity, clock
        self.items: dict[str, Session] = {}
        self.lock = threading.RLock()

    def _expire(self):
        now = self.clock()
        for key, item in list(self.items.items()):
            if now - item.touched_at >= self.ttl:
                item.cancelled = True
                self.items.pop(key, None)

    def create(self, aligned):
        with self.lock:
            self._expire()
            if len(self.items) >= self.capacity:
                raise HTTPException(429, "当前照片会话已满，请清除之前的照片后重试。")
            token = secrets.token_urlsafe(24)
            now = self.clock()
            self.items[token] = Session(aligned, now, now)
            return token

    def get(self, token):
        with self.lock:
            self._expire()
            item = self.items.get(token)
            if item is None or item.cancelled:
                raise HTTPException(410, "本次照片已清除或过期，请重新选择照片。")
            item.touched_at = self.clock()
            return item

    def check(self, token, item):
        with self.lock:
            self._expire()
            if item.cancelled or self.items.get(token) is not item:
                raise HTTPException(410, "本次任务已取消，生成结果已丢弃。")
            item.touched_at = self.clock()

    def delete(self, token):
        with self.lock:
            item = self.items.pop(token, None)
            if item is not None:
                item.cancelled = True

    def clear(self):
        with self.lock:
            for item in self.items.values():
                item.cancelled = True
            self.items.clear()

    def count(self):
        with self.lock:
            self._expire()
            return len(self.items)


def decode_image(value):
    if not isinstance(value, str) or "," not in value:
        raise HTTPException(422, "照片格式无效，请重新上传 JPG、PNG 或 WEBP。")
    header, encoded = value.split(",", 1)
    if header.lower() not in ("data:image/png;base64", "data:image/jpeg;base64", "data:image/webp;base64"):
        raise HTTPException(415, "仅支持 JPG、PNG 或 WEBP 图片。")
    if len(encoded) > 4 * math.ceil(MAX_IMAGE_BYTES / 3):
        raise HTTPException(413, "照片过大，请选择不超过 15 MB 的图片。")
    try:
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "照片过大，请选择不超过 15 MB 的图片。")
        with Image.open(io.BytesIO(raw)) as picture:
            if picture.format not in ("JPEG", "PNG", "WEBP"):
                raise HTTPException(415, "图片内容必须是 JPG、PNG 或 WEBP。")
            if picture.width * picture.height > MAX_PIXELS:
                raise HTTPException(413, "照片像素过多，请缩小到 2400 万像素以内。")
            if min(picture.size) < 64:
                raise HTTPException(422, "照片尺寸太小，请使用清晰的人脸照片。")
            return np.asarray(ImageOps.exif_transpose(picture).convert("RGB")).copy()
    except HTTPException:
        raise
    except (ValueError, binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(422, "无法读取这张照片，请重新上传 JPG、PNG 或 WEBP。") from None


def image_data_url(rgb):
    output = io.BytesIO()
    Image.fromarray(rgb).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def intersection_over_union(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0.0, min(ay + ah, by + bh) - max(ay, by))
    return intersection / max(aw * ah + bw * bh - intersection, 1e-12)


def match_selected_face(detections, selected, width, height):
    """Choose only a detection overlapping the user's selected region.

    Never fall back to the largest face: two distant faces must not interchange.
    YuNet rows contain bbox, five landmarks, then confidence.
    """
    x, y, w, h = (selected.x * width, selected.y * height, selected.w * width, selected.h * height)
    if selected.x + selected.w > 1.01 or selected.y + selected.h > 1.01:
        raise HTTPException(422, "选脸区域超出照片，请重新选择人脸。")
    ranked = []
    for face in detections if detections is not None else []:
        fx, fy, fw, fh = (float(v) for v in face[:4])
        cx, cy = fx + fw / 2, fy + fh / 2
        iou = intersection_over_union((x, y, w, h), (fx, fy, fw, fh))
        # A selected oval can be narrower than YuNet's rectangle. Its center and
        # substantial area must still coincide, instead of nearby faces matching.
        center_inside = x <= cx <= x + w and y <= cy <= y + h
        if not center_inside or iou < 0.08:
            continue
        distance = ((cx - (x + w / 2)) / w) ** 2 + ((cy - (y + h / 2)) / h) ** 2
        ranked.append((iou - 0.15 * distance, face))
    if not ranked:
        raise HTTPException(422, "没有在选中区域检测到清晰人脸，请重新框选该人物或更换照片。")
    return max(ranked, key=lambda item: item[0])[1]


def ffhq_affine(landmarks, size=256):
    """Training-consistent FFHQ alignment from the selected YuNet landmarks.

    Return a 2×3 source-pixel → aligned-pixel matrix. The FFHQ quadrilateral is
    a parallelogram, so this is the affine form of sam_npu.ffhq_align's matrix.
    """
    points = np.asarray(landmarks, dtype=np.float64).reshape(5, 2)
    el, er = sorted(points[:2], key=lambda point: point[0])
    ml, mr = sorted(points[3:5], key=lambda point: point[0])
    eye_avg = (el + er) * 0.5
    e2e = er - el
    e2m = (ml + mr) * 0.5 - eye_avg
    axis_x = e2e - np.flipud(e2m) * [-1, 1]
    norm = np.hypot(*axis_x)
    if norm < 1e-5:
        raise HTTPException(422, "人脸关键点不稳定，请使用清晰正面照片重试。")
    axis_x = axis_x / norm * max(np.hypot(*e2e) * 2.0, np.hypot(*e2m) * 1.8)
    axis_y = np.flipud(axis_x) * [-1, 1]
    center = eye_avg + e2m * 0.1
    origin = center - axis_x - axis_y
    basis = np.stack([2 * axis_x, 2 * axis_y], axis=1)
    try:
        linear = size * np.linalg.inv(basis)
    except np.linalg.LinAlgError:
        raise HTTPException(422, "人脸关键点不稳定，请重新选择照片。") from None
    return np.column_stack([linear, -linear @ origin])


def canvas_matrix(matrix):
    return [float(matrix[0, 0]), float(matrix[1, 0]), float(matrix[0, 1]), float(matrix[1, 1]), float(matrix[0, 2]), float(matrix[1, 2])]


class SamePersonEngine:
    def __init__(self):
        self.ready = False
        self.error = None
        self.lock = threading.RLock()
        self.sr = None
        self.dev = "npu:0"

    def initialize(self):
        try:
            started = time.monotonic()
            # These are intentionally lazy imports so health is available during
            # a cold shared-filesystem / NPU startup.
            import cv2
            import torch
            import torch_npu  # noqa: F401
            sys.path.insert(0, str(BASE / "src"))
            from sam_npu import load_sam
            from sam_losses import AgeEst
            self.cv2, self.torch = cv2, torch
            torch.npu.set_device(self.dev)
            self.net = load_sam(self.dev)
            state = torch.load(BASE / "out/ft_l1/encoder_step1000.pt", map_location="cpu")
            self.net.encoder.load_state_dict(state)
            del state
            self.age_net = AgeEst().to(self.dev).eval()
            self.yunet = str(BASE / "models/yunet/face_detection_yunet_2023mar.onnx")
            with torch.inference_mode():
                warmup = torch.zeros(1, 4, 256, 256, device=self.dev)
                self.net(warmup, randomize_noise=False, resize=False)
                self.age_net(warmup[:, :3])
                torch.npu.synchronize()
            self.ready = True
            LOG.info("SAM model ready after %.1f seconds", time.monotonic() - started)
        except Exception:
            self.error = "模型启动失败，请检查服务日志和昇腾环境后重启。"
            LOG.exception("Model initialization failed")

    def ensure_ready(self):
        if not self.ready:
            raise HTTPException(503, self.error or "年龄模型正在启动，请稍后重试。")

    def tensor(self, rgb):
        return self.torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(127.5).sub(1)[None].to(self.dev)

    def align(self, rgb, selected):
        cv2 = self.cv2
        h, w = rgb.shape[:2]
        scale = min(1.0, 2048 / max(h, w))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if scale < 1:
            detection_img = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            detection_img = bgr
        dh, dw = detection_img.shape[:2]
        detector = cv2.FaceDetectorYN.create(self.yunet, "", (dw, dh), 0.6, 0.3, 100)
        _, detected = detector.detect(detection_img)
        if detected is not None:
            detected = detected.copy()
            detected[:, [0, 2, 4, 6, 8, 10, 12]] *= w / dw
            detected[:, [1, 3, 5, 7, 9, 11, 13]] *= h / dh
        chosen = match_selected_face(detected, selected, w, h)
        matrix = ffhq_affine(chosen[4:14].reshape(5, 2))
        aligned = cv2.warpAffine(rgb, matrix, (256, 256), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)
        inverse = cv2.invertAffineTransform(matrix)
        transform = {"matrix": canvas_matrix(matrix), "inverse": canvas_matrix(inverse), "sourceWidth": w, "sourceHeight": h}
        return aligned, transform, chosen

    def prepare(self, rgb, selected):
        self.ensure_ready()
        with self.lock, self.torch.inference_mode():
            self.torch.npu.set_device(self.dev)
            aligned, transform, chosen = self.align(rgb, selected)
            estimated, _ = self.age_net(self.tensor(aligned))
            self.torch.npu.synchronize()
            reference_age = int(round(float(estimated.item()) * 100))
        return aligned, transform, reference_age, chosen

    def display_reference(self, rgb, transform):
        """Sample the uploaded original directly for viewing; never upscale the
        256 encoder input to masquerade as a high-detail source photograph.
        Pixel-centre mapping matches 1024 -> 256 area pooling (4x + 1.5).
        """
        a,b,c,d,e,f = transform["matrix"]
        matrix = np.array([[a,c,e],[b,d,f]],dtype=np.float64)*4
        matrix[:,2] += 1.5
        return self.cv2.warpAffine(rgb,matrix,(1024,1024),flags=self.cv2.INTER_CUBIC,borderMode=self.cv2.BORDER_REFLECT)

    def transform(self, item, age, enhanced, assert_active):
        self.ensure_ready()
        torch = self.torch
        with self.lock, torch.inference_mode():
            assert_active()
            torch.npu.set_device(self.dev)
            started = time.monotonic()
            tensor = self.tensor(item.aligned)
            condition = torch.full((1, 1, 256, 256), age / 100, device=self.dev)
            generated = self.net(torch.cat([tensor, condition], 1), randomize_noise=False, resize=False)
            torch.npu.synchronize()
            native = ((generated[0].clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).byte().cpu().numpy()
            if native.shape != (1024, 1024, 3):
                raise RuntimeError("Model output must be native 1024 x 1024")
            elapsed = time.monotonic() - started
            output = native
            warning = None
            output_kind = "native"
            if enhanced:
                try:
                    assert_active()
                    if self.sr is None:
                        sys.path.insert(0, str(WEB_BASE))
                        from superres_native import NativeSuperResolution
                        self.sr = NativeSuperResolution(str(WEB_BASE / "models/RealESRGAN_x2plus.pth"), self.dev)
                    output = self.sr.enhance(native)
                    torch.npu.synchronize()
                    output_kind = "super_resolution"
                except HTTPException:
                    raise
                except Exception:
                    LOG.exception("Super-resolution failed; retaining native result")
                    warning = "超分增强未完成，已保留可下载的原生 1024 结果。"
            assert_active()
            return native, output, elapsed, output_kind, warning


store = SessionStore()
engine = SamePersonEngine()
queue_slots = threading.BoundedSemaphore(8)


@contextmanager
def admitted():
    if not queue_slots.acquire(blocking=False):
        raise HTTPException(429, "当前处理任务较多，请稍后重试。")
    try:
        yield
    finally:
        queue_slots.release()


@asynccontextmanager
async def lifespan(app):
    if not os.environ.get("FACE_AGE_SKIP_INIT"):
        threading.Thread(target=engine.initialize, name="sam-initialize", daemon=True).start()
    stop_cleanup = threading.Event()
    def cleanup():
        while not stop_cleanup.wait(30):
            store.count()
    threading.Thread(target=cleanup, name="session-cleanup", daemon=True).start()
    yield
    stop_cleanup.set()
    store.clear()


class RequestSizeLimit:
    """Bound chunked and Content-Length bodies before Starlette parses JSON."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if scope["method"] not in ("POST", "PUT", "PATCH"):
            return await self.app(scope, receive, send)
        chunks = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > MAX_REQUEST_BYTES:
                return await JSONResponse({"detail": "请求过大，请使用不超过 15 MB 的图片。"}, status_code=413)(scope, receive, send)
            chunks.append(message)
            if not message.get("more_body", False):
                break
        iterator = iter(chunks)
        async def buffered_receive():
            try:
                return next(iterator)
            except StopIteration:
                return await receive()
        await self.app(scope, buffered_receive, send)


app = FastAPI(title="时空镜年龄变换接口", version=VERSION, lifespan=lifespan, docs_url=None, redoc_url=None)
app.add_middleware(RequestSizeLimit)


@app.middleware("http")
async def privacy_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    fields = {str(error["loc"][-1]) for error in exc.errors() if error.get("loc")}
    detail = "输入参数无效，请重新选择照片和人脸。"
    if "target_age" in fields:
        detail = "目标年龄必须是 5 至 90 岁之间的整数。"
    elif "session_id" in fields:
        detail = "本次照片会话无效，请重新选择照片。"
    return JSONResponse({"detail": detail}, status_code=422, headers={"Cache-Control": "no-store"})


@app.exception_handler(Exception)
async def unexpected_error(request, exc):
    LOG.error("API operation failed", exc_info=(type(exc), exc, exc.__traceback__))
    return JSONResponse({"detail": "本次处理未完成，请重试或重新选择照片。"}, status_code=500, headers={"Cache-Control": "no-store"})


@app.get("/api/health")
def health():
    return {"ready": engine.ready, "engine": ENGINE_NAME, "device": "Ascend NPU", "model": "ft_l1", "native_size": 1024, "version": VERSION, "status": "ready" if engine.ready else ("failed" if engine.error else "loading"), "detail": engine.error, "active_sessions": store.count(), "session_ttl_seconds": SESSION_TTL, "features": {"source_color_preservation": True, "raw_model_output": True}}


@app.post("/api/prepare")
def prepare(payload: PrepareRequest):
    engine.ensure_ready()
    with admitted():
        image = decode_image(payload.image)
        aligned, transform, age, chosen = engine.prepare(image, payload.face)
        session_id = store.create(aligned)
        response = {"session_id": session_id, "aligned_image": image_data_url(aligned), "age": {"age": age, "source": "model", "model": "FairFace", "regression": None, "probabilities": None}, "transform": transform, "alignment": "sam_ffhq", "meta": {"engine": ENGINE_NAME, "model": "ft_l1", "face_id": payload.face.id, "detected_bbox": [float(v) for v in chosen[:4]], "detector": "YuNet", "age_note": "FairFace 参考估龄，不代表真实年龄；高龄估计上限约 75 岁。", "retention": "内存保存，清除立即失效，闲置 30 分钟过期"}}
        if hasattr(engine,"display_reference"):
            try:
                reference = engine.display_reference(image,transform)
                response["reference_image"] = image_data_url(reference)
                response["meta"]["reference_kind"] = "resampled_uploaded_original_1024"
            except Exception:
                # A viewing enhancement must not invalidate the prepared input.
                LOG.exception("High-resolution original preview unavailable")
        return response


@app.post("/api/transform")
def transform(payload: TransformRequest):
    engine.ensure_ready()
    with admitted():
        item = store.get(payload.session_id)
        native, output, elapsed, kind, warning = engine.transform(item, payload.target_age, payload.enhance, lambda: store.check(payload.session_id, item))
        store.check(payload.session_id, item)
        # Keep the model output available. This bounded chroma correction is an
        # optional presentation step, not a newly trained model or super-resolution.
        from source_color import preserve_source_chroma
        color_started = time.monotonic()
        corrected_native, color_meta = preserve_source_chroma(item.aligned, native, .65 if payload.preserve_color else 0)
        if kind == "super_resolution":
            output, _ = preserve_source_chroma(item.aligned, output, .65 if payload.preserve_color else 0)
        else:
            output = corrected_native
        color_elapsed = time.monotonic()-color_started
        response = {"image": image_data_url(output), "raw_image": image_data_url(native), "width": output.shape[1], "height": output.shape[0], "request_id": payload.request_id, "meta": {"engine": ENGINE_NAME, "model": "ft_l1", "target_age": payload.target_age, "elapsed_seconds": round(elapsed, 3), "output_kind": kind, "alignment": "sam_ffhq", "color_preservation": color_meta, "postprocess_seconds": round(color_elapsed,3), "raw_model_size": 1024}}
        if kind == "super_resolution":
            response["native_image"] = image_data_url(corrected_native)
        if warning:
            response["warning"] = warning
        store.check(payload.session_id, item)
        return response


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    store.delete(session_id)
    return {"deleted": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1, access_log=False)


if __name__ == "__main__":
    main()

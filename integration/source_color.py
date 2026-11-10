"""Conservative, per-photo chroma preservation for FFHQ-aligned age outputs.

This is an optional output correction, not a new learned model or a skin-tone
classifier. It estimates a single bounded Lab a/b offset from two cheek areas
of this same person's source and generated images. Lightness and structure are
not blended with the source. Eyes, mouth, hair and the surrounding background
are excluded geometrically. Raw generated images must remain available.
"""
from __future__ import annotations

import numpy as np

METHOD = "source_cheek_lab_v2"


def _ellipse(x, y, cx, cy, rx, ry, feather=.3):
    radius = ((x-cx)/rx)**2 + ((y-cy)/ry)**2
    t = np.clip((1-radius)/feather, 0, 1)
    return t*t*(3-2*t)


def masks(size=256):
    """FFHQ geometry only; no ethnicity classifier or fixed skin RGB range."""
    y, x = np.mgrid[:size, :size].astype(np.float32)
    x, y = (x+.5)/size, (y+.5)/size
    cheeks = (_ellipse(x,y,.365,.565,.055,.055,.15)
              + _ellipse(x,y,.635,.565,.055,.055,.15)) > .9
    face = _ellipse(x,y,.5,.54,.30,.34,.40)
    # Protect the actual eyes/lips while keeping the surrounding skin continuous.
    # Broad eye/nostril cutouts caused visibly inconsistent islands in v1.
    for cx,cy,rx,ry in ((.375,.455,.083,.022),(.625,.455,.083,.022),
                        (.5,.655,.10,.028)):
        face *= 1-_ellipse(x,y,cx,cy,rx,ry,.35)
    return cheeks, face


def preserve_source_chroma(source_rgb, generated_rgb, strength=.65):
    """Return corrected RGB uint8 + honest metadata; keep input arrays intact.

    Diagnostic cheek statistics are proxies, not perceptual quality scores.
    Extreme/overexposed or inconsistent source patches are left unchanged.
    """
    import cv2

    strength = float(strength)
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Color preservation strength must be between 0 and 1")
    for array in (source_rgb, generated_rgb):
        if not isinstance(array, np.ndarray) or array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("Expected RGB uint8 images")
    metadata = {"enabled": strength > 0, "applied": False, "method": METHOD,
                "strength": strength, "lightness_preserved": True,
                "note": "按本张原照局部校正色度，保留生成亮度与纹理；原始模型结果单独保留。"}
    if strength == 0:
        metadata["reason"] = "disabled"
        return generated_rgb.copy(), metadata
    source = cv2.resize(source_rgb, (256,256), interpolation=cv2.INTER_AREA)
    target = cv2.resize(generated_rgb, (256,256), interpolation=cv2.INTER_AREA)
    source_lab = cv2.cvtColor(source.astype(np.float32)/255, cv2.COLOR_RGB2LAB)
    target_lab = cv2.cvtColor(target.astype(np.float32)/255, cv2.COLOR_RGB2LAB)
    samples, support = masks()
    # Trim each image's local extremes. No particular skin colour is assumed.
    candidate = samples.copy()
    for lab in (source_lab, target_lab):
        lo, hi = np.percentile(lab[...,0][samples], [10,90])
        candidate &= (lab[...,0] >= lo) & (lab[...,0] <= hi)
    if candidate.sum() < 120:
        metadata["reason"] = "insufficient_cheek_samples"
        return generated_rgb.copy(), metadata
    src_med = np.median(source_lab[candidate], axis=0)
    dst_med = np.median(target_lab[candidate], axis=0)
    source_spread = np.median(np.linalg.norm(source_lab[candidate,1:]-src_med[1:],axis=1))
    clipped = np.mean((source[candidate].min(axis=1) >= 248) | (source[candidate].max(axis=1) <= 5))
    metadata.update({"source_cheek_lab": src_med.round(3).tolist(),
                     "raw_cheek_lab": dst_med.round(3).tolist(),
                     "sample_pixels": int(candidate.sum())})
    left = samples.copy(); left[:,128:] = False
    right = samples.copy(); right[:,:128] = False
    left_lab = np.median(source_lab[left],axis=0)
    right_lab = np.median(source_lab[right],axis=0)
    asymmetry = float(np.linalg.norm(left_lab[1:]-right_lab[1:]))
    metadata["reference_cheek_asymmetry"] = round(asymmetry,3)
    if src_med[0] < 30:
        metadata["reason"] = "low_light_reference"
        metadata["note"] = "原照局部亮度不足，保留原始模型结果；建议使用均匀照明的照片。"
        return generated_rgb.copy(), metadata
    if asymmetry > 12:
        metadata["reason"] = "asymmetric_reference"
        metadata["note"] = "原照两侧色彩差异较大，可能有侧脸、遮挡或混合光照，保留原始模型结果。"
        return generated_rgb.copy(), metadata
    if clipped > .25 or source_spread > 15:
        metadata["reason"] = "unreliable_source_colour"
        return generated_rgb.copy(), metadata
    delta = src_med[1:]-dst_med[1:]
    metadata["measured_chroma_distance"] = round(float(np.linalg.norm(delta)),3)
    if np.linalg.norm(delta) < 3:
        metadata["reason"] = "already_close"
        return generated_rgb.copy(), metadata
    # Cap the offset rather than forcing arbitrary images to exactly match.
    delta *= min(1., 14./max(float(np.linalg.norm(delta)),1e-6)) * strength
    # Reduce corrections on source facial hair, deep shadows and highlights.
    luminance_distance = np.abs(source_lab[...,0]-src_med[0])
    confidence = np.clip((36-luminance_distance)/14, 0, 1)
    support *= cv2.GaussianBlur(confidence, (0,0), 2)
    h,w = generated_rgb.shape[:2]
    alpha = cv2.resize(support, (w,h), interpolation=cv2.INTER_LINEAR)
    output_lab = cv2.cvtColor(generated_rgb.astype(np.float32)/255, cv2.COLOR_RGB2LAB)
    output_lab[...,1:] += alpha[...,None]*delta
    rgb = np.clip(cv2.cvtColor(output_lab,cv2.COLOR_LAB2RGB),0,1)
    output = np.rint(rgb*255).astype(np.uint8)
    # Ensure regions outside the support are bit-for-bit the original output.
    output[alpha <= 1e-5] = generated_rgb[alpha <= 1e-5]
    metadata.update({"applied": True, "reason": "corrected",
                     "chroma_offset": delta.round(3).tolist(),
                     "output_kind": "colour_corrected_1024"})
    return output, metadata

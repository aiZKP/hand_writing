"""Preprocessing shared by TRAINING and INFERENCE.

The single most common bug in deployed ML systems is "train/serve skew": the
model is trained on images prepared one way and served images prepared
another way. Here every path (synthetic training words, the dashboard canvas,
user-saved samples) goes through the SAME functions below.

Image convention everywhere: float32 or uint8, INK = bright (1.0/255),
BACKGROUND = 0 — the same as EMNIST.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from . import config as C

INK_THRESHOLD = 0.15     # fraction of max intensity that counts as "ink"


# --------------------------------------------------------------------------
# Decoding what the browser sends
# --------------------------------------------------------------------------
def decode_data_url(data_url: str, max_bytes: int = 5_000_000) -> Image.Image:
    """'data:image/png;base64,....' -> PIL image (rejects huge payloads)."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url, validate=False)
    if len(raw) > max_bytes:
        raise ValueError("image too large")
    img = Image.open(io.BytesIO(raw))
    if img.width * img.height > 8_000_000:
        raise ValueError("image dimensions too large")
    img.load()
    return img


def image_to_ink(img: Image.Image) -> np.ndarray:
    """Any PIL image -> float32 ink map in [0,1] (ink bright, background 0).

    The canvas exports transparent background + dark strokes, so the ALPHA
    channel is exactly the ink. For opaque images (e.g. a scanned photo of
    black ink on white paper) we invert the luminance instead.
    """
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
        alpha = rgba[..., 3]
        lum = rgba[..., :3].mean(-1)
        if alpha.min() < 0.99:                    # really uses transparency
            return alpha * (1.0 - lum)            # dark & opaque = ink
        return 1.0 - lum
    gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    # If the background is bright (paper), invert so ink becomes bright.
    return 1.0 - gray if np.median(gray) > 0.5 else gray


# --------------------------------------------------------------------------
# Word image -> CRNN input
# --------------------------------------------------------------------------
def ink_columns(ink: np.ndarray, thr: float = INK_THRESHOLD) -> tuple[int, int] | None:
    """First and last column containing ink, or None for an empty image."""
    if ink.max() <= 0:
        return None
    cols = np.where((ink > thr * ink.max()).any(axis=0))[0]
    if cols.size == 0:
        return None
    return int(cols[0]), int(cols[-1])


def band_to_model_input(band: np.ndarray, max_w: int = C.MAX_W_INFER) -> np.ndarray | None:
    """Writing band (any resolution, full band height) -> CRNN input [IMG_H, W].

    1. keep the FULL band height (this preserves where letters sit relative
       to the guide lines -> upper/lower case information survives),
    2. crop horizontally to the ink plus a small margin,
    3. resize to height IMG_H keeping the aspect ratio,
    4. pad to at least MIN_W, squash if wider than max_w.
    Returns float32 in [0,1], or None if there is no ink.
    """
    band = band.astype(np.float32)
    if band.max() > 1.0:
        band = band / 255.0
    span = ink_columns(band)
    if span is None:
        return None
    bh = band.shape[0]
    margin = int(round(C.CROP_MARGIN * bh))
    x0, x1 = span[0] - margin, span[1] + 1 + margin
    pad_l, pad_r = max(0, -x0), max(0, x1 - band.shape[1])
    if pad_l or pad_r:
        band = np.pad(band, ((0, 0), (pad_l, pad_r)))
        x0, x1 = x0 + pad_l, x1 + pad_l
    crop = band[:, x0:x1]

    scale = C.IMG_H / bh
    new_w = max(1, int(round(crop.shape[1] * scale)))
    new_w = min(new_w, max_w)
    img = Image.fromarray((crop * 255).astype(np.uint8)).resize((new_w, C.IMG_H), Image.LANCZOS)
    out = np.asarray(img, dtype=np.float32) / 255.0
    if out.shape[1] < C.MIN_W:
        out = np.pad(out, ((0, 0), (0, C.MIN_W - out.shape[1])))
    return out


# --------------------------------------------------------------------------
# Single character -> CharCNN input (EMNIST style)
# --------------------------------------------------------------------------
def normalize_char_emnist_style(ink: np.ndarray) -> np.ndarray:
    """Crop of ONE character (ink bright) -> 28x28 float32, EMNIST-like.

    EMNIST's own pipeline: crop to the bounding box, center it in a square
    keeping the aspect ratio, leave a 2-px border (content fits 24x24),
    downsample to 28x28. We replicate that so the CharCNN sees familiar input.
    Note that this throws away the character's SIZE — 'o' and 'O' become the
    same picture. That is the main weakness of approach A.
    """
    ink = ink.astype(np.float32)
    if ink.max() > 1.0:
        ink = ink / 255.0
    ys, xs = np.where(ink > INK_THRESHOLD * max(ink.max(), 1e-6))
    if ys.size == 0:
        return np.zeros((C.CHAR_SIZE, C.CHAR_SIZE), np.float32)
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    side = max(h, w)
    square = np.zeros((side, side), np.float32)
    square[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = crop
    inner = C.CHAR_SIZE - 4                                   # 24 px content
    img = Image.fromarray((square * 255).astype(np.uint8)).resize((inner, inner), Image.LANCZOS)
    out = np.zeros((C.CHAR_SIZE, C.CHAR_SIZE), np.float32)
    out[2:2 + inner, 2:2 + inner] = np.asarray(img, dtype=np.float32) / 255.0
    if out.max() > 0:
        out /= out.max()                                       # full contrast like EMNIST
    return out

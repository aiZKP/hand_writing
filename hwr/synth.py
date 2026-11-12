"""Synthetic handwritten WORDS composed from EMNIST characters.

Why synthetic? Model B (CRNN+CTC) needs *word* images with *word* labels.
Public word datasets (IAM, ...) need registration and contain scanned pen
ink, not canvas strokes. EMNIST gives us 700k real handwritten characters, so
we glue them into words ourselves — with infinite variety (a new random word
for every training step).

Realism tricks (each one closes part of the gap to real canvas writing):
  * glyphs are placed on the same guide-line geometry as the dashboard
    (config.char_extent), so lowercase 'o' is small and uppercase 'O' is tall;
  * stroke width is equalized inside a word (one pen), matching the canvas
    brush, by dilating/eroding each scaled glyph;
  * random spacing including NEGATIVE gaps, so letters touch/overlap
    (that is exactly where segmentation-based approach A breaks);
  * random global size, vertical offset, slant (shear) and blur.
"""
from __future__ import annotations

import string
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from . import config as C
from .preprocess import band_to_model_input

GLYPH_THRESHOLD = 40          # uint8 level that counts as ink when cropping


# --------------------------------------------------------------------------
# Character bank
# --------------------------------------------------------------------------
class CharBank:
    """EMNIST glyphs grouped by class with pre-computed tight bounding boxes."""

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images
        mask = images > GLYPH_THRESHOLD
        rows, cols = mask.any(2), mask.any(1)
        self.y0 = rows.argmax(1)
        self.y1 = images.shape[1] - rows[:, ::-1].argmax(1)
        self.x0 = cols.argmax(1)
        self.x1 = images.shape[2] - cols[:, ::-1].argmax(1)
        valid = rows.any(1) & ((self.y1 - self.y0) >= 4)       # skip empty/tiny glyphs
        self.by_class = [np.where((labels == c) & valid)[0] for c in range(C.NUM_CLASSES)]

    def sample(self, ch: str, rng: np.random.Generator) -> np.ndarray:
        idx = self.by_class[C.CHAR_TO_IDX[ch]]
        i = idx[rng.integers(len(idx))]
        return self.images[i, self.y0[i]:self.y1[i], self.x0[i]:self.x1[i]]


def estimate_stroke_width(glyph: np.ndarray) -> float:
    """Average stroke width in pixels ~= 2 * area / perimeter.

    (A stroke of width w and length L has area w*L and a boundary of ~2L.)
    """
    m = glyph > 127
    area = m.sum()
    if area == 0:
        return 1.0
    inner = m.copy()
    inner[1:-1, 1:-1] = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    inner[0, :] = inner[-1, :] = inner[:, 0] = inner[:, -1] = False
    perimeter = max((m & ~inner).sum(), 1)
    return float(2.0 * area / perimeter)


# --------------------------------------------------------------------------
# Text sampling
# --------------------------------------------------------------------------
def load_wordlist(path: Path = C.WORDLIST) -> list[str]:
    if path.exists():
        return [w for w in path.read_text().split() if w]
    return []


def build_wordlist(src: Path = Path("/usr/share/dict/american-english"),
                   out: Path = C.WORDLIST, min_len: int = 2, max_len: int = 12) -> int:
    """Keep only plain ASCII letter words (no apostrophes/accents), lowercased."""
    words = set()
    if src.exists():
        for w in src.read_text(errors="ignore").split():
            if min_len <= len(w) <= max_len and w.isascii() and w.isalpha():
                words.add(w.lower())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sorted(words)))
    return len(words)


class TextSampler:
    """Mixture of text sources.

    Dictionary words alone would teach the BiLSTM an English "language model"
    so strong that it would refuse to read 'xq7Z'. Random strings keep the
    model honest: it must look at the pixels.
    """

    def __init__(self, words: list[str], max_len: int = 12):
        self.words = words
        self.max_len = max_len
        self.letters = string.ascii_letters
        self.digits = string.digits

    def _case(self, w: str, rng) -> str:
        r = rng.random()
        if r < 0.55:
            return w.lower()
        if r < 0.85:
            return w.capitalize()
        return w.upper()

    def sample(self, rng: np.random.Generator) -> str:
        r = rng.random()
        if r < 0.40 and self.words:                                    # dictionary word
            text = self._case(self.words[rng.integers(len(self.words))], rng)
        elif r < 0.75:                                                 # random 62-symbol string
            n = int(rng.integers(1, 11))
            text = "".join(rng.choice(list(C.CHARSET), n))
        elif r < 0.87:                                                 # number
            n = int(rng.integers(1, 9))
            text = "".join(rng.choice(list(self.digits), n))
        else:                                                          # alphanumeric mix
            n1, n2 = int(rng.integers(1, 6)), int(rng.integers(1, 5))
            if self.words and rng.random() < 0.6:
                w = self._case(self.words[rng.integers(len(self.words))], rng)[:n1 + 3]
            else:
                w = "".join(rng.choice(list(self.letters), n1))
            d = "".join(rng.choice(list(self.digits), n2))
            text = w + d if rng.random() < 0.7 else d + w
        return text[: self.max_len]


# --------------------------------------------------------------------------
# Word rendering
# --------------------------------------------------------------------------
def render_word(text: str, bank: CharBank, rng: np.random.Generator,
                band_h: int = C.SYNTH_BAND_H) -> tuple[np.ndarray, dict]:
    """Compose `text` into a writing band of height band_h (uint8, ink bright).

    Returns (band, meta) where meta["touching"] says whether at least one pair
    of neighbouring characters overlaps horizontally.
    """
    scale = rng.uniform(0.85, 1.12)                 # writer's size vs. guide lines
    dy = rng.uniform(-0.05, 0.05)                   # writer's vertical offset
    gap_mean = rng.uniform(-0.035, 0.10)            # spacing (fraction of band)
    # target pen width in band pixels: canvas brush scaled to this band
    pen = C.BRUSH_PX / C.CANVAS_H * band_h * rng.uniform(0.75, 1.35)

    pieces, xs = [], []
    x, touching = 0, False
    for k, ch in enumerate(text):
        top, bottom = C.char_extent(ch)
        top = C.BASE_LINE - (C.BASE_LINE - top) * scale + dy + rng.normal(0, 0.012)
        bottom = C.BASE_LINE + (bottom - C.BASE_LINE) * scale + dy + rng.normal(0, 0.012)
        th = max(4, int(round((bottom - top) * band_h * rng.uniform(0.93, 1.07))))

        g = bank.sample(ch, rng)
        s = th / g.shape[0]
        tw = max(2, int(round(g.shape[1] * s * rng.uniform(0.85, 1.15))))
        stroke_after = estimate_stroke_width(g) * s
        img = Image.fromarray(g).resize((tw, th), Image.LANCZOS)
        # equalize stroke width (one pen for the whole word)
        diff = pen - stroke_after
        if diff >= 3.0:
            img = img.filter(ImageFilter.MaxFilter(5))
        elif diff >= 1.0:
            img = img.filter(ImageFilter.MaxFilter(3))
        elif diff <= -2.0 and stroke_after > 4.0:
            img = img.filter(ImageFilter.MinFilter(3))
        glyph = np.asarray(img)

        if k > 0:
            gap = int(round((gap_mean + rng.normal(0, 0.025)) * band_h))
            gap = max(gap, -int(0.45 * min(glyph.shape[1], pieces[-1][0].shape[1])))
            if gap < 0:
                touching = True
            x += gap
        y = int(round(top * band_h))
        pieces.append((glyph, x, y))
        x += glyph.shape[1]

    # paste everything (np.maximum = overlapping ink merges like real ink)
    pad = band_h // 2                                      # room for slant
    width = x + 2 * pad
    band = np.zeros((band_h, width), np.uint8)
    for glyph, gx, gy in pieces:
        gy0, gy1 = max(gy, 0), min(gy + glyph.shape[0], band_h)
        if gy1 <= gy0:
            continue
        sub = glyph[gy0 - gy:gy1 - gy]
        region = band[gy0:gy1, pad + gx:pad + gx + sub.shape[1]]
        np.maximum(region, sub, out=region)

    # slant: x_in = x_out + k * (y - baseline)   (PIL affine maps output->input)
    k = rng.uniform(-0.25, 0.25)
    y_base = C.BASE_LINE * band_h
    img = Image.fromarray(band).transform(band.shape[::-1], Image.AFFINE,
                                          (1, k, -k * y_base, 0, 1, 0), resample=Image.BILINEAR)
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    return np.asarray(img), {"touching": touching}


def make_sample(text: str, bank: CharBank, rng: np.random.Generator,
                max_w: int = C.MAX_W_TRAIN) -> tuple[np.ndarray, np.ndarray, dict]:
    """(band uint8, model_input float32 [IMG_H, W], meta) for one word."""
    band, meta = render_word(text, bank, rng)
    x = band_to_model_input(band, max_w=max_w)
    assert x is not None, "rendered word has no ink"
    return band, x, meta

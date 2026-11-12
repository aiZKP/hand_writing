"""One-time data preparation.

    python -m hwr.prepare_data

1. download EMNIST (536 MB) and cache the ByClass split as .npz
2. build the dictionary word list (from /usr/share/dict)
3. render FIXED synthetic validation and test word sets.
   They use characters from the EMNIST *test* split only (the first half of
   it for validation, the second half for test), so no handwriting sample
   used in training ever appears in evaluation — no data leakage.
4. save picture grids to reports/ so you can look at the data.
"""
from __future__ import annotations

import argparse

import numpy as np
from PIL import Image

from . import config as C
from .datasets import save_fixed_set
from .emnist import build_cache, download_emnist, load_emnist
from .synth import CharBank, TextSampler, build_wordlist, load_wordlist, render_word


def build_word_set(path, bank: CharBank, sampler: TextSampler, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    bands, texts, touching = [], [], []
    for _ in range(n):
        text = sampler.sample(rng)
        band, meta = render_word(text, bank, rng)
        bands.append(band)
        texts.append(text)
        touching.append(meta["touching"])
    save_fixed_set(path, bands, texts, touching)
    print(f"  {path.name}: {n} words, {np.mean(touching):.0%} with touching letters")


def save_emnist_grid(x: np.ndarray, y: np.ndarray, out) -> None:
    rows = []
    for ch in "AaBbCcGgJjKkOo0Qq1lI2Ee":
        idx = np.where(y == C.CHAR_TO_IDX[ch])[0][:14]
        rows.append(np.concatenate([x[i] for i in idx], 1))
    g = 255 - np.concatenate(rows, 0)
    Image.fromarray(g).resize((g.shape[1] * 2, g.shape[0] * 2), Image.NEAREST).save(out)


def save_word_grid(path, out, n: int = 24) -> None:
    from .datasets import FixedWordSet

    ds = FixedWordSet(path)
    w = 400
    canvas = np.full((n * (C.IMG_H + 4), w), 255, np.uint8)
    for i in range(min(n, len(ds))):
        x, _ = ds[i]
        im = (255 - x.numpy() * 255).astype(np.uint8)[:, :w]
        canvas[i * (C.IMG_H + 4): i * (C.IMG_H + 4) + C.IMG_H, : im.shape[1]] = im
    Image.fromarray(canvas).resize((w * 2, canvas.shape[0] * 2), Image.NEAREST).save(out)
    print(f"  labels of {out.name}: {ds.texts[:n]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=3000)
    args = ap.parse_args()

    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/4] EMNIST")
    download_emnist()
    if not C.EMNIST_NPZ.exists():
        build_cache()
    x_te, y_te = load_emnist("test")
    x_tr, y_tr = load_emnist("train")
    f = x_tr.astype(np.float32) / 255.0
    print(f"  train {x_tr.shape}, test {x_te.shape}; pixel mean {f.mean():.4f} std {f.std():.4f} "
          f"(config: {C.EMNIST_MEAN} / {C.EMNIST_STD})")

    print("[2/4] word list")
    print(f"  {build_wordlist()} words -> {C.WORDLIST}")
    sampler = TextSampler(load_wordlist())

    print("[3/4] fixed synthetic word sets (EMNIST test characters only)")
    half = len(x_te) // 2
    build_word_set(C.SYNTH_VAL, CharBank(x_te[:half], y_te[:half]), sampler, args.n_val, seed=1)
    build_word_set(C.SYNTH_TEST, CharBank(x_te[half:], y_te[half:]), sampler, args.n_test, seed=2)

    print("[4/4] sample grids -> reports/")
    save_emnist_grid(x_tr, y_tr, C.REPORTS_DIR / "emnist_samples.png")
    save_word_grid(C.SYNTH_VAL, C.REPORTS_DIR / "synth_word_samples.png")


if __name__ == "__main__":
    main()

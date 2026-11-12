"""Evaluate both models and compare them on the SAME test words.

    python -m hwr.evaluate

Outputs (in reports/):
    evaluation.json            all numbers
    comparison.md              tables (also shown on the dashboard)
    char_cnn_confusion.png     62x62 confusion matrix of the CharCNN
    comparison_examples.png    words where the two approaches disagree
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from . import config as C
from .ctc import prefix_beam_search
from .datasets import FixedWordSet, ctc_collate
from .emnist import load_emnist
from .metrics import classification_metrics, confusion_matrix, text_metrics
from .models import CharCNN, CRNN
from .preprocess import normalize_char_emnist_style
from .segment import segment_characters
from .train_char_cnn import predict_logits
from .train_crnn import decode_dataset
from .utils import load_checkpoint, write_json

LOWER = np.array([C.CHAR_TO_IDX[c.lower()] for c in C.CHARSET])   # class -> lower-case class


def load_model(cls, path):
    m = cls()
    m.load_state_dict(load_checkpoint(path)["state_dict"])
    return m.eval()


# --------------------------------------------------------------------------
# Model A alone: single characters (EMNIST test set)
# --------------------------------------------------------------------------
def eval_char_cnn(cnn: CharCNN) -> dict:
    x, y = load_emnist("test")
    logits = predict_logits(cnn, torch.from_numpy(x))
    pred = logits.argmax(1)
    m = classification_metrics(logits, y, C.NUM_CLASSES)
    m["acc_nocase"] = float((LOWER[pred] == LOWER[y]).mean())
    cm = confusion_matrix(pred, y, C.NUM_CLASSES)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = [(C.CHARSET[i], C.CHARSET[j], int(off[i, j]), float(off[i, j] / max(cm[i].sum(), 1)))
             for i, j in zip(*np.unravel_index(np.argsort(-off, axis=None)[:15], off.shape))]
    m["top_confusions"] = pairs        # (true, predicted, count, fraction of true class)
    plot_confusion(cm)
    return m


def plot_confusion(cm: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(np.log10(norm + 1e-4), cmap="viridis")
    ax.set_xticks(range(C.NUM_CLASSES), list(C.CHARSET), fontsize=7)
    ax.set_yticks(range(C.NUM_CLASSES), list(C.CHARSET), fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("CharCNN confusion matrix (row-normalized, log10)\n"
                 "bright off-diagonal cells = confused pairs such as o/O/0, l/I/1, c/C")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "char_cnn_confusion.png", dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------
# Word level: both pipelines on the same synthetic test words
# --------------------------------------------------------------------------
@torch.inference_mode()
def seg_cnn_predict(cnn: CharCNN, band: np.ndarray) -> tuple[str, int]:
    segs = segment_characters(band)
    if not segs:
        return "", 0
    chars = np.stack([normalize_char_emnist_style(s.crop) for s in segs])
    pred = cnn(torch.from_numpy(chars)[:, None]).argmax(1).numpy()
    return "".join(C.CHARSET[k] for k in pred), len(segs)


@torch.inference_mode()
def crnn_beam_predict(crnn: CRNN, ds: FixedWordSet, beam: int = 10) -> list[str]:
    preds = []
    for i in range(len(ds)):
        x, _ = ds[i]
        lp = crnn(x[None, None])[:, 0].numpy()
        preds.append(prefix_beam_search(lp, beam=beam, top_k=1)[0][0])
    return preds


def split_metrics(preds, refs, touching) -> dict:
    t = np.asarray(touching)
    return {"all": text_metrics(preds, refs),
            "touching": text_metrics([p for p, m in zip(preds, t) if m], [r for r, m in zip(refs, t) if m]),
            "separated": text_metrics([p for p, m in zip(preds, t) if not m], [r for r, m in zip(refs, t) if not m])}


def save_examples(ds: FixedWordSet, a_preds, b_preds, out, n: int = 8) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = [i for i in range(len(ds)) if b_preds[i] == ds.texts[i] and a_preds[i] != ds.texts[i]][: n // 2]
    idx += [i for i in range(len(ds)) if b_preds[i] != ds.texts[i]][: n - len(idx)]
    if not idx:
        return
    fig, axes = plt.subplots(len(idx), 1, figsize=(7, 1.25 * len(idx)))
    for ax, i in zip(np.atleast_1d(axes), idx):
        ax.imshow(255 - ds.band(i), cmap="gray", vmin=0, vmax=255, aspect="equal")
        ax.set_title(f"true '{ds.texts[i]}'   A (seg+CNN) '{a_preds[i]}'   B (CRNN) '{b_preds[i]}'"
                     f"{'   [touching]' if ds.touching[i] else ''}", fontsize=9, loc="left")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def pct(v: float) -> str:
    return f"{100 * v:.1f} %"


def write_markdown(res: dict) -> str:
    lines = ["# Model comparison (synthetic test words, EMNIST test characters)", ""]
    w = res.get("words")
    if w:
        lines += [f"{w['n']} test words, {pct(w['touching_fraction'])} of them with touching letters.", "",
                  "| Model | Word accuracy | Word acc. (ignoring case) | CER | Word acc. touching | Word acc. separated |",
                  "|---|---|---|---|---|---|"]
        for name, key in [("A: segmentation + CharCNN", "seg_cnn"), ("B: CRNN + CTC (greedy)", "crnn_greedy"),
                          ("B: CRNN + CTC (beam search)", "crnn_beam")]:
            if key in w:
                m = w[key]
                lines.append(f"| {name} | {pct(m['all']['word_acc'])} | {pct(m['all']['word_acc_nocase'])} | "
                             f"{pct(m['all']['cer'])} | {pct(m['touching']['word_acc'])} | "
                             f"{pct(m['separated']['word_acc'])} |")
        if "seg_count_correct" in w:
            lines += ["", f"Model A found the right NUMBER of characters in {pct(w['seg_count_correct'])} of words "
                          f"(touching: {pct(w['seg_count_correct_touching'])}, "
                          f"separated: {pct(w['seg_count_correct_separated'])})."]
    c = res.get("char_cnn")
    if c:
        lines += ["", "# CharCNN on single EMNIST characters (116,323 test images)", "",
                  "| Top-1 | Top-3 | Balanced acc. | Ignoring case |", "|---|---|---|---|",
                  f"| {pct(c['acc'])} | {pct(c['top3'])} | {pct(c['balanced_acc'])} | {pct(c['acc_nocase'])} |",
                  "", "Most frequent confusions (true -> predicted):", "",
                  "| true | predicted | count | share of true class |", "|---|---|---|---|"]
        lines += [f"| {a} | {b} | {n} | {pct(f)} |" for a, b, n, f in c["top_confusions"][:10]]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-beam", action="store_true", help="skip the (slower) beam-search evaluation")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    res: dict = {}
    cnn = load_model(CharCNN, C.CHAR_CNN_CKPT) if C.CHAR_CNN_CKPT.exists() else None
    crnn = load_model(CRNN, C.CRNN_CKPT) if C.CRNN_CKPT.exists() else None

    if cnn is not None:
        t0 = time.time()
        res["char_cnn"] = eval_char_cnn(cnn)
        print(f"CharCNN EMNIST test: acc {res['char_cnn']['acc']:.4f}  ignoring case "
              f"{res['char_cnn']['acc_nocase']:.4f}  ({time.time() - t0:.0f}s)")

    ds = FixedWordSet(C.SYNTH_TEST)
    words = {"n": len(ds), "touching_fraction": float(np.mean(ds.touching))}
    a_preds = b_preds = None
    if cnn is not None:
        t0 = time.time()
        out = [seg_cnn_predict(cnn, ds.band(i)) for i in range(len(ds))]
        a_preds = [p for p, _ in out]
        ok = np.array([n == len(t) for (_, n), t in zip(out, ds.texts)])
        touch = np.array(ds.touching)
        words["seg_cnn"] = split_metrics(a_preds, ds.texts, ds.touching)
        words["seg_count_correct"] = float(ok.mean())
        words["seg_count_correct_touching"] = float(ok[touch].mean())
        words["seg_count_correct_separated"] = float(ok[~touch].mean())
        print(f"A seg+CNN: word acc {words['seg_cnn']['all']['word_acc']:.4f}  ({time.time() - t0:.0f}s)")
    if crnn is not None:
        t0 = time.time()
        b_preds = decode_dataset(crnn, ds)
        words["crnn_greedy"] = split_metrics(b_preds, ds.texts, ds.touching)
        print(f"B CRNN greedy: word acc {words['crnn_greedy']['all']['word_acc']:.4f}  ({time.time() - t0:.0f}s)")
        if not args.no_beam:
            t0 = time.time()
            beam_preds = crnn_beam_predict(crnn, ds)
            words["crnn_beam"] = split_metrics(beam_preds, ds.texts, ds.touching)
            print(f"B CRNN beam:   word acc {words['crnn_beam']['all']['word_acc']:.4f}  ({time.time() - t0:.0f}s)")
    if a_preds is not None or b_preds is not None:
        res["words"] = words
    if a_preds is not None and b_preds is not None:
        save_examples(ds, a_preds, b_preds, C.REPORTS_DIR / "comparison_examples.png")

    write_json(C.REPORTS_DIR / "evaluation.json", res)
    md = write_markdown(res)
    (C.REPORTS_DIR / "comparison.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()

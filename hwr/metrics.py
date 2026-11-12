"""Evaluation metrics.

Character Error Rate (CER) is the standard metric for text recognition:

    CER = (substitutions + deletions + insertions) / len(reference)
        = Levenshtein(prediction, reference) / len(reference)

The Levenshtein distance is computed with dynamic programming:
D[i][j] = edit distance between the first i chars of a and first j chars of b

    D[i][j] = min( D[i-1][j]   + 1,              # delete a[i]
                   D[i][j-1]   + 1,              # insert b[j]
                   D[i-1][j-1] + (a[i] != b[j])) # substitute (or match)
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def levenshtein(a: Sequence, b: Sequence) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))           # row D[0][*]
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def text_metrics(preds: Sequence[str], refs: Sequence[str]) -> dict[str, float]:
    """CER, word accuracy and case-insensitive word accuracy over a corpus."""
    assert len(preds) == len(refs)
    edits = sum(levenshtein(p, r) for p, r in zip(preds, refs))
    chars = sum(len(r) for r in refs)
    exact = sum(p == r for p, r in zip(preds, refs))
    exact_ci = sum(p.lower() == r.lower() for p, r in zip(preds, refs))
    n = max(len(refs), 1)
    return {
        "cer": edits / max(chars, 1),
        "word_acc": exact / n,
        "word_acc_nocase": exact_ci / n,
        "n": len(refs),
    }


def classification_metrics(logits: np.ndarray, labels: np.ndarray,
                           num_classes: int) -> dict[str, float]:
    """Top-1/top-3 accuracy and *balanced* accuracy (mean per-class recall).

    Balanced accuracy matters for EMNIST ByClass, where '1' has 20x more
    samples than 'j': plain accuracy is dominated by frequent classes.
    """
    pred = logits.argmax(1)
    top3 = np.argsort(-logits, axis=1)[:, :3]
    recalls = []
    for c in range(num_classes):
        m = labels == c
        if m.any():
            recalls.append((pred[m] == c).mean())
    return {
        "acc": float((pred == labels).mean()),
        "top3": float((top3 == labels[:, None]).any(1).mean()),
        "balanced_acc": float(np.mean(recalls)),
    }


def confusion_matrix(pred: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (labels, pred), 1)
    return cm

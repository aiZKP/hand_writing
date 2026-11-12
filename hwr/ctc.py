"""CTC decoding: turn the CRNN's per-frame distributions into text.

The CRNN outputs, for every time step t = 1..T, a distribution over
63 symbols (62 characters + blank '-'). A *path* picks one symbol per frame,
e.g. for T=8:   "-HH-e-ll-l-oo"  (not that length, but you get the idea)

The collapse function B maps a path to text:
    1. merge consecutive repeats     "-HH-e-ll-l-oo" -> "-H-e-l-l-o"
    2. remove blanks                                 -> "Hello"
The blank between the two l's is what allows double letters.

    P(text | image) = sum over all paths π with B(π) = text of  Π_t p_t(π_t)

Training (nn.CTCLoss) computes -log of that sum with dynamic programming.
Decoding wants argmax_text P(text | image), which is intractable exactly, so:
  * greedy / best-path decoding: take the argmax symbol per frame, collapse.
    Fast, usually good, but it finds the most likely PATH, not the most
    likely TEXT.
  * prefix beam search: keep the `beam` most probable text *prefixes*, and
    sum probabilities of all paths that lead to the same prefix.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import config as C

NEG_INF = -np.inf


def greedy_decode(log_probs: np.ndarray) -> tuple[str, list[float], list[int]]:
    """log_probs: [T, C].  Returns (text, per-char confidence, best path)."""
    best = log_probs.argmax(1)
    probs = np.exp(log_probs.max(1))
    text, confs = [], []
    prev = C.CTC_BLANK
    for t, k in enumerate(best):
        if k != C.CTC_BLANK and k != prev:
            text.append(C.CHARSET[k - 1])
            confs.append(float(probs[t]))
        elif k != C.CTC_BLANK and k == prev:           # same char continues
            confs[-1] = max(confs[-1], float(probs[t]))
        prev = k
    return "".join(text), confs, best.tolist()


def path_to_string(path: list[int]) -> str:
    """Best path as a readable string, blank shown as '·'  (for the dashboard)."""
    return "".join("·" if k == C.CTC_BLANK else C.CHARSET[k - 1] for k in path)


def _lse(*xs: float) -> float:
    """log(sum(exp(x))) computed stably."""
    m = max(xs)
    if m == NEG_INF:
        return NEG_INF
    return m + np.log(sum(np.exp(x - m) for x in xs))


def prefix_beam_search(log_probs: np.ndarray, beam: int = 10, top_k: int = 5,
                       prune: float = np.log(1e-3)) -> list[tuple[str, float]]:
    """CTC prefix beam search (Hannun et al., 2014), no language model.

    For every prefix we track two log-probabilities:
        pb  = P(prefix, path so far ends in BLANK)
        pnb = P(prefix, path so far ends in the prefix's LAST CHARACTER)
    We need both because a repeated character only creates a new letter if a
    blank separated it from the previous one.
    Returns up to top_k (text, log_prob) pairs, best first.
    """
    T, _ = log_probs.shape
    beams: dict[tuple, tuple[float, float]] = {(): (0.0, NEG_INF)}
    for t in range(T):
        lp = log_probs[t]
        cands = [int(c) for c in np.argsort(-lp)[:8] if lp[c] > prune or c == C.CTC_BLANK]
        nxt: dict[tuple, list[float]] = defaultdict(lambda: [NEG_INF, NEG_INF])
        for prefix, (pb, pnb) in beams.items():
            for c in cands:
                p = float(lp[c])
                if c == C.CTC_BLANK:                     # prefix unchanged, now ends in blank
                    e = nxt[prefix]
                    e[0] = _lse(e[0], pb + p, pnb + p)
                    continue
                last = prefix[-1] if prefix else None
                new = prefix + (c,)
                e_new = nxt[new]
                if c == last:
                    # "a" + "a" without blank in between -> still "a"
                    e_same = nxt[prefix]
                    e_same[1] = _lse(e_same[1], pnb + p)
                    # "a" + "-" + "a" -> "aa"
                    e_new[1] = _lse(e_new[1], pb + p)
                else:
                    e_new[1] = _lse(e_new[1], pb + p, pnb + p)
        scored = sorted(nxt.items(), key=lambda kv: _lse(*kv[1]), reverse=True)
        beams = {k: (v[0], v[1]) for k, v in scored[:beam]}
    results = [("".join(C.CHARSET[k - 1] for k in prefix), _lse(pb, pnb))
               for prefix, (pb, pnb) in beams.items()]
    results.sort(key=lambda r: r[1], reverse=True)
    return results[:top_k]

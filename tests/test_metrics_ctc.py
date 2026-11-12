"""Metrics and CTC decoding — small cases you can check by hand."""
import numpy as np

from hwr import config as C
from hwr.ctc import greedy_decode, path_to_string, prefix_beam_search
from hwr.metrics import classification_metrics, confusion_matrix, levenshtein, text_metrics


def test_levenshtein_classic():
    assert levenshtein("kitten", "sitting") == 3      # k->s, e->i, +g
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "abc") == 0


def test_text_metrics():
    m = text_metrics(["Hello", "w0rld", "TEST"], ["Hello", "world", "test"])
    assert m["word_acc"] == 1 / 3
    assert m["word_acc_nocase"] == 2 / 3                 # TEST == test ignoring case
    assert abs(m["cer"] - (0 + 1 + 4) / 14) < 1e-9


def test_classification_metrics_and_confusion():
    logits = np.array([[5, 1, 0], [0, 5, 1], [0, 4, 5], [1, 0, 0]], float)
    labels = np.array([0, 1, 1, 0])
    m = classification_metrics(logits, labels, 3)
    assert m["acc"] == 0.75 and m["top3"] == 1.0
    assert abs(m["balanced_acc"] - (1.0 + 0.5) / 2) < 1e-9
    cm = confusion_matrix(logits.argmax(1), labels, 3)
    assert cm[1, 2] == 1 and cm.sum() == 4


def _logp(rows):
    """rows: list of {class_index: prob}; the rest share the leftover mass."""
    out = np.full((len(rows), C.NUM_CTC_CLASSES), 1e-6)
    for t, r in enumerate(rows):
        for k, p in r.items():
            out[t, k] = p
    out /= out.sum(1, keepdims=True)
    return np.log(out)


def test_encode_decode_roundtrip():
    s = "Ab9zZ0"
    assert C.decode_ctc_indices(C.encode_ctc(s)) == s
    assert min(C.encode_ctc(s)) >= 1                      # 0 is reserved for blank


def test_greedy_collapses_repeats_and_blanks():
    h, e, l, o = (C.CHAR_TO_IDX[c] + 1 for c in "helo")
    blank = C.CTC_BLANK
    path = [blank, h, h, blank, e, l, l, blank, l, o, o, blank]   # "-hh-ell-loo-"
    text, conf, best = greedy_decode(_logp([{k: 0.99} for k in path]))
    assert text == "hello"                                # blank between the l's keeps both
    assert len(conf) == 5 and best == path
    assert path_to_string(best) == "·hh·ell·loo·"


def test_beam_search_beats_greedy_on_classic_example():
    # Two frames, P(blank)=0.6, P(a)=0.4 each.
    # Greedy picks "--" -> ""  with prob 0.36.
    # But P("a") = P(aa)+P(a-)+P(-a) = 0.16+0.24+0.24 = 0.64 > 0.36.
    a = C.CHAR_TO_IDX["a"] + 1
    lp = _logp([{C.CTC_BLANK: 0.6, a: 0.4}] * 2)
    assert greedy_decode(lp)[0] == ""
    beams = prefix_beam_search(lp, beam=5)
    assert beams[0][0] == "a"
    assert abs(np.exp(beams[0][1]) - 0.64) < 1e-3
    assert abs(np.exp(beams[1][1]) - 0.36) < 1e-3

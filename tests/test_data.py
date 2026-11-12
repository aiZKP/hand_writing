"""EMNIST parsing, preprocessing, synthetic words, segmentation."""
import gzip
import struct

import numpy as np
import pytest
from PIL import Image, ImageDraw

from hwr import config as C
from hwr.emnist import check_mapping, parse_idx
from hwr.preprocess import band_to_model_input, image_to_ink, normalize_char_emnist_style
from hwr.segment import segment_characters
from hwr.synth import CharBank, TextSampler, estimate_stroke_width, render_word


def test_parse_idx_roundtrip():
    arr = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    raw = bytes([0, 0, 0x08, 3]) + struct.pack(">3I", 2, 3, 4) + arr.tobytes()
    assert np.array_equal(parse_idx(gzip.decompress(gzip.compress(raw))), arr)
    with pytest.raises(ValueError):
        parse_idx(b"\x01\x02\x03\x04")


def test_mapping_matches_charset():
    check_mapping("\n".join(f"{i} {ord(c)}" for i, c in enumerate(C.CHARSET)))
    with pytest.raises(ValueError):
        check_mapping("0 65")                               # label 0 is '0', not 'A'


def test_band_to_model_input_shapes():
    assert band_to_model_input(np.zeros((300, 1400))) is None   # empty canvas
    band = np.zeros((300, 1400), np.float32)
    band[100:200, 500:700] = 1.0
    x = band_to_model_input(band)
    assert x.shape[0] == C.IMG_H and x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0
    # ink stays at the same RELATIVE height (case information preserved)
    rows = np.where(x.max(1) > 0.5)[0]
    assert abs(rows.mean() / C.IMG_H - 0.5) < 0.05


def test_image_to_ink_transparent_canvas():
    img = Image.new("RGBA", (200, 60), (0, 0, 0, 0))
    ImageDraw.Draw(img).line([(20, 30), (180, 30)], fill=(17, 17, 17, 255), width=8)
    ink = image_to_ink(img)
    assert ink[30, 100] > 0.9 and ink[5, 5] == 0.0


def test_char_normalization_is_centered_28x28():
    crop = np.zeros((90, 30), np.float32)
    crop[:, 10:20] = 1.0                                     # tall thin "l"
    out = normalize_char_emnist_style(crop)
    assert out.shape == (28, 28)
    rows, cols = np.where(out > 0.5)
    assert rows.min() >= 2 and rows.max() <= 25              # 2-px border like EMNIST
    assert abs(cols.mean() - 13.5) < 1.5                     # horizontally centered


def _fake_bank():
    rng = np.random.default_rng(0)
    imgs = np.zeros((C.NUM_CLASSES * 2, 28, 28), np.uint8)
    for i in range(len(imgs)):
        h, w = rng.integers(14, 24), rng.integers(6, 20)
        imgs[i, 2:2 + h, 4:4 + w] = 255
    return CharBank(imgs, np.repeat(np.arange(C.NUM_CLASSES), 2))


def test_render_word_deterministic_and_case_sizes():
    bank = _fake_bank()
    b1, _ = render_word("aA", bank, np.random.default_rng(7))
    b2, _ = render_word("aA", bank, np.random.default_rng(7))
    assert np.array_equal(b1, b2) and b1.shape[0] == C.SYNTH_BAND_H
    rows = lambda ch: np.where(render_word(ch, bank, np.random.default_rng(1))[0].max(1) > 100)[0]
    # lowercase 'o' starts lower (smaller) than uppercase 'O'
    assert rows("o").min() > rows("O").min() + 0.15 * C.SYNTH_BAND_H


def test_text_sampler_uses_only_charset():
    s = TextSampler(["hello", "world"])
    rng = np.random.default_rng(0)
    for _ in range(300):
        t = s.sample(rng)
        assert 1 <= len(t) <= 12 and set(t) <= set(C.CHARSET)


def test_stroke_width_estimate():
    g = np.zeros((40, 40), np.uint8)
    g[5:35, 18:22] = 255                                     # 4-px wide stroke
    assert 3.0 <= estimate_stroke_width(g) <= 5.0


def test_segmentation_separate_dot_and_touching():
    band = np.zeros((300, 900), np.float32)
    band[110:210, 100:120] = 1                                # 'l'
    band[150:210, 200:215] = 1; band[120:132, 200:215] = 1   # 'i' = stem + dot
    band[150:210, 300:360] = 1                                # blob 3
    band[150:210, 600:660] = 1; band[150:210, 655:720] = 1   # two touching letters
    segs = segment_characters(band)
    assert [s.x0 for s in segs] == [100, 200, 300, 600]
    assert segs[1].y0 == 120                                  # dot merged into the 'i'
    assert segs[3].x1 == 720                                  # touching pair = ONE segment
    assert segment_characters(np.zeros((300, 900))) == []

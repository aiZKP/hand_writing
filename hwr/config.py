"""Central configuration — the *single source of truth* shared by training,
evaluation, the inference server and (via /api/info) the browser dashboard.

Keeping these numbers in one place is what prevents "train/serve skew":
the dashboard draws its guide lines at exactly the same relative heights that
the synthetic training words were composed with.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EMNIST_ZIP = RAW_DIR / "emnist_gzip.zip"
EMNIST_NPZ = DATA_DIR / "emnist_byclass.npz"      # parsed + transposed cache
WORDLIST = DATA_DIR / "wordlist.txt"              # filtered dictionary words
SYNTH_VAL = DATA_DIR / "synth_val.npz"            # fixed validation words
SYNTH_TEST = DATA_DIR / "synth_test.npz"          # fixed test words
USER_SAMPLES_DIR = DATA_DIR / "user_samples"      # drawings saved from dashboard
CKPT_DIR = ROOT / "checkpoints"
CHAR_CNN_CKPT = CKPT_DIR / "char_cnn_best.pt"
CRNN_CKPT = CKPT_DIR / "crnn_best.pt"
REPORTS_DIR = ROOT / "reports"

# --------------------------------------------------------------------------
# Alphabet (62 symbols). Order == EMNIST ByClass label order:
#   labels 0-9 -> '0'-'9', 10-35 -> 'A'-'Z', 36-61 -> 'a'-'z'
# --------------------------------------------------------------------------
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
NUM_CLASSES = len(CHARSET)                        # 62  (CharCNN outputs)
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARSET)}

# CTC uses one extra class, the "blank", at index 0.
#   ctc index = CHARSET index + 1
CTC_BLANK = 0
NUM_CTC_CLASSES = NUM_CLASSES + 1                 # 63  (CRNN outputs)


def encode_ctc(text: str) -> list[int]:
    """'Ab1' -> [11, 38, 2]  (CHARSET index + 1, because 0 is the blank)."""
    return [CHAR_TO_IDX[c] + 1 for c in text]


def decode_ctc_indices(indices) -> str:
    """Inverse of encode_ctc. Blanks (0) are skipped."""
    return "".join(CHARSET[i - 1] for i in indices if i != CTC_BLANK)


# --------------------------------------------------------------------------
# Writing-band geometry (fractions of the band height, measured from the TOP)
# Like ruled school paper:
#
#   0.12 ─────────────  ascender / cap line   (tops of A-Z, 0-9, b d h k l f)
#   0.40 - - - - - - -  x-height line         (tops of a c e m n o r s u ...)
#   0.70 ─────────────  baseline              (bottoms of almost everything)
#   0.92 ─────────────  descender line        (bottoms of g j p q y)
# --------------------------------------------------------------------------
ASC_LINE = 0.12
XH_LINE = 0.40
BASE_LINE = 0.70
DESC_LINE = 0.92
MID_ASC = 0.24      # top of 'i' (with dot), 'j' and 't' — between ASC and XH

_X_HEIGHT = set("acemnorsuvwxz")
_ASCENDERS = set("bdfhkl")
_DESCENDERS = set("gpqy")


def char_extent(ch: str) -> tuple[float, float]:
    """(top, bottom) of a character inside the writing band, as fractions.

    This is what lets the CRNN tell 'c' from 'C' or 'o' from 'O': the *size
    relative to the guide lines* differs, even though the shapes are the same.
    """
    if ch in _X_HEIGHT:
        return XH_LINE, BASE_LINE
    if ch in _ASCENDERS:
        return ASC_LINE, BASE_LINE
    if ch in _DESCENDERS:
        return XH_LINE, DESC_LINE
    if ch in "it":
        return MID_ASC, BASE_LINE
    if ch == "j":
        return MID_ASC, DESC_LINE
    return ASC_LINE, BASE_LINE          # A-Z and 0-9


# --------------------------------------------------------------------------
# Image sizes
# --------------------------------------------------------------------------
CHAR_SIZE = 28            # CharCNN input (EMNIST native)
IMG_H = 48                # CRNN input height (band is resized to this)
CRNN_STRIDE = 4           # CRNN downsamples width by 4  ->  T = W // 4
SYNTH_BAND_H = 96         # synthetic words are composed at 2x model height
MIN_W = 32                # minimum model-input width (pad right if smaller)
MAX_W_TRAIN = 400         # cap on training width (squash if wider)
MAX_W_INFER = 1200        # cap at inference (only for absurd inputs)
CROP_MARGIN = 0.06        # horizontal margin around ink, fraction of band height

# Dashboard canvas (internal pixel size). Band height == canvas height.
CANVAS_W = 1400
CANVAS_H = 300
# Brush width on the canvas. EMNIST strokes are ~10 % of the character
# height; cap height on the canvas is (BASE-ASC)*300 = 174 px -> ~16-17 px.
BRUSH_PX = 16

# Normalization of [0,1] pixel values for the CharCNN (EMNIST statistics,
# computed by `python -m hwr.prepare_data`; values are hard-coded here so
# inference never needs the dataset).
EMNIST_MEAN = 0.1736
EMNIST_STD = 0.3317


def geometry_dict() -> dict:
    """Everything the browser needs to draw the canvas exactly like training."""
    return {
        "canvas_w": CANVAS_W,
        "canvas_h": CANVAS_H,
        "brush_px": BRUSH_PX,
        "asc_line": ASC_LINE,
        "xh_line": XH_LINE,
        "base_line": BASE_LINE,
        "desc_line": DESC_LINE,
        "charset": CHARSET,
    }

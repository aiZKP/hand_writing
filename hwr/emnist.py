"""EMNIST ByClass: download, parse the IDX binary format, fix orientation.

EMNIST (Cohen et al., 2017) converts the NIST Special Database 19 handwritten
characters into an MNIST-like format: 28x28 grayscale, white ink on black.
ByClass = 62 classes (digits + upper + lower case), 814,255 images, highly
unbalanced.

Two quirks you must know about:
1. The raw images are stored TRANSPOSED (column-major) compared to MNIST.
   We fix this with `images.transpose(0, 2, 1)`.
2. Many upper/lower pairs (c/C, o/O, s/S, ...) are visually identical once
   every character is scaled to fill a 28x28 box, which caps single-character
   accuracy at roughly 88 %.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import struct
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from . import config as C

EMNIST_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
EMNIST_MD5 = "58c8d27c78d21e728a6bc7b3cc06412e"  # same value torchvision uses

_MEMBERS = {
    "train_images": "gzip/emnist-byclass-train-images-idx3-ubyte.gz",
    "train_labels": "gzip/emnist-byclass-train-labels-idx1-ubyte.gz",
    "test_images": "gzip/emnist-byclass-test-images-idx3-ubyte.gz",
    "test_labels": "gzip/emnist-byclass-test-labels-idx1-ubyte.gz",
    "mapping": "gzip/emnist-byclass-mapping.txt",
}

# IDX type codes -> numpy dtypes (big-endian where it matters)
_IDX_DTYPES = {0x08: np.uint8, 0x09: np.int8, 0x0B: ">i2", 0x0C: ">i4",
               0x0D: ">f4", 0x0E: ">f8"}


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download_emnist(dest: Path = C.EMNIST_ZIP) -> Path:
    """Download the official EMNIST zip (536 MB) unless it already exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        from tqdm import tqdm

        tmp = dest.with_suffix(".part")
        print(f"Downloading EMNIST from {EMNIST_URL} ...")
        with urllib.request.urlopen(EMNIST_URL) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
            with tqdm(total=total, unit="B", unit_scale=True) as bar:
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
                    bar.update(len(chunk))
        tmp.rename(dest)
    digest = md5sum(dest)
    status = "OK" if digest == EMNIST_MD5 else "MISMATCH (file may be corrupt!)"
    print(f"EMNIST zip md5 = {digest}  [{status}]")
    return dest


# --------------------------------------------------------------------------
# IDX parsing
# --------------------------------------------------------------------------
def parse_idx(data: bytes) -> np.ndarray:
    """Parse the IDX format used by (E)MNIST.

    Layout:  [0x00 0x00 <type> <ndim>] [ndim x uint32 big-endian sizes] [data]
    """
    if len(data) < 4 or data[0] != 0 or data[1] != 0:
        raise ValueError("Not an IDX file (bad magic number)")
    type_code, ndim = data[2], data[3]
    if type_code not in _IDX_DTYPES:
        raise ValueError(f"Unknown IDX type code 0x{type_code:02x}")
    shape = struct.unpack(f">{ndim}I", data[4:4 + 4 * ndim])
    arr = np.frombuffer(data, dtype=_IDX_DTYPES[type_code], offset=4 + 4 * ndim)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(f"IDX size mismatch: header says {expected}, got {arr.size}")
    return arr.reshape(shape)


def _read_member(zf: zipfile.ZipFile, name: str) -> bytes:
    # We read named members into memory instead of zf.extractall(): a
    # downloaded archive is untrusted, and extractall can be abused with
    # "../" paths (zip-slip).
    raw = zf.read(name)
    return gzip.decompress(raw) if name.endswith(".gz") else raw


def check_mapping(mapping_txt: str) -> None:
    """Verify the official label->ASCII mapping equals our CHARSET order."""
    for line in mapping_txt.strip().splitlines():
        label, ascii_code = map(int, line.split())
        if C.CHARSET[label] != chr(ascii_code):
            raise ValueError(f"Label {label}: EMNIST says {chr(ascii_code)!r}, "
                             f"CHARSET says {C.CHARSET[label]!r}")


def build_cache(zip_path: Path = C.EMNIST_ZIP, out: Path = C.EMNIST_NPZ) -> Path:
    """Parse the ByClass split from the zip and cache it as a compressed .npz."""
    with zipfile.ZipFile(zip_path) as zf:
        check_mapping(_read_member(zf, _MEMBERS["mapping"]).decode())
        arrays = {}
        for split in ("train", "test"):
            imgs = parse_idx(_read_member(zf, _MEMBERS[f"{split}_images"]))
            labels = parse_idx(_read_member(zf, _MEMBERS[f"{split}_labels"]))
            # Quirk #1: stored transposed -> make characters upright.
            imgs = np.ascontiguousarray(imgs.transpose(0, 2, 1))
            arrays[f"{split}_x"] = imgs.astype(np.uint8)
            arrays[f"{split}_y"] = labels.astype(np.int64)
            print(f"  {split}: images {imgs.shape}, labels {labels.shape}")
    np.savez_compressed(out, **arrays)
    return out


def load_emnist(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (images uint8 [N,28,28] upright white-on-black, labels int64 [N])."""
    if not C.EMNIST_NPZ.exists():
        raise FileNotFoundError(
            f"{C.EMNIST_NPZ} not found — run `python -m hwr.prepare_data` first.")
    with np.load(C.EMNIST_NPZ) as d:
        return d[f"{split}_x"], d[f"{split}_y"]


def load_mapping_from_bytes(data: bytes) -> dict[int, str]:
    """Helper used by tests."""
    out = {}
    for line in io.StringIO(data.decode()):
        if line.strip():
            label, code = map(int, line.split())
            out[label] = chr(code)
    return out

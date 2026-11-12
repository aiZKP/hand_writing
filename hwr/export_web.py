"""Build the static website (GitHub Pages) with in-browser inference.

    python -m hwr.export_web                 # -> build/site/

GitHub Pages can only serve static files, so the Python server cannot run
there. Instead we
  1. export both trained networks to ONNX (an open, framework-independent
     model format) and check that ONNX Runtime gives the same numbers as
     PyTorch,
  2. copy the dashboard (app/static) plus `inference.js`, a JavaScript port of
     the preprocessing, segmentation and CTC decoding,
  3. write config.json (canvas geometry + model metrics) and the report images.
The browser then runs the models itself with ONNX Runtime Web (WebAssembly).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from . import config as C
from .models import CharCNN, CRNN
from .utils import count_params, load_checkpoint

ORT_WEB = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js"
REPO_URL = "https://github.com/aiZKP/hand_writing"


def _load(cls, path: Path):
    ckpt = load_checkpoint(path)
    model = cls()
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt


def export_onnx(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {}
    cnn, ck_cnn = _load(CharCNN, C.CHAR_CNN_CKPT)
    torch.onnx.export(cnn, (torch.rand(2, 1, 28, 28),), out_dir / "char_cnn.onnx",
                      input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    crnn, ck_crnn = _load(CRNN, C.CRNN_CKPT)
    torch.onnx.export(crnn, (torch.rand(1, 1, C.IMG_H, 160),), out_dir / "crnn.onnx",
                      input_names=["image"], output_names=["log_probs"],
                      dynamic_axes={"image": {3: "width"}, "log_probs": {0: "time"}},
                      opset_version=17, dynamo=False)
    verify_onnx(out_dir, cnn, crnn)
    for name, ck, model in (("crnn", ck_crnn, crnn), ("char_cnn", ck_cnn, cnn)):
        meta[name] = {"loaded": True, "params": count_params(model), "epoch": ck.get("epoch"),
                      "val_metrics": ck.get("val_metrics"), "test_metrics": ck.get("test_metrics"),
                      "finetuned": bool(ck.get("finetuned", False)),
                      "path": f"models/{name}.onnx", "bytes": (out_dir / f"{name}.onnx").stat().st_size}
    return meta


def verify_onnx(out_dir: Path, cnn: CharCNN, crnn: CRNN) -> None:
    """ONNX Runtime must reproduce PyTorch's outputs (several input widths)."""
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    s = ort.InferenceSession(str(out_dir / "char_cnn.onnx"))
    x = rng.random((5, 1, 28, 28), dtype=np.float32)
    with torch.inference_mode():
        ref = cnn(torch.from_numpy(x)).numpy()
    d1 = np.abs(s.run(None, {"image": x})[0] - ref).max()
    s = ort.InferenceSession(str(out_dir / "crnn.onnx"))
    d2 = 0.0
    for w in (32, 77, 160, 413):
        x = rng.random((1, 1, C.IMG_H, w), dtype=np.float32)
        with torch.inference_mode():
            ref = crnn(torch.from_numpy(x)).numpy()
        got = s.run(None, {"image": x})[0]
        assert got.shape == ref.shape, (got.shape, ref.shape)
        d2 = max(d2, float(np.abs(got - ref).max()))
    print(f"ONNX check: max |onnx - torch| CharCNN {d1:.2e}, CRNN {d2:.2e}")
    assert d1 < 1e-3 and d2 < 1e-3, "ONNX export does not match PyTorch"


def build_site(site: Path) -> None:
    if site.exists():
        shutil.rmtree(site)
    (site / "reports").mkdir(parents=True)
    models = export_onnx(site / "models")

    static = Path(__file__).resolve().parent.parent / "app" / "static"
    for f in ("app.js", "style.css", "inference.js"):
        shutil.copy(static / f, site / f)
    html = (static / "index.html").read_text()
    html = html.replace('href="/static/style.css"', 'href="style.css"')
    html = html.replace('<script src="/static/app.js"></script>',
                        '<script>window.HWR_STATIC = true;</script>\n'
                        f'  <script src="{ORT_WEB}"></script>\n'
                        '  <script src="inference.js"></script>\n'
                        '  <script src="app.js"></script>')
    assert 'window.HWR_STATIC' in html and "/static/" not in html
    (site / "index.html").write_text(html)

    reports = sorted(p.name for p in C.REPORTS_DIR.glob("*.png") if not p.name.startswith("_"))
    for name in reports:
        shutil.copy(C.REPORTS_DIR / name, site / "reports" / name)
    comparison = C.REPORTS_DIR / "comparison.md"
    cfg = {
        "geometry": C.geometry_dict(),
        "constants": {"img_h": C.IMG_H, "min_w": C.MIN_W, "max_w": C.MAX_W_INFER,
                      "crop_margin": C.CROP_MARGIN, "ink_threshold": 0.15, "char_size": C.CHAR_SIZE,
                      "crnn_stride": C.CRNN_STRIDE, "seg_min_area_frac": 0.0004, "seg_merge_overlap": 0.5},
        "models": models,
        "reports": reports,
        "comparison_md": comparison.read_text() if comparison.exists() else None,
        "repo": REPO_URL,
    }
    (site / "config.json").write_text(json.dumps(cfg, indent=1))
    (site / ".nojekyll").write_text("")     # serve files as-is (no Jekyll processing)
    total = sum(p.stat().st_size for p in site.rglob("*") if p.is_file())
    print(f"site built in {site}  ({total / 1e6:.1f} MB, {len(reports)} report images)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=C.ROOT / "build" / "site")
    build_site(ap.parse_args().out)


if __name__ == "__main__":
    main()

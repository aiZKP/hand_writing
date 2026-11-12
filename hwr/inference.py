"""Inference: run BOTH recognizers on one drawing and explain the result.

    rec = Recognizer()                  # loads checkpoints/*.pt (if present)
    out = rec.recognize_image(pil_img)  # dict, JSON-serializable

The result contains not only the text but also the intermediate data you
want to SEE when studying: the exact model input, the CTC best path per
frame, beam-search alternatives, the character boxes found by segmentation,
the 28x28 images the CharCNN received and its top-3 probabilities.
"""
from __future__ import annotations

import base64
import io
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import config as C
from .ctc import greedy_decode, path_to_string, prefix_beam_search
from .models import CharCNN, CRNN
from .preprocess import band_to_model_input, image_to_ink, normalize_char_emnist_style
from .segment import segment_characters
from .utils import count_params, load_checkpoint


def to_png_data_url(img01: np.ndarray, invert: bool = True) -> str:
    """[0,1] array -> data URL. invert=True shows dark ink on white."""
    arr = (1.0 - img01) if invert else img01
    buf = io.BytesIO()
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class Recognizer:
    def __init__(self, crnn_path: Path = C.CRNN_CKPT, cnn_path: Path = C.CHAR_CNN_CKPT,
                 threads: int = 2):
        torch.set_num_threads(threads)
        self.crnn_path, self.cnn_path = crnn_path, cnn_path
        self._lock = threading.Lock()          # one inference at a time
        self.reload()

    # ------------------------------------------------------------------
    def reload(self) -> None:
        """(Re)load checkpoints — e.g. after training or fine-tuning."""
        self.crnn, self.crnn_meta = self._load(CRNN, self.crnn_path)
        self.cnn, self.cnn_meta = self._load(CharCNN, self.cnn_path)

    @staticmethod
    def _load(cls, path: Path):
        if not path.exists():
            return None, {"loaded": False, "path": str(path)}
        ckpt = load_checkpoint(path)
        model = cls()
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        meta = {"loaded": True, "path": str(path), "params": count_params(model),
                "epoch": ckpt.get("epoch"), "val_metrics": ckpt.get("val_metrics"),
                "test_metrics": ckpt.get("test_metrics"), "finetuned": ckpt.get("finetuned", False)}
        return model, meta

    def info(self) -> dict:
        return {"crnn": self.crnn_meta, "char_cnn": self.cnn_meta}

    # ------------------------------------------------------------------
    def recognize_image(self, img: Image.Image, beam_width: int = 10) -> dict:
        return self.recognize_ink(image_to_ink(img), beam_width)

    @torch.inference_mode()
    def recognize_ink(self, ink: np.ndarray, beam_width: int = 10) -> dict:
        with self._lock:
            result: dict = {"empty": False, "crnn": None, "cnn": None, "timing_ms": {}}
            x = band_to_model_input(ink)
            if x is None:
                result["empty"] = True
                return result
            result["model_input"] = to_png_data_url(x)
            result["model_input_size"] = [int(x.shape[1]), int(x.shape[0])]
            if self.crnn is not None:
                t0 = time.perf_counter()
                result["crnn"] = self._run_crnn(x, beam_width)
                result["timing_ms"]["crnn"] = round(1000 * (time.perf_counter() - t0), 1)
            if self.cnn is not None:
                t0 = time.perf_counter()
                result["cnn"] = self._run_cnn(ink)
                result["timing_ms"]["cnn"] = round(1000 * (time.perf_counter() - t0), 1)
            return result

    # ------------------------------------------------------------------
    def _run_crnn(self, x: np.ndarray, beam_width: int) -> dict:
        log_probs = self.crnn(torch.from_numpy(x)[None, None])[:, 0].numpy()   # T x 63
        greedy_text, char_conf, path = greedy_decode(log_probs)
        beams = prefix_beam_search(log_probs, beam=beam_width, top_k=5)
        best_text, best_logp = beams[0] if beams else (greedy_text, 0.0)
        frame_conf = np.exp(log_probs.max(1))
        return {
            "text": best_text,                          # beam search result
            "confidence": float(np.exp(best_logp)),     # P(text | image) under CTC
            "greedy_text": greedy_text,
            "char_conf": [round(c, 4) for c in char_conf] if best_text == greedy_text else None,
            "beams": [{"text": t, "prob": float(np.exp(lp))} for t, lp in beams],
            "frames": path_to_string(path),             # best path, '·' = blank
            "frame_conf": [round(float(c), 3) for c in frame_conf],
            "num_frames": int(log_probs.shape[0]),
        }

    def _run_cnn(self, ink: np.ndarray) -> dict:
        segs = segment_characters(ink)
        if not segs:
            return {"text": "", "chars": [], "confidence": 0.0}
        chars28 = np.stack([normalize_char_emnist_style(s.crop) for s in segs])
        probs = self.cnn(torch.from_numpy(chars28)[:, None]).softmax(1).numpy()
        chars = []
        for seg, img28, p in zip(segs, chars28, probs):
            top = np.argsort(-p)[:3]
            chars.append({"box": [int(v) for v in seg.box],
                          "top3": [[C.CHARSET[k], float(p[k])] for k in top],
                          "image": to_png_data_url(img28)})
        text = "".join(c["top3"][0][0] for c in chars)
        conf = float(np.prod([c["top3"][0][1] for c in chars]))
        return {"text": text, "chars": chars, "confidence": conf}

"""Dashboard server (FastAPI).

    python -m app.server               # http://127.0.0.1:8000
    python -m app.server --port 8080

Endpoints
    GET  /              the dashboard (static/index.html)
    GET  /api/info      canvas geometry (shared with training!) + model info
    POST /api/predict   {"image": "<PNG data URL>"} -> recognition result
    POST /api/samples   {"image": ..., "text": "Hello"} -> save for fine-tuning
    POST /api/reload    reload checkpoints after (re)training
    GET  /reports/...   training curves and evaluation plots
    GET  /docs          automatic API documentation (Swagger UI)

The server binds to 127.0.0.1 by default, so it is NOT reachable from the
internet — only from a browser on this machine.
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from hwr import config as C
from hwr.inference import Recognizer
from hwr.preprocess import decode_data_url, image_to_ink, ink_columns

STATIC = Path(__file__).parent / "static"
state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state["rec"] = Recognizer()           # load models once at startup
    yield


app = FastAPI(title="Handwritten Word Recognition", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/reports", StaticFiles(directory=C.REPORTS_DIR), name="reports")


class PredictRequest(BaseModel):
    image: str = Field(..., description="PNG data URL exported from the canvas")


class SampleRequest(BaseModel):
    image: str
    text: str = Field(..., min_length=1, max_length=32)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    reports = sorted(p.name for p in C.REPORTS_DIR.glob("*.png") if not p.name.startswith("_"))
    comparison = C.REPORTS_DIR / "comparison.md"
    return {"geometry": C.geometry_dict(), "models": state["rec"].info(), "reports": reports,
            "comparison_md": comparison.read_text() if comparison.exists() else None,
            "user_samples": _count_samples()}


@app.post("/api/predict")
def predict(req: PredictRequest):          # sync def -> runs in a worker thread
    try:
        img = decode_data_url(req.image)
    except Exception as e:                 # malformed / huge input
        raise HTTPException(400, f"invalid image: {e}") from e
    return state["rec"].recognize_image(img)


@app.post("/api/samples")
def save_sample(req: SampleRequest):
    text = req.text.strip()
    bad = sorted(set(c for c in text if c not in C.CHAR_TO_IDX))
    if not text or bad:
        raise HTTPException(400, f"label may only contain A-Z, a-z, 0-9 (invalid: {''.join(bad)!r})")
    try:
        img = decode_data_url(req.image)
    except Exception as e:
        raise HTTPException(400, f"invalid image: {e}") from e
    if ink_columns(image_to_ink(img)) is None:
        raise HTTPException(400, "the canvas is empty")
    C.USER_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}.png"
    img.save(C.USER_SAMPLES_DIR / name)
    with open(C.USER_SAMPLES_DIR / "labels.jsonl", "a") as f:
        f.write(json.dumps({"file": name, "text": text}) + "\n")
    return {"saved": name, "text": text, "total": _count_samples()}


@app.post("/api/reload")
def reload_models():
    state["rec"].reload()
    return state["rec"].info()


def _count_samples() -> int:
    index = C.USER_SAMPLES_DIR / "labels.jsonl"
    return sum(1 for line in index.read_text().splitlines() if line.strip()) if index.exists() else 0


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    print(f"Dashboard: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

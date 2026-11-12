"""Model shapes and the HTTP API."""
import base64
import io

import pytest
import torch
from PIL import Image, ImageDraw

from hwr import config as C
from hwr.models import CRNN, CharCNN


def test_char_cnn_shape():
    assert CharCNN()(torch.rand(5, 1, 28, 28)).shape == (5, C.NUM_CLASSES)


def test_crnn_shapes_and_packing():
    m = CRNN().eval()
    x = torch.rand(3, 1, C.IMG_H, 160)
    out = m(x)
    assert out.shape == (160 // C.CRNN_STRIDE, 3, C.NUM_CTC_CLASSES)
    assert torch.allclose(out.exp().sum(-1), torch.ones(40, 3), atol=1e-4)   # log-softmax
    # with no padding, the packed path must give the same result
    assert torch.allclose(m(x, torch.tensor([160, 160, 160])), out, atol=1e-5)


def _canvas_png(draw: bool) -> str:
    img = Image.new("RGBA", (C.CANVAS_W, C.CANVAS_H), (0, 0, 0, 0))
    if draw:
        d = ImageDraw.Draw(img)
        d.line([(300, 50), (300, 210)], fill=(17, 17, 17, 255), width=C.BRUSH_PX)   # 'l'
        d.ellipse([(380, 125), (460, 210)], outline=(17, 17, 17, 255), width=C.BRUSH_PX)  # 'o'
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.server import app
    with TestClient(app) as c:
        yield c


def test_api_info(client):
    r = client.get("/api/info")
    assert r.status_code == 200
    g = r.json()["geometry"]
    assert g["canvas_w"] == C.CANVAS_W and g["base_line"] == C.BASE_LINE


def test_api_predict_empty_and_drawn(client):
    assert client.post("/api/predict", json={"image": _canvas_png(False)}).json()["empty"] is True
    r = client.post("/api/predict", json={"image": _canvas_png(True)}).json()
    assert r["empty"] is False and r["model_input_size"][1] == C.IMG_H
    if r["crnn"]:                                   # only when a checkpoint exists
        assert len(r["crnn"]["frames"]) == r["crnn"]["num_frames"]
    if r["cnn"]:
        assert len(r["cnn"]["chars"]) == 2          # 'l' and 'o' are separate segments


def test_api_rejects_bad_input(client):
    assert client.post("/api/predict", json={"image": "not-base64!!"}).status_code == 400
    bad = client.post("/api/samples", json={"image": _canvas_png(True), "text": "héllo!"})
    assert bad.status_code == 400

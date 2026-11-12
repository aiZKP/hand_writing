"""hwr — Handwritten Word Recognition (A-Z, a-z, 0-9) with PyTorch.

Two recognizers are implemented so they can be compared:
  A) segmentation + CharCNN   (classic per-character classification)
  B) CRNN + CTC               (segmentation-free sequence recognition)
"""

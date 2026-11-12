"""Data augmentation on whole batches (pure torch, no per-image Python loop).

Augmentation = a regularizer: we show the network plausible variations of the
training images (slightly rotated, scaled, slanted, thicker/thinner strokes),
so it learns features that are invariant to these nuisances instead of
memorizing pixels.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def random_affine(x: torch.Tensor, max_rot_deg: float = 10.0,
                  scale: tuple[float, float] = (0.85, 1.1), max_shear: float = 0.2,
                  max_translate: float = 0.08) -> torch.Tensor:
    """Apply a different random affine transform to every image in the batch.

    `F.affine_grid` takes a 2x3 matrix theta per image that maps OUTPUT pixel
    coordinates (normalized to [-1,1]) to INPUT sampling positions;
    `F.grid_sample` then bilinearly interpolates the input at those positions.
    """
    b = x.shape[0]
    dev = x.device
    ang = (torch.rand(b, device=dev) * 2 - 1) * math.radians(max_rot_deg)
    s = torch.empty(b, device=dev).uniform_(*scale)
    sh = (torch.rand(b, device=dev) * 2 - 1) * max_shear
    tx = (torch.rand(b, device=dev) * 2 - 1) * max_translate * 2   # [-1,1] units
    ty = (torch.rand(b, device=dev) * 2 - 1) * max_translate * 2
    cos, sin = torch.cos(ang), torch.sin(ang)
    # theta = Rotation @ Shear(x) / scale   (inverse-scale because theta maps
    # output->input: sampling a *smaller* region makes the glyph *bigger*)
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0] = cos / s
    theta[:, 0, 1] = (cos * sh - sin) / s
    theta[:, 1, 0] = sin / s
    theta[:, 1, 1] = (sin * sh + cos) / s
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def random_stroke_width(x: torch.Tensor, p_dilate: float = 0.25,
                        p_erode: float = 0.1) -> torch.Tensor:
    """Grey-scale morphology with a 3x3 window, implemented with max-pooling.

    dilation(x) = maxpool(x)       -> thicker strokes
    erosion(x)  = -maxpool(-x)     -> thinner strokes
    Erosion is softened (averaged with the original) because EMNIST strokes
    are only ~2-3 px wide and a hard erosion would delete them.
    """
    dil = F.max_pool2d(x, 3, stride=1, padding=1)
    ero = 0.5 * (x + (-F.max_pool2d(-x, 3, stride=1, padding=1)))
    r = torch.rand(x.shape[0], 1, 1, 1, device=x.device)
    return torch.where(r < p_dilate, dil, torch.where(r < p_dilate + p_erode, ero, x))


def augment_chars(x: torch.Tensor) -> torch.Tensor:
    """Full augmentation pipeline for 28x28 character batches in [0,1]."""
    x = random_affine(x)
    x = random_stroke_width(x)
    return x.clamp_(0.0, 1.0)

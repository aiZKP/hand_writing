"""Model A — CharCNN: a small VGG-style CNN that classifies ONE 28x28 character.

Shape trace (B = batch size):

    input                              B x   1 x 28 x 28   values in [0,1]
    normalize (x-mean)/std             B x   1 x 28 x 28
    Conv3x3(1->32)  BN ReLU            B x  32 x 28 x 28   params 1*32*9+32   =    320
    MaxPool 2x2                        B x  32 x 14 x 14
    Conv3x3(32->64) BN ReLU            B x  64 x 14 x 14   params 32*64*9+64  = 18,496
    Conv3x3(64->64) BN ReLU            B x  64 x 14 x 14   params 64*64*9+64  = 36,928
    MaxPool 2x2                        B x  64 x  7 x  7
    Conv3x3(64->128) BN ReLU           B x 128 x  7 x  7   params 64*128*9+128= 73,856
    MaxPool 2x2                        B x 128 x  3 x  3
    Flatten                            B x 1152
    Dropout(0.5) Linear(1152->256) ReLU                     params 295,168
    Dropout(0.5) Linear(256->62)       B x 62  (logits)     params  15,934

Softmax is NOT applied inside the model: nn.CrossEntropyLoss expects raw
logits and applies log-softmax internally (numerically more stable).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .. import config as C


def conv_bn_relu(c_in: int, c_out: int) -> nn.Sequential:
    # bias=False because BatchNorm's beta already provides a learnable shift.
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class CharCNN(nn.Module):
    def __init__(self, num_classes: int = C.NUM_CLASSES, dropout: float = 0.5):
        super().__init__()
        # Normalization constants live inside the model, so inference code
        # only has to provide [0,1] images (no chance to forget this step).
        self.register_buffer("mean", torch.tensor(C.EMNIST_MEAN))
        self.register_buffer("std", torch.tensor(C.EMNIST_STD))
        self.features = nn.Sequential(
            conv_bn_relu(1, 32),
            nn.MaxPool2d(2),                 # 28 -> 14
            conv_bn_relu(32, 64),
            conv_bn_relu(64, 64),
            nn.MaxPool2d(2),                 # 14 -> 7
            conv_bn_relu(64, 128),
            nn.MaxPool2d(2),                 # 7 -> 3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.classifier(self.features(x))

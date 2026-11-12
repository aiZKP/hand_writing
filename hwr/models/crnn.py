"""Model B — CRNN (Convolutional Recurrent Neural Network) for CTC.

Shi, Bai & Yao (2016), "An End-to-End Trainable Neural Network for Image-based
Sequence Recognition". Three stages:

  1. CNN   : image -> feature map. Pooling shrinks height to 3 but width only
             by 4, so every feature COLUMN describes a narrow vertical strip.
  2. RNN   : the columns are read left->right as a sequence by a
             bidirectional LSTM, so every time step sees context on both sides.
  3. CTC   : a per-time-step softmax over 62 characters + 1 "blank".

Shape trace for an input of width W (B = batch, T = W/4):

    input                       B x   1 x 48 x W
    Conv(1->32)   BN ReLU       B x  32 x 48 x W
    MaxPool(2,2)                B x  32 x 24 x W/2
    Conv(32->64)  BN ReLU       B x  64 x 24 x W/2
    MaxPool(2,2)                B x  64 x 12 x W/4
    Conv(64->128) BN ReLU       B x 128 x 12 x W/4
    MaxPool(2,1)                B x 128 x  6 x W/4     <- height only
    Conv(128->160) BN ReLU      B x 160 x  6 x W/4
    MaxPool(2,1)                B x 160 x  3 x W/4
    Dropout2d
    map-to-sequence             B x T x 480            (160 channels * 3 rows)
    Linear(480->256) ReLU       B x T x 256
    BiLSTM x2 (hidden 128)      B x T x 256            (128 forward + 128 backward)
    Linear(256->63)             B x T x 63
    log_softmax, transpose      T x B x 63             <- layout nn.CTCLoss wants

Receptive field of one output column: 46 px vertically (almost the whole
48-px band, so the network can judge a letter's height = its case) and 26 px
horizontally (about one character; the BiLSTM adds wider context).
Cost: ~105 M multiply-adds for an average 143-px-wide word (CPU friendly).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .. import config as C
from .char_cnn import conv_bn_relu


class CRNN(nn.Module):
    def __init__(self, num_classes: int = C.NUM_CTC_CLASSES, img_h: int = C.IMG_H,
                 rnn_hidden: int = 128, rnn_layers: int = 2, dropout: float = 0.25):
        super().__init__()
        assert img_h % 16 == 0, "height is pooled by 2 four times"
        self.cnn = nn.Sequential(
            conv_bn_relu(1, 32),
            nn.MaxPool2d((2, 2)),
            conv_bn_relu(32, 64),
            nn.MaxPool2d((2, 2)),
            conv_bn_relu(64, 128),
            nn.MaxPool2d((2, 1)),            # shrink height, keep width
            conv_bn_relu(128, 160),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.1),
        )
        feat_h = img_h // 16
        self.proj = nn.Sequential(nn.Linear(160 * feat_h, 256), nn.ReLU(inplace=True))
        self.rnn = nn.LSTM(256, rnn_hidden, num_layers=rnn_layers, bidirectional=True,
                           batch_first=True, dropout=dropout if rnn_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * rnn_hidden, num_classes))

    @staticmethod
    def output_lengths(widths: torch.Tensor) -> torch.Tensor:
        """Number of valid time steps for each (unpadded) input width."""
        return torch.div(widths, C.CRNN_STRIDE, rounding_mode="floor")

    def forward(self, x: torch.Tensor, widths: torch.Tensor | None = None) -> torch.Tensor:
        """x: B x 1 x H x W in [0,1].  widths: true widths before right-padding.

        Returns log-probabilities with shape T x B x C.
        """
        f = self.cnn(x)                                   # B x 192 x 3 x T
        b, c, h, t = f.shape
        f = f.permute(0, 3, 1, 2).reshape(b, t, c * h)    # B x T x 576
        f = self.proj(f)                                  # B x T x 256
        if widths is not None:
            # Packing tells the LSTM where each sequence really ends, so the
            # backward direction starts at the last *real* column instead of
            # reading the zero padding first.
            lengths = self.output_lengths(widths).clamp(min=1, max=t).cpu()
            packed = pack_padded_sequence(f, lengths, batch_first=True, enforce_sorted=False)
            out, _ = self.rnn(packed)
            f, _ = pad_packed_sequence(out, batch_first=True, total_length=t)
        else:
            f, _ = self.rnn(f)                            # B x T x 256
        logits = self.head(f)                             # B x T x 63
        return logits.log_softmax(-1).permute(1, 0, 2)    # T x B x 63

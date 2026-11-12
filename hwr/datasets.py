"""PyTorch datasets for the CRNN.

* BucketedSynthBatches — an IterableDataset that generates brand-new random
  word images on the fly (inside DataLoader worker processes) and yields
  ready-made batches of SIMILAR WIDTH ("bucketing", see below).
* FixedWordSet         — pre-rendered validation/test words stored in .npz, so
  that every evaluation is on exactly the same images (comparable numbers).
* UserSampleDataset    — drawings saved from the dashboard ("Correct & save").
* ctc_collate          — pads a list of variable-width images into one batch.

Why bucketing?  A batch tensor must be rectangular, so every image is padded
to the widest one in the batch. With random batches the widest word is ~2x
the average width -> half of all convolution work is spent on padding.
Generating a pool of 8 batches, sorting it by width and cutting it into
batches makes neighbours similar in width, so padding (and wasted compute)
almost disappears. The batch ORDER is shuffled again, so training still sees
short and long words in random order.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from . import config as C
from .preprocess import band_to_model_input, image_to_ink
from .synth import CharBank, TextSampler, make_sample


def ctc_collate(batch):
    """List of (image [H, W_i] tensor, text) -> padded batch for nn.CTCLoss.

    Returns:
      images         B x 1 x H x W_max   (right-padded with 0 = background)
      widths         B                   true widths (for packing / lengths)
      targets        sum(L_i)            all label indices concatenated
      target_lengths B
      texts          list[str]
    """
    images, texts = zip(*batch)
    widths = torch.tensor([im.shape[1] for im in images])
    w_max = int(widths.max())
    out = torch.zeros(len(images), 1, C.IMG_H, w_max)
    for i, im in enumerate(images):
        out[i, 0, :, : im.shape[1]] = im
    targets = torch.tensor([C.CHAR_TO_IDX[c] + 1 for t in texts for c in t], dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in texts], dtype=torch.long)
    return out, widths, targets, target_lengths, list(texts)


class BucketedSynthBatches(IterableDataset):
    """Infinite-variety synthetic words, yielded as width-bucketed batches.

    Use with DataLoader(batch_size=None): every item is already a batch.
    `extra` (optional) is another dataset — e.g. the user's own drawings —
    from which a sample is drawn with probability `extra_prob`.
    """

    def __init__(self, bank: CharBank, sampler: TextSampler, batch_size: int,
                 batches_per_epoch: int, seed: int = 0, pool_batches: int = 8,
                 extra: Dataset | None = None, extra_prob: float = 0.0):
        self.bank, self.sampler = bank, sampler
        self.batch_size, self.batches_per_epoch = batch_size, batches_per_epoch
        self.seed, self.pool_batches = seed, pool_batches
        self.extra, self.extra_prob = extra, extra_prob
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        # Workers are re-created every epoch (persistent_workers=False),
        # so they pick up the new value -> new random words every epoch.
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batches_per_epoch

    def _sample(self, rng: np.random.Generator):
        if self.extra is not None and len(self.extra) and rng.random() < self.extra_prob:
            return self.extra[int(rng.integers(len(self.extra)))]
        text = self.sampler.sample(rng)
        _, x, _ = make_sample(text, self.bank, rng)
        return torch.from_numpy(x), text

    def __iter__(self):
        info = get_worker_info()
        wid, nworkers = (info.id, info.num_workers) if info else (0, 1)
        # split the epoch's batches between workers, unique RNG stream each
        n_mine = len(range(wid, self.batches_per_epoch, nworkers))
        rng = np.random.default_rng([self.seed, self.epoch, wid])
        done = 0
        while done < n_mine:
            k = min(self.pool_batches, n_mine - done)
            pool = [self._sample(rng) for _ in range(k * self.batch_size)]
            pool.sort(key=lambda s: s[0].shape[1])                 # bucket by width
            batches = [pool[j * self.batch_size:(j + 1) * self.batch_size] for j in range(k)]
            for j in rng.permutation(k):                           # shuffle batch order
                yield ctc_collate(batches[j])
            done += k


class FixedWordSet(Dataset):
    """Pre-rendered word bands (uint8, full band height) + labels."""

    def __init__(self, path: Path):
        with np.load(path) as d:
            self.bands, self.band_w = d["bands"], d["band_w"]
            self.texts = [str(t) for t in d["texts"]]
            self.touching = d["touching"].astype(bool)
        self._cache: dict[int, torch.Tensor] = {}

    def band(self, i: int) -> np.ndarray:
        return self.bands[i, :, : self.band_w[i]]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int):
        if i not in self._cache:                      # model input is deterministic
            self._cache[i] = torch.from_numpy(band_to_model_input(self.band(i), max_w=C.MAX_W_TRAIN))
        return self._cache[i], self.texts[i]


class UserSampleDataset(Dataset):
    """data/user_samples/labels.jsonl lines: {"file": "xxx.png", "text": "Hello"}"""

    def __init__(self, root: Path = C.USER_SAMPLES_DIR):
        self.items = []
        index = root / "labels.jsonl"
        if index.exists():
            for line in index.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    if (root / rec["file"]).exists() and rec["text"]:
                        self.items.append((root / rec["file"], rec["text"]))

    @property
    def texts(self) -> list[str]:
        return [t for _, t in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, text = self.items[i]
        ink = image_to_ink(Image.open(path))
        x = band_to_model_input(ink, max_w=C.MAX_W_TRAIN)
        return torch.from_numpy(x), text


def save_fixed_set(path: Path, bands: list[np.ndarray], texts: list[str],
                   touching: list[bool]) -> None:
    h = bands[0].shape[0]
    wmax = max(b.shape[1] for b in bands)
    arr = np.zeros((len(bands), h, wmax), np.uint8)
    for i, b in enumerate(bands):
        arr[i, :, : b.shape[1]] = b
    np.savez_compressed(path, bands=arr, band_w=np.array([b.shape[1] for b in bands]),
                        texts=np.array(texts), touching=np.array(touching))

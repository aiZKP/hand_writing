"""Character segmentation for Model A (classic, rule-based — no learning).

Pipeline:
  1. binarize the ink image (threshold),
  2. find 8-connected components (scipy.ndimage.label): every blob of
     touching ink pixels becomes one component,
  3. drop tiny components (noise),
  4. merge components that overlap strongly in x — the dot of 'i'/'j', or a
     letter drawn with several separate strokes ('E', 't', '=') — because
     for left-to-right writing they belong to the same character,
  5. sort left -> right.

Known failure modes (on purpose, they motivate Model B):
  * touching / cursive letters form ONE component -> two letters are
    classified as one, the word gets shorter;
  * a letter broken into pieces that do NOT overlap in x is split into two.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .preprocess import INK_THRESHOLD


@dataclass
class CharSegment:
    x0: int
    y0: int
    x1: int          # exclusive
    y1: int          # exclusive
    crop: np.ndarray  # ink of THIS character only (other components masked out)

    @property
    def box(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]


def segment_characters(ink: np.ndarray, min_area_frac: float = 0.0004,
                       merge_overlap: float = 0.5) -> list[CharSegment]:
    """ink: float/uint8 image of the whole writing band (ink bright).

    min_area_frac  components smaller than min_area_frac * band_height^2
                   pixels are treated as noise
    merge_overlap  merge two components if their x-ranges overlap by more
                   than this fraction of the narrower one
    """
    ink = ink.astype(np.float32)
    if ink.max() <= 0:
        return []
    binary = ink > INK_THRESHOLD * ink.max()
    labels, n = ndimage.label(binary, structure=np.ones((3, 3)))     # 8-connectivity
    if n == 0:
        return []
    min_area = min_area_frac * ink.shape[0] ** 2
    areas = ndimage.sum_labels(binary, labels, index=np.arange(1, n + 1))
    comps = []                                   # [x0, x1, y0, y1, [label ids]]
    for lab, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None or areas[lab - 1] < min_area:
            continue
        comps.append([sl[1].start, sl[1].stop, sl[0].start, sl[0].stop, [lab]])

    # merge strongly x-overlapping components until nothing changes
    merged = True
    while merged:
        merged = False
        comps.sort(key=lambda c: c[0])
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                a, b = comps[i], comps[j]
                overlap = min(a[1], b[1]) - max(a[0], b[0])
                narrow = min(a[1] - a[0], b[1] - b[0])
                if overlap > merge_overlap * narrow:
                    comps[i] = [min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]),
                                max(a[3], b[3]), a[4] + b[4]]
                    del comps[j]
                    merged = True
                    break
            if merged:
                break

    segments = []
    for x0, x1, y0, y1, labs in sorted(comps, key=lambda c: c[0]):
        mask = np.isin(labels[y0:y1, x0:x1], labs)
        segments.append(CharSegment(x0, y0, x1, y1, ink[y0:y1, x0:x1] * mask))
    return segments

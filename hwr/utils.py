"""Small shared helpers: seeding, logging, checkpoints, plots."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path: Path, model: torch.nn.Module, **extra) -> None:
    """Checkpoints hold only tensors + plain Python types, so they can be
    loaded with torch.load(weights_only=True) (no arbitrary code execution)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


class CSVLogger:
    def __init__(self, path: Path, fields: list[str]):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.fields = path, fields
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(fields)

    def log(self, row: dict) -> None:
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(k, "") for k in self.fields])


def read_csv(path: Path) -> dict[str, list[float]]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows if r[k] != ""] for k in rows[0]} if rows else {}


def plot_curves(csv_path: Path, out_png: Path, panels: list[tuple[str, list[str]]],
                title: str) -> None:
    """panels = [(y-label, [column names...]), ...] -> one subplot per panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = read_csv(csv_path)
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3.6))
    axes = np.atleast_1d(axes)
    x = data.get("epoch", [])
    for ax, (ylabel, cols) in zip(axes, panels):
        for col in cols:
            if col in data and len(data[col]) == len(x):
                ax.plot(x, data[col], marker="o", label=col)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))

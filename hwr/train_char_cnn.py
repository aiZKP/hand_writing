"""Train Model A — the CharCNN — on EMNIST ByClass (62 classes).

    python -m hwr.train_char_cnn                 # full training
    python -m hwr.train_char_cnn --max-steps 50  # quick speed benchmark

The training loop is the textbook one:
    for each mini-batch:
        logits = model(x)                 # forward pass
        loss   = cross_entropy(logits, y) # how wrong are we?
        loss.backward()                   # backprop: d loss / d every weight
        optimizer.step()                  # gradient-based weight update
        scheduler.step()                  # adjust the learning rate
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, TensorDataset

from . import config as C
from .augment import augment_chars
from .emnist import load_emnist
from .metrics import classification_metrics
from .models import CharCNN
from .utils import (CSVLogger, count_params, load_checkpoint, plot_curves, save_checkpoint,
                    seed_everything, write_json)


def to_input(xb_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 [B,28,28] -> float [B,1,28,28] in [0,1]."""
    return xb_uint8.float().div_(255.0).unsqueeze(1)


@torch.inference_mode()
def predict_logits(model: nn.Module, x: torch.Tensor, batch: int = 4096) -> np.ndarray:
    model.eval()
    out = [model(to_input(x[i:i + batch])) for i in range(0, len(x), batch)]
    return torch.cat(out).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3, help="peak LR of the OneCycle schedule")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--val-size", type=int, default=20000)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (benchmark/debug)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_everything(args.seed)
    torch.set_num_threads(args.threads)

    # ---------------- data ----------------
    x, y = load_emnist("train")
    perm = np.random.default_rng(args.seed).permutation(len(x))
    val_idx, tr_idx = perm[:args.val_size], perm[args.val_size:]
    x_tr, y_tr = torch.from_numpy(x[tr_idx]), torch.from_numpy(y[tr_idx])
    x_val, y_val = torch.from_numpy(x[val_idx]), y[val_idx]
    print(f"train {len(x_tr):,}  val {len(x_val):,}")

    ds = TensorDataset(x_tr, y_tr)
    # A BatchSampler as `sampler` makes the DataLoader fetch a whole shuffled
    # batch with one tensor-indexing call (much faster than 256 single items).
    loader = DataLoader(ds, sampler=BatchSampler(RandomSampler(ds), args.batch_size, drop_last=True),
                        batch_size=None)
    steps_per_epoch = len(loader)

    # ---------------- model / optimization ----------------
    model = CharCNN()
    print(f"CharCNN parameters: {count_params(model):,}")
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.max_steps or args.epochs * steps_per_epoch
    # OneCycle: LR warms up (15 % of steps) then anneals with a cosine to ~0.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                    total_steps=total_steps, pct_start=0.15)

    log = CSVLogger(C.REPORTS_DIR / "char_cnn_log.csv",
                    ["epoch", "train_loss", "train_acc", "val_acc", "val_top3",
                     "val_balanced_acc", "lr", "seconds"])
    best_acc, step = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot_loss = tot_correct = tot_n = 0
        for xb, yb in loader:
            xb = to_input(xb)
            if not args.no_augment:
                xb = augment_chars(xb)
            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1
            tot_loss += loss.item() * len(yb)
            tot_correct += (logits.argmax(1) == yb).sum().item()
            tot_n += len(yb)
            if step % 200 == 0 or step == args.max_steps:
                el = time.time() - t0
                print(f"epoch {epoch} step {step}/{total_steps}  loss {tot_loss / tot_n:.4f}  "
                      f"acc {tot_correct / tot_n:.4f}  {tot_n / el:.0f} img/s", flush=True)
            if args.max_steps and step >= args.max_steps:
                break

        val = classification_metrics(predict_logits(model, x_val), y_val, C.NUM_CLASSES)
        row = {"epoch": epoch, "train_loss": tot_loss / tot_n, "train_acc": tot_correct / tot_n,
               "val_acc": val["acc"], "val_top3": val["top3"], "val_balanced_acc": val["balanced_acc"],
               "lr": scheduler.get_last_lr()[0], "seconds": time.time() - t0}
        log.log(row)
        print(f"== epoch {epoch}: val acc {val['acc']:.4f}  top3 {val['top3']:.4f}  "
              f"balanced {val['balanced_acc']:.4f}  ({row['seconds']:.0f}s)", flush=True)
        if val["acc"] > best_acc:
            best_acc = val["acc"]
            save_checkpoint(C.CHAR_CNN_CKPT, model, model_name="CharCNN", charset=C.CHARSET,
                            epoch=epoch, val_metrics=val, params=count_params(model))
            print(f"   saved best checkpoint -> {C.CHAR_CNN_CKPT}")
        if args.max_steps and step >= args.max_steps:
            break

    # ---------------- final test evaluation with the BEST checkpoint ----------------
    ckpt = load_checkpoint(C.CHAR_CNN_CKPT)
    model.load_state_dict(ckpt["state_dict"])
    x_te, y_te = load_emnist("test")
    test = classification_metrics(predict_logits(model, torch.from_numpy(x_te)), y_te, C.NUM_CLASSES)
    print(f"TEST: acc {test['acc']:.4f}  top3 {test['top3']:.4f}  balanced {test['balanced_acc']:.4f}")
    ckpt["test_metrics"] = test
    torch.save(ckpt, C.CHAR_CNN_CKPT)
    write_json(C.REPORTS_DIR / "char_cnn_metrics.json",
               {"val": ckpt["val_metrics"], "test": test, "best_epoch": ckpt["epoch"],
                "params": ckpt["params"], "args": vars(args)})
    plot_curves(C.REPORTS_DIR / "char_cnn_log.csv", C.REPORTS_DIR / "char_cnn_curves.png",
                [("loss", ["train_loss"]), ("accuracy", ["train_acc", "val_acc", "val_balanced_acc"])],
                "CharCNN on EMNIST ByClass")


if __name__ == "__main__":
    main()

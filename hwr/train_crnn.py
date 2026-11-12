"""Train Model B — the CRNN — with CTC loss on synthetic words.

    python -m hwr.train_crnn                     # full training
    python -m hwr.train_crnn --max-steps 30      # speed benchmark
    python -m hwr.train_crnn --finetune          # adapt to YOUR handwriting
                                                 # (samples saved from the dashboard)

nn.CTCLoss needs four things:
    log_probs       T x B x C   (log-softmax output of the network)
    targets         concatenated label indices of the whole batch
    input_lengths   B           valid time steps per sample (W_i / 4)
    target_lengths  B           number of characters per sample
"""
from __future__ import annotations

import argparse
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config as C
from .ctc import greedy_decode
from .datasets import BucketedSynthBatches, FixedWordSet, UserSampleDataset, ctc_collate
from .emnist import load_emnist
from .metrics import text_metrics
from .models import CRNN
from .synth import CharBank, TextSampler, load_wordlist
from .utils import (CSVLogger, count_params, load_checkpoint, plot_curves, save_checkpoint,
                    seed_everything, write_json)


@torch.inference_mode()
def decode_dataset(model: CRNN, dataset, batch_size: int = 128) -> list[str]:
    """Greedy-decode every sample; processes them sorted by width (little
    padding) and returns predictions in the ORIGINAL order."""
    model.eval()
    items = [dataset[i] for i in range(len(dataset))]
    order = sorted(range(len(items)), key=lambda i: items[i][0].shape[1])
    preds: list[str] = [""] * len(items)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        images, widths, _, _, _ = ctc_collate([items[i] for i in idx])
        log_probs = model(images, widths)                   # T x B x C
        lengths = model.output_lengths(widths)
        lp = log_probs.permute(1, 0, 2).numpy()             # B x T x C
        for b, i in enumerate(idx):
            preds[i] = greedy_decode(lp[b, : lengths[b]])[0]
    return preds


def evaluate(model: CRNN, dataset) -> tuple[dict, list[str]]:
    """Returns (metrics, predictions) for a FixedWordSet or UserSampleDataset."""
    preds = decode_dataset(model, dataset)
    return text_metrics(preds, dataset.texts), preds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--samples-per-epoch", type=int, default=32000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3, help="peak LR of the OneCycle schedule")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=5.0, help="gradient-norm clipping")
    ap.add_argument("--workers", type=int, default=1, help="processes generating synthetic words")
    ap.add_argument("--threads", type=int, default=6, help="torch compute threads")
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (benchmark/debug)")
    ap.add_argument("--finetune", action="store_true",
                    help="start from checkpoints/crnn_best.pt and mix in data/user_samples")
    ap.add_argument("--user-prob", type=float, default=0.3,
                    help="(finetune) fraction of training samples taken from your own drawings")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.finetune:
        # sensible fine-tuning defaults unless the user overrides them
        if args.epochs == ap.get_default("epochs"):
            args.epochs = 3
        if args.lr == ap.get_default("lr"):
            args.lr = 2e-4
        if args.samples_per_epoch == ap.get_default("samples_per_epoch"):
            args.samples_per_epoch = 6400

    seed_everything(args.seed)
    torch.set_num_threads(args.threads)

    # ---------------- data ----------------
    x, y = load_emnist("train")
    bank = CharBank(x, y)
    sampler = TextSampler(load_wordlist())
    user = None
    if args.finetune:
        user = UserSampleDataset()
        if len(user) == 0:
            raise SystemExit("No user samples found in data/user_samples — save some from the dashboard first.")
        # Mixing synthetic data back in prevents "catastrophic forgetting":
        # training only on 30 user words would quickly destroy what the
        # network learned from hundreds of thousands of synthetic words.
        print(f"fine-tuning: {len(user)} user samples, {args.user_prob:.0%} of every batch")
    stream = BucketedSynthBatches(bank, sampler, args.batch_size,
                                  args.samples_per_epoch // args.batch_size,
                                  seed=args.seed + 1000 * args.finetune,
                                  extra=user, extra_prob=args.user_prob if user else 0.0)
    loader = DataLoader(stream, batch_size=None, num_workers=args.workers, persistent_workers=False)
    val_set = FixedWordSet(C.SYNTH_VAL)

    # ---------------- model / optimization ----------------
    model = CRNN()
    if args.finetune:
        model.load_state_dict(load_checkpoint(C.CRNN_CKPT)["state_dict"])
        shutil.copy(C.CRNN_CKPT, C.CKPT_DIR / "crnn_before_finetune.pt")
    print(f"CRNN parameters: {count_params(model):,}")
    # "channels_last" memory layout (N,H,W,C) lets the CPU convolution
    # kernels (oneDNN) vectorize better: ~20 % faster here.
    model = model.to(memory_format=torch.channels_last)
    criterion = nn.CTCLoss(blank=C.CTC_BLANK, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.max_steps or args.epochs * len(stream)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps,
                                                    pct_start=0.2 if args.finetune else 0.1)

    tag = "crnn_finetune" if args.finetune else "crnn"
    log = CSVLogger(C.REPORTS_DIR / f"{tag}_log.csv",
                    ["epoch", "train_loss", "val_cer", "val_word_acc", "val_word_acc_nocase",
                     "user_cer", "lr", "seconds"])
    best_cer, step = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        stream.set_epoch(epoch)
        model.train()
        t0 = time.time()
        tot_loss, n_batches, n_samples = 0.0, 0, 0
        for images, widths, targets, target_lengths, _ in loader:
            images = images.contiguous(memory_format=torch.channels_last)
            # No LSTM packing during training: bucketing keeps the padding to
            # a few pixels, and packing costs ~20 % speed on CPU. The CTC loss
            # still ignores padded frames because it gets the true lengths.
            log_probs = model(images)                                  # T x B x C
            input_lengths = model.output_lengths(widths).clamp(max=log_probs.shape[0])
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # LSTMs can produce exploding gradients; clipping the global
            # gradient norm keeps every update bounded.
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()

            step += 1
            n_batches += 1
            n_samples += images.shape[0]
            tot_loss += loss.item()
            if step % 50 == 0 or step == args.max_steps:
                el = time.time() - t0
                print(f"epoch {epoch} step {step}/{total_steps}  ctc-loss {tot_loss / n_batches:.4f}  "
                      f"{n_samples / el:.0f} words/s", flush=True)
            if args.max_steps and step >= args.max_steps:
                break

        val, _ = evaluate(model, val_set)
        row = {"epoch": epoch, "train_loss": tot_loss / max(n_batches, 1), "val_cer": val["cer"],
               "val_word_acc": val["word_acc"], "val_word_acc_nocase": val["word_acc_nocase"],
               "lr": scheduler.get_last_lr()[0], "seconds": time.time() - t0}
        if user is not None:
            u, _ = evaluate(model, user)
            row["user_cer"] = u["cer"]
            print(f"   your samples: CER {u['cer']:.4f}  word acc {u['word_acc']:.3f}")
        log.log(row)
        print(f"== epoch {epoch}: val CER {val['cer']:.4f}  word acc {val['word_acc']:.4f}  "
              f"(ignoring case {val['word_acc_nocase']:.4f})  ({row['seconds']:.0f}s)", flush=True)
        if val["cer"] < best_cer:
            best_cer = val["cer"]
            save_checkpoint(C.CRNN_CKPT, model, model_name="CRNN", charset=C.CHARSET, epoch=epoch,
                            val_metrics=val, params=count_params(model), img_h=C.IMG_H,
                            finetuned=bool(args.finetune))
            print(f"   saved best checkpoint -> {C.CRNN_CKPT}")
        if args.max_steps and step >= args.max_steps:
            break

    # ---------------- final test evaluation with the BEST checkpoint ----------------
    ckpt = load_checkpoint(C.CRNN_CKPT)
    model.load_state_dict(ckpt["state_dict"])
    test_set = FixedWordSet(C.SYNTH_TEST)
    test, preds = evaluate(model, test_set)
    touch = np.array(test_set.touching)
    t_touch = text_metrics([p for p, m in zip(preds, touch) if m],
                           [t for t, m in zip(test_set.texts, touch) if m])
    t_sep = text_metrics([p for p, m in zip(preds, touch) if not m],
                         [t for t, m in zip(test_set.texts, touch) if not m])
    print(f"TEST: CER {test['cer']:.4f}  word acc {test['word_acc']:.4f}  "
          f"(ignoring case {test['word_acc_nocase']:.4f})")
    print(f"      touching letters: word acc {t_touch['word_acc']:.4f}   "
          f"separated: word acc {t_sep['word_acc']:.4f}")
    ckpt["test_metrics"] = test
    torch.save(ckpt, C.CRNN_CKPT)
    write_json(C.REPORTS_DIR / f"{tag}_metrics.json",
               {"val": ckpt["val_metrics"], "test": test, "test_touching": t_touch,
                "test_separated": t_sep, "best_epoch": ckpt["epoch"], "params": ckpt["params"],
                "args": vars(args)})
    plot_curves(C.REPORTS_DIR / f"{tag}_log.csv", C.REPORTS_DIR / f"{tag}_curves.png",
                [("CTC loss", ["train_loss"]), ("validation", ["val_cer", "val_word_acc"])],
                "CRNN + CTC on synthetic EMNIST words")


if __name__ == "__main__":
    main()

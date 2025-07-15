# src/train.py
#!/usr/bin/env python3
# Copyright (C) 2025 Robert Senatorov
# All rights reserved.
#
# Pretrain BERT-style masked LM model on OHLC token streams.
# • context window = 100 VQ-VAE tokens + [CLS] = 101 tokens
# • target       = only the 15 masked tokens (cross-entropy, ignore others)
# • codebook size= 1536, special tokens [MASK]=1536, [CLS]=1537

import os, glob, random
from pathlib import Path

import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import MaskedLanguageDataset, BertMLMCollator
from network.model  import BertTimeSeries
from utils.misc      import set_seed, save_ckpt
from utils.scheduler import CosineWarmupScheduler

# ───────────────────────── CONFIG ─────────────────────────
CFG = {
    "train_root":  os.path.join("data", "tokens", "train"),
    "val_root":    os.path.join("data", "tokens", "validation"),

    "context":     100,
    "vocab":       1536,
    "batch":       64,
    "workers":     4,
    "seed":        42,

    "lr":          3e-4,
    "weight_decay":1e-2,
    "warmup":      500,
    "max_steps":   50_000,

    "check_dir":   "checkpoints",
    "log_file":    "logs/pretrain_mlm.csv",
    "plot_file":   "logs/mlm_metrics.png",

    "quick_test":  False,
    "mask_count":  15,
}
# ───────────────────────────────────────────────────────────

def make_loader(folder: str, batch: int, workers: int,
                context: int, fraction: float,
                shuffle: bool, mask_count: int):
    paths = sorted(glob.glob(os.path.join(folder, "**", "*.csv"),
                             recursive=True))
    if shuffle:
        random.shuffle(paths)

    ds = MaskedLanguageDataset(paths, context, fraction, mask_count)
    collator = BertMLMCollator(
        vocab_size=CFG["vocab"],
        mask_token_id=CFG["vocab"],
        cls_token_id=CFG["vocab"] + 1,
        mask_count=mask_count
    )
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collator,
        drop_last=True,
    )

def run_epoch(model, loader, opt, scaler, device,
              sched, train: bool, tag: str):
    model.train() if train else model.eval()

    tot_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=tag, leave=False, ncols=140)

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)

        with torch.set_grad_enabled(train), autocast(device_type="cuda"):
            logits = model(input_ids)  # (B, seq_len, vocab)
            loss   = nn.functional.cross_entropy(
                        logits.view(-1, CFG["vocab"]),
                        labels.view(-1),
                        ignore_index=-100)

        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            sched.step()

        pred = logits.argmax(dim=-1)
        mask = labels != -100
        correct += (pred[mask] == labels[mask]).sum().item()
        total   += mask.sum().item()
        tot_loss += loss.item() * mask.sum().item()

        avg_loss   = tot_loss / total if total else 0.0
        avg_mlm_acc = correct / total if total else 0.0

        pbar.set_postfix(
            avg_loss=f"{avg_loss:.4f}",
            mlm_acc=f"{avg_mlm_acc:.2%}",
            lr=f"{opt.param_groups[0]['lr']:.3e}",
        )

    return tot_loss / total if total else 0.0, correct / total if total else 0.0

def ensure_log_header(path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fp:
            fp.write("epoch,train_loss,val_loss,train_mlm_acc,val_mlm_acc\n")

def update_plots(csv_path, png_out):
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9,8), sharex=True)

    ax1.plot(df["epoch"], df["train_loss"], label="train")
    ax1.plot(df["epoch"], df["val_loss"],   label="val")
    ax1.set_ylabel("MLM loss"); ax1.legend()

    ax2.plot(df["epoch"], df["train_mlm_acc"], label="mlm acc (train)")
    ax2.plot(df["epoch"], df["val_mlm_acc"],   label="mlm acc (val)")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy"); ax2.legend()

    fig.tight_layout()
    Path(os.path.dirname(png_out)).mkdir(parents=True, exist_ok=True)
    fig.savefig(png_out, dpi=120)
    plt.close(fig)

def main():
    set_seed(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(CFG["check_dir"], exist_ok=True)
    ensure_log_header(CFG["log_file"])

    model = BertTimeSeries(
        orig_vocab_size=CFG["vocab"],
        context_size=CFG["context"] + 1
    ).to(device)

    opt    = optim.AdamW(
                 model.parameters(),
                 lr=CFG["lr"],
                 weight_decay=CFG["weight_decay"])
    scaler = GradScaler()
    sched  = CosineWarmupScheduler(opt, CFG["warmup"], CFG["max_steps"])

    ckpts = glob.glob(os.path.join(CFG["check_dir"], "ckpt*.pth"))
    if ckpts:
        last = max(int(Path(p).stem[4:]) for p in ckpts)
        st   = torch.load(
                  os.path.join(CFG["check_dir"], f"ckpt{last}.pth"),
                  map_location=device)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        sched.load_state_dict(st["sched"])
        epoch = st["epoch"] + 1
        print(f"Resumed from epoch {st['epoch']}")
    else:
        epoch = 1

    quick = 0.01 if CFG["quick_test"] else 1.0

    while True:
        fold = ((epoch - 1) % 10) + 1
        train_folder = os.path.join(CFG["train_root"], f"epoch_{fold}")
        print(f"\nEpoch {epoch} → {os.path.basename(train_folder)}")

        train_loader = make_loader(
            train_folder, CFG["batch"], CFG["workers"],
            CFG["context"], quick,
            shuffle=True, mask_count=CFG["mask_count"]
        )
        val_loader   = make_loader(
            CFG["val_root"], CFG["batch"], CFG["workers"],
            CFG["context"], quick,
            shuffle=False, mask_count=CFG["mask_count"]
        )

        tr_loss, tr_acc = run_epoch(
            model, train_loader, opt, scaler, device,
            sched, train=True,  tag=f"train {epoch}"
        )
        vl_loss, vl_acc = run_epoch(
            model, val_loader,   opt, scaler, device,
            sched, train=False, tag=f"val   {epoch}"
        )

        save_ckpt({
            "epoch": epoch,
            "model": model.state_dict(),
            "opt":   opt.state_dict(),
            "scaler":scaler.state_dict(),
            "sched": sched.state_dict()
        }, os.path.join(CFG["check_dir"], f"ckpt{epoch}.pth"))

        with open(CFG["log_file"], "a") as fp:
            fp.write(f"{epoch},{tr_loss:.6f},{vl_loss:.6f},"
                     f"{tr_acc:.4f},{vl_acc:.4f}\n")
        update_plots(CFG["log_file"], CFG["plot_file"])

        print(
            f"  train loss {tr_loss:.4f} | mlm acc {tr_acc:.2%}\n"
            f"  val   loss {vl_loss:.4f} | mlm acc {vl_acc:.2%}\n"
            f"  plot → {CFG['plot_file']}"
        )
        epoch += 1

if __name__ == "__main__":
    main()

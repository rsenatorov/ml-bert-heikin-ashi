#!/usr/bin/env python3
# Copyright (C) 2025 Robert Senatorov
# All rights reserved.
#
# Fine-tune the pre-trained BERT model for Long/Short/Hold classification.
# Loads every *_signal.csv under each epoch’s market subfolder (no extra scanning),
# maps to the matching token CSV, and reports per-class precision in tqdm.

import os
import glob
import random
from pathlib import Path

import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from network.model import BertTimeSeries
from utils.misc import set_seed, save_ckpt
from utils.scheduler import CosineWarmupScheduler

# ------------------------ CONFIG ------------------------
CFG = {
    "signal_root":  "data/signal",
    "token_root":   "data/tokens",
    "context":      100,
    "num_classes":  3,      # 0=long,1=short,2=hold
    "batch":        32,
    "workers":      4,
    "seed":         42,
    "lr":           3e-5,
    "weight_decay": 1e-2,
    "warmup":       100,
    "max_steps":    10000,
    "quick_test":   False,
    "check_dir":    "checkpoints",
    "log_file":     "logs/finetune_signal.csv",
    "plot_file":    "logs/signal_metrics.png",
    "base_ckpt":    "checkpoints/ckpt5.pth",
}
# --------------------------------------------------------

class HeikinAshiClassifier(nn.Module):
    def __init__(self, pretrained_model: BertTimeSeries, num_classes: int):
        super().__init__()
        self.bert = pretrained_model
        d_model = self.bert.ln_f.normalized_shape[0]
        for p in self.bert.head.parameters():
            p.requires_grad = False
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        cls_id = self.bert.cls_token_id
        cls = torch.full(
            (input_ids.size(0), 1),
            cls_id,
            dtype=torch.long,
            device=input_ids.device
        )
        x = torch.cat([cls, input_ids], dim=1)
        out = self.bert.bert_forward(x)
        return self.classifier(out[:, 0, :])

def bert_forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
    x = self.token_emb(input_ids)
    x = self.pos_enc(x)
    x = self.drop(x)
    for blk in self.blocks:
        x = blk(x)
    return self.ln_f(x)

BertTimeSeries.bert_forward = bert_forward

class SignalDataset(Dataset):
    def __init__(
        self,
        signal_paths: list[str],
        context: int,
        token_root: str,
        signal_root: str,
        fraction: float = 1.0
    ):
        self.context = context
        self.token_arrays = {}
        self.window_map = []

        for fidx, spath in enumerate(signal_paths):
            try:
                rel = os.path.relpath(spath, signal_root)
                token_rel = rel.replace("_signal.csv", ".csv")
                tpath = os.path.join(token_root, token_rel)
                if not os.path.exists(tpath):
                    continue

                df_sig = pd.read_csv(spath)
                tokens = pd.read_csv(
                    tpath, usecols=["token"]
                )["token"].to_numpy(np.int64)
                self.token_arrays[fidx] = tokens

                for idx, lab in enumerate(df_sig["signal"]):
                    if np.isnan(lab):
                        continue
                    lab = int(lab)
                    if idx >= context - 1:
                        self.window_map.append((fidx, idx, lab))

            except Exception as e:
                print(f"Warning: failed {spath}: {e}")

        if not self.window_map:
            print("Warning: no valid windows.")

        if 0.0 < fraction < 1.0:
            keep = int(len(self.window_map) * fraction)
            self.window_map = self.window_map[:keep]

    def __len__(self):
        return len(self.window_map)

    def __getitem__(self, i):
        fidx, end_idx, lab = self.window_map[i]
        arr = self.token_arrays[fidx]
        start = end_idx - self.context + 1
        seq = arr[start : end_idx + 1]
        return {
            "tokens": torch.tensor(seq, dtype=torch.long),
            "label": torch.tensor([lab], dtype=torch.long)
        }

def make_loader(
    folder: str,
    batch: int,
    workers: int,
    context: int,
    signal_root: str,
    token_root: str,
    shuffle: bool,
    fraction: float
):
    pattern = os.path.join(folder, "**", "*_signal.csv")
    paths = sorted(glob.glob(pattern, recursive=True))
    if shuffle:
        random.shuffle(paths)

    ds = SignalDataset(paths, context, token_root, signal_root, fraction)
    if len(ds) == 0:
        return None

    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=True
    )

def run_epoch(
    model,
    loader,
    opt,
    scaler,
    device,
    sched,
    train: bool,
    tag: str
):
    model.train(train)
    tot_loss = 0.0
    correct = 0
    total = 0
    batch_prec = {0: [], 1: [], 2: []}
    pbar = tqdm(loader, desc=tag, leave=False, ncols=140)

    for batch in pbar:
        tokens = batch["tokens"].to(device)
        labels = batch["label"].to(device).squeeze(-1)
        with torch.set_grad_enabled(train), autocast(device_type="cuda"):
            logits = model(tokens)
            loss = nn.functional.cross_entropy(logits, labels)

        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            if sched:
                sched.step()

        preds = logits.argmax(dim=-1)
        for cls in (0, 1, 2):
            tp = int(((preds == cls) & (labels == cls)).sum().item())
            pc = int((preds == cls).sum().item())
            batch_prec[cls].append(tp / pc if pc > 0 else 0.0)

        correct += (preds == labels).sum().item()
        total += labels.size(0)
        tot_loss += loss.item() * labels.size(0)

        avg_loss = tot_loss / total if total else 0.0
        avg_acc = correct / total if total else 0.0
        long_p = sum(batch_prec[0]) / len(batch_prec[0]) if batch_prec[0] else 0.0
        short_p = sum(batch_prec[1]) / len(batch_prec[1]) if batch_prec[1] else 0.0
        hold_p = sum(batch_prec[2]) / len(batch_prec[2]) if batch_prec[2] else 0.0

        pbar.set_postfix(
            avg_loss=f"{avg_loss:.4f}",
            accuracy=f"{avg_acc:.2%}",
            long=f"{long_p:.2%}",
            short=f"{short_p:.2%}",
            hold=f"{hold_p:.2%}",
        )

    avg_prec = [
        sum(batch_prec[c]) / len(batch_prec[c]) if batch_prec[c] else 0.0
        for c in (0, 1, 2)
    ]
    return avg_loss, avg_acc, avg_prec

def ensure_log_header(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(
            "epoch,train_loss,val_loss,train_acc,val_acc,"
            "train_prec_long,train_prec_short,train_prec_hold,"
            "val_prec_long,val_prec_short,val_prec_hold\n"
        )

def update_plots(csv_path, png_out):
    if not os.path.exists(csv_path):
        return
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.ParserError as e:
        print(f"Warning: could not parse {csv_path}, skipping plot update: {e}")
        return
    if df.empty:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax1.plot(df["epoch"], df["train_loss"], label="train")
    ax1.plot(df["epoch"], df["val_loss"], label="val")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(df["epoch"], df["train_acc"], label="train acc")
    ax2.plot(df["epoch"], df["val_acc"], label="val acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Acc")
    ax2.legend()

    fig.tight_layout()
    Path(os.path.dirname(png_out)).mkdir(parents=True, exist_ok=True)
    fig.savefig(png_out, dpi=120)
    plt.close(fig)

def main():
    set_seed(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CFG["check_dir"], exist_ok=True)

    if not os.path.exists(CFG["base_ckpt"]):
        raise FileNotFoundError(f"Missing {CFG['base_ckpt']}")
    print(f"Loading pre-trained BERT from {CFG['base_ckpt']}…")
    base = BertTimeSeries(
        orig_vocab_size=1536,
        context_size=CFG["context"] + 1
    ).to(device)
    st = torch.load(CFG["base_ckpt"], map_location=device, weights_only=False)
    base.load_state_dict(st["model"])
    print("Pre-trained weights loaded.")

    model = HeikinAshiClassifier(base, CFG["num_classes"]).to(device)

    # initial freeze: only blocks 10-11 + classifier trainable
    for n, p in model.bert.named_parameters():
        if ("classifier" not in n) and ("head" not in n):
            if not any(f"blocks.{i}" in n for i in (10, 11)):
                p.requires_grad = False

    # optimizer & scheduler on current trainable params
    opt = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"]
    )
    scaler = GradScaler()
    sched = CosineWarmupScheduler(opt, CFG["warmup"], CFG["max_steps"])

    # resume?
    epoch = 1
    cks = glob.glob(os.path.join(CFG["check_dir"], "finetune_ckpt_*.pth"))
    if cks:
        last = max(cks, key=os.path.getctime)
        print(f"Resuming from {last}")
        st = torch.load(last, map_location=device)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        sched.load_state_dict(st["sched"])
        epoch = st["epoch"] + 1

        # if we've already reached epoch 5 or beyond, unfreeze all
        if epoch >= 5:
            print(">>> Unfreezing all BERT layers on resume <<<")
            for p in model.bert.parameters():
                p.requires_grad = True
            # re-init optimizer & scheduler over all params
            opt = optim.AdamW(
                model.parameters(),
                lr=CFG["lr"],
                weight_decay=CFG["weight_decay"]
            )
            sched = CosineWarmupScheduler(opt, CFG["warmup"], CFG["max_steps"])
    else:
        print("Starting fresh.")

    ensure_log_header(CFG["log_file"])
    frac = 0.01 if CFG["quick_test"] else 1.0

    val_folder = os.path.join(CFG["signal_root"], "validation")
    val_loader = make_loader(
        val_folder,
        CFG["batch"],
        CFG["workers"],
        CFG["context"],
        CFG["signal_root"],
        CFG["token_root"],
        shuffle=False,
        fraction=frac
    )

    # infinite cycle: epoch -> epoch_1…epoch_10, then back to epoch_1…
    while True:
        # if we hit epoch 5 in a fresh run, unfreeze now
        if epoch == 5:
            print(">>> Unfreezing all BERT layers at epoch 5 <<<")
            for p in model.bert.parameters():
                p.requires_grad = True
            opt = optim.AdamW(
                model.parameters(),
                lr=CFG["lr"],
                weight_decay=CFG["weight_decay"]
            )
            sched = CosineWarmupScheduler(opt, CFG["warmup"], CFG["max_steps"])

        fold = ((epoch - 1) % 10) + 1
        train_folder = os.path.join(
            CFG["signal_root"],
            "train",
            f"epoch_{fold}"
        )
        print(f"\nEpoch {epoch} -> data from {train_folder}")
        train_loader = make_loader(
            train_folder,
            CFG["batch"],
            CFG["workers"],
            CFG["context"],
            CFG["signal_root"],
            CFG["token_root"],
            shuffle=True,
            fraction=frac
        )
        if not train_loader or not val_loader:
            print("No data loader -> stopping.")
            break

        tr_loss, tr_acc, tr_prec = run_epoch(
            model, train_loader, opt, scaler, device, sched,
            train=True, tag=f"Train {epoch}"
        )
        vl_loss, vl_acc, vl_prec = run_epoch(
            model, val_loader, opt, scaler, device, sched=None,
            train=False, tag=f" Val  {epoch}"
        )

        ck = os.path.join(
            CFG["check_dir"],
            f"finetune_ckpt_{epoch}.pth"
        )
        save_ckpt({
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "sched": sched.state_dict()
        }, ck)

        with open(CFG["log_file"], "a") as f:
            f.write(
                f"{epoch},{tr_loss:.6f},{vl_loss:.6f},"
                f"{tr_acc:.4f},{vl_acc:.4f},"
                f"{tr_prec[0]:.4f},{tr_prec[1]:.4f},{tr_prec[2]:.4f},"
                f"{vl_prec[0]:.4f},{vl_prec[1]:.4f},{vl_prec[2]:.4f}\n"
            )

        update_plots(CFG["log_file"], CFG["plot_file"])

        print(
            f"  Train L {tr_loss:.4f} Acc {tr_acc:.2%} "
            f"Prec(L,S,H) {tr_prec[0]:.2%},{tr_prec[1]:.2%},{tr_prec[2]:.2%}\n"
            f"  Val   L {vl_loss:.4f} Acc {vl_acc:.2%} "
            f"Prec(L,S,H) {vl_prec[0]:.2%},{vl_prec[1]:.2%},{vl_prec[2]:.2%}\n"
            f"  Saved -> {ck}"
        )

        epoch += 1
        # keep cycling indefinitely

if __name__ == "__main__":
    main()

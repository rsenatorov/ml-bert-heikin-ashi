#!/usr/bin/env python3
# Tokenise 100-bar Heikin-Ashi mid-price OHLC CSVs → integer-code sequences
# using your new VQ-VAE (192-dim / 1 536-code) encoder + codebook.

import os
import glob
import random
import pandas as pd
import torch
from torch import nn
from vector_quantize_pytorch import VectorQuantize

# ─── CONFIG ────────────────────────────────────────────────────────────────
NORM_DIR     = os.path.join("data", "norm")          # Heikin-Ashi normalized CSVs
TOKENS_DIR   = os.path.join("data", "tokens")
MODELS_DIR   = "models"
ENC_PATH     = os.path.join(MODELS_DIR, "vqvae_encoder.pth")
VQ_PATH      = os.path.join(MODELS_DIR, "vqvae_vq.pth")

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_D      = 192
K_CODES       = 1_536
DECAY_GAMMA   = 0.99
BETA          = 0.25

BATCH_SZ_GPU  = 2_048    # tune down if you hit OOM
FEATURE_COLS  = ["open", "high", "low", "close"]
random.seed(42)
# ────────────────────────────────────────────────────────────────────────────


class Encoder(nn.Module):
    def __init__(self, in_dim: int, d_latent: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, d_latent)
        )
    def forward(self, x):
        return self.net(x)


def load_models():
    # 1) rebuild same architecture
    enc = Encoder(4, LATENT_D).to(DEVICE)
    vq  = VectorQuantize(
        dim=LATENT_D,
        codebook_size=K_CODES,
        decay=DECAY_GAMMA,
        kmeans_init=True,
        kmeans_iters=10,
        commitment_weight=BETA
    ).to(DEVICE)

    # 2) load state dicts
    enc_sd = torch.load(ENC_PATH, map_location=DEVICE)
    enc.load_state_dict(enc_sd)
    vq_sd  = torch.load(VQ_PATH, map_location=DEVICE)
    vq.load_state_dict(vq_sd)

    enc.eval()
    vq.eval()
    return enc, vq


def tokenize_file(path: str, enc, vq, batch=BATCH_SZ_GPU):
    dfs = []
    for chunk in pd.read_csv(path, usecols=["time"] + FEATURE_COLS,
                             chunksize=batch):
        times = chunk["time"].copy()
        arr   = chunk[FEATURE_COLS].to_numpy(dtype="float32")
        with torch.no_grad():
            x    = torch.from_numpy(arr).to(DEVICE)
            z_e  = enc(x)
            _, idxs, _ = vq(z_e)
        dfs.append(pd.DataFrame({
            "time":  times,
            "token": idxs.cpu().numpy()
        }))
        del x, z_e, idxs
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return pd.concat(dfs, ignore_index=True)


def partition_files():
    paths = glob.glob(os.path.join(NORM_DIR, "*", "*", "*.csv"))
    by_mkt, splits = {}, {"train": [], "validation": [], "test": []}
    for p in paths:
        m = os.path.normpath(p).split(os.sep)[2]
        by_mkt.setdefault(m, []).append(p)
    for m, lst in by_mkt.items():
        random.shuffle(lst)
        n = len(lst)
        n_train = int(0.8 * n)
        n_val   = int(0.1 * n)
        splits["train"].extend(lst[:n_train])
        splits["validation"].extend(lst[n_train:n_train + n_val])
        splits["test"].extend(lst[n_train + n_val:])
    return splits


def split_list(lst, n):
    k, m = divmod(len(lst), n)
    out, i = [], 0
    for _ in range(n):
        size = k + (1 if i < m else 0)
        out.append(lst[i:i + size])
        i += size
    return out


def main():
    enc, vq = load_models()
    splits  = partition_files()

    # TRAIN: 10 sub-epochs per full pass
    train_files = splits["train"]
    random.shuffle(train_files)
    epochs = split_list(train_files, 10)
    for ep, files in enumerate(epochs, 1):
        for full in files:
            rel    = os.path.relpath(full, NORM_DIR)
            market, tf, fname = rel.split(os.sep)
            out_dir = os.path.join(TOKENS_DIR, "train", f"epoch_{ep}", market)
            os.makedirs(out_dir, exist_ok=True)
            tokenize_file(full, enc, vq).to_csv(
                os.path.join(out_dir, fname),
                index=False
            )
            print(f"[train/ep{ep}] → {market}/{tf}/{fname}")

    # VALIDATION & TEST
    for split in ("validation", "test"):
        for full in splits[split]:
            rel    = os.path.relpath(full, NORM_DIR)
            market, tf, fname = rel.split(os.sep)
            out_dir = os.path.join(TOKENS_DIR, split, market)
            os.makedirs(out_dir, exist_ok=True)
            tokenize_file(full, enc, vq).to_csv(
                os.path.join(out_dir, fname),
                index=False
            )
            print(f"[{split:>10}] → {market}/{tf}/{fname}")


if __name__ == "__main__":
    main()

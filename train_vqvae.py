#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# train_vqvae_mid_ha.py – VQ-VAE (2048-code codebook) with stronger MLP
#                       encoder/decoder + 3-way direction head for
#                       stochastically-normalised Heikin-Ashi mid-price
#                       OHLC candles in data/norm.
#
# Each CSV row’s [open, high, low, close] is a training sample.
# An auxiliary 3-way direction loss steers the network to
# reconstruct the candle direction (up/down/flat).
# A structural-consistency loss enforces high ≥ max(open,close)
# and low ≤ min(open,close).
#
# Direction labels:
#   0 = up    (close > open + EPS)
#   1 = down  (close < open - EPS)
#   2 = flat  (otherwise)

import os
import glob
import math
import json
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from vector_quantize_pytorch import VectorQuantize

# ─── CONFIG ────────────────────────────────────────────────────────────────
NORM_DIR           = os.path.join("data", "norm")
MODELS_DIR         = "models"
ENC_PATH           = os.path.join(MODELS_DIR, "vqvae_encoder.pth")
DEC_PATH           = os.path.join(MODELS_DIR, "vqvae_decoder.pth")
VQ_PATH            = os.path.join(MODELS_DIR, "vqvae_vq.pth")
DIR_CLS_PATH       = os.path.join(MODELS_DIR, "vqvae_dir_clf.pth")
CFG_PATH           = os.path.join(MODELS_DIR, "vqvae_cfg.json")
VOCAB_PATH         = os.path.join(MODELS_DIR, "vocab.json")

DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

CHUNK_ROWS         = 50_000
BATCH_SZ           = 4_096
N_EPOCHS           = 50
K_CODES            = 1_536
LATENT_D           = 192
BETA               = 0.25
DECAY_GAMMA        = 0.99
LR                 = 3e-4
CLIP_NORM          = 1.0

DIR_LOSS_WEIGHT    = 5.0   # up/down/flat
STRUCT_LOSS_WEIGHT = 10.0   # HA structure
FEATURE_COLS       = ["open", "high", "low", "close"]

EPS                = 1e-9   # flat threshold
# ────────────────────────────────────────────────────────────────────────────


class Encoder(nn.Module):
    def __init__(self, in_dim: int, d_latent: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 512),    nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256),    nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, d_latent)
        )
    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, d_latent: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 512),      nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 512),      nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, out_dim)
        )
    def forward(self, z):
        return self.net(z)


class DirectionHead(nn.Module):
    def __init__(self, d_latent: int, n_classes: int = 3):
        super().__init__()
        self.fc = nn.Linear(d_latent, n_classes)
    def forward(self, z):
        return self.fc(z)


def count_csv_rows(paths):
    total = 0
    for p in paths:
        with open(p, "r") as f:
            total += sum(1 for _ in f) - 1
    return total


def infinite_csv_loader(paths, chunk_rows, batch_size, shuffle=True):
    while True:
        np.random.shuffle(paths)
        for path in paths:
            for chunk in pd.read_csv(path, usecols=FEATURE_COLS,
                                     chunksize=chunk_rows):
                if shuffle:
                    chunk = chunk.sample(frac=1).reset_index(drop=True)
                arr = chunk.values.astype(np.float32)
                for start in range(0, len(arr), batch_size):
                    yield torch.from_numpy(arr[start:start + batch_size]).to(DEVICE)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(NORM_DIR, "*", "*", "*.csv")))
    if not files:
        raise FileNotFoundError(f"No normalized files in {NORM_DIR}")

    # instantiate models
    enc     = Encoder(in_dim=4, d_latent=LATENT_D).to(DEVICE)
    dec     = Decoder(d_latent=LATENT_D, out_dim=4).to(DEVICE)
    vq      = VectorQuantize(
                dim=LATENT_D,
                codebook_size=K_CODES,
                decay=DECAY_GAMMA,
                kmeans_init=True,
                kmeans_iters=10,
                commitment_weight=BETA
             ).to(DEVICE)
    dir_clf = DirectionHead(d_latent=LATENT_D, n_classes=3).to(DEVICE)

    # optimizer, scheduler, scaler
    optim  = torch.optim.AdamW(
                list(enc.parameters()) +
                list(dec.parameters()) +
                list(vq.parameters()) +
                list(dir_clf.parameters()),
                lr=LR, betas=(0.9, 0.95), weight_decay=1e-4
             )
    sched  = CosineAnnealingLR(optim, T_max=N_EPOCHS)
    scaler = GradScaler()

    total_rows   = count_csv_rows(files)
    steps_per_ep = math.ceil(total_rows / BATCH_SZ)
    print(f"Total rows: {total_rows:,}  •  Steps/epoch: {steps_per_ep:,}")

    loader = infinite_csv_loader(files, CHUNK_ROWS, BATCH_SZ, shuffle=True)

    best_loss = float('inf')
    for epoch in range(1, N_EPOCHS + 1):
        enc.train(); dec.train(); vq.train(); dir_clf.train()
        running_loss = 0.0
        used_codes   = set()
        pbar = tqdm(range(steps_per_ep),
                    desc=f"Epoch {epoch}/{N_EPOCHS}",
                    dynamic_ncols=True)

        for _ in pbar:
            batch = next(loader)
            optim.zero_grad()
            # fixed: pass device_type into autocast
            with autocast(device_type=DEVICE):
                # encode → quantize → decode
                z_e   = enc(batch)
                z_q, indices, commit_loss = vq(z_e)
                recon = dec(z_q)

                # 1) reconstruction loss
                recon_loss = nn.functional.mse_loss(recon, batch)

                # 2) 3-way direction loss
                true_dir  = torch.where(
                    batch[:,3] > batch[:,0] + EPS, 0,
                    torch.where(batch[:,3] < batch[:,0] - EPS, 1, 2)
                )
                logits_dir = dir_clf(z_q)  # [B,3]
                dir_loss   = nn.functional.cross_entropy(
                                logits_dir,
                                true_dir.long()
                             )

                # 3) structural-consistency loss
                max_oc      = torch.max(recon[:,0], recon[:,3])
                min_oc      = torch.min(recon[:,0], recon[:,3])
                viol_h      = torch.relu(max_oc - recon[:,1])
                viol_l      = torch.relu(recon[:,2] - min_oc)
                struct_loss = (viol_h + viol_l).mean()

                loss = (
                    recon_loss
                    + commit_loss
                    + DIR_LOSS_WEIGHT    * dir_loss
                    + STRUCT_LOSS_WEIGHT * struct_loss
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(
                list(enc.parameters()) +
                list(dec.parameters()) +
                list(vq.parameters()) +
                list(dir_clf.parameters()),
                CLIP_NORM
            )
            scaler.step(optim)
            scaler.update()

            running_loss += loss.item()
            used_codes.update(indices.cpu().tolist())
            pbar.set_postfix(
                loss   = f"{loss.item():.6f}",
                recon  = f"{recon_loss.item():.6f}",
                commit = f"{commit_loss.item():.6f}",
                dir    = f"{dir_loss.item():.6f}",
                struct = f"{struct_loss.item():.6f}",
                codes  = f"{len(used_codes)}/{K_CODES}"
            )

        sched.step()
        avg_loss = running_loss / steps_per_ep
        print(f"Epoch {epoch}: avg_loss={avg_loss:.6f}  •  unique_codes={len(used_codes)}")

        # checkpoint best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(enc.state_dict(),     ENC_PATH)
            torch.save(dec.state_dict(),     DEC_PATH)
            torch.save(vq.state_dict(),      VQ_PATH)
            torch.save(dir_clf.state_dict(), DIR_CLS_PATH)

    # save final config
    with open(CFG_PATH, "w") as fp:
        json.dump({
            "latent_dim":        LATENT_D,
            "codebook_size":     K_CODES,
            "beta":              BETA,
            "decay_gamma":       DECAY_GAMMA,
            "dir_loss_weight":   DIR_LOSS_WEIGHT,
            "struct_loss_weight": STRUCT_LOSS_WEIGHT,
            "best_loss":         best_loss
        }, fp, indent=2)

    # build vocab
    sums   = np.zeros((K_CODES, 4), dtype=np.float64)
    counts = np.zeros(K_CODES,    dtype=np.int64)
    for path in files:
        for chunk in pd.read_csv(path, usecols=FEATURE_COLS,
                                 chunksize=CHUNK_ROWS):
            data = torch.from_numpy(chunk.values.astype(np.float32)).to(DEVICE)
            with torch.no_grad():
                _, idxs, _ = vq(enc(data))
            for code, feat in zip(idxs.cpu().numpy(), chunk.values):
                sums[code]   += feat
                counts[code] += 1

    vocab = {}
    for code in range(K_CODES):
        if counts[code] == 0:
            avg       = [0.0] * 4
            direction = 2
        else:
            avg       = (sums[code] / counts[code]).tolist()
            o, h, l, c = avg
            if c > o + EPS:
                direction = 0
            elif c < o - EPS:
                direction = 1
            else:
                direction = 2
        vocab[str(code)] = {
            "open":  avg[0],
            "high":  avg[1],
            "low":   avg[2],
            "close": avg[3],
            "dir":   direction
        }

    with open(VOCAB_PATH, "w") as fp:
        json.dump(vocab, fp, indent=2)

    print(f"✔ Training complete. Saved models, dir-clf, config, and vocab in “{MODELS_DIR}”.")
    

if __name__ == "__main__":
    main()

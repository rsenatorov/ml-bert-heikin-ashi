#!/usr/bin/env python3
# Copyright (C) 2025 Robert Senatorov
# All rights reserved.
#
# normalise_mid_heikin_ashi.py – Stochastic min-max normalisation of
# Heikin-Ashi mid-price OHLC bars.
# Saves to data/norm/{market}/{tf}/… and stores the fitted normaliser
# in models/mid_ha_norm_normalizer.joblib.

import os
import glob
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

# ── CONFIG ─────────────────────────────────────────────────────────────
RAW_ROOT    = os.path.join("data", "raw")
NORM_ROOT   = os.path.join("data", "norm")
MODEL_DIR   = "models"
WINDOW_SIZE = 100      # 99 past + current
EPS         = 1e-9
COLUMNS     = [
    "time",
    "open", "high", "low", "close",
    "low_min", "high_max",
]
# ────────────────────────────────────────────────────────────────────────


class MidHeikinAshiStochasticNormalizer:
    """Min-max normaliser of Heikin-Ashi mid-price candles over a fixed rolling window."""

    def __init__(self, window_size: int, eps: float = 1e-9):
        self.window_size = window_size
        self.eps = eps
        self.columns = COLUMNS.copy()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # ─── compute mid-prices ──────────────────────────────────────────
        mid_open  = (df["bid_open"]  + df["ask_open"])  / 2.0
        mid_high  = (df["bid_high"]  + df["ask_high"])  / 2.0
        mid_low   = (df["bid_low"]   + df["ask_low"])   / 2.0
        mid_close = (df["bid_close"] + df["ask_close"]) / 2.0

        df_mid = pd.DataFrame({
            "time":  df["time"],
            "open":  mid_open,
            "high":  mid_high,
            "low":   mid_low,
            "close": mid_close
        })

        # ─── compute Heikin-Ashi OHLC ─────────────────────────────────
        ha_close = (df_mid[["open", "high", "low", "close"]].sum(axis=1)) / 4.0
        ha_open = pd.Series(index=df_mid.index, dtype="float64")
        # first HA-open = midpoint of first bar
        ha_open.iloc[0] = (df_mid["open"].iloc[0] + df_mid["close"].iloc[0]) / 2.0
        for i in range(1, len(df_mid)):
            ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0

        ha_high = pd.concat([df_mid["high"], ha_open, ha_close], axis=1).max(axis=1)
        ha_low  = pd.concat([df_mid["low"],  ha_open, ha_close], axis=1).min(axis=1)

        df_ha = pd.DataFrame({
            "time":  df_mid["time"],
            "open":  ha_open.values,
            "high":  ha_high.values,
            "low":   ha_low.values,
            "close": ha_close.values
        })

        # ─── rolling window extremes (back-ward looking) ────────────────
        low_min  = df_ha["low"].rolling(window=self.window_size, min_periods=self.window_size).min()
        high_max = df_ha["high"].rolling(window=self.window_size, min_periods=self.window_size).max()

        df_ha["low_min"]  = low_min.values
        df_ha["high_max"] = high_max.values

        # ─── drop rows without full window ───────────────────────────────
        df_ha = df_ha.dropna(subset=["low_min", "high_max"]).reset_index(drop=True)

        # ─── stochastic min–max normalise HA OHLC ───────────────────────
        price_range = (df_ha["high_max"] - df_ha["low_min"]).clip(lower=self.eps)
        for col in ["open", "high", "low", "close"]:
            df_ha[col] = (df_ha[col] - df_ha["low_min"]) / price_range

        return df_ha[self.columns]


def normalise_file(path: str, normaliser: MidHeikinAshiStochasticNormalizer) -> None:
    rel = os.path.relpath(path, RAW_ROOT)
    market, tf, fname = rel.split(os.sep)
    out_dir = os.path.join(NORM_ROOT, market, tf)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)

    df = pd.read_csv(path)
    if df.empty:
        print(f"Skipping {rel}: no data")
        return

    df_norm = normaliser.transform(df)
    if df_norm.empty:
        print(f"Skipping {rel}: fewer than {WINDOW_SIZE} rows")
        return

    df_norm.to_csv(out_path, index=False, float_format="%.6f")
    print(f"Saved {out_path} ({len(df_norm)} rows)")


def main() -> None:
    # 1) instantiate & save normaliser
    os.makedirs(MODEL_DIR, exist_ok=True)
    normaliser = MidHeikinAshiStochasticNormalizer(WINDOW_SIZE, EPS)
    joblib_path = os.path.join(MODEL_DIR, "mid_ha_norm_normalizer.joblib")
    joblib.dump(normaliser, joblib_path)
    print(f"✔ Normaliser saved → {joblib_path}")

    # 2) process every raw CSV
    pattern = os.path.join(RAW_ROOT, "*", "*", "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No raw files found under {RAW_ROOT}")
        return

    for path in tqdm(files, desc=f"Normalising (mid-HA / {WINDOW_SIZE}-bar window)"):
        try:
            normalise_file(path, normaliser)
        except Exception as e:
            print(f"[ERROR] {path}: {e}")


if __name__ == "__main__":
    main()

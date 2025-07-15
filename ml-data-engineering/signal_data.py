#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo
from tqdm import tqdm

# CONFIG --------------------------------------------------------------
RAW_ROOT       = "data/raw"         # raw bid/ask CSVs under data/raw/{market}/m1/*.csv
TOKENS_ROOT    = "data/tokens"      # token CSVs under data/tokens/{split}/... or train/epoch_{e}/...
SIGNAL_ROOT    = "data/signal"      # output signals will go under data/signal/{split}/{market}/
TIMEFRAME      = "m1"
TIMEZONE       = "America/Vancouver"
CONTEXT_SIZE   = 100
FUTURE_SIZE    = 24
ATR_LENGTH     = 14
ATR_MULTIPLIER = 1.5
TP_MULTIPLIER  = 1.5
# ---------------------------------------------------------------------

tz = ZoneInfo(TIMEZONE)

token_paths = glob.glob(os.path.join(TOKENS_ROOT, "**", "*.csv"), recursive=True)
if not token_paths:
    raise RuntimeError(f"No token files found under {TOKENS_ROOT}")

for token_path in tqdm(token_paths, desc="Processing signals"):
    rel    = os.path.relpath(token_path, TOKENS_ROOT)
    parts  = rel.split(os.sep)
    if parts[0] == "train":
        split   = "train"
        epoch   = parts[1]      # e.g. "epoch_1"
        market  = parts[2]
        fname   = parts[3]
    else:
        split   = parts[0]      # "validation" or "test"
        market  = parts[1]
        fname   = parts[2]

    raw_path    = os.path.join(RAW_ROOT, market, TIMEFRAME, fname)
    out_dir     = os.path.join(SIGNAL_ROOT, split, market)
    os.makedirs(out_dir, exist_ok=True)
    base_name   = os.path.splitext(fname)[0]
    out_path    = os.path.join(out_dir, f"{base_name}_signal.csv")

    # 1) LOAD & mid‐price Heikin‐Ashi
    df_raw = pd.read_csv(raw_path)
    for col in ("open", "high", "low", "close"):
        df_raw[f"mid_{col}"] = (df_raw[f"bid_{col}"] + df_raw[f"ask_{col}"]) / 2.0

    ha_close = df_raw[["mid_open","mid_high","mid_low","mid_close"]].mean(axis=1)
    ha_open  = ha_close.copy()
    ha_open.iloc[0] = (df_raw["mid_open"].iloc[0] + df_raw["mid_close"].iloc[0]) / 2.0
    for i in range(1, len(df_raw)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2.0

    ha_high = pd.concat([df_raw["mid_high"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low  = pd.concat([df_raw["mid_low"],  ha_open, ha_close], axis=1).min(axis=1)

    df_ha = pd.DataFrame({
        "time":  df_raw["time"],
        "open":  ha_open.values,
        "high":  ha_high.values,
        "low":   ha_low.values,
        "close": ha_close.values
    })
    df_ha["time_utc"]   = pd.to_datetime(df_ha["time"], format="%Y-%m-%d-%H:%M", utc=True)
    df_ha["time_local"] = df_ha["time_utc"].dt.tz_convert(tz)

    # 2) LOAD TOKENS & merge
    df_tok = pd.read_csv(token_path)
    df_tok["time_utc"] = pd.to_datetime(df_tok["time"], utc=True, errors="coerce")

    df = pd.merge(
        df_ha,
        df_tok[["time_utc","token"]],
        on="time_utc", how="left"
    ).reset_index(drop=True)

    # 3) TRUE‐RANGE & ATR
    hl  = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift(1)).abs()
    lpc = (df["low"]  - df["close"].shift(1)).abs()
    df["tr"]  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    alpha     = 1.0 / ATR_LENGTH
    df["atr"] = df["tr"].ewm(alpha=alpha, adjust=False).mean()

    # prepare output frame
    df_sig = pd.DataFrame({
        "time":   df["time_local"],
        "token":  df["token"],
        "signal": np.nan
    })

    # 4) SL/TP & SIGNAL SLIDING WINDOW
    for i in range(CONTEXT_SIZE-1, len(df)-FUTURE_SIZE):
        atr = df.at[i, "atr"]
        if np.isnan(atr):
            continue

        entry   = df.at[i, "close"]
        risk    = atr * ATR_MULTIPLIER
        sl_l    = entry - risk
        tp_l    = entry + risk * TP_MULTIPLIER
        sl_s    = entry + risk
        tp_s    = entry - risk * TP_MULTIPLIER

        long_hit = None
        short_hit = None
        for j in range(1, FUTURE_SIZE+1):
            hi = df.at[i+j, "high"]
            lo = df.at[i+j, "low"]
            if long_hit is None:
                if lo <= sl_l:
                    long_hit = -1
                elif hi >= tp_l:
                    long_hit = j
            if short_hit is None:
                if hi >= sl_s:
                    short_hit = -1
                elif lo <= tp_s:
                    short_hit = j
            if long_hit is not None and short_hit is not None:
                break

        long_ok  = (long_hit is not None and long_hit > 0)
        short_ok = (short_hit is not None and short_hit > 0)
        if   long_ok and not short_ok:
            sig = 0
        elif short_ok and not long_ok:
            sig = 1
        elif long_ok and short_ok:
            sig = 0 if long_hit < short_hit else (1 if short_hit < long_hit else 2)
        else:
            sig = 2

        df_sig.at[i, "signal"] = sig

    # 5) SAVE (time,token,signal)
    df_sig.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

print("All signals generated.")

#!/usr/bin/env python3
# bnb_dataset.py
# Generate BNB 1m signals from 2020-01-01 to 2025-07-10,
# then split into train/validation/test under data/bnb/signal.

import os
from datetime import datetime, date, timezone
import pandas as pd

import simulation  # Assumes simulation.py is on PYTHONPATH or same folder

def main():
    # --- Configure simulation for BNB 1m --------------------------------
    simulation.CFG['fsym'] = 'BNB'
    simulation.CFG['market'] = 'BNBUSDT'
    simulation.CFG['timeframe'] = '1m'
    simulation.CFG['start_dt'] = datetime(2020, 1, 1, tzinfo=timezone.utc)
    simulation.CFG['end_dt'] = datetime(2025, 7, 10, tzinfo=timezone.utc)
    base = os.path.join('data', 'bnb')
    simulation.CFG['raw_root'] = os.path.join(base, 'raw')
    simulation.CFG['norm_root'] = os.path.join(base, 'norm')
    simulation.CFG['tokens_root'] = os.path.join(base, 'tokens')
    simulation.CFG['signal_root'] = os.path.join(base, 'signal')

    for key in ('raw_root', 'norm_root', 'tokens_root', 'signal_root'):
        os.makedirs(simulation.CFG[key], exist_ok=True)

    # --- Run pipeline ----------------------------------------------------
    print("1/4 Fetching raw data…")
    raw_csv = simulation.fetch_historical_data()
    print("2/4 Normalising…")
    norm_csv = simulation.normalize_all(raw_csv)[0]
    print("3/4 Tokenising…")
    tok_csv = simulation.tokenize_all([norm_csv])[0]
    print(" hardware4/4 Generating signals…")
    sig_csv = simulation.generate_signals([tok_csv])[0]
    print(f"\nSignal file written to: {sig_csv}")

    # --- Load & split ----------------------------------------------------
    df = pd.read_csv(sig_csv)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df['date'] = df['time'].dt.date

    splits = {
        'train': df['date'] < date(2025, 1, 1),
        'validation': (df['date'] >= date(2025, 1, 1)) & (df['date'] <= date(2025, 4, 1)),
        'test': (df['date'] >= date(2025, 4, 2)) & (df['date'] <= date(2025, 7, 10)),
    }

    market = simulation.CFG['market']
    tf = simulation.CFG['timeframe']
    fname = os.path.basename(sig_csv)

    for split, mask in splits.items():
        out_dir = os.path.join(simulation.CFG['signal_root'], split, market, tf)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, fname)

        df.loc[mask].to_csv(out_path, index=True)

        total = int(mask.sum())
        counts = df.loc[mask, 'signal'].value_counts().reindex([0, 1, 2], fill_value=0)
        print(f"{split.capitalize()} set ({total}): Long {counts[0]}, Short {counts[1]}, Hold {counts[2]}")
        print(f"  -> Saved to {out_path}")

if __name__ == "__main__":
    main()
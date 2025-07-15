#!/usr/bin/env python3
import os
import csv
import subprocess
from datetime import datetime, timedelta, timezone
from collections import defaultdict

# ─── CONFIG ────────────────────────────────────────────────────────────────
TIMEFRAME   = "m1"                  # now using 1-minute bars
QUICK_TEST  = 0                     # number of instruments to process; 0 = all
INSTR_CSV   = os.path.join("data", "raw", "instruments.csv")
TMP_DIR     = "tmp"
# ────────────────────────────────────────────────────────────────────────────

def load_instruments(path):
    """
    Return a list of (market, instrument) tuples from CSV.
    CSV must have headers: market,instrument
    """
    instruments = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            market = row.get("market", "").strip()
            inst   = row.get("instrument", "").strip().lower()
            if market and inst:
                instruments.append((market, inst))
    return instruments

def download_side(inst, side, start_date, end_date):
    """
    Download one side (bid/ask) via dukascopy-node CLI,
    return list of (ts_ms, o, h, l, c).
    """
    os.makedirs(TMP_DIR, exist_ok=True)
    base = f"{inst}_{TIMEFRAME}_{side}"
    cmd = (
        f"npx dukascopy-node "
        f"-i {inst} -from {start_date} -to {end_date} "
        f"-t {TIMEFRAME} -p {side} -f csv "
        f"-dir {TMP_DIR} -fn {base}"
    )
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Error fetching {inst} {side}: {proc.stderr.strip()}")

    path = os.path.join(TMP_DIR, f"{base}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected {path} but it does not exist")

    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            ts = int(row[0])
            o, h, l, c = map(float, row[1:5])
            rows.append((ts, o, h, l, c))
    return rows

def merge_and_filter(bid, ask):
    """
    Align bid & ask on timestamp, drop bars where both sides are flat.
    """
    bmap = {r[0]: r[1:] for r in bid}
    amap = {r[0]: r[1:] for r in ask}
    merged = []
    for ts in sorted(bmap.keys() & amap.keys()):
        bo, bh, bl, bc = bmap[ts]
        ao, ah, al, ac = amap[ts]
        # drop only if both bid & ask are perfectly flat
        if (bo == bc and bh == bl) and (ao == ac and ah == al):
            continue
        merged.append((ts, bo, bh, bl, bc, ao, ah, al, ac))
    return merged

def save_csv(market, inst, rows):
    """
    Write merged rows to data/raw/{market}/m1/{inst}_m1.csv
    with time formatted YYYY-MM-DD-HH:MM (UTC).
    """
    out_dir = os.path.join("data", "raw", market, TIMEFRAME)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{inst}_{TIMEFRAME}.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time",
            "bid_open","bid_high","bid_low","bid_close",
            "ask_open","ask_high","ask_low","ask_close"
        ])
        for ts, bo, bh, bl, bc, ao, ah, al, ac in rows:
            dt = datetime.fromtimestamp(ts/1000, timezone.utc)
            time_str = dt.strftime("%Y-%m-%d-%H:%M")
            writer.writerow([time_str, bo, bh, bl, bc, ao, ah, al, ac])

    return len(rows)

def generate_readme(per_mkt, per_tf):
    """
    Generate data/raw/readme.txt summarizing sample counts.
    """
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    total = sum(per_mkt.values())

    lines = [f"Total samples: {total}", "", "By market:"]
    for m, cnt in per_mkt.items():
        lines.append(f"  {m}: {cnt}")
    lines += ["", "By timeframe:"]
    for tf, cnt in per_tf.items():
        lines.append(f"  {tf}: {cnt}")

    with open(os.path.join(raw_dir, "readme.txt"), "w") as f:
        f.write("\n".join(lines))

def main():
    today = datetime.now(timezone.utc).date()
    start = (today - timedelta(days=5*365)).isoformat()
    end   = (today - timedelta(days=1)).isoformat()

    instruments = load_instruments(INSTR_CSV)
    if QUICK_TEST > 0:
        instruments = instruments[:QUICK_TEST]

    per_mkt = defaultdict(int)
    per_tf  = defaultdict(int)

    for market, inst in instruments:
        bid    = download_side(inst, "bid",  start, end)
        ask    = download_side(inst, "ask",  start, end)
        merged = merge_and_filter(bid, ask)

        count = save_csv(market, inst, merged)
        per_mkt[market] += count
        per_tf[TIMEFRAME] += count
        print(f"[OK]  {market}/{inst}: saved {count} rows")

    generate_readme(per_mkt, per_tf)
    print("✔ data/raw/readme.txt generated")

if __name__ == "__main__":
    main()

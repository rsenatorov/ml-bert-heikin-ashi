#!/usr/bin/env python3
# Copyright (C) 2025 Robert Senatorov
# All rights reserved.

import os, time, math, csv, requests, pandas as pd, numpy as np
import torch, torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from tqdm import tqdm
from vector_quantize_pytorch import VectorQuantize
from network.model import BertTimeSeries
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

matplotlib.use('Agg')

# ------------------------------ CONFIG ---------------------------------
CFG = dict(
    pairs            = ['BNB', 'DOGE', 'BTC', 'ETH', 'SOL', 'ADA', 'XRP'],
    tsym             = 'USDT',
    candles_pull     = 300,
    context_size     = 100,
    future_size      = 24,
    atr_length       = 14,
    sl_multiplier    = 0.5,
    tp_multiplier    = 1.5,
    other_conf_max   = 1.0,
    conf_thresh      = 0.0,
    min_tp_pct       = 0.0,
    tz_local         = 'America/Vancouver',
    enc_path         = os.path.join('models','vqvae_encoder.pth'),
    vq_path          = os.path.join('models','vqvae_vq.pth'),
    finetune_ckpt    = 'checkpoints/finetune_ckpt_1.pth',
    trade_root       = 'trades',
    start_year       = 2025,
    start_month      = 6,
    start_day        = 1,
    end_year         = 2025,
    end_month        = 7,
    end_day          = 1,
    log_dir          = 'logs',
    classif_report   = 'logs/classification_report.txt',
    equity_curve_png = 'logs/equity_curve.png',
    confusion_mat_png= 'logs/confusion_matrix.png',
    initial_equity   = 1000.0,
    fee_pct          = 0.0,
    batch            = 64,
    workers          = 4,
    seed             = 42,
    supertrend_enable = True,
    supertrend_period = 10,
    supertrend_multiplier = 3,
)

timeframes = ['15m']

BINANCE_URL = 'https://api.binance.com/api/v3/klines'
local_tz = ZoneInfo(CFG['tz_local'])
device   = 'cuda' if torch.cuda.is_available() else 'cpu'

tf_map = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400
}

COLORS = plt.get_cmap('tab10').colors

# --------------------------- BERT PATCH --------------------------------
def _bert_forward(self, ids):
    x = self.token_emb(ids)
    x = self.pos_enc(x)
    x = self.drop(x)
    for blk in self.blocks:
        x = blk(x)
    return self.ln_f(x)
BertTimeSeries.bert_forward = _bert_forward

# ------------------------- BERT + HEAD WRAPPER -------------------------
class BertClassifierEval(nn.Module):
    def __init__(self, vocab_sz, ctx_sz, num_cls=3):
        super().__init__()
        self.bert = BertTimeSeries(orig_vocab_size=vocab_sz,
                                   context_size=ctx_sz)
        hd = self.bert.token_emb.embedding_dim
        self.classifier  = nn.Linear(hd, num_cls)
        self.cls_token_id = self.bert.cls_token_id
    def forward(self, ids):
        seq = self.bert.bert_forward(ids)
        return self.classifier(seq[:, 0, :])

# ------------------------- LOAD STATIC MODELS -------------------------
LATENT_D, CODEBOOK = 192, 1536

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4,512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512,512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512,256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256,LATENT_D))
    def forward(self, x): return self.net(x)

enc = Encoder().to(device)
vq  = VectorQuantize(dim=LATENT_D, codebook_size=CODEBOOK, decay=0.99,
                     kmeans_init=True, kmeans_iters=10,
                     commitment_weight=0.25).to(device)
enc.load_state_dict(torch.load(CFG['enc_path'], map_location=device, weights_only=True))
vq.load_state_dict(torch.load(CFG['vq_path'],  map_location=device, weights_only=True))
enc.eval(); vq.eval()

model = BertClassifierEval(CODEBOOK, CFG['context_size'] + 1).to(device)
state_dict = torch.load(CFG['finetune_ckpt'], map_location=device, weights_only=True)
state_dict = state_dict['model'] if 'model' in state_dict else state_dict
model.load_state_dict(state_dict, strict=True)
model.eval()

# --------------------------- HELPER FUNCS ------------------------------
def pre_fetch_data(start_ms, end_ms):
    data = {}
    all_tfs = set(timeframes)
    pbar = tqdm(total=len(CFG['pairs']) * len(all_tfs), desc="Pre-fetching data", ncols=80)
    for sym in CFG['pairs']:
        pair = sym + CFG['tsym']
        for tf in all_tfs:
            data[(pair, tf)] = fetch_all_klines(pair, tf, start_ms, end_ms)
            pbar.update(1)
    pbar.close()
    return data

def fetch_all_klines(pair, tf, start_ms, end_ms):
    url = BINANCE_URL
    limit = 1000
    current_end = end_ms
    klines = []
    while current_end > start_ms:
        params = {
            'symbol': pair,
            'interval': tf,
            'limit': limit,
            'endTime': current_end
        }
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            klines = batch + klines
            current_end = batch[0][0] - 1
            time.sleep(0.1)
        except Exception as e:
            print(f"Error fetching {pair} {tf}: {e}")
            break
    if not klines:
        return pd.DataFrame(columns=['open_time', 'open', 'high', 'low', 'close'])
    cols = ['open_time','open','high','low','close']
    df = pd.DataFrame([[d[0],*(float(v) for v in d[1:5])] for d in klines], columns=cols)
    df = df[(df['open_time'] >= start_ms) & (df['open_time'] <= end_ms)]
    return df.sort_values('open_time').reset_index(drop=True)

def sim_fetch_closed(sym, timeframe, now_ms, pre_data):
    pair = sym + CFG['tsym']
    tf_ms = tf_map[timeframe] * 1000
    end_ms = now_ms - (now_ms % tf_ms) - 1
    full_df = pre_data.get((pair, timeframe))
    if full_df is None:
        return pd.DataFrame(columns=['open_time', 'open', 'high', 'low', 'close'])
    candidates = full_df[full_df['open_time'] <= end_ms]
    if len(candidates) == 0:
        return pd.DataFrame(columns=['open_time', 'open', 'high', 'low', 'close'])
    return candidates.tail(CFG['candles_pull']).copy()

def heikin(df):
    ha_close = df[['open','high','low','close']].mean(axis=1)
    ha_open  = ha_close.copy()
    ha_open.iloc[0] = (df.iloc[0]['open']+df.iloc[0]['close'])/2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    ha_high = pd.concat([df['high'],ha_open,ha_close],axis=1).max(axis=1)
    ha_low  = pd.concat([df['low'],ha_open,ha_close],axis=1).min(axis=1)
    return pd.DataFrame(dict(open=ha_open,high=ha_high,low=ha_low,close=ha_close))

def get_rolling_normalized(ha, context_size):
    if len(ha) < context_size:
        return None
    low_min = ha['low'].rolling(context_size, min_periods=context_size).min()
    high_max = ha['high'].rolling(context_size, min_periods=context_size).max()
    ha_norm = ha.iloc[context_size - 1:].copy().reset_index(drop=True)
    ha_norm['low_min'] = low_min.iloc[context_size - 1:].values
    ha_norm['high_max'] = high_max.iloc[context_size - 1:].values
    rng = (ha_norm['high_max'] - ha_norm['low_min']).clip(lower=1e-9)
    for c in ['open', 'high', 'low', 'close']:
        ha_norm[c] = (ha_norm[c] - ha_norm['low_min']) / rng
    return ha_norm[['open', 'high', 'low', 'close']]

def to_tokens(arr):
    with torch.no_grad():
        _, idx, _ = vq(enc(torch.from_numpy(arr).to(device)))
    return idx.cpu().numpy()

def atr_last(ha):
    hl  = ha['high'] - ha['low']
    hpc = (ha['high'] - ha['close'].shift(1)).abs()
    lpc = (ha['low']  - ha['close'].shift(1)).abs()
    tr  = pd.concat([hl,hpc,lpc], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/CFG['atr_length'], adjust=False).mean().iloc[-1]

def calculate_atr(df, period):
    hl = df['high'] - df['low']
    hpc = abs(df['high'] - df['close'].shift())
    lpc = abs(df['low'] - df['close'].shift())
    tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def calculate_supertrend(df, period=10, multiplier=3):
    atr = calculate_atr(df, period)
    hl2 = (df['high'] + df['low']) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    supertrend = pd.Series(np.nan, index=df.index)
    in_uptrend = pd.Series(True, index=df.index)
    for i in range(1, len(df)):
        if in_uptrend.iloc[i-1]:
            supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i-1] if not pd.isna(supertrend.iloc[i-1]) else lower_band.iloc[i])
            if df['close'].iloc[i] < supertrend.iloc[i]:
                in_uptrend.iloc[i] = False
                supertrend.iloc[i] = upper_band.iloc[i]
            else:
                in_uptrend.iloc[i] = True
        else:
            supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i-1] if not pd.isna(supertrend.iloc[i-1]) else upper_band.iloc[i])
            if df['close'].iloc[i] > supertrend.iloc[i]:
                in_uptrend.iloc[i] = True
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                in_uptrend.iloc[i] = False
    supertrend.iloc[0] = lower_band.iloc[0]
    in_uptrend.iloc[0] = df['close'].iloc[0] > supertrend.iloc[0]
    return supertrend, in_uptrend

def trade_file(pair, tf):
    path_dir = os.path.join(CFG['trade_root'], pair.lower(), tf)
    os.makedirs(path_dir, exist_ok=True)
    return os.path.join(path_dir, f'{pair.lower()}_trades_{tf}.csv')

def log_trade(pair, row, tf):
    path = trade_file(pair, tf)
    write_header = not os.path.isfile(path)
    with open(path,'a',newline='',encoding='utf-8') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['pair','date','position','entry','tp','sl','result'])
        w.writerow(row)

def update_open_trades(pair, raw_df, tf):
    path = trade_file(pair, tf)
    if not os.path.isfile(path):
        return
    df = pd.read_csv(path)
    if 'result' not in df.columns:
        return
    if df['result'].notna().all():
        return

    raw_df = raw_df.sort_values('open_time')
    changed = False

    for idx, row in df[df['result'].isna() | (df['result']=='')].iterrows():
        open_dt = datetime.strptime(row['date'], '%Y-%m-%d %H:%M').replace(tzinfo=local_tz)
        open_ms = int(open_dt.astimezone(timezone.utc).timestamp() * 1000)
        future = raw_df[raw_df['open_time'] >= open_ms].head(CFG['future_size'])

        if future.empty:
            continue

        pos, tp, sl = row['position'], float(row['tp']), float(row['sl'])
        outcome = ''
        for _, bar in future.iterrows():
            hi, lo = bar['high'], bar['low']
            if pos == 'long':
                if lo <= sl:
                    outcome = 'SL'
                    break
                elif hi >= tp:
                    outcome = 'TP'
                    break
            else:
                if hi >= sl:
                    outcome = 'SL'
                    break
                elif lo <= tp:
                    outcome = 'TP'
                    break
        if not outcome and len(future) >= CFG['future_size']:
            outcome = 'HOLD'

        if outcome:
            df.at[idx, 'result'] = outcome
            changed = True

    if changed:
        df.to_csv(path, index=False)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SignalDataset(Dataset):
    def __init__(self, trades_path, raw_df, tf):
        self.tf = tf
        self.df_trades = pd.read_csv(trades_path)
        self.df_trades['date'] = pd.to_datetime(self.df_trades['date'])
        self.df_raw = raw_df
        self.window_map = []
        self.seqs = []
        tf_ms = tf_map[tf] * 1000
        for i, row in self.df_trades.iterrows():
            entry_time = row['date']
            entry_ms = int(entry_time.timestamp() * 1000)
            signal_open_ms = entry_ms - tf_ms
            past = self.df_raw[self.df_raw['open_time'] <= signal_open_ms].tail(CFG['candles_pull'])
            if len(past) < CFG['context_size'] + 1:
                continue
            ha = heikin(past)
            ha_norm = get_rolling_normalized(ha, CFG['context_size'])
            if ha_norm is None or len(ha_norm) < CFG['context_size']:
                continue
            arr = ha_norm.tail(CFG['context_size']).to_numpy(np.float32)
            seq = to_tokens(arr)
            if row['result'] not in ['TP', 'SL']:
                continue
            lab = 2 if row['result'] == 'SL' else (0 if row['position'] == 'long' else 1)
            self.seqs.append(seq)
            self.window_map.append((
                i,
                lab,
                float(row['entry']),
                float(row['tp']) if row['result'] == 'TP' else float(row['sl']),
                row['result'],
                -1,
                entry_time.strftime('%Y-%m-%d %H:%M'),
                '',
                0.0,
                past.index[-1]
            ))

    def __len__(self):
        return len(self.window_map)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        end, lab, entry_price, exit_price, exit_reason, exit_idx, tin, tout, atr, raw_idx = self.window_map[idx]
        return (
            torch.tensor(seq, dtype=torch.long),
            lab, entry_price, exit_price,
            exit_reason, exit_idx, tin, tout,
            atr, raw_idx
        )

def collate_fn(batch):
    seqs, labs, entries, exits, reasons, exit_idxs, t_in, t_out, atrs, raw_idxs = [], [], [], [], [], [], [], [], [], []
    for seq, lab, entry, exit_price, exit_reason, exit_idx, tin, tout, atr, raw_idx in batch:
        seqs.append(seq)
        labs.append(lab)
        entries.append(entry)
        exits.append(exit_price)
        reasons.append(exit_reason)
        exit_idxs.append(exit_idx)
        t_in.append(tin)
        t_out.append(tout)
        atrs.append(atr)
        raw_idxs.append(raw_idx)
    return (
        torch.stack(seqs, dim=0),
        torch.tensor(labs, dtype=torch.long),
        entries, exits, reasons, exit_idxs, t_in, t_out, atrs, raw_idxs
    )

def make_loader(trades_path, raw_df, tf):
    ds = SignalDataset(trades_path, raw_df, tf)
    if len(ds) == 0:
        return None
    return DataLoader(
        ds,
        batch_size=CFG['batch'],
        shuffle=False,
        num_workers=CFG['workers'],
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )

def evaluate_all(pre_data):
    set_seed(CFG['seed'])
    os.makedirs(CFG['log_dir'], exist_ok=True)
    all_labels, all_preds = [], []
    curves = {}

    for sym_idx, sym in enumerate(CFG['pairs']):
        pair = sym + CFG['tsym']
        equity = CFG['initial_equity']
        eq_curve = [equity]
        labels, preds = [], []

        for tf in timeframes:
            trades_path = trade_file(pair, tf)
            if not os.path.exists(trades_path):
                continue
            raw_df = pre_data.get((pair, tf))
            if raw_df is None:
                continue
            loader = make_loader(trades_path, raw_df, tf)
            if loader is None:
                continue

            for batch in loader:
                seqs, batch_labs, entries, exits, reasons, exit_idxs, t_in, t_out, atrs, raw_idxs = batch
                if seqs.numel() == 0:
                    continue
                seqs = seqs.to(device)
                cls = torch.full((seqs.size(0), 1), model.cls_token_id, dtype=torch.long, device=device)
                with torch.no_grad():
                    logits = model(torch.cat([cls, seqs], dim=1))
                batch_preds = torch.argmax(logits, dim=-1).cpu().tolist()
                labels.extend(batch_labs.tolist())
                preds.extend(batch_preds)

                for i in range(len(batch_preds)):
                    if batch_preds[i] not in (0, 1):
                        continue
                    entry = entries[i]
                    exit_price = exits[i]
                    position = 'long' if batch_preds[i] == 0 else 'short'
                    gross_pct = (exit_price - entry) / entry if position == 'long' else (entry - exit_price) / entry
                    net_pct = gross_pct - 2 * CFG['fee_pct']
                    equity *= 1 + net_pct
                    eq_curve.append(equity)

        curves[sym] = eq_curve
        all_labels.extend(labels)
        all_preds.extend(preds)

    report = classification_report(all_labels, all_preds, labels=[0, 1, 2], target_names=['Long', 'Short', 'Hold'], zero_division=0)
    with open(CFG['classif_report'], 'w', encoding='utf-8') as f:
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap=LinearSegmentedColormap.from_list('wb', ['white', 'blue']),
                xticklabels=['Long', 'Short', 'Hold'], yticklabels=['Long', 'Short', 'Hold'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(CFG['confusion_mat_png'])
    plt.close()

    plt.figure(figsize=(10, 5))
    best_pair, best_end = None, -np.inf
    for i, (sym, curve) in enumerate(curves.items()):
        if not curve:
            continue
        plt.plot(curve, color=COLORS[i % len(COLORS)], label=sym)
        if len(curve) > 1 and curve[-1] > best_end:
            best_end, best_pair = curve[-1], sym
    plt.title(f'Equity Curves – best: {best_pair} ≈ ${best_end:,.0f}')
    plt.xlabel('Executed Trade #')
    plt.ylabel('Equity (USD)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(CFG['equity_curve_png'])
    plt.close()

# ------------------------------- MAIN ----------------------------------
if __name__ == '__main__':
    start_dt = datetime(CFG['start_year'], CFG['start_month'], CFG['start_day'], tzinfo=timezone.utc)
    end_dt = datetime(CFG['end_year'], CFG['end_month'], CFG['end_day'], tzinfo=timezone.utc)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    print('Pre-fetching data...')
    pre_data = pre_fetch_data(start_ms, end_ms)
    print('Data fetched.')

    tf_loop = '3m'
    tf_loop_seconds = tf_map[tf_loop]
    tf_loop_ms = tf_loop_seconds * 1000

    # Start from the next full interval after start_ms
    current_ms = ((start_ms // tf_loop_ms) + 1) * tf_loop_ms

    print('Simulation started')

    while current_ms <= end_ms:
        simulated_now_ms = current_ms
        tick = datetime.fromtimestamp(current_ms / 1000, tz=timezone.utc).astimezone(local_tz)
        pbar = tqdm(total=len(CFG['pairs']) * len(timeframes),
                    desc=f'{tick:%Y-%m-%d %H:%M}', ncols=80)
        for sym in CFG['pairs']:
            pair = f'{sym}{CFG["tsym"]}'
            for tf in timeframes:
                try:
                    raw = sim_fetch_closed(sym, tf, simulated_now_ms, pre_data)
                    raw = raw[~(
                        (raw['open']==raw['high']) &
                        (raw['open']==raw['low'])  &
                        (raw['open']==raw['close'])
                    )]
                    if len(raw) < CFG['context_size'] + 1:
                        pbar.update(1); continue

                    update_open_trades(pair, raw, tf)

                    ha  = heikin(raw)
                    ha_norm = get_rolling_normalized(ha, CFG['context_size'])
                    if ha_norm is None or len(ha_norm) < CFG['context_size']:
                        pbar.update(1); continue
                    arr = ha_norm.tail(CFG['context_size']).to_numpy(np.float32)
                    tokens = to_tokens(arr)
                    ids = torch.tensor(tokens, dtype=torch.long,
                                       device=device).unsqueeze(0)
                    cls = torch.full((1,1), model.cls_token_id,
                                     dtype=torch.long, device=device)

                    with torch.no_grad():
                        probs = torch.softmax(
                            model(torch.cat([cls, ids], dim=1)), dim=1
                        )[0].cpu().numpy()

                    pred  = int(np.argmax(probs))
                    pred_conf = probs[pred]
                    other = probs[1] if pred==0 else probs[0] if pred==1 else max(probs[0], probs[1])

                    if pred not in (0,1) or pred_conf < CFG['conf_thresh'] or other >= CFG['other_conf_max']:
                        pbar.update(1); continue

                    if CFG['supertrend_enable']:
                        _, trend = calculate_supertrend(raw, CFG['supertrend_period'], CFG['supertrend_multiplier'])
                        is_uptrend = trend.iloc[-1]
                        if (pred == 0 and not is_uptrend) or (pred == 1 and is_uptrend):
                            pbar.update(1); continue

                    entry = raw.iloc[-1]['close']
                    atr_v = atr_last(ha)
                    if math.isnan(atr_v):
                        pbar.update(1); continue

                    r_sl = atr_v * CFG['sl_multiplier']
                    r_tp = atr_v * CFG['tp_multiplier']

                    potential_tp_pct = r_tp / entry
                    if potential_tp_pct < CFG['min_tp_pct']:
                        pbar.update(1); continue

                    if pred == 0:
                        sl = entry - r_sl
                        tp = entry + r_tp
                        pos = 'long'
                    else:
                        sl = entry + r_sl
                        tp = entry - r_tp
                        pos = 'short'

                    open_ms = int(raw.iloc[-1]['open_time'])
                    trade_t = datetime.fromtimestamp(
                        (open_ms // 1000 + tf_map[tf]), tz=timezone.utc
                    ).astimezone(local_tz)

                    # Check for duplicate trade log for this timestamp
                    trade_path = trade_file(pair, tf)
                    trade_date_str = trade_t.strftime('%Y-%m-%d %H:%M')
                    if os.path.exists(trade_path):
                        df_trades = pd.read_csv(trade_path)
                        if not df_trades.empty and df_trades.iloc[-1]['date'] == trade_date_str:
                            pbar.update(1)
                            continue

                    log_trade(pair, [
                        pair.lower(),
                        trade_date_str,
                        pos,
                        f'{entry:.8f}',
                        f'{tp:.8f}',
                        f'{sl:.8f}',
                        ''
                    ], tf)

                    pbar.write(f'{pair} {tf} {trade_t:%H:%M} {pos.upper()} '
                               f'@{entry:.8f}  tp={tp:.8f}  sl={sl:.8f}')

                except Exception as e:
                    pbar.write(f'{pair} {tf} ERROR: {e}')
                finally:
                    pbar.update(1)

        pbar.close()
        current_ms += tf_loop_ms

    print('Simulation completed')
    print('Evaluating results...')
    evaluate_all(pre_data)
    print(f"Done. Report → {CFG['classif_report']} | Matrix → {CFG['confusion_mat_png']} | Equity → {CFG['equity_curve_png']}")
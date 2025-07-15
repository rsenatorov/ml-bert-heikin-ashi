#!/usr/bin/env python3
# Copyright (C) 2025 Robert Senatorov
# All rights reserved.

import os, time, math, csv, requests, pandas as pd, numpy as np
import torch, torch.nn as nn
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from tqdm import tqdm
from vector_quantize_pytorch import VectorQuantize
from network.model import BertTimeSeries

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
    supertrend_enable = True,
    supertrend_period = 10,
    supertrend_multiplier = 3,
)

timeframes = ['3m', '15m', '1h', '4h']

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
def next_wakeup():
    tf_seconds = tf_map['3m']  # Loop every 3 minutes
    tf_ms = tf_seconds * 1000
    now = time.time() * 1000
    return (tf_ms - (now % tf_ms)) / 1000

def fetch_closed(sym, timeframe):
    tf_seconds = tf_map[timeframe]
    tf_ms = tf_seconds * 1000
    now_ms = int(time.time()*1000)
    end_ms = now_ms - (now_ms % tf_ms) - 1
    r = requests.get(BINANCE_URL, params=dict(
            symbol=f'{sym}{CFG["tsym"]}',
            interval=timeframe,
            limit=CFG['candles_pull'],
            endTime=end_ms), timeout=10)
    r.raise_for_status()
    cols = ['open_time','open','high','low','close']
    return pd.DataFrame([[d[0],*(map(float,d[1:5]))] for d in r.json()],
                        columns=cols)

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

# ------------------------------- LOOP ----------------------------------
print('Realtime runner started — Ctrl+C to stop')
try:
    while True:
        tick = datetime.now(tz=local_tz)
        pbar = tqdm(total=len(CFG['pairs']) * len(timeframes),
                    desc=f'{tick:%Y-%m-%d %H:%M}', ncols=80)
        for sym in CFG['pairs']:
            pair = f'{sym}{CFG["tsym"]}'
            for tf in timeframes:
                try:
                    raw = fetch_closed(sym, tf)
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
        time.sleep(next_wakeup())

except KeyboardInterrupt:
    print('\nStopped by user')
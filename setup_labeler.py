"""
Step 1: Data + setup detection + outcome labeling.

Setup logic (long; short is the mirror image):
  1. A 1h bullish FVG forms (candle3.low > candle1.high). Zone = [candle1.high, candle3.low].
  2. Price on the 5m chart comes back DOWN into that zone (touch).
  3. After the touch, a 5m bearish FVG forms (a small gap down).
  4. A 5m candle CLOSES above the top of that bearish FVG -> IFVG (inverse FVG) -> ENTRY.
  5. SL  = lowest low since the touch (minus a small buffer).
     TP  = the swing high made before the pullback (the "target" on your chart).
  6. Outcome is labeled by walking forward candle by candle: SL first = loss,
     TP first = win. If both hit in the same candle we assume SL (conservative).

Usage:
  pip install pandas numpy requests
  python setup_labeler.py --days 365            # real Binance data
  python setup_labeler.py --synthetic           # offline self-test with fake data
Output: setups.csv  (one row per setup, with features + result in R)
"""
import argparse
import time

import numpy as np
import pandas as pd
import requests

BASE_URLS = ["https://api.binance.com", "https://data-api.binance.vision"]


# ----------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------
def download_klines(symbol="BTCUSDT", interval="5m", days=365):
    end = int(time.time() * 1000)
    start = end - days * 24 * 3600 * 1000
    rows, cur = [], start
    while cur < end:
        data = None
        for base in BASE_URLS:
            try:
                r = requests.get(
                    f"{base}/api/v3/klines",
                    params={"symbol": symbol, "interval": interval,
                            "startTime": cur, "limit": 1000},
                    timeout=20,
                )
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:  # try next mirror
                err = e
        if data is None:
            raise RuntimeError(f"Binance download failed: {err}")
        if not data:
            break
        rows += data
        cur = data[-1][0] + 1
        time.sleep(0.15)  # be polite to the rate limit
    df = pd.DataFrame(rows, columns=[
        "t", "open", "high", "low", "close", "volume", "ct", "qv", "n", "tb", "tq", "x"])
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.set_index("t")[["open", "high", "low", "close", "volume"]].astype(float)
    return df[~df.index.duplicated()]


def synthetic_klines(n=60000, seed=1):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.0008, n)
    close = 60000 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.0006, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": 1.0}, index=idx)


# ----------------------------------------------------------------------------
# 2. DETECTION + LABELING
# ----------------------------------------------------------------------------
def label_trade(entry, sl, tp, h, l, c, start, horizon, fee_r):
    """Walk forward from candle `start`. Returns (R result, outcome, bars)."""
    risk = entry - sl
    rr = (tp - entry) / risk
    end = min(start + horizon, len(h))
    for k in range(start, end):
        if l[k] <= sl:                       # SL checked first = conservative
            return -1.0 - fee_r, "loss", k - start + 1
        if h[k] >= tp:
            return rr - fee_r, "win", k - start + 1
    if end <= start:
        return np.nan, "open", 0
    r_mtm = (c[end - 1] - entry) / risk      # timed out: mark to market
    return r_mtm - fee_r, "timeout", end - start


def find_setups(df5, side="long", htf="1h", lookback=48, max_wait=60, max_age_htf=72,
                horizon=288, buffer=0.0002, min_rr=1.0, fee=0.0005, ema_len=288, live=False):
    """Detect HTF FVG -> LTF IFVG setups. df5 = the LTF candles (any timeframe), htf = pandas rule like "1h","4h","1D".
    side='short' runs on a mirrored price series. max_age_htf = how many HTF candles a zone stays valid."""
    htf_td = pd.Timedelta(htf)
    sgn = 1 if side == "long" else -1
    if sgn == 1:
        d = df5[["open", "high", "low", "close"]].copy()
    else:  # mirror: highs become -lows, so a short setup looks like a long one
        d = pd.DataFrame({"open": -df5["open"], "high": -df5["low"],
                          "low": -df5["high"], "close": -df5["close"]})
    idx = d.index
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    ema = d["close"].ewm(span=ema_len, adjust=False).mean().values

    # ---- 1h FVGs (known only once the 3rd 1h candle has closed)
    h1 = d.resample(htf).agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    zones = []
    H, L = h1["high"].values, h1["low"].values
    for i in range(2, len(h1)):
        if L[i] > H[i - 2]:
            zones.append({"avail": h1.index[i] + htf_td,
                          "lo": H[i - 2], "hi": L[i]})
    zones.sort(key=lambda z: z["avail"])

    out, active, zp = [], [], 0
    for j in range(max(lookback, 3), len(d)):
        t = idx[j]
        while zp < len(zones) and zones[zp]["avail"] <= t:
            z = zones[zp]
            z.update(touch=None, fvgs=[], born=t)
            active.append(z)
            zp += 1

        survivors = []
        for z in active:
            if c[j] < z["lo"] or (t - z["born"]) > max_age_htf * htf_td:
                continue                                   # zone broken / too old
            if z["touch"] is None:
                if l[j] <= z["hi"]:                        # price tapped the 1h FVG
                    z["touch"] = j
                    z["pre_high"] = h[j - lookback:j].max()
                else:
                    survivors.append(z)
                    continue
            if j - z["touch"] > max_wait:
                continue                                   # no IFVG in time -> drop

            # IFVG: a close above the top of an earlier 5m bearish FVG
            fired = False
            for f in z["fvgs"]:
                if j > f["k"] and c[j] > f["hi"]:
                    entry = c[j]
                    sl = l[z["touch"]:j + 1].min() - buffer * abs(entry)
                    tp = z["pre_high"]
                    risk = entry - sl
                    if risk > 0 and (tp - entry) >= min_rr * risk:
                        fee_r = 2 * fee * abs(entry) / risk
                        if live:   # scanner mode: no future candles exist, so no outcome yet
                            res, outcome, bars = np.nan, "live", 0
                        else:
                            res, outcome, bars = label_trade(entry, sl, tp, h, l, c, j + 1, horizon, fee_r)
                        if outcome != "open":
                            out.append({
                                "time": idx[j], "side": side,
                                "entry": sgn * entry, "sl": sgn * sl, "tp": sgn * tp,
                                "fvg_lo": sgn * (z["hi"] if sgn == -1 else z["lo"]),
                                "fvg_hi": sgn * (z["lo"] if sgn == -1 else z["hi"]),
                                # features (all known at entry time)
                                "fvg_size_pct": (z["hi"] - z["lo"]) / abs(entry) * 100,
                                "risk_pct": risk / abs(entry) * 100,
                                "rr_planned": (tp - entry) / risk,
                                "wait_bars": j - z["touch"],
                                "trend_align": (entry - ema[j]) / abs(entry) * 100,
                                "hour_utc": t.hour, "dow": t.dayofweek,
                                "killzone": int(7 <= t.hour < 10 or 12 <= t.hour < 15),
                                # label
                                "result_R": res, "outcome": outcome, "bars_to_exit": bars,
                            })
                    fired = True
                    break
            if fired:
                continue                                   # one trade per zone touch

            # record a new 5m bearish FVG (gap down) formed by candles j-2..j
            if h[j] < l[j - 2] and (j - 2) >= z["touch"] - 3:
                z["fvgs"].append({"k": j, "hi": l[j - 2], "lo": h[j]})
            survivors.append(z)
        active = survivors

    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# 3. REPORT
# ----------------------------------------------------------------------------
def report(df):
    if df.empty:
        print("No setups found.")
        return
    n = len(df)
    wins = (df["outcome"] == "win").sum()
    print(f"\nSetups: {n}  | Win rate: {wins / n:.1%}  | Avg R (after fees): {df['result_R'].mean():.3f}"
          f"  | Total R: {df['result_R'].sum():.1f}")
    for side, g in df.groupby("side"):
        print(f"  {side:5s}: n={len(g):4d}  win={(g['outcome'] == 'win').mean():.1%}  avgR={g['result_R'].mean():.3f}")
    kz = df.groupby("killzone")["result_R"].agg(["count", "mean"])
    print("\nBy killzone (1 = London/NY window, UTC approx):\n", kz.round(3))
    # simple out-of-sample sanity check: first 70% vs last 30% by time
    df = df.sort_values("time")
    cut = int(n * 0.7)
    print(f"\nFirst 70% avgR: {df['result_R'].iloc[:cut].mean():.3f} | "
          f"Last 30% avgR: {df['result_R'].iloc[cut:].mean():.3f}")
    print("(If the last 30% is much worse than the first 70%, the 'edge' is probably noise.)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--synthetic", action="store_true", help="offline test with fake data")
    ap.add_argument("--out", default="setups.csv")
    a = ap.parse_args()

    df5 = synthetic_klines() if a.synthetic else download_klines(a.symbol, "5m", a.days)
    print(f"Loaded {len(df5)} 5m candles: {df5.index[0]} -> {df5.index[-1]}")
    res = pd.concat([find_setups(df5, "long"), find_setups(df5, "short")], ignore_index=True)
    res = res.sort_values("time").reset_index(drop=True) if not res.empty else res
    res.to_csv(a.out, index=False)
    print(f"Saved {len(res)} labeled setups -> {a.out}")
    report(res)
"""
msnr_v3.py  -  Multi-Timeframe MSNR (Fractal version)

HTF: 1h, 4h, Daily
LTF: 5m, 15m, 30m
Conflict rule: Best R:R (highest R3)
Filters: Longs → Q2+Q4 | Shorts → Q3
"""

import argparse
import sys
import numpy as np
import pandas as pd
from itertools import product


# ----------------------------------------------------------------- data
def load_csv(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    tcol = next((cols[c] for c in ("time", "timestamp", "open_time", "datetime", "date") if c in cols), None)
    if tcol is None:
        raise SystemExit("no time column found")
    t = df[tcol]
    if pd.api.types.is_numeric_dtype(t):
        unit = "ms" if float(t.iloc[0]) > 1e11 else "s"
        t = pd.to_datetime(t, unit=unit, utc=True)
    else:
        t = pd.to_datetime(t, utc=True)
    out = pd.DataFrame({k: df[cols[k]].astype(float).values for k in ("open", "high", "low", "close")},
                       index=pd.DatetimeIndex(t))
    return out[~out.index.duplicated()].sort_index()


def synthetic(n=200_000, seed=1):
    rng = np.random.default_rng(seed)
    r = rng.standard_t(4, n) * 0.0012
    close = 30000 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.0008, n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0008, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def shuffle_bars(df, seed=0):
    rng = np.random.default_rng(seed)
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    prev_c = np.r_[c[0], c[:-1]]
    rel = np.column_stack([o / prev_c, h / o, l / o, c / o])[1:]
    rel = rel[rng.permutation(len(rel))]
    n = len(df)
    no, nh, nl, nc = (np.empty(n) for _ in range(4))
    nc[0], no[0], nh[0], nl[0] = c[0], o[0], h[0], l[0]
    for i in range(1, n):
        ro, rh, rl, rc = rel[i - 1]
        no[i] = nc[i - 1] * ro
        nh[i] = no[i] * rh
        nl[i] = no[i] * rl
        nc[i] = no[i] * rc
    return pd.DataFrame({"open": no, "high": nh, "low": nl, "close": nc}, index=df.index)


def resample(df, rule):
    return df.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


# ----------------------------------------------------------------- HTF levels
def htf_signals(h, k=3, max_age=400):
    o, hi, lo, c = (h[x].values for x in ("open", "high", "low", "close"))
    n = len(c)
    cand = {}
    for i in range(k, n - 1 - k):
        win = c[i - k:i + k + 1]
        if c[i] > o[i] and c[i + 1] < o[i + 1] and c[i] == win.max():
            lvl = max(c[i], o[i + 1])
            if not (hi[i + 2:i + k + 1] >= lvl).any():
                cand.setdefault(i + k + 1, []).append(("S", lvl, i))
        elif c[i] < o[i] and c[i + 1] > o[i + 1] and c[i] == win.min():
            lvl = min(c[i], o[i + 1])
            if not (lo[i + 2:i + k + 1] <= lvl).any():
                cand.setdefault(i + k + 1, []).append(("L", lvl, i))
    active, sigs = [], []
    for j in range(n):
        active += cand.get(j, [])
        keep, done = [], set()
        body_hi, body_lo = max(o[j], c[j]), min(o[j], c[j])
        for side, lvl, born in active:
            if j - born > max_age:
                continue
            if side == "S" and hi[j] >= lvl:
                if body_hi < lvl and "S" not in done:
                    sigs.append((j, "S", lvl, hi[j]))
                    done.add("S")
                continue
            if side == "L" and lo[j] <= lvl:
                if body_lo > lvl and "L" not in done:
                    sigs.append((j, "L", lvl, lo[j]))
                    done.add("L")
                continue
            keep.append((side, lvl, born))
        active = keep
    return sigs


# ----------------------------------------------------------------- LTF entry + sim
def find_entry(side, start, extreme, hi, lo, cl, kp=3, window=72, min_risk=0.001, max_risk=0.02):
    n = len(cl)
    ext, piv = extreme, None
    for m in range(start, min(start + window, n - 1)):
        p = m - kp
        if side == "S":
            ext = max(ext, hi[m])
            if p >= start and lo[p] == lo[max(0, p - kp):m + 1].min():
                piv = lo[p]
            if piv is not None and cl[m] < piv:
                risk = ext - cl[m]
                if min_risk <= risk / cl[m] <= max_risk:
                    return m, cl[m], ext, risk
                piv = None
        else:
            ext = min(ext, lo[m])
            if p >= start and hi[p] == hi[max(0, p - kp):m + 1].max():
                piv = hi[p]
            if piv is not None and cl[m] > piv:
                risk = cl[m] - ext
                if min_risk <= risk / cl[m] <= max_risk:
                    return m, cl[m], ext, risk
                piv = None
    return None


def simulate(side, m, entry, sl, risk, R, hi, lo, cl, fee, maxhold=288):
    n = len(cl)
    tp = entry - R * risk if side == "S" else entry + R * risk
    end = min(n, m + 1 + maxhold)
    res = None
    for t in range(m + 1, end):
        if side == "S":
            if hi[t] >= sl:
                res = -1.0
                break
            if lo[t] <= tp:
                res = float(R)
                break
        else:
            if lo[t] <= sl:
                res = -1.0
                break
            if hi[t] >= tp:
                res = float(R)
                break
    if res is None:
        last = cl[end - 1]
        res = (entry - last) / risk if side == "S" else (last - entry) / risk
    return res - 2 * fee * entry / risk


# ----------------------------------------------------------------- Multi-TF run
def run_multi(ltf_base, htf_rules=("1h", "1D"), ltf_rules=("5min", "15min", "30min"),
              k=3, fee=0.0, targets=(1, 2, 3), tz=0.0, window=72, kp=3):
    
    all_trades = []
    
    for htf_rule, ltf_rule in product(htf_rules, ltf_rules):
        ltf = resample(ltf_base, ltf_rule)
        h = resample(ltf_base, htf_rule)
        
        if len(ltf) < 100 or len(h) < 50:
            continue
            
        hi, lo, cl = ltf["high"].values, ltf["low"].values, ltf["close"].values
        times = ltf.index.values
        htf_delta = pd.Timedelta(htf_rule).to_timedelta64()
        sigs = htf_signals(h, k=k)
        
        for j, side, lvl, ext in sigs:
            end_t = h.index[j].to_datetime64() + htf_delta
            start = int(np.searchsorted(times, end_t))
            if start >= len(cl) - 2:
                continue
            e = find_entry(side, start, ext, hi, lo, cl, kp=kp, window=window)
            if e is None:
                continue
            m, entry, sl, risk = e
            ts = ltf.index[m] + pd.Timedelta(hours=tz)
            q = ((ts.hour * 60 + ts.minute) % 360) // 90 + 1
            
            row = {
                "side": side,
                "q": int(q),
                "entry_time": ts,
                "htf": htf_rule,
                "ltf": ltf_rule,
                "risk": risk,
                "entry": entry
            }
            for R in targets:
                row[f"R{R}"] = simulate(side, m, entry, sl, risk, R, hi, lo, cl, fee)
            all_trades.append(row)
    
    if not all_trades:
        return pd.DataFrame(), 0
    
    df = pd.DataFrame(all_trades)
    
    # ===== Conflict resolution: Best R:R (highest R3) =====
    df = df.sort_values("entry_time").reset_index(drop=True)
    selected = []
    last_time = None
    last_side = None
    
    for _, row in df.iterrows():
        # If too close to previous selected trade (within 2 hours) and same side → keep better R3
        if last_time is not None and (row["entry_time"] - last_time).total_seconds() < 7200 and row["side"] == last_side:
            if row["R3"] > selected[-1]["R3"]:
                selected[-1] = row
            continue
        selected.append(row)
        last_time = row["entry_time"]
        last_side = row["side"]
    
    df = pd.DataFrame(selected)
    return df, len(df)


def table(df, targets, title):
    print(f"\n{title}")
    print(f"{'target':>8}{'n':>8}{'win%':>8}{'avgR':>9}{'sumR':>10}")
    for R in targets:
        x = df[f"R{R}"]
        if len(x) == 0:
            print(f"{R:>7}R{0:>8}{'nan':>8}{'nan':>9}{0.0:>10.1f}")
            continue
        print(f"{R:>7}R{len(x):>8}{(x > 0).mean() * 100:>8.1f}{x.mean():>9.3f}{x.sum():>10.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--fee", type=float, default=0.0)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--tz", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ltf_base = synthetic() if a.synthetic or not a.csv else load_csv(a.csv)
    if a.shuffle:
        ltf_base = shuffle_bars(ltf_base, a.seed)
    
    targets = (1, 2, 3)
    df, nsig = run_multi(ltf_base, fee=a.fee, targets=targets, tz=a.tz, k=a.k)

    # ===== PERMANENT FILTERS =====
    # Longs: 1h + Daily  |  Q2 + Q4
    # Shorts: Only Daily HTF  |  Q3
    df = df[
        ((df.side == "L") & (df.q.isin([2, 4]))) |
        ((df.side == "S") & (df.htf == "1D") & (df.q == 3))
    ]
    # =============================

    print(
        f"bars={len(ltf_base)}  signals≈{nsig}  trades={len(df)}  "
        f"{'SHUFFLED' if a.shuffle else 'REAL'}  fee/side={a.fee}  "
        f"[MULTI-TF | Longs Q2/Q4 + Shorts Daily Q3]"
    )

    if df.empty:
        return
    
    table(df, targets, "ALL")
    for s, name in (("L", "LONG (V-level)"), ("S", "SHORT (A-level)")):
        table(df[df.side == s], targets, name)
    
    print("\nBy QT quarter (R=2 target):")
    print(f"{'Q':>3}{'n':>8}{'avgR':>9}")
    for q in (1, 2, 3, 4):
        x = df[df.q == q]["R2"]
        if len(x):
            print(f"{q:>3}{len(x):>8}{x.mean():>9.3f}")
    
    print("\nBy HTF:")
    for htf in ("1h", "4h", "1D"):
        sub = df[df.htf == htf]
        if len(sub):
            print(f"  {htf}: n={len(sub)}  avgR2={sub['R2'].mean():.3f}  avgR3={sub['R3'].mean():.3f}")


class Print:
    """Simple console printer with useful row and table output helpers."""

    def __init__(self, stream=None):
        self.stream = stream if stream is not None else sys.stdout
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def flush(self):
        if self._closed:
            return
        if hasattr(self.stream, "flush"):
            self.stream.flush()

    def write(self, message="", *, end="\n"):
        if self._closed:
            raise ValueError("Print is closed")
        self.stream.write(str(message))
        if end:
            self.stream.write(end)
        self.flush()

    def write_row(self, row):
        self.write(" | ".join(str(v) for v in row))

    def write_table(self, rows, headers=None):
        if headers is not None:
            self.write_row(headers)
        for row in rows:
            self.write_row(row)

    def close(self):
        if self._closed:
            return
        self.flush()
        if hasattr(self.stream, "close"):
            self.stream.close()
        self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
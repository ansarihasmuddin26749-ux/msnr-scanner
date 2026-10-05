"""
msnr_v3.py  -  MSNR (Alchemist PDF + Scribd guide) as a strict, testable checklist.

Checklist (no discretion, everything is a number):
 1. HTF level (default 1h), built from the CLOSE-LINE (body only, wicks ignored):
      A-level (resistance): bullish candle i, next candle bearish, close[i] is the
                            highest close in +-k bars.  level = max(close[i], open[i+1])
      V-level (support)   : mirror image.
 2. Fresh: no wick/body touched the level since it formed.
 3. First touch must be a WICK (candle body stays on the rejecting side).
    If the body crosses the level it becomes unfresh and is dropped.
 4. After the HTF touch candle CLOSES, switch to LTF (your data, e.g. 5m) and wait for a
    market-structure shift: LTF close beyond the last confirmed LTF pivot that formed
    after the touch.
 5. Entry = close of the shift candle.  SL = extreme since the touch (the swept liquidity).
    TP = R x risk.  Same-bar SL+TP counts as a loss (conservative).
 6. Diagnostic only: results are grouped by direction and by QT quarter (90-min blocks of
    the 6h session, PDF page "QUARTERLY"), so you can SEE whether Q3 matters, not assume it.

NOT coded (not objectively defined in the sources): IDM sweep, engulfing-OB quality, SMT,
fibo storyline entries, "H1 POI" beyond the level itself.

Usage:
  python msnr_v3.py --csv btc5m.csv --fee 0.0002
  python msnr_v3.py --csv btc5m.csv --fee 0.0002 --shuffle      # baseline: same candles, random order
  python msnr_v3.py --synthetic                                  # smoke test, no data needed

CSV columns: time (ms/s epoch or datetime), open, high, low, close   (5m recommended)
"""
import argparse
import numpy as np
import pandas as pd


# ----------------------------------------------------------------- data
def load_csv(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    tcol = next((cols[c] for c in ("time", "timestamp", "open_time", "datetime", "date") if c in cols), None)
    if tcol is None:
        raise SystemExit("no time column found")
    t = df[tcol]
    if np.issubdtype(t.dtype, np.number):
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
    """Baseline: keep candle shapes, destroy sequence/structure."""
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
    """Return list of (htf_bar_index, side, level, touch_extreme). side: 'S' (A-level) or 'L' (V-level)."""
    o, hi, lo, c = (h[x].values for x in ("open", "high", "low", "close"))
    n = len(c)
    cand = {}
    for i in range(k, n - 1 - k):
        win = c[i - k:i + k + 1]
        if c[i] > o[i] and c[i + 1] < o[i + 1] and c[i] == win.max():
            lvl = max(c[i], o[i + 1])
            if not (hi[i + 2:i + k + 1] >= lvl).any():          # still fresh at confirmation
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
                if body_hi < lvl and "S" not in done:            # wick-only first touch
                    sigs.append((j, "S", lvl, hi[j]))
                    done.add("S")
                continue                                          # touched -> no longer fresh
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


# ----------------------------------------------------------------- run
def run(ltf, htf_rule="1h", k=3, fee=0.0, targets=(1, 2, 3), tz=0.0, window=72, kp=3):
    h = resample(ltf, htf_rule)
    hi, lo, cl = ltf["high"].values, ltf["low"].values, ltf["close"].values
    times = ltf.index.values
    htf_delta = pd.Timedelta(htf_rule).to_timedelta64()
    sigs = htf_signals(h, k=k)
    rows = []
    for j, side, lvl, ext in sigs:
        end_t = h.index[j].to_datetime64() + htf_delta
        start = int(np.searchsorted(times, end_t))             # first LTF bar after HTF close (no lookahead)
        if start >= len(cl) - 2:
            continue
        e = find_entry(side, start, ext, hi, lo, cl, kp=kp, window=window)
        if e is None:
            continue
        m, entry, sl, risk = e
        ts = ltf.index[m] + pd.Timedelta(hours=tz)
        q = ((ts.hour * 60 + ts.minute) % 360) // 90 + 1
        row = {"side": side, "q": int(q)}
        for R in targets:
            row[f"R{R}"] = simulate(side, m, entry, sl, risk, R, hi, lo, cl, fee)
        rows.append(row)
    return pd.DataFrame(rows), len(sigs)


def table(df, targets, title):
    print(f"\n{title}")
    print(f"{'target':>8}{'n':>8}{'win%':>8}{'avgR':>9}{'sumR':>10}")
    for R in targets:
        x = df[f"R{R}"]
        print(f"{R:>7}R{len(x):>8}{(x > 0).mean() * 100:>8.1f}{x.mean():>9.3f}{x.sum():>10.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--htf", default="1h")
    ap.add_argument("--k", type=int, default=3, help="pivot width on HTF close-line")
    ap.add_argument("--fee", type=float, default=0.0, help="per side, fraction (0.0002 = 0.02)")
    ap.add_argument("--shuffle", action="store_true", help="baseline on shuffled candles")
    ap.add_argument("--tz", type=float, default=0.0, help="hours added to UTC for the QT clock")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ltf = synthetic() if a.synthetic or not a.csv else load_csv(a.csv)
    if a.shuffle:
        ltf = shuffle_bars(ltf, a.seed)
    targets = (1, 2, 3)
    df, nsig = run(ltf, a.htf, a.k, a.fee, targets, a.tz)
    print(f"bars={len(ltf)}  HTF touch signals={nsig}  trades={len(df)}  "
          f"{'SHUFFLED' if a.shuffle else 'REAL'}  fee/side={a.fee}")
    if df.empty:
        return
    table(df, targets, "ALL")
    for s, name in (("L", "LONG (V-level)"), ("S", "SHORT (A-level)")):
        table(df[df.side == s], targets, name)
    print("\nBy QT quarter (R=2 target)  - diagnostic, do NOT cherry-pick:")
    print(f"{'Q':>3}{'n':>8}{'avgR':>9}")
    for q in (1, 2, 3, 4):
        x = df[df.q == q]["R2"]
        if len(x):
            print(f"{q:>3}{len(x):>8}{x.mean():>9.3f}")


if __name__ == "__main__":
    main()
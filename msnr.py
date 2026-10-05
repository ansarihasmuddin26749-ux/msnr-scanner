"""
msnr.py - Malaysian SnR ("Alchemist SnR") levels + a first-touch rejection model.

Levels (all taken from the LINE chart = closing prices):
  A-level   : a peak of the close-line   (close[i-1] > close[i-2] and close[i-1] > close[i])
  V-level   : a valley of the close-line (close[i-1] < close[i-2] and close[i-1] < close[i])
  Gap level : two consecutive candles of the SAME colour with a gap between the first close and the
              second open. Level price = close of the first candle.   <-- assumption, check vs. your course
Fresh / unfresh:
  A level is FRESH until a candle WICK touches it. A candle BODY crossing it makes it fresh again.

Model (long side; short = mirror image through the same code):
  * A fresh level lies below price (previous close above it).
  * A candle wicks down into the level but its BODY stays above it  -> first-touch rejection.
  * Entry = close of that candle. SL = low of that candle (minus a tiny buffer).
  * Targets scored: 1R and 2R. Fees are subtracted in R.
Anything beyond this (storyline, engulfing/flag-limit confirmation, QM...) is NOT included until the
course rules are confirmed.
"""
import numpy as np
import pandas as pd

from smc import session_of

KINDS = {0: "A", 1: "V", 2: "gap"}


def _label_market(E, sl, tps, h, l, c, start, horizon):
    N = len(h)
    if start + horizon > N:
        return None
    risk = E - sl
    res = []
    for tp in tps:
        out = None
        for k in range(start, start + horizon):
            if l[k] <= sl:
                out = (-1.0, "loss")
                break
            if h[k] >= tp:
                out = ((tp - E) / risk, "win")
                break
        if out is None:
            out = ((c[start + horizon - 1] - E) / risk, "timeout")
        res.append(out)
    return res


def find_msnr_setups(df, side="long", cap=1500, horizon=288, buffer=0.0002, fee=0.0005,
                     dedupe_tol=1e-4, ema_len=288, live=False):
    sgn = 1 if side == "long" else -1
    if sgn == 1:
        d = df[["open", "high", "low", "close"]].copy()
    else:
        d = pd.DataFrame({"open": -df["open"], "high": -df["low"], "low": -df["high"], "close": -df["close"]})
    idx = d.index
    o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
    N = len(d)
    ema = d["close"].ewm(span=ema_len, adjust=False).mean().values

    P = np.zeros(cap)                 # level prices
    valid = np.zeros(cap, bool)
    fresh = np.zeros(cap, bool)
    born = np.zeros(cap, int)
    kind = np.zeros(cap, int)
    ptr = 0
    rows = []

    def add(price, k, j):
        nonlocal ptr
        if valid.any() and np.any(valid & (np.abs(P - price) <= dedupe_tol * abs(price))):
            return
        P[ptr], valid[ptr], fresh[ptr], born[ptr], kind[ptr] = price, True, True, j, k
        ptr = (ptr + 1) % cap

    for j in range(3, N):
        lo, hi = l[j], h[j]
        b_lo, b_hi = min(o[j], c[j]), max(o[j], c[j])

        # 1) first-touch rejection of a fresh level that sits below price
        m = valid & fresh & (P >= lo) & (P < b_lo) & (c[j - 1] > P)
        if m.any():
            cand = np.where(m)[0]
            q = cand[np.argmax(P[cand])]                 # level nearest to the candle body
            E = c[j]
            sl = lo - buffer * abs(E)
            risk = E - sl
            if risk > 0:
                kk = KINDS[kind[q]]
                if sgn == -1 and kk != "gap":
                    kk = "V" if kk == "A" else "A"       # mirror back to original chart
                ses = session_of(idx[j])
                row = {
                    "time": idx[j], "side": side, "level_kind": kk, "level_age": j - born[q],
                    "session": ses, "risk_pct": risk / abs(E) * 100,
                    "wick_pct": (b_lo - lo) / abs(E) * 100,
                    "trend_align": (c[j] - ema[j]) / abs(c[j]) * 100,
                    "entry": sgn * E, "sl": sgn * sl, "level": sgn * P[q],
                }
                if live:
                    row.update({"outcome_1R": "pending", "result_1R": np.nan, "result_2R": np.nan})
                    rows.append(row)
                else:
                    res = _label_market(E, sl, [E + risk, E + 2 * risk], h, l, c, j + 1, horizon)
                    if res is not None:
                        fee_r = 2 * fee * abs(E) / risk
                        row.update({"outcome_1R": res[0][1], "result_1R": res[0][0] - fee_r,
                                    "outcome_2R": res[1][1], "result_2R": res[1][0] - fee_r})
                        rows.append(row)

        # 2) new levels confirmed by this candle
        if c[j - 1] > c[j - 2] and c[j - 1] > c[j]:
            add(c[j - 1], 0, j)                           # A-level
        elif c[j - 1] < c[j - 2] and c[j - 1] < c[j]:
            add(c[j - 1], 1, j)                           # V-level
        same = (c[j - 1] > o[j - 1]) == (c[j] > o[j])
        if same and ((c[j] > o[j] and o[j] > c[j - 1]) or (c[j] < o[j] and o[j] < c[j - 1])):
            add(c[j - 1], 2, j)                           # gap level

        # 3) freshness update (wick touch -> unfresh, body cross -> fresh again)
        touched = valid & (P >= lo) & (P <= hi)
        cross = valid & (P > b_lo) & (P < b_hi)
        fresh[cross] = True
        fresh[touched & ~cross] = False

    return pd.DataFrame(rows)
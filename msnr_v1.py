#!/usr/bin/env python3
"""
msnr_v1.py  -  MSNR v1 "backbone" backtest (from the Alchemist MSNR Overview PDF)

Flow coded (SETUP BUY / SETUP SELL pages of the PDF):
  1. H1 POI (key level): Classic A (peak of close-line), Classic V (valley of close-line),
     plus flipped levels SBR / RBS (broken level that turns into resistance/support).
     A level is "fresh" until a wick touches it.
  2. Liquidity sweep: a 5m wick pierces the fresh level and a 5m candle closes back
     on the original side within --rej-bars (rejection).  No rejection = breakout -> level flips.
  3. M5 market-structure shift (MSS/CHOCH): after the sweep, a 5m close breaks the last
     confirmed M5 swing low (for shorts) / swing high (for longs).
  4. Entry (3 variants tested side by side):
        mss          : market entry at the close of the MSS candle
        retest       : limit order at the broken M5 level (SBR / RBS), waits max --retest-bars
        retest+idm   : same, but only if an inducement (pullback pivot) exists before the touch
  5. SL beyond the sweep wick (+ small buffer), TP at 2R / 3R / 5R, max hold --hold-bars.

NOT coded yet (next step, one at a time): Asia-range sweep, Quarterly Theory (QT) cycle,
SMT divergence, trendline key, fib circle / storyline, QML, engulfing OB.

Conservative rules: if SL and TP are both inside one candle, SL is counted first.
On a limit-fill candle only the SL is checked (TP is counted from the next candle).
One position at a time (overlapping setups are skipped).

Run:
    python msnr_v1.py --data path\\to\\btc_5m.csv
    python msnr_v1.py --synthetic        # random-walk data = baseline for comparison
"""
import argparse
import glob
import sys
import time

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- data


def load_5m(path):
    if path is None:
        cands = []
        for pat in ("*5m*.csv", "*5m*.parquet", "**/*5m*.csv", "**/*5m*.parquet"):
            cands += glob.glob(pat, recursive=True)
        cands = sorted(set(cands))
        if not cands:
            sys.exit("5m data file nahi mila. Chalao: python msnr_v1.py --data <5m csv/parquet ka path>")
        path = cands[0]
        print(f"data file: {path}" + (f"  (aur {len(cands) - 1} mile)" if len(cands) > 1 else ""))
    df = pd.read_parquet(path) if path.lower().endswith(".parquet") else pd.read_csv(path)
    cols = {str(c).lower().strip(): c for c in df.columns}
    tcol = next((cols[c] for c in ("open_time", "timestamp", "time", "datetime", "date", "open time", "ts")
                 if c in cols), None)
    if tcol is None:
        if isinstance(df.index, pd.DatetimeIndex):
            tser = pd.Series(df.index, index=range(len(df)))
        else:
            tcol = df.columns[0]
            tser = df[tcol]
    else:
        tser = df[tcol]
    if pd.api.types.is_numeric_dtype(tser):
        v = float(tser.iloc[0])
        unit = "ms" if v > 1e11 else "s"
        if v > 1e14:
            unit = "us"
        t = pd.to_datetime(tser, unit=unit, utc=True)
    else:
        t = pd.to_datetime(tser, utc=True)
    need = {}
    for k in ("open", "high", "low", "close"):
        if k not in cols:
            sys.exit(f"column '{k}' nahi mila. Columns: {list(df.columns)}")
        need[k] = df[cols[k]].astype(float).to_numpy()
    out = pd.DataFrame(need, index=pd.DatetimeIndex(t))
    out = out[~out.index.duplicated()].sort_index()
    return out


def synthetic(n_bars=420000, seed=7):
    """Random walk with volatility clustering and wicks = 'zero edge' baseline."""
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n_bars)
    lv = np.zeros(n_bars)
    for i in range(1, n_bars):
        lv[i] = 0.995 * lv[i - 1] + 0.1 * e[i]
    sigma = 0.0007 * np.exp(lv)
    close = 20000 * np.exp(np.cumsum(rng.normal(size=n_bars) * sigma))
    open_ = np.r_[close[0], close[:-1]]
    up = np.abs(rng.normal(size=n_bars)) * sigma * 0.6 * close
    dn = np.abs(rng.normal(size=n_bars)) * sigma * 0.6 * close
    high = np.maximum(open_, close) + up
    low = np.minimum(open_, close) - dn
    idx = pd.date_range("2022-10-05 18:30", periods=n_bars, freq="5min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


# ----------------------------------------------------------------------------- helpers


def pivot_high_mask(x, k):
    s = pd.Series(x)
    left = s.shift(1).rolling(k).max()
    right = s[::-1].shift(1).rolling(k).max()[::-1]
    return ((s >= left) & (s > right)).fillna(False).to_numpy()


def last_confirmed(price, mask, k):
    """price of the latest pivot already confirmed (pivot idx + k) at each bar."""
    n = len(price)
    out = np.full(n, np.nan)
    idx = np.nonzero(mask)[0]
    conf = idx + k
    ok = conf < n
    out[conf[ok]] = price[idx[ok]]
    return pd.Series(out).ffill().to_numpy()


def first_cmp(a, start, end, thr, op):
    n = len(a) if end is None else min(end, len(a))
    i, step = start, 2048
    while i < n:
        j = min(n, i + step)
        seg = op(a[i:j], thr)
        if seg.any():
            return i + int(seg.argmax())
        i, step = j, min(step * 2, 1 << 18)
    return -1


def make_spaces(h, l, c, k5):
    """space 0 = normal (shorts from resistance sweeps), space 1 = negated (longs from support sweeps)."""
    spaces = []
    for H, L, C in ((h, l, c), (-l, -h, -c)):
        spaces.append(dict(
            H=H, L=L, C=C,
            PL=last_confirmed(L, pivot_high_mask(-L, k5), k5),   # last confirmed swing low (space coords)
            PHm=pivot_high_mask(H, k5),                          # swing-high mask (inducement)
        ))
    return spaces


def make_levels(t5, h5, l5, c5, args):
    n = len(t5)
    df = pd.DataFrame({"h": h5, "l": l5, "c": c5}, index=t5)
    g = df.resample("1h", label="left", closed="left").agg({"h": "max", "l": "min", "c": "last"}).dropna()
    T1 = g.index.values
    C1 = g["c"].to_numpy()
    n1, k = len(g), args.k1
    tv = t5.values
    hour = np.timedelta64(1, "h")
    levels = []
    for space, sgn in ((0, 1.0), (1, -1.0)):
        X = sgn * C1
        Hs = h5 if space == 0 else -l5
        Ls = l5 if space == 0 else -h5
        for j in np.nonzero(pivot_high_mask(X, k))[0]:
            if j + k >= n1:
                continue
            p = X[j]
            st = int(np.searchsorted(tv, T1[j] + hour, "left"))
            a = int(np.searchsorted(tv, T1[j + k] + hour, "left"))
            if a >= n:
                continue
            if st >= a:
                continue
            # price must first move away from the level by --min-away, and after that
            # no wick may touch the level before it becomes active (= still fresh)
            away = np.nonzero(Ls[st:a] <= p - args.min_away * abs(p))[0]
            if away.size == 0:
                continue
            q = st + int(away[0])
            if Hs[q:a].max() >= p:
                continue
            levels.append((space, p, a, "A" if space == 0 else "V"))
    levels.sort(key=lambda x: x[2])
    return levels


# ----------------------------------------------------------------------------- trade sim


def simulate(H, L, C, entry, sl, risk, start, fill_bar, rr_list, hold):
    n = len(C)
    end = min(n, start + hold)
    if start >= end:
        return []
    hs, ls = H[start:end], L[start:end]
    slh = np.nonzero(hs >= sl)[0]
    sl_i = int(slh[0]) if slh.size else 10 ** 9
    out = []
    for rr in rr_list:
        tp = entry - rr * risk
        tph = np.nonzero(ls <= tp)[0]
        if fill_bar:
            tph = tph[tph >= 1]
        tp_i = int(tph[0]) if tph.size else 10 ** 9
        if sl_i == 10 ** 9 and tp_i == 10 ** 9:
            ex_i, ex, R = end - 1, C[end - 1], (entry - C[end - 1]) / risk
        elif sl_i <= tp_i:
            ex_i, ex, R = start + sl_i, sl, -1.0
        else:
            ex_i, ex, R = start + tp_i, tp, float(rr)
        out.append((rr, ex_i, R, (abs(entry) + abs(ex)) / risk))
    return out


FUN = dict(levels=0, pierced=0, rejected=0, breakout=0, mss=0)


def process(space, p, a, kind, S, args, n, rr_list):
    H, L, C = S[space]["H"], S[space]["L"], S[space]["C"]
    FUN["levels"] += 1
    s0 = first_cmp(H, a, None, p, np.greater)
    if s0 < 0:
        return [], None
    FUN["pierced"] += 1
    W = args.rej_bars
    seg = C[s0:min(n, s0 + W + 1)] < p
    if not seg.any():                                   # breakout -> level flips (SBR / RBS)
        FUN["breakout"] += 1
        e = s0 + W
        flip = None
        if args.flip and kind in ("A", "V") and e + 1 < n and L[s0 + 1:e + 1].min() > p:
            flip = (1 - space, -p, e + 1, "RBS" if space == 0 else "SBR")
        return [], flip
    r = s0 + int(seg.argmax())                          # rejection candle
    FUN["rejected"] += 1

    ref = S[space]["PL"][s0]                            # last confirmed M5 swing (known at the sweep)
    if not np.isfinite(ref) or ref >= p:
        return [], None
    hi = min(n, r + args.mss_bars + 1)
    hits = np.nonzero(C[r:hi] < ref)[0]
    if hits.size == 0:
        return [], None
    m = r + int(hits[0])                                # MSS candle
    if m > r + 1 and (C[r + 1:m] > p).any():            # level reclaimed -> sweep failed
        return [], None

    FUN["mss"] += 1
    sweep = H[s0:m + 1].max()
    sl = sweep + args.sl_buf * abs(sweep)
    direction = "short" if space == 0 else "long"
    trades = []

    def add(variant, eidx, entry, start, fill_bar):
        risk = sl - entry
        if risk <= 0:
            return
        rp = risk / abs(entry)
        if rp < args.min_risk or rp > args.max_risk:
            return
        for rr, ex_i, R, fu in simulate(H, L, C, entry, sl, risk, start, fill_bar, rr_list, args.hold_bars):
            trades.append(dict(eidx=eidx, xidx=ex_i, dir=direction, kind=kind, variant=variant,
                               rr=rr, Rg=R, fu=fu, risk_pct=rp * 100))

    add("mss", m, C[m], m + 1, False)                   # variant 1: market entry at MSS close

    end = min(n, m + 1 + args.retest_bars)              # variant 2/3: limit at broken M5 level
    j = first_cmp(H, m + 1, end, ref, np.greater_equal)
    if j >= 0:
        add("retest", j, ref, j, True)
        k5 = args.k5
        if j - k5 > m + 1:
            cand = np.nonzero(S[space]["PHm"][m + 1:j - k5])[0] + m + 1
            if cand.size and (H[cand] < ref).any():
                add("retest+idm", j, ref, j, True)
    return trades, None


# ----------------------------------------------------------------------------- reporting


def stats(R):
    n = len(R)
    if n == 0:
        return dict(n=0, win=np.nan, avgR=np.nan, t=np.nan, pf=np.nan)
    sd = R.std(ddof=1) if n > 1 else np.nan
    t = R.mean() / (sd / np.sqrt(n)) if sd and sd > 0 else np.nan
    pos, neg = R[R > 0].sum(), -R[R < 0].sum()
    return dict(n=n, win=(R > 0).mean() * 100, avgR=R.mean(), t=t, pf=pos / neg if neg > 0 else np.nan)


def session_of(ts):
    h = pd.DatetimeIndex(ts).hour
    return np.where(h < 8, "asia", np.where(h < 13, "london", np.where(h < 21, "ny", "late")))


def non_overlap(df):
    df = df.sort_values("eidx")
    keep, last = [], -1
    for i, (e, x) in enumerate(zip(df["eidx"].to_numpy(), df["xidx"].to_numpy())):
        if e > last:
            keep.append(i)
            last = x
    return df.iloc[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--fee", type=float, default=0.0002, help="fee per side (fraction)")
    ap.add_argument("--rr", default="2,3,5")
    ap.add_argument("--k1", type=int, default=3, help="H1 pivot width for A/V levels (close-line)")
    ap.add_argument("--k5", type=int, default=3, help="M5 pivot width for structure")
    ap.add_argument("--min-away", type=float, default=0.002, help="price must leave the level by this fraction before it counts as fresh")
    ap.add_argument("--rej-bars", type=int, default=12)
    ap.add_argument("--mss-bars", type=int, default=36)
    ap.add_argument("--retest-bars", type=int, default=48)
    ap.add_argument("--hold-bars", type=int, default=288)
    ap.add_argument("--sl-buf", type=float, default=0.0002)
    ap.add_argument("--min-risk", type=float, default=0.0005)
    ap.add_argument("--max-risk", type=float, default=0.02)
    ap.add_argument("--no-flip", dest="flip", action="store_false")
    ap.add_argument("--save", default="msnr_v1_trades.csv")
    args = ap.parse_args()
    rr_list = [float(x) for x in args.rr.split(",")]

    t0 = time.time()
    df = synthetic() if args.synthetic else load_5m(args.data)
    t5 = df.index
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    n = len(df)
    print(f"{'SYNTHETIC ' if args.synthetic else ''}5m candles: {n}  {t5[0]} -> {t5[-1]}   fee/side: {args.fee}")

    S = make_spaces(h, l, c, args.k5)
    queue = make_levels(t5, h, l, c, args)
    print(f"H1 fresh levels (A/V): {len(queue)}")
    qi, rows = 0, []
    while qi < len(queue):
        space, p, a, kind = queue[qi]
        qi += 1
        tr, flip = process(space, p, a, kind, S, args, n, rr_list)
        rows += tr
        if flip:
            queue.append(flip)
    print(f"levels processed (incl. flips): {qi}   ({time.time() - t0:.0f}s)")
    print("funnel:", FUN)
    if not rows:
        print("koi setup nahi bana.")
        return
    T = pd.DataFrame(rows)

    mid = t5[n // 2]
    res, kept = [], {}
    for (variant, rr), g in T.groupby(["variant", "rr"]):
        g = non_overlap(g).copy()
        g["time"] = t5[g["eidx"].to_numpy()]
        g["R0"] = g["Rg"]
        g["Rf"] = g["Rg"] - args.fee * g["fu"]
        g["half"] = np.where(g["time"] < mid, "H1", "H2")
        g["session"] = session_of(g["time"])
        kept[(variant, int(rr) if rr == int(rr) else rr)] = g
        s0_, sf = stats(g["R0"].to_numpy()), stats(g["Rf"].to_numpy())
        a1 = stats(g.loc[g.half == "H1", "Rf"].to_numpy())
        a2 = stats(g.loc[g.half == "H2", "Rf"].to_numpy())
        days = (t5[-1] - t5[0]).days or 1
        res.append(dict(variant=variant, RR=rr, n=sf["n"], per_day=sf["n"] / days, win_pct=sf["win"],
                        avgR_fee0=s0_["avgR"], avgR_fee=sf["avgR"], t_fee=sf["t"], PF_fee=sf["pf"],
                        avgR_H1=a1["avgR"], avgR_H2=a2["avgR"]))
    R = pd.DataFrame(res).sort_values(["variant", "RR"])
    pd.set_option("display.width", 200)
    print(f"\n=== Variants x RR  (one trade at a time; fee/side {args.fee}; H1/H2 = first/second half of data) ===")
    print(R.round(3).to_string(index=False))

    best = R.sort_values("avgR_fee", ascending=False).iloc[0]
    key = (best["variant"], int(best["RR"]) if best["RR"] == int(best["RR"]) else best["RR"])
    g = kept[key]
    print(f"\n=== Detail: {key[0]}, RR {key[1]}  (best avgR_fee on full data - cherry-picked, trust only if H1 and H2 both positive) ===")
    for col in ("dir", "kind", "session", "half"):
        rows2 = []
        for name, gg in g.groupby(col):
            st = stats(gg["Rf"].to_numpy())
            rows2.append(dict(**{col: name}, n=st["n"], win_pct=st["win"], avgR_fee=st["avgR"], t=st["t"]))
        print(pd.DataFrame(rows2).round(3).to_string(index=False))
        print()
    g.drop(columns=["fu"]).to_csv(args.save, index=False)
    print(f"trades saved: {args.save}")
    print("Rule of thumb: n < ~100 ya |t| < 2 wali rows ko ignore karo. Bahut saari combos dikh rahi hain, ek achhi row luck ho sakti hai.")


if __name__ == "__main__":
    main()
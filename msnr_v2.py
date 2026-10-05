#!/usr/bin/env python3
"""
msnr_v2.py  -  MSNR full-rules backtest (Alchemist MSNR Overview PDF), multi-asset, multi-timeframe (fractal)

Core flow (same rules on every timeframe pair HTF -> LTF, all bar-counts scale with HTF/LTF ratio):
  1. HTF POI: Classic A / Classic V (peak / valley of the close-line), plus flipped SBR / RBS levels.
     Level is "fresh" until a wick touches it (after price first left it by --away-atr x HTF ATR).
  2. Liquidity sweep: LTF wick pierces the level, an LTF candle closes back (rejection) within 1 HTF bar.
  3. LTF market-structure shift (MSS/CHOCH): close breaks the last confirmed LTF swing low/high.
  4. Entry variants: mss = market at MSS close | retest = limit at the broken LTF level (SBR/RBS).
  5. SL beyond the sweep wick, TP 2R/3R/5R, max hold 24 HTF bars, one position at a time per asset+pair.

Confluences from the PDF, each stored as a True/False flag per trade:
  asia  : the day's Asia-range high/low (UTC 00-06) is swept during the setup, after the Asia window
  qt    : entry candle is in daily Q3 (UTC 12-18)                       (Quarterly Theory, "Q3 ENTRY")
  qt90  : entry candle is in Q3 of the 90-minute micro cycle (LTF <= 15m only)
  qtM2  : the sweep happened in daily Q2 (UTC 06-12) = "M manipulation"
  smt   : SMT divergence vs the partner asset (exactly one of the two takes the lookback high/low)
  tl    : sweep is the 3rd touch of a trendline through the last two LTF swing highs/lows
  fib   : entry lies in a PDF fib zone (0.618-0.654 or 0.786-0.822) of the last HTF impulse leg
  idm   : an inducement (pullback pivot) exists before the entry
Output: baseline per asset/pair, each flag alone, and a dose-response table (does edge grow with # confluences?).

NOT coded: QML, engulfing OB, circle fibonacci, CRT.   Session times are UTC (shift with --tz-shift).
Conservative sim: SL first if SL and TP share a candle; on a limit-fill candle only SL is checked.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

TFMIN = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "1D": 1440}
FIB_ZONES = [(0.618, 0.654), (0.786, 0.822)]
CONF_FLAGS = ["asia", "qt", "smt", "tl", "fib"]
SOLO_FLAGS = ["asia", "qt", "qt90", "qtM2", "smt", "tl", "fib", "idm"]


# ----------------------------------------------------------------------------- data
def load_ohlc(path):
    df = pd.read_parquet(path) if path.lower().endswith(".parquet") else pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).lower().strip().strip("<>") for c in df.columns]
    cols = set(df.columns)
    if "date" in cols and "time" in cols and not (cols & {"open_time", "timestamp", "datetime"}):
        tser = df["date"].astype(str) + " " + df["time"].astype(str)
    else:
        tname = next((c for c in ("open_time", "timestamp", "time", "datetime", "date", "open time", "ts") if c in cols), None)
        tser = df[tname] if tname else df[df.columns[0]]
    if pd.api.types.is_numeric_dtype(tser):
        v = float(tser.iloc[0])
        unit = "us" if v > 1e14 else ("ms" if v > 1e11 else "s")
        t = pd.to_datetime(tser, unit=unit, utc=True)
    else:
        t = pd.to_datetime(tser, utc=True)
    out = {}
    for k in ("open", "high", "low", "close"):
        if k not in cols:
            sys.exit(f"{path}: column '{k}' nahi mila. Columns: {list(df.columns)}")
        out[k] = df[k].astype(float).to_numpy()
    o = pd.DataFrame(out, index=pd.DatetimeIndex(t))
    return o[~o.index.duplicated()].sort_index()


def synthetic_assets(n_bars=210000, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-10-05 18:30", periods=n_bars, freq="5min", tz="UTC")

    def vol():
        e = rng.normal(size=n_bars)
        lv = np.zeros(n_bars)
        for i in range(1, n_bars):
            lv[i] = 0.995 * lv[i - 1] + 0.1 * e[i]
        return np.exp(lv)

    common = rng.normal(size=n_bars) * vol()

    def build(start, s, mix):
        ret = (mix * common + np.sqrt(1 - mix ** 2) * rng.normal(size=n_bars) * vol()) * s
        close = start * np.exp(np.cumsum(ret))
        op = np.r_[close[0], close[:-1]]
        up = np.abs(rng.normal(size=n_bars)) * s * 0.6 * close
        dn = np.abs(rng.normal(size=n_bars)) * s * 0.6 * close
        return pd.DataFrame({"open": op, "high": np.maximum(op, close) + up,
                             "low": np.minimum(op, close) - dn, "close": close}, index=idx)

    return {"BTC": build(20000, 0.0007, 0.8), "GOLD": build(1700, 0.0003, 0.1)}, {"BTC": build(1300, 0.0009, 0.8)}


def to_tf(df, minutes, day_off=0):
    if minutes == 5:
        return df
    kw = dict(label="left", closed="left")
    if minutes >= 1440:
        kw["offset"] = f"{day_off}h"
    g = df.resample(f"{minutes}min" if minutes < 1440 else "1D", **kw).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"})
    return g.dropna(subset=["close"])


# ----------------------------------------------------------------------------- helpers
def pivot_high_mask(x, k):
    s = pd.Series(x)
    left = s.shift(1).rolling(k).max()
    right = s[::-1].shift(1).rolling(k).max()[::-1]
    return ((s >= left) & (s > right)).fillna(False).to_numpy()


def last_confirmed(price, mask, k):
    n = len(price)
    out = np.full(n, np.nan)
    idx = np.nonzero(mask)[0]
    conf = idx + k
    ok = conf < n
    out[conf[ok]] = price[idx[ok]]
    return pd.Series(out).ffill().to_numpy()


def atr_arr(h, l, c, n=14):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).rolling(n, min_periods=n).mean().to_numpy()


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


class Ctx:
    pass


def asia_ranges(df5, shift_h, a0, a1):
    lt = df5.index + pd.Timedelta(hours=shift_h)
    m = (lt.hour >= a0) & (lt.hour < a1)
    d = lt.normalize()
    g = pd.DataFrame({"h": df5["high"].to_numpy()[m], "l": df5["low"].to_numpy()[m]}, index=d[m])
    return g.groupby(level=0).agg({"h": "max", "l": "min"})


def build_ctx(df5, partner5, asia_daily, ltf, htf, args):
    c = Ctx()
    lm, hm = TFMIN[ltf], TFMIN[htf]
    L = to_tf(df5, lm, args.day_offset)
    Hf = to_tf(df5, hm, args.day_offset)
    c.ltf, c.htf, c.lm, c.hm = ltf, htf, lm, hm
    c.ratio = max(1, hm // lm)
    c.t = L.index
    h, l, cl = (L[x].to_numpy() for x in ("high", "low", "close"))
    n = len(L)
    c.n = n
    c.atrL = atr_arr(h, l, cl)
    hh, hl, hc = (Hf[x].to_numpy() for x in ("high", "low", "close"))
    c.Hf, c.hc = Hf, hc
    c.atrH = atr_arr(hh, hl, hc)
    dur = np.timedelta64(hm, "m")
    c.dur = dur
    pos = np.searchsorted(Hf.index.values + dur, c.t.values, "right") - 1
    c.atrH_L = np.where(pos >= 0, c.atrH[np.clip(pos, 0, None)], np.nan)
    c.W = max(2, int(round(args.rej_mult * c.ratio)))
    c.MSS = max(3, int(round(args.mss_mult * c.ratio)))
    c.RET = max(3, int(round(args.retest_mult * c.ratio)))
    c.HOLD = max(5, int(round(args.hold_mult * c.ratio)))
    c.smt_lb = max(6, int(round(args.smt_lb_mult * c.ratio)))
    c.k5 = args.k5
    c.has_partner = partner5 is not None
    if c.has_partner:
        P = to_tf(partner5, lm, args.day_offset).reindex(L.index)
        ph, pl = P["high"].to_numpy(), P["low"].to_numpy()
    else:
        ph = pl = np.full(n, np.nan)
    lt = c.t + pd.Timedelta(hours=args.tz_shift)
    c.lhour = (lt.hour + lt.minute / 60.0).to_numpy()
    ready = lt.hour.to_numpy() >= args.asia_end
    day = lt.normalize()
    ah = asia_daily["h"].reindex(day).to_numpy()
    al = asia_daily["l"].reindex(day).to_numpy()
    ah = np.where(ready, ah, np.nan)
    al = np.where(ready, al, np.nan)
    c.S = []
    for sp, (H, Lo, C, PH, PLo, AH) in enumerate(((h, l, cl, ph, pl, ah), (-l, -h, -cl, -pl, -ph, -al))):
        phm = pivot_high_mask(H, c.k5)
        c.S.append(dict(H=H, L=Lo, C=C, PHp=PH, AH=AH,
                        PL=last_confirmed(Lo, pivot_high_mask(-Lo, c.k5), c.k5),
                        PHm=phm, PHidx=np.nonzero(phm)[0]))
    # HTF pivots for the fib flag
    tv = c.t.values
    T1 = Hf.index.values
    n1, k = len(Hf), args.k1
    c.FIB = []
    for Hh, Ll in ((hh, hl), (-hl, -hh)):
        def collect(mask, price):
            js = np.nonzero(mask)[0]
            js = js[js + k < n1]
            return np.searchsorted(tv, T1[js + k] + dur, "left"), js, price[js]
        hc_, hj_, hp_ = collect(pivot_high_mask(Hh, k), Hh)
        lc_, lj_, lp_ = collect(pivot_high_mask(-Ll, k), Ll)
        c.FIB.append(dict(hc=hc_, hj=hj_, hp=hp_, lc=lc_, lj=lj_, lp=lp_))
    return c


def make_levels(c, args):
    tv = c.t.values
    T1 = c.Hf.index.values
    n1, k, n = len(c.Hf), args.k1, c.n
    levels = []
    for sp, sgn in ((0, 1.0), (1, -1.0)):
        X = sgn * c.hc
        S = c.S[sp]
        for j in np.nonzero(pivot_high_mask(X, k))[0]:
            if j + k >= n1:
                continue
            p = X[j]
            a = int(np.searchsorted(tv, T1[j + k] + c.dur, "left"))
            st = int(np.searchsorted(tv, T1[j] + c.dur, "left"))
            atr_j = c.atrH[j]
            if a >= n or st >= a or not np.isfinite(atr_j):
                continue
            away = np.nonzero(S["L"][st:a] <= p - args.away_atr * atr_j)[0]
            if away.size == 0:
                continue
            q = st + int(away[0])
            if S["H"][q:a].max() >= p:
                continue
            levels.append((sp, p, a, "A" if sp == 0 else "V"))
    levels.sort(key=lambda x: x[2])
    return levels


# ----------------------------------------------------------------------------- flags
def flag_asia(c, sp, s0, m):
    ah = c.S[sp]["AH"][s0]
    return bool(np.isfinite(ah) and c.S[sp]["H"][s0:m + 1].max() > ah)


def flag_smt(c, sp, s0, m):
    if not c.has_partner:
        return False
    lb = c.smt_lb
    if s0 - lb < 0:
        return False
    S = c.S[sp]
    HA, HB = S["H"], S["PHp"]
    wb, lbb = HB[s0:m + 1], HB[s0 - lb:s0]
    if np.isnan(wb).any() or np.isnan(lbb).any():
        return False
    a_sw = HA[s0:m + 1].max() > HA[s0 - lb:s0].max()
    b_sw = wb.max() > lbb.max()
    return bool(a_sw != b_sw)


def flag_tl(c, sp, s0, m, args):
    S = c.S[sp]
    H, C = S["H"], S["C"]
    pidx = S["PHidx"]
    cnt = int(np.searchsorted(pidx + c.k5, s0, "right"))
    if cnt < 2:
        return False
    i1, i2 = int(pidx[cnt - 2]), int(pidx[cnt - 1])
    if i2 - i1 < args.tl_gap or s0 - i1 > args.tl_back_mult * c.ratio:
        return False
    slope = (H[i2] - H[i1]) / (i2 - i1)
    w = H[s0:m + 1]
    kmax = s0 + int(w.argmax())
    atr = c.atrL[kmax]
    if not np.isfinite(atr):
        return False
    val = H[i1] + slope * (kmax - i1)
    if abs(w.max() - val) > args.tl_tol * atr:
        return False
    if s0 > i2 + 1:
        line = H[i1] + slope * (np.arange(i2 + 1, s0) - i1)
        if (C[i2 + 1:s0] > line).any():
            return False
    return True


def flag_fib(c, sp, E, s0):
    F = c.FIB[sp]
    il = int(np.searchsorted(F["lc"], s0, "right")) - 1
    if il < 0:
        return False
    Lp, jl = F["lp"][il], F["lj"][il]
    ih = int(np.searchsorted(F["hj"], jl, "left")) - 1
    if ih < 0:
        return False
    Hp = F["hp"][ih]
    if Hp <= Lp or E < Lp:
        return False
    r = (E - Lp) / (Hp - Lp)
    return any(lo <= r <= hi for lo, hi in FIB_ZONES)


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


def process(sp, p, a, kind, c, args, rr_list, fun):
    S = c.S[sp]
    H, L, C = S["H"], S["L"], S["C"]
    n = c.n
    fun["levels"] += 1
    s0 = first_cmp(H, a, None, p, np.greater)
    if s0 < 0:
        return [], None
    fun["pierced"] += 1
    seg = C[s0:min(n, s0 + c.W + 1)] < p
    if not seg.any():
        fun["breakout"] += 1
        e = s0 + c.W
        flip = None
        if args.flip and kind in ("A", "V") and e + 1 < n and L[s0 + 1:e + 1].min() > p:
            flip = (1 - sp, -p, e + 1, "RBS" if sp == 0 else "SBR")
        return [], flip
    r = s0 + int(seg.argmax())
    fun["rejected"] += 1
    ref = S["PL"][s0]
    if not np.isfinite(ref) or ref >= p:
        return [], None
    hi = min(n, r + c.MSS + 1)
    hits = np.nonzero(C[r:hi] < ref)[0]
    if hits.size == 0:
        return [], None
    m = r + int(hits[0])
    if m > r + 1 and (C[r + 1:m] > p).any():
        return [], None
    atrL, atrH = c.atrL[m], c.atrH_L[m]
    if not (np.isfinite(atrL) and np.isfinite(atrH)):
        return [], None
    fun["mss"] += 1
    sweep = H[s0:m + 1].max()
    sl = sweep + args.sl_buf_atr * atrL
    direction = "short" if sp == 0 else "long"
    lh = c.lhour
    f_asia = flag_asia(c, sp, s0, m)
    f_smt = flag_smt(c, sp, s0, m)
    f_tl = flag_tl(c, sp, s0, m, args)
    f_m2 = bool(args.q2_start <= lh[s0] < args.q2_end)
    k5 = c.k5
    trades = []

    def add(variant, eidx, entry, start, fill_bar, idm):
        risk = sl - entry
        if risk <= 0 or risk < args.min_risk_atr * atrH or risk > args.max_risk_atr * atrH:
            return
        eh = lh[eidx]
        flags = dict(asia=f_asia, qt=bool(args.q3_start <= eh < args.q3_end),
                     qt90=bool(c.lm <= 15 and 45 <= (eh * 60) % 90 < 67.5),
                     qtM2=f_m2, smt=f_smt, tl=f_tl, fib=flag_fib(c, sp, entry, s0), idm=bool(idm))
        for rr, ex_i, R, fu in simulate(H, L, C, entry, sl, risk, start, fill_bar, rr_list, c.HOLD):
            trades.append(dict(eidx=eidx, xidx=ex_i, dir=direction, kind=kind, variant=variant,
                               rr=rr, Rg=R, fu=fu, risk_atr=risk / atrH, **flags))

    lo_i, hi_i = r, m - k5
    cand = np.nonzero(S["PHm"][lo_i:hi_i + 1])[0] + lo_i if hi_i >= lo_i else np.array([], dtype=int)
    idm_m = bool(cand.size and (H[cand] < p).any())
    add("mss", m, C[m], m + 1, False, idm_m)

    end = min(n, m + 1 + c.RET)
    j = first_cmp(H, m + 1, end, ref, np.greater_equal)
    if j >= 0:
        idm_r = False
        if j - k5 > m + 1:
            cd = np.nonzero(S["PHm"][m + 1:j - k5])[0] + m + 1
            idm_r = bool(cd.size and (H[cd] < ref).any())
        add("retest", j, ref, j, True, idm_r)
    return trades, None


# ----------------------------------------------------------------------------- reporting
def non_overlap(df):
    df = df.sort_values("eidx")
    keep, last = [], -1
    for i, (e, x) in enumerate(zip(df["eidx"].to_numpy(), df["xidx"].to_numpy())):
        if e > last:
            keep.append(i)
            last = x
    return df.iloc[keep]


def pick(T, variant, rr, cond=None):
    g = T[(T["variant"] == variant) & (T["rr"] == rr)]
    if cond is not None:
        g = g[cond(g)]
    parts = [non_overlap(gg) for _, gg in g.groupby(["asset", "pair"])]
    return pd.concat(parts) if parts else g.iloc[0:0]


def st(R):
    n = len(R)
    if n == 0:
        return dict(n=0, avg=np.nan, t=np.nan, win=np.nan)
    sd = R.std(ddof=1) if n > 1 else np.nan
    t = R.mean() / (sd / np.sqrt(n)) if sd and sd > 0 else np.nan
    return dict(n=n, avg=R.mean(), t=t, win=(R > 0).mean() * 100)


def row(name, g):
    a, b = st(g["Rf"].to_numpy()), st(g["Rg"].to_numpy())
    h1, h2 = st(g.loc[g["half"] == "H1", "Rf"].to_numpy()), st(g.loc[g["half"] == "H2", "Rf"].to_numpy())
    return dict(name=name, n=a["n"], win=a["win"], avgR_fee0=b["avg"], avgR_fee=a["avg"], t_fee=a["t"],
                H1=h1["avg"], H2=h2["avg"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", action="append", default=[], help="NAME=5m_csv  (repeatable)")
    ap.add_argument("--partner", action="append", default=[], help="NAME=5m_csv  SMT partner (BTC->ETH, GOLD->silver)")
    ap.add_argument("--asset-fee", action="append", default=[], help="NAME=fee_per_side (repeatable)")
    ap.add_argument("--synthetic", action="store_true", help="random-walk baseline (3 fake assets)")
    ap.add_argument("--pairs", default="1D:1h,4h:15m,1h:5m", help="HTF:LTF list")
    ap.add_argument("--fee", type=float, default=0.0002)
    ap.add_argument("--rr", default="2,3,5")
    ap.add_argument("--detail-rr", type=float, default=3.0)
    ap.add_argument("--k1", type=int, default=3)
    ap.add_argument("--k5", type=int, default=3)
    ap.add_argument("--away-atr", type=float, default=0.5)
    ap.add_argument("--rej-mult", type=float, default=1.0)
    ap.add_argument("--mss-mult", type=float, default=3.0)
    ap.add_argument("--retest-mult", type=float, default=4.0)
    ap.add_argument("--hold-mult", type=float, default=24.0)
    ap.add_argument("--smt-lb-mult", type=float, default=6.0)
    ap.add_argument("--sl-buf-atr", type=float, default=0.1)
    ap.add_argument("--min-risk-atr", type=float, default=0.1)
    ap.add_argument("--max-risk-atr", type=float, default=3.0)
    ap.add_argument("--tl-gap", type=int, default=5)
    ap.add_argument("--tl-tol", type=float, default=0.5)
    ap.add_argument("--tl-back-mult", type=float, default=10.0)
    ap.add_argument("--tz-shift", type=float, default=0.0, help="hours added to UTC for session/QT times")
    ap.add_argument("--day-offset", type=int, default=0)
    ap.add_argument("--asia-start", type=int, default=0)
    ap.add_argument("--asia-end", type=int, default=6)
    ap.add_argument("--q2-start", type=float, default=6)
    ap.add_argument("--q2-end", type=float, default=12)
    ap.add_argument("--q3-start", type=float, default=12)
    ap.add_argument("--q3-end", type=float, default=18)
    ap.add_argument("--no-flip", dest="flip", action="store_false")
    ap.add_argument("--save", default="msnr_v2_trades.csv")
    args = ap.parse_args()
    rr_list = [float(x) for x in args.rr.split(",")]
    pairs = [tuple(x.split(":")) for x in args.pairs.split(",")]
    for hp_, lp_ in pairs:
        if hp_ not in TFMIN or lp_ not in TFMIN or TFMIN[hp_] <= TFMIN[lp_]:
            sys.exit(f"bad pair {hp_}:{lp_}  (HTF bada, LTF chhota; allowed {list(TFMIN)})")

    assets, partners = {}, {}
    if args.synthetic:
        assets, partners = synthetic_assets()
    else:
        for s in args.asset:
            k, v = s.split("=", 1)
            assets[k.strip()] = load_ohlc(v.strip())
        for s in args.partner:
            k, v = s.split("=", 1)
            partners[k.strip()] = load_ohlc(v.strip())
        if not assets:
            auto = [("BTC", ["btc_5m.csv"], ["eth_5m.csv"]),
                    ("GOLD", ["paxg_5m.csv", "xauusd_5m.csv", "xau_5m.csv", "gold_5m.csv"],
                     ["xagusd_5m.csv", "xag_5m.csv", "silver_5m.csv"])]
            for nm, files, pfiles in auto:
                f = next((x for x in files if os.path.exists(x)), None)
                if f:
                    assets[nm] = load_ohlc(f)
                    print(f"asset {nm}: {f}")
                    pf = next((x for x in pfiles if os.path.exists(x)), None)
                    if pf and nm not in partners:
                        partners[nm] = load_ohlc(pf)
                        print(f"  SMT partner {nm}: {pf}")
            if not assets:
                sys.exit("koi data file nahi mili. --asset BTC=btc_5m.csv --asset GOLD=paxg_5m.csv do (fetch_data.py se banao)")
    fee_map = {k: args.fee for k in assets}
    for s in args.asset_fee:
        k, v = s.split("=", 1)
        fee_map[k.strip()] = float(v)

    t0 = time.time()
    all_rows, funnel = [], []
    for name, df5 in assets.items():
        p5 = partners.get(name)
        adaily = asia_ranges(df5, args.tz_shift, args.asia_start, args.asia_end)
        print(f"\n{name}: {len(df5)} x 5m  {df5.index[0]} -> {df5.index[-1]}   SMT partner: {'yes' if p5 is not None else 'no'}   fee/side {fee_map[name]}")
        for htf, ltf in pairs:
            c = build_ctx(df5, p5, adaily, ltf, htf, args)
            levels = make_levels(c, args)
            fun = dict(levels=0, pierced=0, rejected=0, breakout=0, mss=0)
            queue, qi, rows = list(levels), 0, []
            while qi < len(queue):
                sp, p, a, kind = queue[qi]
                qi += 1
                tr, flip = process(sp, p, a, kind, c, args, rr_list, fun)
                rows += tr
                if flip:
                    queue.append(flip)
            fun.update(asset=name, pair=f"{htf}>{ltf}", fresh_levels=len(levels), raw=len(rows))
            funnel.append(fun)
            print(f"  {htf}>{ltf}: fresh levels {len(levels):5d} | mss setups {fun['mss']:4d} | raw trades {len(rows):5d}   ({time.time() - t0:.0f}s)")
            if rows:
                d = pd.DataFrame(rows)
                d["asset"], d["pair"] = name, f"{htf}>{ltf}"
                d["time"] = c.t[d["eidx"].to_numpy()]
                all_rows.append(d)
    if not all_rows:
        print("koi setup nahi bana.")
        return
    T = pd.concat(all_rows, ignore_index=True)
    T["fee"] = T["asset"].map(fee_map)
    T["Rf"] = T["Rg"] - T["fee"] * T["fu"]
    mid = T["time"].min() + (T["time"].max() - T["time"].min()) / 2
    T["half"] = np.where(T["time"] < mid, "H1", "H2")
    T["conf"] = T[CONF_FLAGS].sum(axis=1)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    # ---- funnel
    F = pd.DataFrame(funnel)[["asset", "pair", "fresh_levels", "pierced", "rejected", "breakout", "mss"]]
    print("\n=== Funnel (levels -> sweep -> rejection -> structure shift) ===")
    print(F.to_string(index=False))

    # ---- A: baseline per asset / pair
    print(f"\n=== A) Baseline, no filters  (net avgR after fee; one trade at a time) ===")
    rowsA = []
    for variant in ("mss", "retest"):
        for asset in assets:
            for _, pr in [(0, f"{h}>{l}") for h, l in pairs]:
                r_ = dict(variant=variant, asset=asset, pair=pr)
                for rr in rr_list:
                    g = pick(T, variant, rr, lambda x: (x["asset"] == asset) & (x["pair"] == pr))
                    s_ = st(g["Rf"].to_numpy())
                    r_[f"avgR@{rr:g}R"] = s_["avg"]
                    if rr == args.detail_rr:
                        r_["n"] = s_["n"]
                        r_["avgR_fee0"] = st(g["Rg"].to_numpy())["avg"]
                        r_["t"] = s_["t"]
                rowsA.append(r_)
    print(pd.DataFrame(rowsA).round(3).to_string(index=False))
    print("(n / avgR_fee0 / t sirf RR", args.detail_rr, "ke liye)")

    # ---- B: each flag alone, pooled
    rr0 = args.detail_rr
    print(f"\n=== B) Each confluence alone, all assets + pairs pooled, RR {rr0:g}  (with = flag True, without = flag False) ===")
    for variant in ("mss", "retest"):
        rowsB = [row("ALL (no filter)", pick(T, variant, rr0))]
        for f in SOLO_FLAGS:
            w = pick(T, variant, rr0, lambda x, f=f: x[f])
            wo = pick(T, variant, rr0, lambda x, f=f: ~x[f])
            a = row(f"{f}: with", w)
            b = st(wo["Rf"].to_numpy())
            a["n_without"], a["avgR_without"] = b["n"], b["avg"]
            rowsB.append(a)
        print(f"\n-- entry variant: {variant}")
        print(pd.DataFrame(rowsB).round(3).to_string(index=False))

    # ---- C: dose response
    print(f"\n=== C) Dose-response: number of confluences present {CONF_FLAGS}, RR {rr0:g} ===")
    for variant in ("mss", "retest"):
        rowsC = []
        for lo, hi, nm in ((0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 9, "3+")):
            g = pick(T, variant, rr0, lambda x, lo=lo, hi=hi: (x["conf"] >= lo) & (x["conf"] <= hi))
            rowsC.append(row(f"conf {nm}", g))
        print(f"\n-- entry variant: {variant}")
        print(pd.DataFrame(rowsC).round(3).to_string(index=False))

    T.drop(columns=["fu"]).to_csv(args.save, index=False)
    print(f"\ntrades saved: {args.save}   (total {time.time() - t0:.0f}s)")
    print("Rule of thumb: n < ~100 ya |t| < 2 wali rows ignore karo. Bahut saari rows dikh rahi hain, ek achhi row luck ho sakti hai.")
    print("Asli sawal: conf badhne par avgR_fee lagatar badhta hai aur H1 aur H2 dono positive hain? Tab hi bharosa karo.")


if __name__ == "__main__":
    main()
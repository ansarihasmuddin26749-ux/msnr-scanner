"""
smc.py  -  ICT/SMC building blocks + one complete model, written as precise, testable rules.

Concepts implemented (batch 1):
  swing pivots, market structure (BOS / CHoCH-MSS), liquidity pools (swing highs/lows, equal highs/lows,
  previous-day high/low), liquidity sweeps, order blocks, FVG (and overlap of OB+FVG), premium/discount
  (dealing range), sessions/killzones (NY time, incl. London, NY AM, Silver Bullet, NY PM).

Model A  ("liquidity sweep -> MSS -> retrace into OB/FVG"), long side (short = mirror image):
  1. Sell-side liquidity is swept: a candle wicks below a known low (swing low / equal lows / PDL)
     and CLOSES back above it.
  2. Within `mss_window` candles price closes above the last swing high (BOS / CHoCH = market structure shift).
  3. Entry zone = order block (last bearish candle before the up-leg) and/or FVG created by the leg.
     Overlap of both is preferred. Limit entry at the middle of the zone.
  4. SL = lowest low since the sweep (minus a tiny buffer). Target = highest high of the last `lookback_tp`
     candles (external range high). Setup needs RR >= `min_rr`.
  5. Outcome: entry must fill within `fill_wait` candles (else 'nofill'); then SL first = loss,
     target first = win (same candle = loss). Two targets are scored: the range high and a fixed 2R.
"""
import numpy as np
import pandas as pd

KILLZONES = ("london", "ny_am", "silver_bullet", "ny_pm")


# ------------------------------------------------------------------ helpers
def _pivots(h, l, n):
    """Swing pivots. A pivot at i is only *confirmed* n candles later (no look-ahead)."""
    N, piv = len(h), []
    for i in range(n, N - n):
        if h[i] > h[i - n:i].max() and h[i] >= h[i + 1:i + n + 1].max():
            piv.append((i + n, i, h[i], "H"))
        if l[i] < l[i - n:i].min() and l[i] <= l[i + 1:i + n + 1].min():
            piv.append((i + n, i, l[i], "L"))
    piv.sort()
    return piv


def _prev_day_levels(d):
    """Previous New-York-day high/low for every candle."""
    ny = d.tz_convert("America/New_York")
    key = ny.index.normalize()
    g = ny.groupby(key).agg({"high": "max", "low": "min"})
    return g["high"].shift(1).reindex(key).values, g["low"].shift(1).reindex(key).values, key.values


def session_of(ts):
    ny = ts.tz_convert("America/New_York")
    m = ny.hour * 60 + ny.minute
    if 120 <= m < 300:
        return "london"
    if 420 <= m < 600:
        return "ny_am"
    if 600 <= m < 660:
        return "silver_bullet"
    if 810 <= m < 960:
        return "ny_pm"
    if m >= 1200:
        return "asia"
    return "off"


def _fill_and_label(E, sl, tps, h, l, c, start, fill_wait, horizon):
    """Limit order at E. Returns (fill_bar, [(R, outcome) per target]) or (None, status)."""
    N = len(h)
    fill = None
    for i in range(start, min(start + fill_wait, N)):
        if l[i] <= E:
            fill = i
            break
        if h[i] >= tps[0]:
            return None, "missed"            # target hit without us being filled
    if fill is None:
        return None, ("nofill" if start + fill_wait <= N else "open")
    if fill + 1 + horizon > N:
        return None, "open"                  # not enough future data to judge yet
    risk = E - sl
    res = []
    for tp in tps:
        if l[fill] <= sl:                    # stopped on the fill candle itself
            res.append((-1.0, "loss"))
            continue
        out = None
        for k in range(fill + 1, fill + 1 + horizon):
            if l[k] <= sl:
                out = (-1.0, "loss")
                break
            if h[k] >= tp:
                out = ((tp - E) / risk, "win")
                break
        if out is None:
            out = ((c[fill + horizon] - E) / risk, "timeout")
        res.append(out)
    return fill, res


# ------------------------------------------------------------------ Model A
def find_smc_setups(df, side="long", n=3, mss_window=24, fill_wait=24, horizon=288, lookback_tp=100,
                    min_rr=1.5, buffer=0.0002, fee=0.0005, eq_tol=0.0003, ema_len=288,
                    require_choch=False, live=False):
    sgn = 1 if side == "long" else -1
    if sgn == 1:
        d = df[["open", "high", "low", "close"]].copy()
    else:   # mirror prices so a short looks like a long
        d = pd.DataFrame({"open": -df["open"], "high": -df["low"], "low": -df["high"], "close": -df["close"]})
    idx = d.index
    o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
    N = len(d)
    ema = d["close"].ewm(span=ema_len, adjust=False).mean().values
    piv = _pivots(h, l, n)
    pdh, pdl, daykey = _prev_day_levels(d)

    pp, lows, recent_pl = 0, [], []
    last_sh = last_sl = None
    trend = 0
    sweep = None
    last_pdl_day = None
    rows = []

    for j in range(n + 3, N):
        # 1) newly confirmed pivots
        while pp < len(piv) and piv[pp][0] < j:
            cf, i, price, kind = piv[pp]
            pp += 1
            if kind == "H":
                last_sh = (price, i)
            else:
                last_sl = (price, i)
                eq = any(abs(price - p) / abs(price) <= eq_tol for p in recent_pl[-10:])
                lows.append({"price": price, "kind": "equal" if eq else "swing", "born": i})
                recent_pl.append(price)

        # 2) liquidity sweeps of known lows
        keep, swept = [], []
        for lv in lows:
            if j - lv["born"] > 600:
                continue
            if l[j] < lv["price"]:
                if c[j] > lv["price"]:
                    swept.append(lv)         # wick below, close back above = sweep
                continue                     # swept or broken: level is used up
            keep.append(lv)
        lows = keep
        if swept:
            best = next((s for s in swept if s["kind"] == "equal"), swept[0])
            sweep = {"idx": j, "level": best["price"], "kind": best["kind"], "minlow": l[j]}
        elif not np.isnan(pdl[j]) and l[j] < pdl[j] < c[j] and daykey[j] != last_pdl_day:
            last_pdl_day = daykey[j]
            sweep = {"idx": j, "level": pdl[j], "kind": "PDL", "minlow": l[j]}
        elif sweep is not None:
            sweep["minlow"] = min(sweep["minlow"], l[j])
        if sweep is not None and j - sweep["idx"] > mss_window:
            sweep = None

        # 3) structure
        if last_sl is not None and c[j] < last_sl[0]:
            trend = -1
            last_sl = None
        if last_sh is not None and c[j] > last_sh[0]:
            is_choch = trend == -1
            trend = 1
            last_sh = None
            if sweep is not None and sweep["idx"] < j and (is_choch or not require_choch):
                s0 = sweep["idx"]
                leg_low = l[s0:j + 1].min()
                leg_low_i = s0 + int(np.argmin(l[s0:j + 1]))
                # FVG made during the leg (largest bullish gap)
                fvg = None
                for k in range(s0 + 2, j + 1):
                    if l[k] > h[k - 2] and (fvg is None or l[k] - h[k - 2] > fvg[1] - fvg[0]):
                        fvg = (h[k - 2], l[k])
                # order block: last bearish candle at/before the low of the leg
                ob = None
                for k in range(leg_low_i, max(leg_low_i - 6, 0), -1):
                    if c[k] < o[k]:
                        ob = (l[k], h[k])
                        break
                overlap = None
                if fvg and ob:
                    lo_, hi_ = max(fvg[0], ob[0]), min(fvg[1], ob[1])
                    if hi_ > lo_:
                        overlap = (lo_, hi_)
                zone = overlap or fvg or ob
                if zone is not None:
                    ztype = "fvg+ob" if overlap else ("fvg" if fvg else "ob")
                    E = (zone[0] + zone[1]) / 2
                    sl = leg_low - buffer * abs(c[j])
                    tp = h[max(0, j - lookback_tp):j + 1].max()
                    risk = E - sl
                    if E < c[j] and risk > 0 and (tp - E) >= min_rr * risk:
                        rng_hi, rng_lo = h[max(0, j - lookback_tp):j + 1].max(), l[max(0, j - lookback_tp):j + 1].min()
                        mid = (rng_hi + rng_lo) / 2
                        ses = session_of(idx[j])
                        discount = int(E < mid)
                        liq_good = int(sweep["kind"] in ("equal", "PDL"))
                        kz = int(ses in KILLZONES)
                        row = {
                            "time": idx[j], "side": side, "sweep_kind": sweep["kind"], "zone": ztype,
                            "choch": int(is_choch), "session": ses, "killzone": kz,
                            "discount": discount, "liq_good": liq_good,
                            "entry": sgn * E, "sl": sgn * sl, "tp": sgn * tp,
                            "risk_pct": risk / abs(E) * 100, "rr_planned": (tp - E) / risk,
                            "trend_align": (c[j] - ema[j]) / abs(c[j]) * 100,
                            "score": liq_good + discount + kz + int(ztype == "fvg+ob") + int(is_choch),
                        }
                        if live:
                            row.update({"outcome": "pending", "result_R": np.nan, "result_2R": np.nan})
                            rows.append(row)
                        else:
                            fee_r = 2 * fee * abs(E) / risk
                            tps = [tp, E + 2 * risk]
                            fill, res = _fill_and_label(E, sl, tps, h, l, c, j + 1, fill_wait, horizon)
                            if fill is None:
                                if res != "open":
                                    row.update({"outcome": res, "result_R": np.nan, "result_2R": np.nan})
                                    rows.append(row)
                            else:
                                row.update({"outcome": res[0][1], "result_R": res[0][0] - fee_r,
                                            "outcome_2R": res[1][1], "result_2R": res[1][0] - fee_r,
                                            "fill_delay": fill - j})
                                rows.append(row)
            if sweep is not None and sweep["idx"] < j:
                sweep = None   # one setup attempt per sweep
    return pd.DataFrame(rows)
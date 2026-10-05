"""
msnr_chart.py  -  paints an MSNR (msnr_v3.py rules) trade on a chart with numbered reason tags.

Put this file next to msnr_v3.py. It reuses msnr_v3.htf_signals / resample / load_csv, so the
levels are exactly the ones your backtest uses. The entry search is the same as
msnr_v3.find_entry, but also returns the LTF pivot that was broken (needed for the chart).

Tags painted on the chart (same numbers as the list under it):
  1 HTF level      A-level (resistance, SHORT) or V-level (support, LONG) from the close-line
  2 Fresh          level untouched since it formed
  3 HTF touch      first touch was a WICK only (grey band = the HTF touch candle)
  4 LTF pivot      last confirmed LTF pivot after the touch
  5 Shift candle   LTF close beyond that pivot = ENTRY
  6 SL             extreme since the touch (swept liquidity)
  7 TP             R x risk
QT quarter is shown in the title (diagnostic only, as in msnr_v3).

Usage:
  python msnr_chart.py --csv btc5m.csv --last 3            # last 3 trades -> msnr_alert_1.png ...
  python msnr_chart.py --csv btc5m.csv --last 1 --R 2 --htf 1h --k 3
  python msnr_chart.py --synthetic --last 2                # smoke test, no data needed

From the scanner:
  from msnr_chart import build_trades, render_msnr_chart
  trades = build_trades(ltf)                  # ltf = 5m DataFrame (open, high, low, close, DatetimeIndex UTC)
  t = trades[-1]
  if t["m"] >= len(ltf) - 2:                  # shift candle is the latest closed bar -> fresh signal
      render_msnr_chart(ltf, t, "alert.png")  # send alert.png to ntfy
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from msnr_v3 import htf_signals, load_csv, resample, synthetic

UP, DOWN, TAG_FC = "#26a69a", "#ef5350", "#f5c542"


# ----------------------------------------------------------------- trade detail (same rules as msnr_v3)
def _find_born(o, c, side, lvl, j, k, max_age=400):
    """HTF bar index where this level was formed (for the 'fresh since' text)."""
    for i in range(j - k - 1, max(k, j - max_age - k - 2), -1):
        if i + 1 >= len(c):
            continue
        win = c[i - k:i + k + 1]
        if side == "S" and c[i] > o[i] and c[i + 1] < o[i + 1] and c[i] == win.max() \
                and abs(max(c[i], o[i + 1]) - lvl) < 1e-9:
            return i
        if side == "L" and c[i] < o[i] and c[i + 1] > o[i + 1] and c[i] == win.min() \
                and abs(min(c[i], o[i + 1]) - lvl) < 1e-9:
            return i
    return None


def find_entry_detail(side, start, extreme, hi, lo, cl, kp=3, window=72, min_risk=0.001, max_risk=0.02):
    """Identical to msnr_v3.find_entry, but also returns the pivot that was broken."""
    n = len(cl)
    ext, piv, piv_i = extreme, None, None
    for m in range(start, min(start + window, n - 1)):
        p = m - kp
        if side == "S":
            ext = max(ext, hi[m])
            if p >= start and lo[p] == lo[max(0, p - kp):m + 1].min():
                piv, piv_i = lo[p], p
            if piv is not None and cl[m] < piv:
                risk = ext - cl[m]
                if min_risk <= risk / cl[m] <= max_risk:
                    return dict(m=m, entry=cl[m], sl=ext, risk=risk, piv=piv, piv_i=piv_i)
                piv = None
        else:
            ext = min(ext, lo[m])
            if p >= start and hi[p] == hi[max(0, p - kp):m + 1].max():
                piv, piv_i = hi[p], p
            if piv is not None and cl[m] > piv:
                risk = cl[m] - ext
                if min_risk <= risk / cl[m] <= max_risk:
                    return dict(m=m, entry=cl[m], sl=ext, risk=risk, piv=piv, piv_i=piv_i)
                piv = None
    return None


def build_trades(ltf, htf_rule="1h", k=3, kp=3, window=72, R=2, tz=0.0):
    h = resample(ltf, htf_rule)
    ho, hc = h["open"].values, h["close"].values
    hi, lo, cl = ltf["high"].values, ltf["low"].values, ltf["close"].values
    times = ltf.index.values
    delta = pd.Timedelta(htf_rule).to_timedelta64()
    out = []
    for j, side, lvl, ext in htf_signals(h, k=k):
        t0 = h.index[j].to_datetime64()
        touch_i = int(np.searchsorted(times, t0))
        start = int(np.searchsorted(times, t0 + delta))      # first LTF bar after the HTF close
        if start >= len(cl) - 2:
            continue
        e = find_entry_detail(side, start, ext, hi, lo, cl, kp=kp, window=window)
        if e is None:
            continue
        m = e["m"]
        seg = hi[touch_i:m + 1] if side == "S" else lo[touch_i:m + 1]
        ext_i = touch_i + int(seg.argmax() if side == "S" else seg.argmin())
        born = _find_born(ho, hc, side, lvl, j, k)
        tp = e["entry"] - R * e["risk"] if side == "S" else e["entry"] + R * e["risk"]
        ts = ltf.index[m] + pd.Timedelta(hours=tz)
        q = ((ts.hour * 60 + ts.minute) % 360) // 90 + 1
        out.append(dict(side=side, level=lvl, htf=htf_rule, R=R, tp=tp, q=int(q),
                        born_time=h.index[born] if born is not None else None,
                        touch_i=touch_i, start=start, ext_i=ext_i, **e))
    return out


# ----------------------------------------------------------------- drawing
def _tag(ax, n, x, y, dx=0, dy=24):
    ax.annotate(str(n), xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                ha="center", va="center", fontsize=9, fontweight="bold", zorder=10,
                bbox=dict(boxstyle="circle,pad=0.3", fc=TAG_FC, ec="black", lw=1),
                arrowprops=dict(arrowstyle="-", color="black", lw=0.8))


def render_msnr_chart(ltf, tr, out_path="msnr_alert.png", before=30, after=25, title=None):
    S = tr["side"] == "S"
    up = 1 if S else -1          # +1: level / touch wick are ABOVE price (SHORT); -1: below (LONG)
    a = max(0, tr["touch_i"] - before)
    b = min(len(ltf), tr["m"] + after + 1)
    w = ltf.iloc[a:b]
    o, h, l, c = (w[k].values for k in ("open", "high", "low", "close"))
    n = len(w)

    def X(i):
        return i - a

    fig = plt.figure(figsize=(11, 8.8), dpi=130)
    gs = fig.add_gridspec(2, 1, height_ratios=[4.2, 1.9], hspace=0.12)
    ax, tx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    tx.axis("off")

    for i in range(n):
        col = UP if c[i] >= o[i] else DOWN
        ax.plot([i, i], [l[i], h[i]], color=col, lw=1, zorder=2)
        ax.add_patch(Rectangle((i - 0.3, min(o[i], c[i])), 0.6, max(abs(c[i] - o[i]), 1e-9),
                               fc=col, ec=col, zorder=3))

    reasons = []

    def add(text):
        reasons.append((len(reasons) + 1, text))
        return len(reasons)

    # 1) HTF level
    name = "A-level (resistance)" if S else "V-level (support)"
    ax.axhline(tr["level"], color="#1e88e5", lw=1.3, ls="--", zorder=1)
    n1 = add(f"{name} {tr['level']:,.1f}  -  {tr['htf']} close-line (body only, wicks ignored)")
    _tag(ax, n1, 1, tr["level"], dy=up * 28)

    # 2) fresh
    born = tr.get("born_time")
    n2 = add("Fresh: no wick/body touch since it formed" + (f" ({born:%m-%d %H:%M} UTC)" if born is not None else ""))
    _tag(ax, n2, 6, tr["level"], dy=up * 28)

    # 3) HTF touch candle (wick only)
    s0, s1 = X(tr["touch_i"]), X(tr["start"])
    ax.axvspan(s0 - 0.5, s1 - 0.5, color="#90a4ae", alpha=0.18, zorder=0)
    ei = X(tr["ext_i"])
    n3 = add(f"{tr['htf']} touch candle ({ltf.index[tr['touch_i']]:%m-%d %H:%M}): wick-only touch, body stayed on the rejecting side")
    _tag(ax, n3, ei, tr["sl"], dy=up * 28)

    # 4) LTF pivot
    pi = X(tr["piv_i"])
    ax.hlines(tr["piv"], pi, X(tr["m"]), colors="#8e24aa", lw=1.1, ls=":", zorder=1)
    n4 = add(f"LTF pivot {'low' if S else 'high'} {tr['piv']:,.1f} (last confirmed pivot after the touch)")
    _tag(ax, n4, pi, tr["piv"], dy=-up * 28)

    # 5) shift candle = entry
    mi = X(tr["m"])
    ax.scatter([mi], [tr["entry"]], marker="v" if S else "^", s=70, color="#1565c0", zorder=6)
    n5 = add(f"Shift candle closed {'below' if S else 'above'} the pivot  =  ENTRY {tr['entry']:,.1f}")
    _tag(ax, n5, mi, tr["entry"], dx=24, dy=-up * 30)

    # 6) SL, 7) TP
    for val, col, label, txt in (
        (tr["sl"], DOWN, "SL", f"SL {tr['sl']:,.1f}  =  extreme since the touch (swept liquidity), risk {tr['risk'] / tr['entry'] * 100:.2f}%"),
        (tr["tp"], UP, "TP", f"TP {tr['tp']:,.1f}  =  {tr['R']}R"),
    ):
        ax.hlines(val, mi, n - 1, colors=col, lw=1.1, zorder=1)
        nn = add(txt)
        _tag(ax, nn, n - 4, val, dx=0, dy=24 if val > tr["entry"] else -24)
        ax.text(n + 3, val, f"{label} {val:,.1f}", color=col, fontsize=8, fontweight="bold", va="center")
    ax.text(n + 3, tr["entry"], f"ENTRY {tr['entry']:,.1f}", color="#1565c0", fontsize=8, fontweight="bold", va="center")

    # axes
    lo_, hi_ = min(l.min(), tr["tp"], tr["sl"]), max(h.max(), tr["tp"], tr["sl"], tr["level"])
    lo_ = min(lo_, tr["level"])
    pad = (hi_ - lo_) * 0.16
    ax.set_ylim(lo_ - pad, hi_ + pad)
    ax.set_xlim(-1, n + 20)
    step = max(1, n // 8)
    ticks = list(range(0, n, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{w.index[t]:%m-%d %H:%M}" for t in ticks], fontsize=7)
    ax.grid(alpha=0.15)
    ax.set_title(title or (f"BTCUSDT {'SHORT' if S else 'LONG'}  -  MSNR ({'A' if S else 'V'}-level)   |   "
                           f"QT quarter Q{tr['q']} (diagnostic)   |   {ltf.index[tr['m']]:%Y-%m-%d %H:%M} UTC"),
                 fontsize=11, fontweight="bold")

    # numbered list
    y = 0.98
    for num, text in reasons:
        tx.text(0.01, y, str(num), fontsize=9, fontweight="bold", va="top", ha="center", transform=tx.transAxes,
                bbox=dict(boxstyle="circle,pad=0.3", fc=TAG_FC, ec="black", lw=1))
        tx.text(0.04, y, text, fontsize=9.5, va="top", transform=tx.transAxes)
        y -= 0.135

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------- TradingView-style chart (1h candles like the MSNR indicator)
def htf_levels(h, k=3, max_age=400):
    """All close-line levels: (side, level, born_idx, end_idx). end_idx = first bar that touched it (None = still alive)."""
    o, hi, lo, c = (h[x].values for x in ("open", "high", "low", "close"))
    n, out = len(c), []
    for i in range(k, n - 1 - k):
        win = c[i - k:i + k + 1]
        if c[i] > o[i] and c[i + 1] < o[i + 1] and c[i] == win.max():
            side, lvl = "S", max(c[i], o[i + 1])
            if (hi[i + 2:i + k + 1] >= lvl).any():
                continue
        elif c[i] < o[i] and c[i + 1] > o[i + 1] and c[i] == win.min():
            side, lvl = "L", min(c[i], o[i + 1])
            if (lo[i + 2:i + k + 1] <= lvl).any():
                continue
        else:
            continue
        end = None
        for j in range(i + k + 1, min(n, i + max_age + 1)):
            if (side == "S" and hi[j] >= lvl) or (side == "L" and lo[j] <= lvl):
                end = j
                break
        out.append((side, lvl, i, end))
    return out


def _exit(tr, ltf, maxhold=288):
    """First LTF bar that hits SL or TP after entry (same-bar SL+TP counts as SL, like msnr_v3)."""
    hi, lo = ltf["high"].values, ltf["low"].values
    for t in range(tr["m"] + 1, min(len(ltf), tr["m"] + 1 + maxhold)):
        if tr["side"] == "S":
            if hi[t] >= tr["sl"]:
                return t, "SL"
            if lo[t] <= tr["tp"]:
                return t, "TP"
        else:
            if lo[t] <= tr["sl"]:
                return t, "SL"
            if hi[t] >= tr["tp"]:
                return t, "TP"
    return None, None


def _tag_dark(ax, n, x, y, dx=0, dy=22):
    ax.annotate(str(n), xy=(x, y), xytext=(dx, dy), textcoords="offset points", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=12,
                bbox=dict(boxstyle="circle,pad=0.3", fc="#263238", ec="white", lw=1),
                arrowprops=dict(arrowstyle="-", color="#263238", lw=0.8))


def render_msnr_chart_tv(ltf, tr, out_path="msnr_tv.png", before=45, ctx_levels=True, title=None):
    """HTF candles + level lines + yellow touch circle + black path arrows + grey/pink RR boxes."""
    from matplotlib.patches import Ellipse

    S = tr["side"] == "S"
    up = 1 if S else -1
    h = resample(ltf, tr["htf"])
    delta = pd.Timedelta(tr["htf"])
    ldt = ltf.index[1] - ltf.index[0]
    h0 = h.index[0]

    def tx_(ts):                       # time -> x in HTF-bar units (bar j spans [j, j+1])
        return (ts - h0) / delta

    j = int(h.index.get_indexer([ltf.index[tr["touch_i"]]])[0])
    ex_i, ex_res = _exit(tr, ltf)
    x_entry = tx_(ltf.index[tr["m"]] + ldt)
    x_exit = tx_(ltf.index[ex_i] + ldt) if ex_i is not None else None
    a = max(0, j - before)
    b = min(len(h), int(max(x_exit if x_exit else x_entry + 20, x_entry + 8)) + 3)
    b = max(b, j + 12)
    hw = h.iloc[a:b]
    o, hi, lo, c = (hw[k].values for k in ("open", "high", "low", "close"))
    n = len(hw)

    def X(x):
        return x - a

    bg = "#f0f3fa"
    fig = plt.figure(figsize=(12, 9.4), dpi=130, facecolor=bg)
    gs = fig.add_gridspec(2, 1, height_ratios=[4.8, 1.9], hspace=0.10)
    ax, tx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    ax.set_facecolor(bg)
    tx.axis("off")
    ymin, ymax = lo.min(), hi.max()
    yr = ymax - ymin

    # context levels (every close-line level; line stops where a later bar touched it)
    if ctx_levels:
        s0 = max(0, a - max_age_ctx)
        for side, lvl, born, end in htf_levels(h.iloc[s0:b + 4], k=3):
            born += s0
            end = None if end is None else end + s0
            if (end is not None and end < a) or born >= b or not (ymin - 0.1 * yr <= lvl <= ymax + 0.1 * yr):
                continue
            col = DOWN if side == "S" else UP
            x1 = (end + 0.5) if end is not None else b
            ax.hlines(lvl, X(born + 0.5), X(x1), colors=col, lw=0.9, zorder=1)
            ax.text(X(born + 0.5), lvl, f"{'A' if side == 'S' else 'V'} {lvl:,.0f}", color=col, fontsize=6.5,
                    va="bottom", ha="left", zorder=1)

    # candles
    for i in range(n):
        col = UP if c[i] >= o[i] else DOWN
        ax.plot([i + 0.5, i + 0.5], [lo[i], hi[i]], color=col, lw=1, zorder=3)
        ax.add_patch(Rectangle((i + 0.12, min(o[i], c[i])), 0.76, max(abs(c[i] - o[i]), 1e-9),
                               fc=col, ec=col, zorder=4))

    # the traded level, thicker
    lvl_col = DOWN if S else UP
    born = tr.get("born_time")
    xb = X(tx_(born) + 0.5) if born is not None else 0
    ax.hlines(tr["level"], max(xb, 0), X(j + 1.0 + 0.0), colors=lvl_col, lw=2.0, zorder=2)
    ax.hlines(tr["level"], X(j + 1.0), n, colors=lvl_col, lw=1.0, ls="--", zorder=2)

    reasons = []

    def add(t):
        reasons.append((len(reasons) + 1, t))
        return len(reasons)

    # 1/2 level + fresh
    name = "A-level (resistance)" if S else "V-level (support)"
    n1 = add(f"{name} {tr['level']:,.1f}  -  {tr['htf']} close-line (body only, wicks ignored)")
    _tag_dark(ax, n1, max(xb, 0) + 1.5, tr["level"], dy=up * 24)
    n2 = add("Fresh: no wick/body touch since it formed" + (f" ({born:%m-%d %H:%M} UTC)" if born is not None else ""))
    _tag_dark(ax, n2, max(xb, 0) + 4.5, tr["level"], dy=up * 24)

    # 3 touch circle
    tip = h["high"].values[j] if S else h["low"].values[j]
    cand_range = h["high"].values[j] - h["low"].values[j]
    ax.add_patch(Ellipse((X(j + 0.5), (h["high"].values[j] + h["low"].values[j]) / 2), 2.4,
                         max(cand_range * 1.7, yr * 0.07), fc="#fff59d", alpha=0.55, ec="black", lw=1.4, zorder=6))
    n3 = add(f"{tr['htf']} touch candle ({h.index[j]:%m-%d %H:%M}): wick-only touch, body stayed on the rejecting side")
    _tag_dark(ax, n3, X(j + 0.5), tip, dy=up * 34)

    # 4 pivot, 5 entry
    xp = X(tx_(ltf.index[tr["piv_i"]] + ldt / 2))
    xm = X(x_entry)
    ax.scatter([xp], [tr["piv"]], s=22, color="#8e24aa", zorder=8)
    n4 = add(f"LTF pivot {'low' if S else 'high'} {tr['piv']:,.1f} (last confirmed pivot after the touch)")
    _tag_dark(ax, n4, xp, tr["piv"], dy=-up * 26)
    # black path arrows: touch tip -> pivot -> entry
    for p0, p1 in (((X(j + 0.5), tip), (xp, tr["piv"])), ((xp, tr["piv"]), (xm, tr["entry"]))):
        ax.annotate("", xy=p1, xytext=p0, zorder=9, arrowprops=dict(arrowstyle="-|>", color="black", lw=1.7, shrinkA=0, shrinkB=0))
    n5 = add(f"Shift candle closed {'below' if S else 'above'} the pivot  =  ENTRY {tr['entry']:,.1f}")
    _tag_dark(ax, n5, xm, tr["entry"], dx=22, dy=-up * 24)

    # RR boxes: pink = risk (entry -> SL), grey = reward (entry -> TP)
    xe = X(x_exit) if x_exit is not None else n
    ax.add_patch(Rectangle((xm, min(tr["entry"], tr["sl"])), xe - xm, abs(tr["sl"] - tr["entry"]),
                           fc="#f48fb1", alpha=0.45, ec="none", zorder=0))
    ax.add_patch(Rectangle((xm, min(tr["entry"], tr["tp"])), xe - xm, abs(tr["tp"] - tr["entry"]),
                           fc="#757575", alpha=0.30, ec="none", zorder=0))
    end_y = tr["tp"] if ex_res == "TP" else tr["sl"] if ex_res == "SL" else c[-1]
    ax.plot([xm, xe], [tr["entry"], end_y], color="#616161", lw=0.9, ls=":", zorder=5)

    n6 = add(f"SL {tr['sl']:,.1f}  =  extreme since the touch (swept liquidity), risk {tr['risk'] / tr['entry'] * 100:.2f}%")
    _tag_dark(ax, n6, xm + (xe - xm) * 0.5, tr["sl"], dy=up * 22)
    n7 = add(f"TP {tr['tp']:,.1f}  =  {tr['R']}R   (result so far: {ex_res or 'still open'})")
    _tag_dark(ax, n7, xm + (xe - xm) * 0.5, tr["tp"], dy=-up * 22)

    # TradingView-like price boxes on the right axis
    for val, col, lab in ((tr["tp"], UP, "TP"), (tr["entry"], "#1565c0", "ENTRY"), (tr["sl"], DOWN, "SL")):
        ax.text(1.0, val, f" {lab} {val:,.1f} ", transform=ax.get_yaxis_transform(), color="white", fontsize=7.5,
                fontweight="bold", va="center", ha="left", zorder=15, bbox=dict(fc=col, ec="none", pad=1.5))

    # axes
    lo_all = min(ymin, tr["tp"], tr["sl"])
    hi_all = max(ymax, tr["tp"], tr["sl"])
    pad = (hi_all - lo_all) * 0.10
    ax.set_ylim(lo_all - pad, hi_all + pad)
    ax.set_xlim(0, n + 1)
    ax.yaxis.tick_right()
    for sp in ("top", "left"):
        ax.spines[sp].set_visible(False)
    step = max(1, n // 9)
    ticks = list(range(0, n, step))
    ax.set_xticks([t + 0.5 for t in ticks])
    ax.set_xticklabels([f"{hw.index[t]:%m-%d %H:%M}" for t in ticks], fontsize=7)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title(title or (f"BTCUSDT {tr['htf']}  -  MSNR {'SHORT (A-level)' if S else 'LONG (V-level)'}   |   "
                           f"QT quarter Q{tr['q']} (diagnostic)   |   entry {ltf.index[tr['m']]:%Y-%m-%d %H:%M} UTC"),
                 fontsize=11, fontweight="bold", loc="left")

    y = 0.98
    for num, text in reasons:
        tx.text(0.01, y, str(num), fontsize=8.5, fontweight="bold", color="white", va="top", ha="center",
                transform=tx.transAxes, bbox=dict(boxstyle="circle,pad=0.3", fc="#263238", ec="white", lw=1))
        tx.text(0.04, y, text, fontsize=9.5, va="top", transform=tx.transAxes)
        y -= 0.135
    fig.savefig(out_path, bbox_inches="tight", facecolor=bg)
    plt.close(fig)
    return out_path


max_age_ctx = 420


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--htf", default="1h")
    ap.add_argument("--k", type=int, default=3, help="pivot width on HTF close-line")
    ap.add_argument("--R", type=float, default=2, help="TP as a multiple of risk")
    ap.add_argument("--tz", type=float, default=0.0, help="hours added to UTC for the QT clock")
    ap.add_argument("--last", type=int, default=1, help="how many of the latest trades to draw")
    ap.add_argument("--out", default="msnr_alert", help="output file prefix")
    ap.add_argument("--style", choices=("tv", "ltf"), default="tv",
                    help="tv = HTF candles like the MSNR indicator, ltf = 5m zoom")
    a = ap.parse_args()

    ltf = synthetic() if a.synthetic or not a.csv else load_csv(a.csv)
    trades = build_trades(ltf, a.htf, a.k, R=a.R, tz=a.tz)
    print(f"trades found: {len(trades)}")
    for i, t in enumerate(trades[-a.last:], 1):
        fn = render_msnr_chart_tv if a.style == "tv" else render_msnr_chart
        print(fn(ltf, t, f"{a.out}_{i}.png"))


if __name__ == "__main__":
    main()
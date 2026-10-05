"""
chart_reasons.py  -  Full MSNR-style setup chart (A to Z concepts labelled ON the chart).

Supports: Classic A / Classic V, OCL (Open-Close Level / Gap), IDM (Inducement),
QM / QT (Quasimodo), SBR / RBS role-flips, Fresh / Unfresh, Storyline, Asia range +
Liquidity Sweep, FVG, KEY trendline, SMT, SHIFT LTF, ORDER 1 / ENTRY ORDER 2,
Wick Touch, Risk/Reward box with exact SL/TP distance in pts / % / R.

Usage in scanner:
    from chart_reasons import render_setup_chart
    path = render_setup_chart(df, setup, "alert.png")
    path = render_setup_chart(df, setup, "alert.png", theme="mono")

df: columns time, open, high, low, close ; index 0..n-1
All idx values in `setup` refer to rows of this df.

Required setup keys:
    side        "LONG" | "SHORT"
    entry, sl, tp

Optional (all drawn + numbered in Reasons list):
    symbol, tf          e.g. "XAUUSD", "H1"   -> title
    decimals            int (auto-detected from price if missing)
    unit                "pts" | "pips" | "points"  (default "pts")
    entry_idx, entry_why, sl_why, tp_why
    level       {"price", "label": "H1 POI / Classic V", "vol": float(opt)}
    wick_touch  {"idx"}
    swing_high  {"idx", "price"}                 # SWING HIGH (LONG) / SWING LOW (SHORT)
    shift       {"idx"}                          # SHIFT LTF
    asia        {"start","end","high","low","sweep_idx"}
    fvg         {"start", "top", "bottom"}
    ocl         {"idx", "price"} or {"idx","top","bottom"}  # OCL line or gap band
    idm         {"idx", "price"}                 # Inducement
    qm          {"idx", "price"} or {"left_idx","head_idx","right_idx",...}  # Quasimodo / QT
    trendline   {"p1":{"idx","price"}, "p2":{"idx","price"}, "p3_idx"(opt)}
    smt         {"p1":{"idx","price"}, "p2":{"idx","price"}, "pair": "..."}
    extras      list of dicts for anything else:
                {"idx","price","label","why"(opt),"line"(bool),"band"("top"|"bottom"),
                 "shade"(bool)  -> draws a small horizontal band if True}

Demo:
    python chart_reasons.py --demo
    python chart_reasons.py --demo-short
    python chart_reasons.py --demo-gold
    python chart_reasons.py --demo-mono
"""
import sys
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, FancyBboxPatch

THEMES = {
    "color": dict(BG="#f0f3fa", GRID="#e0e3eb", UP="#26a69a", DOWN="#ef5350"),
    "mono":  dict(BG="#f2f2f2", GRID="#dcdcdc", UP="#8c8c8c", DOWN="#111111"),
}
TXT = "#131722"
MUTED = "#787b86"
GREY_BOX = "#9598a1"
PINK_BOX = "#f7a1ac"
CIRCLE_FILL = "#fff3b0"
FVG_COL = "#ff9800"
OCL_COL = "#7b1fa2"
IDM_COL = "#c62828"
QM_COL  = "#1565c0"
ENTRY_COL = "#2962ff"
SL_COL = "#e53935"
TP_COL = "#1e9e8f"


def _auto_decimals(price):
    p = abs(float(price))
    if p >= 1000:
        return 1
    if p >= 10:
        return 2
    if p >= 1:
        return 3
    return 5


def _fmt(p, decimals=None):
    if decimals is None:
        decimals = _auto_decimals(p)
    return f"{p:,.{decimals}f}"


def _candles(ax, df, up, down):
    w = 0.62
    for i, r in df.iterrows():
        col = up if r["close"] >= r["open"] else down
        ax.plot([i, i], [r["low"], r["high"]], color=col, lw=1, zorder=3)
        lo, hi = sorted([r["open"], r["close"]])
        ax.add_patch(Rectangle((i - w / 2, lo), w, max(hi - lo, 1e-9),
                               facecolor=col, edgecolor=col, zorder=4))


def _tag(ax, x, y, text, color):
    ax.text(x, y, f" {text} ", fontsize=7, color="white", va="center", ha="left",
            zorder=8, clip_on=False,
            bbox=dict(boxstyle="square,pad=0.15", fc=color, ec=color))


def render_setup_chart(df, setup, out_path="alert.png", last_n=62, title=None, theme="color"):
    T = THEMES.get(theme, THEMES["color"])
    BG, GRID, UP, DOWN = T["BG"], T["GRID"], T["UP"], T["DOWN"]
    df = df.reset_index(drop=True)
    n = len(df)
    long_ = str(setup.get("side", "LONG")).upper() == "LONG"
    side = "LONG" if long_ else "SHORT"

    # adaptive formatting
    entry0 = float(setup["entry"])
    decimals = setup.get("decimals")
    if decimals is None:
        decimals = _auto_decimals(entry0)
    unit = setup.get("unit", "pts")

    def fmt(p):
        return _fmt(p, decimals)

    # ---------- window: always show every element of the setup ----------
    lo_i = max(0, n - last_n)
    idxs = []
    for key, sub in setup.items():
        if isinstance(sub, dict):
            for kk in ("idx", "start", "sweep_idx", "left_idx", "head_idx", "right_idx"):
                if sub.get(kk) is not None:
                    idxs.append(int(sub[kk]))
        elif key == "extras":
            idxs += [int(e["idx"]) for e in sub if e.get("idx") is not None]
    for k in ("trendline", "smt"):
        sub = setup.get(k)
        if sub:
            idxs += [int(sub["p1"]["idx"]), int(sub["p2"]["idx"])]
    if idxs:
        lo_i = max(0, min(lo_i, min(idxs) - 4))
    view = df.iloc[lo_i:].reset_index(drop=True)
    sh = lo_i
    m = len(view)

    def X(i):
        return i - sh

    entry = float(setup["entry"])
    sl = float(setup["sl"])
    tp = float(setup["tp"])
    entry_i = int(setup.get("entry_idx", n - 1))
    ex = X(entry_i)
    xmax = m + 16

    d_sl = abs(entry - sl)
    d_tp = abs(tp - entry)
    p_sl = d_sl / entry * 100
    p_tp = d_tp / entry * 100
    rr = d_tp / max(d_sl, 1e-9)

    # ---------- pass 1: reasons + callouts ----------
    mid = (view["high"].max() + view["low"].min()) / 2
    reasons, callouts, nums = [], [], {}

    def reg(name, why):
        k = len(reasons) + 1
        reasons.append((k, why))
        nums[name] = k
        return k

    # Level / Classic A or V / POI
    lv = setup.get("level")
    if lv:
        lab = lv.get("label", "KEY LEVEL")
        reg("level", f"{lab} {fmt(lv['price'])}: Classic {'A (resistance)' if not long_ else 'V (support)'} / POI where orders rest. Fresh levels preferred.")

    # Wick touch
    wt = setup.get("wick_touch")
    if wt:
        r = df.loc[wt["idx"]]
        k = reg("wick", f"Wick went {'below' if long_ else 'above'} the level and was rejected – no body close beyond it (valid rejection).")
        callouts.append((k, X(wt["idx"]), r["low"] if long_ else r["high"], "WICK TOUCH",
                         "bottom" if long_ else "top"))

    # Swing
    sj = setup.get("swing_high")
    sj_name = "SWING HIGH" if long_ else "SWING LOW"
    if sj:
        reg("swing", f"{sj_name} {fmt(sj['price'])}: structure point that must break to confirm the directional move.")

    # Shift LTF
    sf = setup.get("shift")
    if sf:
        r = df.loc[sf["idx"]]
        k = reg("shift", "SHIFT LTF: candle closes through the swing = lower-timeframe structure shift, direction confirmed.")
        callouts.append((k, X(sf["idx"]), r["high"] if long_ else r["low"], "SHIFT LTF",
                         "top" if long_ else "bottom"))

    # Asia + Liquidity Sweep
    a = setup.get("asia")
    if a:
        sw = a.get("sweep_idx")
        why = f"Asia range {fmt(a['low'])} – {fmt(a['high'])}"
        if sw is not None:
            why += f": {'low' if long_ else 'high'} swept → liquidity taken (classic MSNR liquidity grab)."
        k = reg("asia", why)
        if sw is not None:
            callouts.append((k, X(sw), df.loc[sw, "low"] if long_ else df.loc[sw, "high"],
                             "LIQUIDITY SWEEP", "bottom" if long_ else "top"))

    # FVG
    f = setup.get("fvg")
    if f:
        fm = (f["top"] + f["bottom"]) / 2
        k = reg("fvg", f"FVG {fmt(f['bottom'])} – {fmt(f['top'])}: imbalance left by impulse. Entry on retest of the gap.")
        callouts.append((k, X(f["start"]) + 1, fm, "FVG ENTRY", "top" if fm >= mid else "bottom"))

    # OCL (Open-Close Level)
    ocl = setup.get("ocl")
    if ocl:
        if "top" in ocl and "bottom" in ocl:
            mid_ocl = (ocl["top"] + ocl["bottom"]) / 2
            k = reg("ocl", f"OCL Gap {fmt(ocl['bottom'])} – {fmt(ocl['top'])}: Open-Close Level (gap between consecutive candles). Strong reaction zone in MSNR.")
            callouts.append((k, X(ocl["idx"]), mid_ocl, "OCL GAP", "top" if mid_ocl >= mid else "bottom"))
        else:
            k = reg("ocl", f"OCL {fmt(ocl['price'])}: Open-Close Level – body-to-body level (ignores wicks). Core MSNR concept.")
            callouts.append((k, X(ocl["idx"]), ocl["price"], "OCL", "top" if ocl["price"] >= mid else "bottom"))

    # IDM (Inducement)
    idm = setup.get("idm")
    if idm:
        k = reg("idm", f"IDM {fmt(idm['price'])}: Inducement – minor swing that traps early entries. Smart money sweeps it before the real move.")
        callouts.append((k, X(idm["idx"]), idm["price"], "IDM", "top" if idm["price"] >= mid else "bottom"))

    # QM / QT (Quasimodo)
    qm = setup.get("qm")
    if qm:
        if "head_idx" in qm:
            k = reg("qm", "QM / QT (Quasimodo): left shoulder – head – right shoulder. Trade the return to the left shoulder after the head is taken.")
            # will draw later
        else:
            k = reg("qm", f"QM / QT {fmt(qm['price'])}: Quasimodo level – strong reversal structure used heavily with MSNR storyline.")
            callouts.append((k, X(qm["idx"]), qm["price"], "QM / QT", "top" if qm["price"] >= mid else "bottom"))

    # Trendline
    tl = setup.get("trendline")
    tl_end = None
    if tl:
        i1, i2 = int(tl["p1"]["idx"]), int(tl["p2"]["idx"])
        pr1, pr2 = float(tl["p1"]["price"]), float(tl["p2"]["price"])
        i3 = int(tl.get("p3_idx", entry_i))
        pr3 = pr1 + (pr2 - pr1) / max(i2 - i1, 1) * (i3 - i1)
        tl_end = (i3, pr3)
        reg("tl", f"KEY trendline: touches 1 ({fmt(pr1)}) & 2 ({fmt(pr2)}). Entry on 3rd touch / break, preferably with a KEY / OCL level.")

    # SMT
    sm = setup.get("smt")
    if sm:
        spr1, spr2 = float(sm["p1"]["price"]), float(sm["p2"]["price"])
        pair = sm.get("pair", "correlated pair")
        reg("smt", f"SMT ({pair}): {'lower low' if long_ else 'higher high'} here while pair did not → divergence confirms the setup.")

    # Extras (ORDER 1, ENTRY ORDER 2, SBR, RBS, Storyline, Fresh, etc.)
    extra_items = []
    for e in setup.get("extras", []):
        k = reg(f"x{len(extra_items)}", e.get("why", e["label"]))
        extra_items.append((k, e))
        if not e.get("line") and not e.get("shade"):
            callouts.append((k, X(e["idx"]), e["price"], e["label"],
                             e.get("band") or ("top" if e["price"] >= mid else "bottom")))

    # Entry / SL / TP always last so numbers are high
    reg("entry", setup.get("entry_why") or
        f"ENTRY {fmt(entry)}: limit at the {'FVG / OCL / POI' if f or ocl else 'POI'} retest after LTF shift / IDM sweep.")
    reg("sl", setup.get("sl_why") or
        f"SL {fmt(sl)}: {'below' if long_ else 'above'} the zone / sweep / IDM. Close beyond = setup invalid. "
        f"Risk {d_sl:,.{decimals}f} {unit} ({p_sl:.2f}%) = 1R.")
    reg("tp", setup.get("tp_why") or
        f"TP {fmt(tp)}: next liquidity / opposite key level / Classic A or V. Reward {d_tp:,.{decimals}f} {unit} ({p_tp:.2f}%) = {rr:.1f}R.")

    # ---------- y-range ----------
    ys = [view["high"].max(), view["low"].min(), entry, sl]
    if tl:
        ys += [pr1, pr2]
    if sm:
        ys += [spr1, spr2]
    if ocl:
        if "top" in ocl:
            ys += [ocl["top"], ocl["bottom"]]
        else:
            ys.append(ocl["price"])
    if idm:
        ys.append(idm["price"])
    if qm and "price" in qm:
        ys.append(qm["price"])
    for e in setup.get("extras", []):
        if "price" in e:
            ys.append(e["price"])
    span0 = max(ys) - min(ys)
    if (min(ys) - 1.0 * span0) <= tp <= (max(ys) + 1.0 * span0):
        ys.append(tp)
    lo_y, hi_y = min(ys), max(ys)
    span = hi_y - lo_y
    n_top = min(4, sum(1 for c in callouts if c[4] == "top"))
    n_bot = min(4, sum(1 for c in callouts if c[4] == "bottom"))
    ymax = hi_y + span * (0.11 + (0.035 + 0.045 * n_top if n_top else 0))
    ymin = lo_y - span * (0.13 + (0.035 + 0.045 * n_bot if n_bot else 0))
    rng = ymax - ymin

    # ---------- figure ----------
    fig = plt.figure(figsize=(11.2, 9.0), dpi=130, facecolor=BG)
    ax = fig.add_axes([0.03, 0.28, 0.87, 0.62], facecolor=BG)
    ax.grid(color=GRID, lw=0.7, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.yaxis.tick_right()
    ax.tick_params(colors=MUTED, labelsize=7, length=0)
    ax.set_xlim(-1, xmax)
    ax.set_ylim(ymin, ymax)

    def num(name):
        return nums.get(name, "")

    # ---------- zones (behind candles) ----------
    if a:
        x0, x1 = X(a["start"]), X(a["end"])
        ax.add_patch(Rectangle((x0, a["low"]), x1 - x0, a["high"] - a["low"],
                               fc=GREY_BOX, ec="none", alpha=0.33, zorder=1))
        ax.text(x0 + 0.5, a["high"] if long_ else a["low"], f"{num('asia')}  ASIA RANGE",
                fontsize=7, color=TXT, fontweight="bold", va="bottom" if long_ else "top", zorder=6)
        sw = a.get("sweep_idx")
        if sw is not None:
            if long_ and df.loc[sw, "low"] < a["low"]:
                y0 = float(df.loc[sw, "low"])
                ax.add_patch(Rectangle((x0, y0), (X(sw) + 3) - x0, a["low"] - y0,
                                       fc=PINK_BOX, ec="none", alpha=0.55, zorder=1))
            elif (not long_) and df.loc[sw, "high"] > a["high"]:
                y1 = float(df.loc[sw, "high"])
                ax.add_patch(Rectangle((x0, a["high"]), (X(sw) + 3) - x0, y1 - a["high"],
                                       fc=PINK_BOX, ec="none", alpha=0.55, zorder=1))

    if f:
        x0 = X(f["start"])
        ax.add_patch(Rectangle((x0, f["bottom"]), (m - 0.5) - x0, f["top"] - f["bottom"],
                               fc=FVG_COL, ec="none", alpha=0.22, zorder=1))

    # OCL gap band
    if ocl and "top" in ocl and "bottom" in ocl:
        x0 = X(ocl["idx"])
        ax.add_patch(Rectangle((x0 - 0.4, ocl["bottom"]), 4.5, ocl["top"] - ocl["bottom"],
                               fc=OCL_COL, ec="none", alpha=0.18, zorder=1))

    # risk / reward boxes
    edge = rng * 0.02
    tp_draw = min(max(tp, ymin + edge), ymax - edge)
    tp_clamped = abs(tp_draw - tp) > 1e-9
    bx0, bx1 = ex, xmax - 1
    ax.add_patch(Rectangle((bx0, min(entry, tp_draw)), bx1 - bx0, abs(tp_draw - entry),
                           fc=TP_COL, ec="none", alpha=0.13, zorder=1))
    ax.add_patch(Rectangle((bx0, min(entry, sl)), bx1 - bx0, abs(sl - entry),
                           fc=SL_COL, ec="none", alpha=0.16, zorder=1))

    _candles(ax, view, UP, DOWN)

    # ---------- level ----------
    if lv:
        ax.plot([-1, m], [lv["price"]] * 2, color=DOWN if long_ else UP, lw=1.1, zorder=2)
        lab = f"{num('level')}  {lv.get('label', 'KEY LEVEL')}  {fmt(lv['price'])}"
        if "vol" in lv and lv["vol"] is not None:
            lab += f"   Vol: {lv['vol']:.2f}"
        ax.text(-0.5, lv["price"], lab, fontsize=7.5, color=TXT, fontweight="bold", ha="left",
                va="bottom", zorder=7,
                bbox=dict(boxstyle="square,pad=0.15", fc=BG, ec="none", alpha=0.9))

    # ---------- OCL single line ----------
    if ocl and "price" in ocl and "top" not in ocl:
        ax.plot([-1, m], [ocl["price"]] * 2, color=OCL_COL, lw=1.2, ls="--", zorder=2)
        ax.text(-0.5, ocl["price"], f"{num('ocl')}  OCL  {fmt(ocl['price'])}",
                fontsize=7.5, color=OCL_COL, fontweight="bold", ha="left", va="bottom", zorder=7,
                bbox=dict(boxstyle="square,pad=0.15", fc=BG, ec="none", alpha=0.9))

    # ---------- IDM ----------
    if idm:
        ax.plot([X(idm["idx"]) - 1.5, X(idm["idx"]) + 1.5], [idm["price"]] * 2,
                color=IDM_COL, lw=2.0, zorder=6)
        ax.plot(X(idm["idx"]), idm["price"], "D", ms=7, color=IDM_COL, zorder=11)  # diamond

    # ---------- QM simple ----------
    if qm and "price" in qm and "head_idx" not in qm:
        ax.plot([X(qm["idx"]) - 1.5, X(qm["idx"]) + 1.5], [qm["price"]] * 2,
                color=QM_COL, lw=2.0, zorder=6)
        ax.plot(X(qm["idx"]), qm["price"], "s", ms=6, color=QM_COL, zorder=11)

    # ---------- full QM structure if head provided ----------
    if qm and "head_idx" in qm:
        li = int(qm.get("left_idx", qm["head_idx"] - 5))
        hi = int(qm["head_idx"])
        ri = int(qm.get("right_idx", qm["head_idx"] + 5))
        lp = float(qm.get("left_price", df.loc[li, "high" if not long_ else "low"]))
        hp = float(qm.get("head_price", df.loc[hi, "high" if not long_ else "low"]))
        rp = float(qm.get("right_price", df.loc[ri, "high" if not long_ else "low"]))
        ax.plot([X(li), X(hi), X(ri)], [lp, hp, rp], color=QM_COL, lw=1.4, zorder=6)
        for xx, yy, lab in ((X(li), lp, "L"), (X(hi), hp, "H"), (X(ri), rp, "R")):
            ax.plot(xx, yy, "o", ms=5, color=QM_COL, zorder=11)
            ax.annotate(lab, (xx, yy), xytext=(0, 9 if yy >= mid else -12),
                        textcoords="offset points", ha="center", fontsize=7,
                        fontweight="bold", color=QM_COL, zorder=12)
        ax.text(X(ri) + 0.8, rp, f"{num('qm')}  QM / QT", fontsize=7.5, color=QM_COL,
                fontweight="bold", ha="left", va="center", zorder=7,
                bbox=dict(boxstyle="square,pad=0.15", fc=BG, ec="none", alpha=0.9))

    # ---------- wick touch circle ----------
    wt_xy = None
    if wt:
        i = wt["idx"]
        r = df.loc[i]
        cy = (r["high"] + r["low"]) / 2
        h = max(r["high"] - r["low"], rng * 0.05) * 1.7
        ax.add_patch(Ellipse((X(i), cy), 5.6, h, fc=CIRCLE_FILL, ec="black", lw=1.5,
                             alpha=0.6, zorder=5))
        wt_xy = (X(i), r["low"] if long_ else r["high"])

    # ---------- swing line ----------
    sj_xy = None
    if sj:
        sx = X(sj["idx"])
        ax.plot([sx, ex], [sj["price"]] * 2, color=TXT, lw=1, ls=":", zorder=5)
        ax.plot(sx, sj["price"], "o", ms=3.5, color=TXT, zorder=11)
        ax.text(sx + 0.6, sj["price"], f"{num('swing')}  {sj_name}  {fmt(sj['price'])}",
                fontsize=7.5, color=TXT, fontweight="bold", ha="left",
                va="bottom" if long_ else "top", zorder=7,
                bbox=dict(boxstyle="square,pad=0.15", fc=BG, ec="none", alpha=0.9))
        sj_xy = (sx, sj["price"])

    # ---------- KEY trendline ----------
    if tl:
        x1_, x2_, x3_ = X(i1), X(i2), X(tl_end[0])
        ax.plot([x1_, x3_], [pr1, tl_end[1]], color=TXT, lw=1.3, ls=(0, (1, 2.2)), zorder=6)
        for xx, yy, tt in ((x1_, pr1, "1"), (x2_, pr2, "2")):
            ax.plot(xx, yy, "o", ms=8, color=SL_COL, zorder=11)
            ax.annotate(tt, (xx, yy), xytext=(0, 11 if yy >= mid else -15), textcoords="offset points",
                        ha="center", fontsize=8, fontweight="bold", color=TXT, zorder=12)
        if "p3_idx" in tl:
            ax.plot(x3_, tl_end[1], "o", ms=8, color=SL_COL, zorder=11)
            ax.annotate("3", (x3_, tl_end[1]), xytext=(0, 11 if tl_end[1] >= mid else -15),
                        textcoords="offset points", ha="center", fontsize=8, fontweight="bold",
                        color=TXT, zorder=12)
        ax.text(x3_ + 0.6, tl_end[1], f"{num('tl')}  TREND LINE", fontsize=7.5, color=TXT,
                fontweight="bold", ha="left", va="center", zorder=7,
                bbox=dict(boxstyle="square,pad=0.15", fc=BG, ec="none", alpha=0.9))

    # ---------- SMT ----------
    if sm:
        sx1, sx2 = X(int(sm["p1"]["idx"])), X(int(sm["p2"]["idx"]))
        ax.plot([sx1, sx2], [spr1, spr2], color=TXT, lw=1.5, zorder=6)
        ax.plot([sx1, sx2], [spr1, spr2], "o", ms=4.5, color=TXT, zorder=11)
        (dx1, dy1), (dx2, dy2) = ax.transData.transform([(sx1, spr1), (sx2, spr2)])
        ang = float(np.degrees(np.arctan2(dy2 - dy1, dx2 - dx1)))
        ax.text((sx1 + sx2) / 2, (spr1 + spr2) / 2, f"{num('smt')}  SMT", rotation=ang,
                rotation_mode="anchor", ha="center", va="top" if long_ else "bottom",
                fontsize=8, fontweight="bold", color=TXT, zorder=12)

    # ---------- shift candle box ----------
    sf_xy = None
    if sf:
        r = df.loc[sf["idx"]]
        ax.add_patch(Rectangle((X(sf["idx"]) - 0.7, r["low"]), 1.4, r["high"] - r["low"],
                               fc="none", ec=TXT, lw=1.2, ls="--", zorder=6))
        sf_xy = (X(sf["idx"]), r["high"] if long_ else r["low"])

    # ---------- extras (lines + shaded bands) ----------
    for k, e in extra_items:
        x0 = X(e["idx"])
        if e.get("shade") and "top" in e and "bottom" in e:
            ax.add_patch(Rectangle((x0 - 0.3, e["bottom"]), 3.5, e["top"] - e["bottom"],
                                   fc=GREY_BOX, ec="none", alpha=0.25, zorder=1))
        if e.get("line") or e.get("shade"):
            ax.plot([x0, ex], [e["price"]] * 2, color=TXT, lw=1, zorder=5)
            ax.plot(x0, e["price"], "o", ms=3.5, color=TXT, zorder=11)
            ax.text(x0 + 0.6, e["price"], f"{k}  {e['label']}  {fmt(e['price'])}",
                    fontsize=7.5, color=TXT, fontweight="bold", ha="left", va="bottom", zorder=7,
                    bbox=dict(boxstyle="square,pad=0.15", fc=BG, ec="none", alpha=0.9))

    # ---------- path arrows ----------
    pts = []
    if wt_xy:
        pts.append(wt_xy)
    if sj_xy:
        pts.append(sj_xy)
    if a and a.get("sweep_idx") is not None:
        sw = a["sweep_idx"]
        pts.append((X(sw), df.loc[sw, "low"] if long_ else df.loc[sw, "high"]))
    elif sf_xy:
        pts.append(sf_xy)
    for p, q in zip(pts[:-1], pts[1:]):
        ax.annotate("", xy=q, xytext=p, zorder=9,
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3, alpha=0.8))

    # ---------- callouts with leader lines ----------
    for band in ("top", "bottom"):
        items = sorted([c for c in callouts if c[4] == band], key=lambda c: c[1])
        for i, (k, x, y, text, _) in enumerate(items):
            tier = i % 4
            ty = ymax - (0.030 + 0.048 * tier) * rng if band == "top" else ymin + (0.030 + 0.048 * tier) * rng
            ha, tx = ("right", x + 1) if x > m - 18 else ("left", x - 1)
            ax.plot(x, y, "o", ms=3.5, color=TXT, zorder=11)
            ax.annotate(f"{k}  {text}", xy=(x, y), xytext=(tx, ty), ha=ha, va="center",
                        fontsize=7.5, color=TXT, fontweight="bold", zorder=12,
                        arrowprops=dict(arrowstyle="-", color=TXT, lw=0.8, shrinkA=0, shrinkB=2),
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=TXT, lw=0.8))

    # ---------- entry / SL / TP ----------
    ax.plot([bx0, bx1], [entry, entry], color=ENTRY_COL, lw=1.4, zorder=7)
    ax.plot([bx0, bx1], [sl, sl], color=SL_COL, lw=1.2, zorder=7)
    ax.plot([bx0, bx1], [tp_draw, tp_draw], color=TP_COL, lw=1.2, ls="--" if tp_clamped else "-", zorder=7)
    ax.plot([bx0, bx1], [entry, tp_draw], color=TXT, lw=1, ls="--", zorder=6)
    ax.text(bx0 - 0.4, entry, f" {num('entry')} ENTRY ", fontsize=7, color="white", va="center", ha="right", zorder=8,
            bbox=dict(boxstyle="square,pad=0.15", fc=ENTRY_COL, ec=ENTRY_COL))

    tp_arrow = ("" if not tp_clamped else (" ▲" if tp > tp_draw else " ▼"))
    ax.text(bx0 + 0.6, tp_draw,
            f"{num('tp')}  TP{tp_arrow}  {fmt(tp)}\n+{d_tp:,.{decimals}f} {unit}  (+{p_tp:.2f}%)   {rr:.1f}R",
            fontsize=7, color=TP_COL, fontweight="bold", ha="left",
            va="top" if tp_draw > entry else "bottom", zorder=8, linespacing=1.25)
    ax.text(bx0 + 0.6, sl,
            f"{num('sl')}  SL  {fmt(sl)}\n-{d_sl:,.{decimals}f} {unit}  (-{p_sl:.2f}%)   1R",
            fontsize=7, color=SL_COL, fontweight="bold", ha="left",
            va="top" if sl < entry else "bottom", zorder=8, linespacing=1.25)

    tags = sorted([[entry, fmt(entry), ENTRY_COL], [sl, fmt(sl), SL_COL], [tp_draw, fmt(tp), TP_COL]],
                  key=lambda t: t[0])
    gap = rng * 0.032
    for i in range(1, len(tags)):
        if tags[i][0] - tags[i - 1][0] < gap:
            tags[i][0] = tags[i - 1][0] + gap
    for y_, txt_, col_ in tags:
        _tag(ax, xmax + 0.3, y_, txt_, col_)

    # ---------- time axis ----------
    step = max(1, m // 8)
    ticks = list(range(0, m, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([pd.to_datetime(view["time"].iloc[t]).strftime("%d %H:%M") for t in ticks])

    # ---------- header ----------
    sym = title or f"{setup.get('symbol', 'BTCUSDT')} {setup.get('tf', '1h')}"
    fig.text(0.035, 0.965, f"{sym}   |   {side} setup", fontsize=12, color=TXT, fontweight="bold", va="center")
    fig.text(0.035, 0.935,
             f"Entry {fmt(entry)}    SL {fmt(sl)}  (-{d_sl:,.{decimals}f} {unit}, -{p_sl:.2f}%)    "
             f"TP {fmt(tp)}  (+{d_tp:,.{decimals}f} {unit}, +{p_tp:.2f}%)    RR 1:{rr:.2f}",
             fontsize=9, color=TXT, va="center")

    # ---------- reasons list ----------
    fig.text(0.035, 0.245, "Reasons (numbers match the chart)  •  MSNR concepts",
             fontsize=10, color=TXT, fontweight="bold")
    rows = max(1, (len(reasons) + 1) // 2)
    row_h = min(0.052, 0.195 / rows)
    for j, (k, t) in enumerate(reasons):
        col, r_ = divmod(j, rows)
        cx = 0.042 + col * 0.49
        yy = 0.205 - r_ * row_h
        fig.text(cx, yy, str(k), fontsize=8, color="white", fontweight="bold", ha="center", va="center",
                 bbox=dict(boxstyle="circle,pad=0.25", fc=TXT, ec=TXT))
        fig.text(cx + 0.022, yy, textwrap.fill(t, 60), fontsize=8.2, color=TXT, va="center", linespacing=1.15)

    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)
    return out_path


def _make_base_df(seed=7, n=140, base_price=77000):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2026-10-01", periods=n, freq="h")
    anchors_x = [0, 30, 52, 68, 72, 80, 86, 92, 100, 115, 139]
    scale = base_price / 77000
    anchors_y = [a * scale for a in [77300, 77000, 76900, 76850, 76700, 76850, 76620, 77000, 77150, 77080, 77120]]
    noise = 25 * scale
    c = np.interp(np.arange(n), anchors_x, anchors_y) + rng.normal(0, noise, n)
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + rng.uniform(15 * scale, 70 * scale, n)
    l = np.minimum(o, c) - rng.uniform(15 * scale, 70 * scale, n)
    return pd.DataFrame(dict(time=t, open=o, high=h, low=l, close=c))


def _demo(short=False, out="demo_chart.png", theme="color", gold=False):
    base = 3000.0 if gold else 77000.0
    df = _make_base_df(seed=7, base_price=base)
    scale = base / 77000

    touch, sh_i, sweep, shift, fv = 72, 80, 86, 92, 96
    a0, a1 = 52, 68
    asia_hi = float(df.loc[a0:a1, "high"].max())
    asia_lo = float(df.loc[a0:a1, "low"].min())
    level = asia_lo - 20 * scale
    df.loc[touch, "low"] = level - 25 * scale
    df.loc[sweep, "low"] = asia_lo - 110 * scale
    swing_px = float(df.loc[sh_i, "high"])
    df.loc[shift, "open"] = swing_px - 60 * scale
    df.loc[shift, "close"] = swing_px + 90 * scale
    df.loc[shift, "high"] = swing_px + 110 * scale
    gap_bot = float(df.loc[fv - 2, "high"])
    gap_top = gap_bot + 60 * scale
    df.loc[fv - 1, "open"] = gap_bot - 10 * scale
    df.loc[fv - 1, "close"] = gap_top + 40 * scale
    df.loc[fv - 1, "high"] = gap_top + 50 * scale
    df.loc[fv, "open"] = gap_top + 10 * scale
    df.loc[fv, "close"] = max(float(df.loc[fv, "close"]), gap_top + 30 * scale)
    df.loc[fv, "low"] = gap_top
    df.loc[fv, "high"] = max(float(df.loc[fv, "high"]), float(df.loc[fv, "close"]) + 10 * scale)

    entry_idx = fv + 2
    entry = gap_top
    sl = gap_bot - 40 * scale
    tp = entry + 3 * (entry - sl)
    df.loc[entry_idx, "low"] = min(float(df.loc[entry_idx, "low"]), entry - 5 * scale)
    tl_a, tl_b = 50, sh_i
    tl_pa = swing_px + 120 * scale
    df.loc[tl_a, "high"] = tl_pa

    # IDM just before the real POI
    idm_idx = touch - 3
    idm_price = level + 35 * scale

    # OCL near the level
    ocl_idx = a1 - 2
    ocl_price = asia_lo + 5 * scale

    setup = {
        "side": "LONG",
        "symbol": "XAUUSD" if gold else "BTCUSDT",
        "tf": "H1",
        "entry": entry, "sl": sl, "tp": tp, "entry_idx": entry_idx,
        "level": {"price": level, "label": "Classic V / H1 POI", "vol": -1658.96 if not gold else None},
        "wick_touch": {"idx": touch},
        "swing_high": {"idx": sh_i, "price": swing_px},
        "shift": {"idx": shift},
        "asia": {"start": a0, "end": a1, "high": asia_hi, "low": asia_lo, "sweep_idx": sweep},
        "fvg": {"start": fv, "top": gap_top, "bottom": gap_bot},
        "ocl": {"idx": ocl_idx, "price": ocl_price},
        "idm": {"idx": idm_idx, "price": idm_price},
        "qm": {"idx": sh_i - 4, "price": swing_px + 40 * scale},
        "smt": {"p1": {"idx": touch, "price": float(df.loc[touch, "low"])},
                "p2": {"idx": sweep, "price": float(df.loc[sweep, "low"])},
                "pair": "XAUUSD vs XAGUSD" if gold else "BTCUSDT vs ETHUSDT"},
        "trendline": {"p1": {"idx": tl_a, "price": tl_pa}, "p2": {"idx": tl_b, "price": swing_px}},
        "extras": [
            {"idx": touch + 2, "price": level, "label": "ORDER 1",
             "why": "ORDER 1: first small-size entry at the POI before confirmation.", "line": True},
            {"idx": entry_idx - 1, "price": entry, "label": "ENTRY ORDER 2",
             "why": "ENTRY ORDER 2: main position after SHIFT LTF + IDM sweep + FVG retest.", "line": True},
            {"idx": a0 + 3, "price": asia_hi, "label": "SBR",
             "why": "SBR (Support Break Resistance): previous support broken, now acting as resistance (storyline).", "line": True},
        ],
    }
    if gold:
        setup["decimals"] = 2
        setup["unit"] = "pts"

    if short:
        P0 = base
        mir = lambda p: 2 * P0 - p
        d2 = df.copy()
        d2["open"], d2["close"] = mir(df["open"]), mir(df["close"])
        d2["high"], d2["low"] = mir(df["low"]), mir(df["high"])
        df = d2
        for k in ("entry", "sl", "tp"):
            setup[k] = mir(setup[k])
        setup["side"] = "SHORT"
        setup["level"]["price"] = mir(setup["level"]["price"])
        setup["level"]["label"] = "Classic A / H1 POI"
        setup["swing_high"]["price"] = mir(setup["swing_high"]["price"])
        setup["asia"]["high"], setup["asia"]["low"] = mir(asia_lo), mir(asia_hi)
        setup["fvg"]["top"], setup["fvg"]["bottom"] = mir(gap_bot), mir(gap_top)
        setup["ocl"]["price"] = mir(setup["ocl"]["price"])
        setup["idm"]["price"] = mir(setup["idm"]["price"])
        setup["qm"]["price"] = mir(setup["qm"]["price"])
        setup["smt"]["p1"]["price"] = mir(setup["smt"]["p1"]["price"])
        setup["smt"]["p2"]["price"] = mir(setup["smt"]["p2"]["price"])
        setup["trendline"]["p1"]["price"] = mir(tl_pa)
        setup["trendline"]["p2"]["price"] = mir(swing_px)
        for e in setup["extras"]:
            e["price"] = mir(e["price"])
        if setup["extras"][2]["label"] == "SBR":
            setup["extras"][2]["label"] = "RBS"
            setup["extras"][2]["why"] = "RBS (Resistance Break Support): previous resistance broken, now acting as support."

    return render_setup_chart(df, setup, out, theme=theme)


if __name__ == "__main__":
    if "--demo" in sys.argv:
        print("saved", _demo())
    elif "--demo-short" in sys.argv:
        print("saved", _demo(short=True, out="demo_chart_short.png"))
    elif "--demo-gold" in sys.argv:
        print("saved", _demo(gold=True, out="demo_chart_gold.png"))
    elif "--demo-mono" in sys.argv:
        print("saved", _demo(out="demo_chart_mono.png", theme="mono"))
    else:
        print(__doc__)
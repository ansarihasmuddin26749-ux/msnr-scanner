"""
chart_reasons.py  -  Clean MSNR chart (image-3 style) + SMT

Shows:
  - Classic A / Classic V (main POI)
  - Asia range + Liquidity Sweep
  - OCL
  - SHIFT LTF
  - SMT divergence
  - IDM / QM (subtle, if present)
  - Entry / SL / TP with clear R
  - Max 6 reasons

Usage:
    from chart_reasons import render_setup_chart
    path = render_setup_chart(df, setup, "alert.png")
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

THEMES = {
    "color": dict(BG="#f7f8fc", GRID="#e8ebf2", UP="#26a69a", DOWN="#ef5350"),
    "mono":  dict(BG="#f2f2f2", GRID="#dcdcdc", UP="#8c8c8c", DOWN="#111111"),
}
TXT = "#1a1d26"
MUTED = "#6b7280"
ASIA_FILL = "#c5c9d4"
SWEEP_FILL = "#f8b4c0"
ENTRY_COL = "#2563eb"
SL_COL = "#dc2626"
TP_COL = "#059669"
LEVEL_COL = "#7c3aed"
OCL_COL = "#9333ea"
SMT_COL = "#0891b2"
IDM_COL = "#e11d48"
QM_COL = "#4f46e5"


def _auto_decimals(price):
    p = abs(float(price))
    if p >= 1000: return 1
    if p >= 10: return 2
    if p >= 1: return 3
    return 5


def _fmt(p, decimals=None):
    if decimals is None:
        decimals = _auto_decimals(p)
    return f"{p:,.{decimals}f}"


def _candles(ax, df, up, down):
    w = 0.65
    for i, r in df.iterrows():
        col = up if r["close"] >= r["open"] else down
        ax.plot([i, i], [r["low"], r["high"]], color=col, lw=1.1, zorder=3)
        lo, hi = sorted([r["open"], r["close"]])
        ax.add_patch(Rectangle((i - w/2, lo), w, max(hi - lo, 1e-9),
                               facecolor=col, edgecolor=col, zorder=4, linewidth=0))


def render_setup_chart(df, setup, out_path="alert.png", last_n=70, title=None, theme="color"):
    T = THEMES.get(theme, THEMES["color"])
    BG, GRID, UP, DOWN = T["BG"], T["GRID"], T["UP"], T["DOWN"]
    df = df.reset_index(drop=True)
    n = len(df)
    long_ = str(setup.get("side", "LONG")).upper() == "LONG"
    side = "LONG" if long_ else "SHORT"

    entry = float(setup["entry"])
    sl = float(setup["sl"])
    tp = float(setup["tp"])
    decimals = setup.get("decimals") or _auto_decimals(entry)
    unit = setup.get("unit", "pts")
    symbol = setup.get("symbol", "")
    tf = setup.get("tf", "")

    def fmt(p):
        return _fmt(p, decimals)

    # ----- adaptive window -----
    lo_i = max(0, n - last_n)
    idxs = []
    for key, sub in setup.items():
        if isinstance(sub, dict):
            for kk in ("idx", "start", "end", "sweep_idx"):
                if sub.get(kk) is not None:
                    idxs.append(int(sub[kk]))
            if key in ("smt", "trendline"):
                for p in ("p1", "p2"):
                    if sub.get(p) and sub[p].get("idx") is not None:
                        idxs.append(int(sub[p]["idx"]))
        elif key == "extras":
            for e in sub:
                if e.get("idx") is not None:
                    idxs.append(int(e["idx"]))
    if idxs:
        lo_i = max(0, min(lo_i, min(idxs) - 3))
    dfw = df.iloc[lo_i:].reset_index(drop=True)
    offset = lo_i
    nw = len(dfw)

    def adj(i):
        return int(i) - offset

    # ----- price range -----
    lo = min(dfw["low"].min(), sl, tp, entry)
    hi = max(dfw["high"].max(), sl, tp, entry)
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad

    # ----- figure -----
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(10, 9),
                                   gridspec_kw={"height_ratios": [3.2, 1.15]},
                                   facecolor=BG)
    ax.set_facecolor(BG)
    axr.set_facecolor(BG)

    _candles(ax, dfw, UP, DOWN)
    ax.set_xlim(-1.5, nw + 8)
    ax.set_ylim(lo, hi)
    ax.yaxis.tick_right()
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(True, axis="y", color=GRID, lw=0.6, zorder=0)

    # ========== 1. ASIA RANGE ==========
    asia = setup.get("asia")
    if asia:
        a0, a1 = adj(asia["start"]), adj(asia["end"])
        a0, a1 = max(0, a0), min(nw-1, a1)
        ah, al = asia["high"], asia["low"]
        ax.add_patch(Rectangle((a0 - 0.4, al), a1 - a0 + 0.8, ah - al,
                               facecolor=ASIA_FILL, edgecolor="none", alpha=0.45, zorder=1))
        ax.text(a0, ah + (hi-lo)*0.012, "ASIA", fontsize=7, color=MUTED,
                ha="left", va="bottom", fontweight="bold", zorder=6)

        sw = asia.get("sweep_idx")
        if sw is not None:
            sx = adj(sw)
            if 0 <= sx < nw:
                if long_:
                    ax.axhspan(min(al, dfw.iloc[sx]["low"]), al, xmin=0, xmax=1,
                               facecolor=SWEEP_FILL, alpha=0.35, zorder=1)
                else:
                    ax.axhspan(ah, max(ah, dfw.iloc[sx]["high"]), xmin=0, xmax=1,
                               facecolor=SWEEP_FILL, alpha=0.35, zorder=1)
                ay = dfw.iloc[sx]["low"] if long_ else dfw.iloc[sx]["high"]
                ax.annotate("LIQ SWEEP", xy=(sx, ay),
                            xytext=(sx + 5, lo + (hi-lo)*0.07),
                            fontsize=7, color="#be185d", fontweight="bold",
                            arrowprops=dict(arrowstyle="->", color="#be185d", lw=1),
                            zorder=7)

    # ========== 2. MAIN LEVEL (Classic A / V) ==========
    level = setup.get("level")
    if level:
        lp = float(level["price"])
        lbl = level.get("label", "POI")
        if "Classic V" in lbl or lbl.strip().startswith("V"):
            short = "Classic V"
        elif "Classic A" in lbl or lbl.strip().startswith("A"):
            short = "Classic A"
        else:
            short = lbl.split("/")[0].strip()[:12]
        ax.axhline(lp, color=LEVEL_COL, ls="-", lw=1.4, alpha=0.9, zorder=5)
        ax.text(nw + 0.3, lp, f" {short}  {fmt(lp)}", fontsize=8, color=LEVEL_COL,
                va="center", ha="left", fontweight="bold", zorder=8)

    # ========== 3. OCL ==========
    ocl = setup.get("ocl")
    if ocl:
        op = float(ocl["price"])
        ax.axhline(op, color=OCL_COL, ls="--", lw=1, alpha=0.75, zorder=5)
        ax.text(-1.2, op, "OCL", fontsize=7, color=OCL_COL, va="center",
                ha="right", fontweight="bold", zorder=8)

    # ========== 4. SMT divergence ==========
    smt = setup.get("smt")
    if smt and smt.get("p1") and smt.get("p2"):
        x1, y1 = adj(smt["p1"]["idx"]), float(smt["p1"]["price"])
        x2, y2 = adj(smt["p2"]["idx"]), float(smt["p2"]["price"])
        if 0 <= x1 < nw and 0 <= x2 < nw:
            ax.plot([x1, x2], [y1, y2], color=SMT_COL, ls="-.", lw=1.4, zorder=6)
            ax.scatter([x1, x2], [y1, y2], s=28, color=SMT_COL, zorder=7)
            mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
            pair = smt.get("pair", "SMT")
            ax.text(mid_x, mid_y + (hi-lo)*0.02, f"SMT", fontsize=7,
                    color=SMT_COL, ha="center", fontweight="bold", zorder=8)

    # ========== 5. IDM (subtle) ==========
    idm = setup.get("idm")
    if idm and idm.get("idx") is not None:
        ix = adj(idm["idx"])
        ip = float(idm["price"])
        if 0 <= ix < nw:
            ax.scatter([ix], [ip], s=50, marker="D", facecolors="none",
                       edgecolors=IDM_COL, linewidths=1.3, zorder=7)
            ax.text(ix, ip - (hi-lo)*0.025, "IDM", fontsize=6.5, color=IDM_COL,
                    ha="center", va="top", fontweight="bold", zorder=8)

    # ========== 6. QM (subtle) ==========
    qm = setup.get("qm")
    if qm and qm.get("idx") is not None:
        qx = adj(qm["idx"])
        qp = float(qm["price"])
        if 0 <= qx < nw:
            ax.scatter([qx], [qp], s=45, marker="s", facecolors="none",
                       edgecolors=QM_COL, linewidths=1.2, zorder=7)
            ax.text(qx, qp + (hi-lo)*0.02, "QM", fontsize=6.5, color=QM_COL,
                    ha="center", va="bottom", fontweight="bold", zorder=8)

    # ========== 7. SHIFT ==========
    shift = setup.get("shift")
    if shift and shift.get("idx") is not None:
        sx = adj(shift["idx"])
        if 0 <= sx < nw:
            ax.axvline(sx, color="#f59e0b", ls=":", lw=1.1, alpha=0.75, zorder=2)
            ax.text(sx, hi - (hi-lo)*0.025, "SHIFT", fontsize=7, color="#d97706",
                    ha="center", va="top", fontweight="bold", zorder=8)

    # ========== 8. ENTRY / SL / TP ==========
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    rr = reward / risk if risk > 0 else 0
    risk_pct = risk / entry * 100

    ax.fill_between([nw - 0.5, nw + 6], entry, sl, color=SL_COL, alpha=0.10, zorder=2)
    ax.fill_between([nw - 0.5, nw + 6], entry, tp, color=TP_COL, alpha=0.10, zorder=2)

    ax.hlines(entry, nw - 0.5, nw + 6, color=ENTRY_COL, lw=1.8, zorder=6)
    ax.hlines(sl,    nw - 0.5, nw + 6, color=SL_COL,    lw=1.5, zorder=6)
    ax.hlines(tp,    nw - 0.5, nw + 6, color=TP_COL,    lw=1.5, zorder=6)

    ax.text(nw + 6.3, entry, f" ENTRY  {fmt(entry)}", fontsize=8, color=ENTRY_COL,
            va="center", fontweight="bold", zorder=8)
    ax.text(nw + 6.3, sl, f" SL  {fmt(sl)}  (−{risk:.1f} {unit})", fontsize=8,
            color=SL_COL, va="center", fontweight="bold", zorder=8)
    ax.text(nw + 6.3, tp, f" TP  {fmt(tp)}  (+{reward:.1f} {unit})", fontsize=8,
            color=TP_COL, va="center", fontweight="bold", zorder=8)

    # ========== 9. Wick touch ==========
    wt = setup.get("wick_touch")
    if wt and wt.get("idx") is not None:
        wx = adj(wt["idx"])
        if 0 <= wx < nw:
            wy = dfw.iloc[wx]["low"] if long_ else dfw.iloc[wx]["high"]
            ax.scatter([wx], [wy], s=55, facecolors="none", edgecolors="#eab308",
                       linewidths=1.5, zorder=7)

    # ========== title ==========
    ttl = title or f"{symbol}  {tf}  |  {side}"
    ax.set_title(ttl, loc="left", fontsize=11, color=TXT, fontweight="bold", pad=8)
    ax.text(0.99, 1.02, f"RR {rr:.1f}  |  Risk {risk_pct:.2f}%", transform=ax.transAxes,
            fontsize=8, color=MUTED, ha="right", va="bottom")

    # ========== REASONS (max 6) ==========
    axr.axis("off")
    reasons = []

    if level:
        reasons.append(f"{'Classic V (support)' if long_ else 'Classic A (resistance)'} at {fmt(level['price'])} — main POI")
    if asia and asia.get("sweep_idx") is not None:
        if long_:
            reasons.append(f"Asia low swept — liquidity taken below {fmt(asia['low'])}")
        else:
            reasons.append(f"Asia high swept — liquidity taken above {fmt(asia['high'])}")
    if ocl:
        reasons.append(f"OCL at {fmt(ocl['price'])} — body-to-body level")
    if smt:
        pair = smt.get("pair", "correlated pair")
        reasons.append(f"SMT divergence vs {pair}")
    if idm:
        reasons.append(f"IDM at {fmt(idm['price'])} — inducement swept")
    if shift:
        reasons.append("SHIFT LTF confirmed — structure change")
    reasons.append(f"SL {fmt(sl)} ({risk:.1f} {unit}) → TP {fmt(tp)} ({reward:.1f} {unit})  |  RR {rr:.1f}")

    body = f"{side} SETUP\n\n"
    for i, r in enumerate(reasons[:6], 1):
        body += f"{i}.  {r}\n"

    axr.text(0.02, 0.95, body, transform=axr.transAxes, fontsize=9,
             color=TXT, va="top", ha="left", linespacing=1.45,
             family="sans-serif")

    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------- demo
def _demo(short=False, gold=False, out="demo_chart.png", theme="color"):
    np.random.seed(42)
    n = 100
    if gold:
        base, step = 3010.0, 0.8
    else:
        base, step = 85600.0, 45.0

    closes = [base]
    for _ in range(n - 1):
        closes.append(closes[-1] + np.random.randn() * step)
    closes = np.array(closes)
    highs = closes + np.abs(np.random.randn(n) * step * 0.6)
    lows  = closes - np.abs(np.random.randn(n) * step * 0.6)
    opens = closes + np.random.randn(n) * step * 0.3

    lows[55:62] = lows[55:62] - step * 3
    closes[55:62] = (highs[55:62] + lows[55:62]) / 2
    closes[70:] = closes[70:] + np.linspace(0, step * 8, n - 70)
    highs[70:] = closes[70:] + step * 0.5
    lows[70:] = closes[70:] - step * 0.4

    df = pd.DataFrame({
        "time": pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC"),
        "open": opens, "high": highs, "low": lows, "close": closes,
    })

    entry = float(closes[-1])
    sl = float(lows[58:65].min()) * 0.999
    risk = entry - sl
    tp = entry + risk * 2

    setup = {
        "side": "LONG",
        "symbol": "XAUUSD" if gold else "BTCUSDT",
        "tf": "5m",
        "decimals": 2 if gold else 1,
        "unit": "pts",
        "entry": entry, "sl": sl, "tp": tp,
        "entry_idx": n - 1,
        "level": {"price": float(lows[60]), "label": "Classic V / H1 POI"},
        "ocl": {"idx": 48, "price": float(closes[48])},
        "idm": {"idx": 64, "price": float(lows[64])},
        "qm":  {"idx": 52, "price": float(highs[52])},
        "wick_touch": {"idx": 62},
        "shift": {"idx": 78},
        "asia": {
            "start": 20, "end": 40,
            "high": float(highs[20:41].max()),
            "low": float(lows[20:41].min()),
            "sweep_idx": 60,
        },
        "smt": {
            "p1": {"idx": 58, "price": float(lows[58])},
            "p2": {"idx": 66, "price": float(lows[66])},
            "pair": "BTCUSDT vs ETHUSDT",
        },
        "entry_why": "Limit at Classic V after Asia sweep + SHIFT + SMT",
        "sl_why": "Below sweep low / IDM",
        "tp_why": "2R at next liquidity",
    }

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
        setup["ocl"]["price"] = mir(setup["ocl"]["price"])
        setup["idm"]["price"] = mir(setup["idm"]["price"])
        setup["qm"]["price"] = mir(setup["qm"]["price"])
        setup["asia"]["high"], setup["asia"]["low"] = mir(setup["asia"]["low"]), mir(setup["asia"]["high"])
        setup["smt"]["p1"]["price"] = mir(setup["smt"]["p1"]["price"])
        setup["smt"]["p2"]["price"] = mir(setup["smt"]["p2"]["price"])

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
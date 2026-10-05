"""
24/7 setup scanner. ALERTS ONLY: no orders, no execution. You decide and manage the trade.

What it does every minute:
  for each (HTF, LTF) pair -> fetch latest Binance candles -> look for a fresh
  "HTF FVG -> LTF IFVG" setup -> paint a chart -> send it to Telegram (or save PNG if no Telegram set).

Setup:
  python -m pip install pandas numpy requests matplotlib
  (optional) create config.json next to this file:
      {"token": "123456:ABC...", "chat_id": "123456789"}
  python scanner.py            # runs forever, Ctrl+C to stop
  python scanner.py --once     # scan one time and exit (good for testing)
  python scanner.py --demo     # offline: paints one sample alert from fake data (no internet needed)
Needs setup_labeler.py in the same folder. Optional: setups_mtf.csv (from mtf_run.py) adds
historical stats to each alert.
"""
import argparse
import inspect
import json
import os
import time

import numpy as np
import pandas as pd
import requests

try:
    import matplotlib  # pyright: ignore[reportMissingModuleSource]
    matplotlib.use("Agg")                      # draw to files, no window needed
    import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource]
    from matplotlib.patches import Rectangle  # pyright: ignore[reportMissingModuleSource]
except ImportError:
    raise SystemExit("matplotlib is missing. Run:  python -m pip install matplotlib")

import setup_labeler as sl

if "live" not in inspect.signature(sl.find_setups).parameters:
    raise SystemExit("Your setup_labeler.py is the OLD version. Replace it with the latest one "
                     "(it must contain the 'live' option), then run again.")

SYMBOL = "BTCUSDT"
# (HTF where the FVG lives, LTF where the entry is confirmed). Edit freely.
PAIRS = [("1D", "1h"), ("4h", "1h"), ("4h", "15m"), ("1h", "15m"), ("1h", "5m")]
BARS = 1000
OUT_DIR = "alerts"
SEEN_FILE = "alerted.json"


# ----------------------------------------------------------------- data
def fetch(symbol, interval, limit=BARS):
    err = None
    for base in sl.BASE_URLS:
        try:
            r = requests.get(f"{base}/api/v3/klines",
                             params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=20)
            r.raise_for_status()
            rows = r.json()
            df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume",
                                             "ct", "qv", "n", "tb", "tq", "x"])
            df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
            df = df.set_index("t")[["open", "high", "low", "close", "volume"]].astype(float)
            return df.iloc[:-1]          # drop the still-forming candle: only closed candles
        except Exception as e:
            err = e
    raise RuntimeError(f"fetch failed: {err}")


def max_age_for(htf, ltf):
    cover = BARS * pd.Timedelta(ltf) / pd.Timedelta(htf)   # how many HTF candles our window spans
    return int(max(5, min(72, cover - 10)))


# ----------------------------------------------------------------- drawing
def paint(d, row, htf, ltf, path, hist=None):
    t = pd.Timestamp(row["time"])
    loc = d.index.get_loc(t)
    a, b = max(0, loc - 110), min(len(d), loc + 16)
    w = d.iloc[a:b]
    x = np.arange(len(w))
    xs = loc - a

    fig, ax = plt.subplots(figsize=(13, 7), dpi=110)
    lo, hi = row["fvg_lo"], row["fvg_hi"]
    ax.add_patch(Rectangle((0, lo), len(w), hi - lo, color="#fff3a8", alpha=0.7, zorder=0))
    ax.text(1, (lo + hi) / 2, f"{htf} FVG", va="center", fontsize=9, color="#7a6a00")

    e, sl_, tp = row["entry"], row["sl"], row["tp"]
    ax.add_patch(Rectangle((xs, min(e, tp)), len(w) - xs, abs(tp - e), color="#2ecc71", alpha=0.18, zorder=0))
    ax.add_patch(Rectangle((xs, min(e, sl_)), len(w) - xs, abs(e - sl_), color="#e74c3c", alpha=0.18, zorder=0))
    for y, name, col in ((tp, "target", "#1e8449"), (e, "entry", "#34495e"), (sl_, "sl", "#c0392b")):
        ax.hlines(y, xs, len(w), colors=col, linewidth=1.2)
        ax.text(len(w) + 0.3, y, f"{name} {y:,.1f}", va="center", fontsize=9, color=col)

    for i, (o, h, l, c) in enumerate(zip(w["open"], w["high"], w["low"], w["close"])):
        col = "#16a085" if c >= o else "#111111"
        ax.vlines(i, l, h, color=col, linewidth=1)
        ax.add_patch(Rectangle((i - 0.3, min(o, c)), 0.6, max(abs(c - o), (h - l) * 0.02), color=col, zorder=3))
    ax.annotate("signal", (xs, e), xytext=(xs - 8, e + (tp - e) * 0.25),
                arrowprops=dict(arrowstyle="->", color="#34495e"), fontsize=9)

    step = max(1, len(w) // 8)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([ts.strftime("%m-%d %H:%M") for ts in w.index[::step]], fontsize=8)
    ax.set_xlim(-1, len(w) + 12)
    ymin = min(w["low"].min(), sl_, lo)
    ymax = max(w["high"].max(), tp, hi)
    pad = (ymax - ymin) * 0.05
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.grid(alpha=0.2)
    rr = abs(tp - e) / abs(e - sl_)
    ax.set_title(f"{SYMBOL}  {htf} FVG -> {ltf} IFVG   {row['side'].upper()}   (planned RR {rr:.1f})", fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------- alerts
def load_hist():
    if not os.path.exists("setups_mtf.csv"):
        return {}
    h = pd.read_csv("setups_mtf.csv")
    out = {}
    for (htf, ltf), g in h.groupby(["htf", "ltf"]):
        out[(htf, ltf)] = (len(g), (g["outcome"] == "win").mean() * 100, g["result_R"].mean())
    return out


def caption(row, htf, ltf, hist):
    ltf_rule = {"5m": "5min", "15m": "15min", "1h": "1h"}.get(ltf, ltf)
    rr = abs(row["tp"] - row["entry"]) / abs(row["entry"] - row["sl"])
    lines = [f"{SYMBOL} {row['side'].upper()} setup",
             f"{htf} FVG -> {ltf} IFVG | signal candle {pd.Timestamp(row['time']).strftime('%Y-%m-%d %H:%M')} UTC",
             f"Entry {row['entry']:,.1f} | SL {row['sl']:,.1f} | Target {row['tp']:,.1f} | RR {rr:.1f}",
             f"Killzone: {'yes' if row['killzone'] else 'no'}"]
    st = hist.get((htf, ltf_rule))
    if st:
        lines.append(f"History of this pair (past data): {st[0]} setups, win {st[1]:.0f}%, avg {st[2]:+.2f}R after fees")
    lines.append("Alert only. Not advice, no guarantee.")
    return "\n".join(lines)


def send(cfg, img, text):
    if cfg.get("token") and cfg.get("chat_id"):
        with open(img, "rb") as f:
            r = requests.post(f"https://api.telegram.org/bot{cfg['token']}/sendPhoto",
                              data={"chat_id": cfg["chat_id"], "caption": text}, files={"photo": f}, timeout=30)
        print("Telegram:", "sent" if r.ok else f"FAILED {r.status_code} {r.text[:120]}")
    else:
        print("(no Telegram config: saved PNG only)")


def load_json(p, default):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return default


# ----------------------------------------------------------------- main
def scan_once(cfg, seen, hist):
    os.makedirs(OUT_DIR, exist_ok=True)
    for htf, ltf in PAIRS:
        try:
            d = fetch(SYMBOL, ltf)
            ma = max_age_for(htf, ltf)
            rows = pd.concat([sl.find_setups(d, s, htf=htf, max_age_htf=ma, live=True)
                              for s in ("long", "short")], ignore_index=True)
        except Exception as e:
            print(f"[{htf}->{ltf}] error: {e}")
            continue
        if rows.empty:
            continue
        rows = rows[rows["time"] >= d.index[-2]]          # only fresh signals (last 2 closed candles)
        for _, row in rows.iterrows():
            key = f"{htf}|{ltf}|{row['side']}|{pd.Timestamp(row['time']).isoformat()}"
            if key in seen:
                continue
            seen[key] = True
            img = os.path.join(OUT_DIR, key.replace("|", "_").replace(":", "-") + ".png")
            paint(d, row, htf, ltf, img)
            text = caption(row, htf, ltf, hist)
            print("\nNEW SETUP\n" + text)
            send(cfg, img, text)
            with open(SEEN_FILE, "w") as f:
                json.dump(seen, f)


def demo():
    os.makedirs(OUT_DIR, exist_ok=True)
    d = sl.synthetic_klines(n=20000, seed=7)
    rows = pd.concat([sl.find_setups(d, s, live=True) for s in ("long", "short")], ignore_index=True)
    row = rows.sort_values("time").iloc[len(rows) // 2]
    img = os.path.join(OUT_DIR, "demo_alert.png")
    paint(d, row, "1h", "5m", img)
    print(caption(row, "1h", "5m", {}))
    print("Saved:", img)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        demo()
        raise SystemExit
    cfg = load_json("config.json", {})
    seen = load_json(SEEN_FILE, {})
    hist = load_hist()
    print("Scanner started. Telegram:", "ON" if cfg.get("token") else "OFF (PNG only)")
    while True:
        scan_once(cfg, seen, hist)
        if a.once:
            break
        time.sleep(60)
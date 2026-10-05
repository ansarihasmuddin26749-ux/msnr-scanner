"""
context_scanner.py - BTCUSDT CONTEXT alerts only.
No orders, no entries/SL/TP, no edge claim. Backtests showed no mechanical edge,
so this only tells you WHERE price is relative to levels. You decide the trade.

Alerts:
  1) LEVEL TOUCH : 1h fresh A/V level (close-line peak/valley) touched in last hour
  2) 5M SHIFT    : after a touch, 5m close breaks last 5m swing in the expected direction
  3) ASIA SWEEP  : price took out Asia high/low (00:00-08:00 UTC)
  4) 1H FVG      : price inside an unmitigated 1h FVG

Setup:
  Telegram (optional): set env vars TG_TOKEN and TG_CHAT. Otherwise prints to console.
  Run:  python context_scanner.py            (loops every 60s)
        python context_scanner.py --once     (single scan, for testing)
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

SYMBOL = "BTCUSDT"
API = "https://api.binance.com/api/v3/klines"
STATE_FILE = "context_state.json"
TOL = 0.0015          # 0.15 percent: how close counts as "touch"
PIVOT_1H = 3          # bars each side for 1h close-line pivots
PIVOT_5M = 2          # bars each side for 5m swing
LOOKBACK_1H = 400


HOSTS = ["https://api.binance.com", "https://data-api.binance.vision",
         "https://api1.binance.com", "https://api2.binance.com"]


def _get(url_path, params):
    q = urllib.parse.urlencode(params)
    last = None
    for use_proxy in (False, True):
        handler = urllib.request.ProxyHandler() if use_proxy else urllib.request.ProxyHandler({})
        opener = urllib.request.build_opener(handler)
        for h in HOSTS:
            try:
                with opener.open(f"{h}{url_path}?{q}", timeout=15) as r:
                    return json.loads(r.read())
            except Exception as e:
                last = e
    raise last


def klines(interval, limit):
    raw = _get("/api/v3/klines", {"symbol": SYMBOL, "interval": interval, "limit": limit})
    out = []
    for k in raw[:-1]:  # drop the still-forming candle
        out.append({"t": k[0] // 1000, "o": float(k[1]), "h": float(k[2]),
                    "l": float(k[3]), "c": float(k[4])})
    return out


def pivots_close(bars, n):
    """close-line peaks (A) and valleys (V): (index, price, kind)."""
    res = []
    for i in range(n, len(bars) - n):
        c = bars[i]["c"]
        win = [bars[j]["c"] for j in range(i - n, i + n + 1)]
        if c == max(win):
            res.append((i, c, "A"))
        elif c == min(win):
            res.append((i, c, "V"))
    return res


def fresh_levels(bars, n):
    """Levels never touched by a later wick."""
    out = []
    for i, p, kind in pivots_close(bars, n):
        later = bars[i + 1:]
        if kind == "A":
            touched = any(b["h"] >= p for b in later)
        else:
            touched = any(b["l"] <= p for b in later)
        if not touched:
            out.append({"price": p, "kind": kind, "t": bars[i]["t"]})
    return out


def recent_touch(levels, bars5, price_now, n_bars=12):
    """Fresh 1h levels reached by a 5m wick within the last hour (or price within TOL)."""
    recent = bars5[-n_bars:]
    hits = []
    for lv in levels:
        p = lv["price"]
        if lv["kind"] == "A":
            hit = any(b["h"] >= p * (1 - TOL) for b in recent)
        else:
            hit = any(b["l"] <= p * (1 + TOL) for b in recent)
        # ignore levels far away (guards against old levels already blown through)
        if hit and abs(price_now - p) / p < 0.01:
            hits.append(lv)
    return hits


def swing_5m(bars):
    """last confirmed 5m swing high and low (price, index)."""
    sh = sl = None
    for i in range(len(bars) - PIVOT_5M - 1, PIVOT_5M - 1, -1):
        w = bars[i - PIVOT_5M:i + PIVOT_5M + 1]
        if sh is None and bars[i]["h"] == max(b["h"] for b in w):
            sh = (bars[i]["h"], i)
        if sl is None and bars[i]["l"] == min(b["l"] for b in w):
            sl = (bars[i]["l"], i)
        if sh and sl:
            break
    return sh, sl


def shift_5m(bars5):
    """'bull' if last closed 5m candle just closed above last swing high, 'bear' for below swing low."""
    sh, sl = swing_5m(bars5[:-1])
    last, prev = bars5[-1], bars5[-2]
    if sh and last["c"] > sh[0] >= prev["c"]:
        return "bull", sh[0]
    if sl and last["c"] < sl[0] <= prev["c"]:
        return "bear", sl[0]
    return None, None


def asia_range(bars5):
    today = datetime.now(timezone.utc).date()
    rng = [b for b in bars5
           if datetime.fromtimestamp(b["t"], timezone.utc).date() == today
           and datetime.fromtimestamp(b["t"], timezone.utc).hour < 8]
    if not rng:
        return None
    return max(b["h"] for b in rng), min(b["l"] for b in rng)


def open_fvgs(bars1h, max_age=120):
    """Unmitigated 1h FVGs (3-candle gap), newest first."""
    out = []
    start = max(2, len(bars1h) - max_age)
    for i in range(start, len(bars1h)):
        a, c = bars1h[i - 2], bars1h[i]
        later = bars1h[i + 1:]
        if c["l"] > a["h"]:  # bullish gap [a.h, c.l]
            lo, hi = a["h"], c["l"]
            if not any(b["l"] <= lo for b in later):
                out.append(("bull", lo, hi))
        elif c["h"] < a["l"]:  # bearish gap [c.h, a.l]
            lo, hi = c["h"], a["l"]
            if not any(b["h"] >= hi for b in later):
                out.append(("bear", lo, hi))
    return out[::-1]


def send(text):
    print(text + "\n" + "-" * 40)  # console par bhi dikhega
    topic = os.getenv("NTFY_TOPIC")
    if not topic:
        print("NTFY_TOPIC set nahi hai, phone par alert nahi gaya")
        return
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}",
            data=text.encode("utf-8"),
            headers={"Title": "BTC alert"},
        )
        req.add_header("User-Agent", "Mozilla/5.0")
        urllib.request.urlopen(req, timeout=15)
    except Exception as e:
        print("ntfy error:", e)


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(s):
    # keep it small
    keys = sorted(s)[-300:]
    with open(STATE_FILE, "w") as f:
        json.dump({k: s[k] for k in keys}, f)


def fmt(p):
    return f"{p:,.1f}"


def scan():
    b1, b5 = klines("1h", LOOKBACK_1H + 1), klines("5m", 300)
    price = b5[-1]["c"]
    state = load_state()
    msgs = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def once(key, text):
        if key not in state:
            state[key] = int(time.time())
            msgs.append(text)

    levels = fresh_levels(b1, PIVOT_1H)
    touched = recent_touch(levels, b5, price)

    # 1) level touch + 2) 5m shift
    direction, brk = shift_5m(b5)
    for lv in touched:
        name = "V-level (support)" if lv["kind"] == "V" else "A-level (resistance)"
        day_hour = datetime.fromtimestamp(b5[-1]["t"], timezone.utc).strftime("%Y%m%d%H")
        once(f"touch:{lv['kind']}:{round(lv['price'])}:{day_hour}",
             f"LEVEL TOUCH  {SYMBOL}\n1h fresh {name} at {fmt(lv['price'])}\n"
             f"Price now {fmt(price)}\nWatch for 5m reaction. {now}")
        want = "bull" if lv["kind"] == "V" else "bear"
        if direction == want:
            once(f"shift:{lv['kind']}:{round(lv['price'])}:{b5[-1]['t']}",
                 f"5M SHIFT {'UP' if want == 'bull' else 'DOWN'}  {SYMBOL}\n"
                 f"After touch of 1h {name} {fmt(lv['price'])}\n"
                 f"5m closed through swing {fmt(brk)}, price {fmt(price)}\n"
                 f"Structure shift only. Decide entry/SL yourself. {now}")

    # 3) Asia sweep
    ar = asia_range(b5)
    if ar and datetime.now(timezone.utc).hour >= 8:
        hi, lo = ar
        d = datetime.now(timezone.utc).strftime("%Y%m%d")
        after = [b for b in b5
                 if datetime.fromtimestamp(b["t"], timezone.utc).hour >= 8
                 and datetime.fromtimestamp(b["t"], timezone.utc).strftime("%Y%m%d") == d]
        if after and max(b["h"] for b in after) > hi:
            once(f"asia_hi:{d}", f"ASIA HIGH SWEPT  {SYMBOL}\nAsia high {fmt(hi)} taken. {now}")
        if after and min(b["l"] for b in after) < lo:
            once(f"asia_lo:{d}", f"ASIA LOW SWEPT  {SYMBOL}\nAsia low {fmt(lo)} taken. {now}")

    # 4) inside an open 1h FVG
    for side, lo, hi in open_fvgs(b1)[:10]:
        if lo <= price <= hi:
            once(f"fvg:{side}:{round(lo)}:{datetime.now(timezone.utc).strftime('%Y%m%d%H')}",
                 f"INSIDE 1H FVG ({side})  {SYMBOL}\nZone {fmt(lo)} - {fmt(hi)}, price {fmt(price)}. {now}")

    for m in msgs:
        send(m)
    save_state(state)
    return len(msgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="run a single scan and exit")
    ap.add_argument("--every", type=int, default=60, help="seconds between scans")
    a = ap.parse_args()
    while True:
        try:
            n = scan()
            print(datetime.now().strftime("%H:%M:%S"), "scan ok, alerts:", n)
        except Exception as e:
            print("scan error:", e)
        if a.once:
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
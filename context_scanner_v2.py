"""
context_scanner_v2.py - Multi-pair MSNR scanner (conflict-free)
Pairs: BTCUSDT | XAUUSDT (Gold) | NAS100USDT (US100)

Rules to avoid Buy+Sell same zone:
  1. Recent 5m SHIFT direction is boss → only that side alerts
  2. Opposite signal blocked within 0.8% price distance
  3. Only one alert per level per side per day
  4. Touch-only alerts are weaker; full SHIFT alerts preferred

Needs:  pip install matplotlib pandas numpy
Run:    $env:NTFY_TOPIC="hasmuddin-btc-8f3k2"
        python context_scanner_v2.py
"""
import http.server
import socketserver
import threading
import os
def start_fake_server():
    PORT = int(os.environ.get("PORT", 8080))
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd.serve_forever()

# बैकग्राउंड में वेब सर्वर शुरू करने के लिए thread का उपयोग करें
threading.Thread(target=start_fake_server, daemon=True).start()


import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import pandas as pd
from chart_reasons import render_setup_chart

PAIRS = {
    "BTCUSDT": {
        "symbol": "BTCUSDT", "label": "BTC", "decimals": 1, "unit": "pts",
        "tf_chart": "5m", "min_risk": 0.001, "sl_buffer": 0.0005,
        "smt_pair": "ETHUSDT",  # BTC vs ETH
    },
    "XAUUSDT": {
        "symbol": "XAUUSDT", "label": "GOLD", "decimals": 2, "unit": "pts",
        "tf_chart": "5m", "min_risk": 0.0008, "sl_buffer": 0.0004,
        "smt_pair": "XAGUSDT",  # Gold vs Silver
    },
    "NAS100USDT": {
        "symbol": "NAS100USDT", "label": "US100", "decimals": 1, "unit": "pts",
        "tf_chart": "5m", "min_risk": 0.001, "sl_buffer": 0.0005,
        "smt_pair": None,  # no easy correlated pair on Binance
    },
}

HOSTS = [
    "https://api.binance.com",
    "https://data-api.binance.vision",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://fapi.binance.com",
]
STATE_FILE = "context_state.json"
TOL = 0.0015
PIVOT_1H = 3
PIVOT_5M = 2
LOOKBACK_1H = 400
OPPOSITE_BLOCK_PCT = 0.008   # 0.8% — opposite signal blocked inside this range


def _get(path, params, futures=False):
    q = urllib.parse.urlencode(params)
    last = None
    hosts = HOSTS if not futures else ["https://fapi.binance.com"] + HOSTS
    for use_proxy in (False, True):
        handler = urllib.request.ProxyHandler() if use_proxy else urllib.request.ProxyHandler({})
        opener = urllib.request.build_opener(handler)
        for h in hosts:
            try:
                with opener.open(f"{h}{path}?{q}", timeout=12) as r:
                    return json.loads(r.read())
            except Exception as e:
                last = e
    raise last


def klines(symbol, interval, limit, futures=False):
    path = "/fapi/v1/klines" if futures else "/api/v3/klines"
    try:
        raw = _get(path, {"symbol": symbol, "interval": interval, "limit": limit}, futures=futures)
    except Exception:
        path2 = "/api/v3/klines" if futures else "/fapi/v1/klines"
        raw = _get(path2, {"symbol": symbol, "interval": interval, "limit": limit}, futures=not futures)
    return [{"t": k[0]//1000, "o": float(k[1]), "h": float(k[2]),
             "l": float(k[3]), "c": float(k[4])} for k in raw[:-1]]


def bars_to_df(bars):
    df = pd.DataFrame(bars)
    df = df.rename(columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close"})
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.reset_index(drop=True)


def pivots_close(bars, n):
    res = []
    for i in range(n, len(bars) - n):
        c = bars[i]["c"]
        win = [bars[j]["c"] for j in range(i - n, i + n + 1)]
        if c == max(win):
            res.append({"idx": i, "price": c, "kind": "A", "t": bars[i]["t"]})
        elif c == min(win):
            res.append({"idx": i, "price": c, "kind": "V", "t": bars[i]["t"]})
    return res


def fresh_levels(bars, n):
    out = []
    for lv in pivots_close(bars, n):
        later = bars[lv["idx"] + 1:]
        touched = any(b["h"] >= lv["price"] for b in later) if lv["kind"] == "A" \
                  else any(b["l"] <= lv["price"] for b in later)
        if not touched:
            out.append(lv)
    return out


def detect_ocl(bars, lookback=80):
    ocls = []
    start = max(1, len(bars) - lookback)
    for i in range(start, len(bars) - 1):
        c0, o1 = bars[i]["c"], bars[i+1]["o"]
        gap = abs(o1 - c0)
        body0 = abs(bars[i]["c"] - bars[i]["o"])
        if gap <= max(body0 * 0.25, bars[i]["c"] * 0.0003):
            ocls.append({"idx": i, "price": c0, "kind": "OCL",
                         "bull": bars[i]["c"] >= bars[i]["o"], "t": bars[i]["t"]})
    return ocls


def detect_sbr_rbs(bars, levels):
    flips = []
    for lv in levels:
        later = bars[lv["idx"] + 1:]
        if not later:
            continue
        if lv["kind"] == "V":
            if any(b["c"] < lv["price"] for b in later):
                for j, b in enumerate(later):
                    if b["l"] < lv["price"] and b["c"] > lv["price"]:
                        flips.append({"idx": lv["idx"]+1+j, "price": lv["price"],
                                      "kind": "SBR", "t": b["t"]})
                        break
        else:
            if any(b["c"] > lv["price"] for b in later):
                for j, b in enumerate(later):
                    if b["h"] > lv["price"] and b["c"] < lv["price"]:
                        flips.append({"idx": lv["idx"]+1+j, "price": lv["price"],
                                      "kind": "RBS", "t": b["t"]})
                        break
    return flips


def detect_qml(bars, n=3):
    piv = pivots_close(bars, n)
    qms = []
    for i in range(max(0, len(piv)-30), len(piv)-2):
        a, b, c = piv[i], piv[i+1], piv[i+2]
        if a["kind"]=="A" and b["kind"]=="A" and c["kind"]=="A":
            if b["price"] > a["price"] and c["price"] < a["price"]:
                qms.append({"idx": c["idx"], "price": a["price"], "kind": "QML",
                            "side": "bear", "t": c["t"]})
        if a["kind"]=="V" and b["kind"]=="V" and c["kind"]=="V":
            if b["price"] < a["price"] and c["price"] > a["price"]:
                qms.append({"idx": c["idx"], "price": a["price"], "kind": "QML",
                            "side": "bull", "t": c["t"]})
    return qms


def detect_idm(bars, direction, swing_level, n=2, lookback=30):
    if direction is None:
        return None
    end = len(bars) - 1
    start = max(0, end - lookback)
    cands = []
    for i in range(start+n, end-n):
        win = bars[i-n:i+n+1]
        if direction == "bull":
            if bars[i]["l"] == min(b["l"] for b in win):
                if swing_level is None or bars[i]["l"] > swing_level:
                    cands.append({"idx": i, "price": bars[i]["l"], "kind": "IDM"})
        else:
            if bars[i]["h"] == max(b["h"] for b in win):
                if swing_level is None or bars[i]["h"] < swing_level:
                    cands.append({"idx": i, "price": bars[i]["h"], "kind": "IDM"})
    return cands[-1] if cands else None


def recent_touch(levels, bars5, price_now, n_bars=12):
    recent = bars5[-n_bars:]
    hits = []
    for lv in levels:
        p = lv["price"]
        is_res = lv["kind"] in ("A", "SBR") or (lv["kind"]=="QML" and lv.get("side")=="bear")
        hit = any(b["h"] >= p*(1-TOL) for b in recent) if is_res \
              else any(b["l"] <= p*(1+TOL) for b in recent)
        if hit and abs(price_now - p)/p < 0.012:
            hits.append(lv)
    return hits


def swing_5m(bars):
    sh = sl = None
    for i in range(len(bars)-PIVOT_5M-1, PIVOT_5M-1, -1):
        w = bars[i-PIVOT_5M:i+PIVOT_5M+1]
        if sh is None and bars[i]["h"] == max(b["h"] for b in w):
            sh = (bars[i]["h"], i)
        if sl is None and bars[i]["l"] == min(b["l"] for b in w):
            sl = (bars[i]["l"], i)
        if sh and sl:
            break
    return sh, sl


def shift_5m(bars5):
    sh, sl = swing_5m(bars5[:-1])
    last, prev = bars5[-1], bars5[-2]
    if sh and last["c"] > sh[0] >= prev["c"]:
        return "bull", sh[0], len(bars5)-1
    if sl and last["c"] < sl[0] <= prev["c"]:
        return "bear", sl[0], len(bars5)-1
    return None, None, None


def detect_smt(bars_main, bars_corr, direction, lookback=40):
    """
    SMT divergence vs correlated pair.
    Bullish SMT: main makes lower low, correlated does NOT (higher low).
    Bearish SMT: main makes higher high, correlated does NOT (lower high).
    Returns dict for chart or None.
    """
    if not bars_corr or len(bars_corr) < lookback or len(bars_main) < lookback:
        return None
    n = min(len(bars_main), len(bars_corr), lookback)
    m = bars_main[-n:]
    c = bars_corr[-n:]
    # find last two swing lows / highs on main
    def swings(bars, mode="low", pivot=2):
        out = []
        for i in range(pivot, len(bars) - pivot):
            if mode == "low":
                if bars[i]["l"] == min(b["l"] for b in bars[i-pivot:i+pivot+1]):
                    out.append((i, bars[i]["l"]))
            else:
                if bars[i]["h"] == max(b["h"] for b in bars[i-pivot:i+pivot+1]):
                    out.append((i, bars[i]["h"]))
        return out

    if direction == "bull":
        ms = swings(m, "low")
        cs = swings(c, "low")
        if len(ms) < 2 or len(cs) < 2:
            return None
        (i1, p1), (i2, p2) = ms[-2], ms[-1]
        # main lower low
        if p2 >= p1:
            return None
        # correlated should NOT make lower low
        c1 = min((x[1] for x in cs if abs(x[0] - i1) <= 3), default=None)
        c2 = min((x[1] for x in cs if abs(x[0] - i2) <= 3), default=None)
        if c1 is None or c2 is None:
            return None
        if c2 < c1:  # correlated also lower low → no SMT
            return None
        # map indices back to full bars_main
        off = len(bars_main) - n
        return {
            "p1": {"idx": off + i1, "price": p1},
            "p2": {"idx": off + i2, "price": p2},
            "side": "bull",
        }
    elif direction == "bear":
        ms = swings(m, "high")
        cs = swings(c, "high")
        if len(ms) < 2 or len(cs) < 2:
            return None
        (i1, p1), (i2, p2) = ms[-2], ms[-1]
        if p2 <= p1:
            return None
        c1 = max((x[1] for x in cs if abs(x[0] - i1) <= 3), default=None)
        c2 = max((x[1] for x in cs if abs(x[0] - i2) <= 3), default=None)
        if c1 is None or c2 is None:
            return None
        if c2 > c1:
            return None
        off = len(bars_main) - n
        return {
            "p1": {"idx": off + i1, "price": p1},
            "p2": {"idx": off + i2, "price": p2},
            "side": "bear",
        }
    return None



def asia_status(bars5):
    now = datetime.now(timezone.utc)
    today = now.date()
    def dt(b): return datetime.fromtimestamp(b["t"], timezone.utc)
    asia_bars = [(i,b) for i,b in enumerate(bars5) if dt(b).date()==today and dt(b).hour < 8]
    if not asia_bars:
        return None
    start_idx, end_idx = asia_bars[0][0], asia_bars[-1][0]
    hi = max(b["h"] for _,b in asia_bars)
    lo = min(b["l"] for _,b in asia_bars)
    after = [(i,b) for i,b in enumerate(bars5) if dt(b).date()==today and dt(b).hour >= 8]
    sweep_idx = None
    swept_high = any(b["h"] > hi for _,b in after)
    swept_low  = any(b["l"] < lo for _,b in after)
    if swept_low:
        for i,b in after:
            if b["l"] < lo: sweep_idx = i; break
    elif swept_high:
        for i,b in after:
            if b["h"] > hi: sweep_idx = i; break
    return {"start": start_idx, "end": end_idx, "high": hi, "low": lo,
            "sweep_idx": sweep_idx, "swept_high": swept_high, "swept_low": swept_low}


def open_fvgs(bars1h, max_age=120):
    out = []
    for i in range(max(2, len(bars1h)-max_age), len(bars1h)):
        a, c = bars1h[i-2], bars1h[i]
        later = bars1h[i+1:]
        if c["l"] > a["h"]:
            lo, hi = a["h"], c["l"]
            if not any(b["l"] <= lo for b in later):
                out.append({"side":"bull", "bottom":lo, "top":hi, "start":i})
        elif c["h"] < a["l"]:
            lo, hi = c["h"], a["l"]
            if not any(b["h"] >= hi for b in later):
                out.append({"side":"bear", "bottom":lo, "top":hi, "start":i})
    return out[::-1]


def make_plan(side, recent, price, levels, cfg):
    sl_buffer = cfg["sl_buffer"]
    min_risk  = cfg["min_risk"]
    if side == "long":
        sl = min(b["l"] for b in recent) * (1 - sl_buffer)
        risk = max(price - sl, price * min_risk)
        sl = price - risk
        tp1, tp2 = price + risk, price + 2*risk
        nxt = [l["price"] for l in levels if l.get("kind") in ("A","SBR","QML") and l["price"] > price+risk*0.5]
        ref = min(nxt) if nxt else None
    else:
        sl = max(b["h"] for b in recent) * (1 + sl_buffer)
        risk = max(sl - price, price * min_risk)
        sl = price + risk
        tp1, tp2 = price - risk, price - 2*risk
        nxt = [l["price"] for l in levels if l.get("kind") in ("V","RBS","QML") and l["price"] < price-risk*0.5]
        ref = max(nxt) if nxt else None
    return {"side":side, "entry":price, "sl":sl, "tp1":tp1, "tp2":tp2, "risk":risk, "ref":ref}


def fmt(p, decimals=1):
    return f"{p:,.{decimals}f}"


def utc(ts, f="%d %b %H:%M"):
    return datetime.fromtimestamp(ts, timezone.utc).strftime(f) + " UTC"


# ---------- CONFLICT CONTROL ----------
def is_opposite_blocked(state, symbol, side, price, now_ts):
    """
    Block opposite-side alert if we recently sent a signal
    within OPPOSITE_BLOCK_PCT of this price.
    """
    want_opposite = "SHORT" if side == "LONG" else "LONG"
    window = 6 * 3600  # last 6 hours
    for k, ts in state.items():
        if not k.startswith(f"{symbol}:"):
            continue
        if abs(now_ts - ts) > window:
            continue
        # key formats: symbol:shift:KIND:price:ts  or symbol:touch:...
        parts = k.split(":")
        if len(parts) < 4:
            continue
        # find if this key was opposite side
        if "shift" in k or "touch" in k:
            try:
                lvl_price = float(parts[3])
            except Exception:
                continue
            if abs(lvl_price - price) / price < OPPOSITE_BLOCK_PCT:
                # check stored side hint
                if f":{want_opposite}" in k or want_opposite.lower() in k:
                    return True
                # also block if any recent opposite exists nearby
                if "shift" in k:
                    return True
    return False


def build_setup(side, lv, price, plan, bars5, asia, fvgs, shift_idx, extreme, cfg,
                ocl=None, idm=None, qml=None, flips=None, smt=None):
    is_long = side == "long"
    kind = lv.get("kind", "V")
    label_map = {"V":"Classic V / H1 POI", "A":"Classic A / H1 POI",
                 "OCL":"OCL", "SBR":"SBR", "RBS":"RBS", "QML":"QML"}
    label = label_map.get(kind, str(kind))
    dec = cfg["decimals"]

    touch_idx = None
    for i in range(len(bars5)-1, max(0, len(bars5)-25), -1):
        b = bars5[i]
        if is_long and b["l"] <= lv["price"]*(1+TOL):
            touch_idx = i; break
        if not is_long and b["h"] >= lv["price"]*(1-TOL):
            touch_idx = i; break

    setup = {
        "side": "LONG" if is_long else "SHORT",
        "symbol": cfg["symbol"], "tf": cfg["tf_chart"],
        "decimals": dec, "unit": cfg["unit"],
        "entry": plan["entry"], "sl": plan["sl"], "tp": plan["tp2"],
        "entry_idx": len(bars5)-1,
        "level": {"price": lv["price"], "label": label},
        "entry_why": f"ENTRY {fmt(plan['entry'],dec)}: {label} + SHIFT. Reference only.",
        "sl_why": f"SL {fmt(plan['sl'],dec)}: beyond {fmt(extreme,dec)}. Risk {plan['risk']:.2f} (1R).",
        "tp_why": f"TP {fmt(plan['tp2'],dec)}: 2R (TP1 {fmt(plan['tp1'],dec)})."
                  + (f" Next ~{fmt(plan['ref'],dec)}." if plan.get("ref") else ""),
    }
    if touch_idx is not None: setup["wick_touch"] = {"idx": touch_idx}
    if shift_idx is not None: setup["shift"] = {"idx": shift_idx}
    if asia and asia.get("sweep_idx") is not None:
        setup["asia"] = {"start":asia["start"], "end":asia["end"],
                         "high":asia["high"], "low":asia["low"],
                         "sweep_idx":asia["sweep_idx"]}
    for f in fvgs[:8]:
        if f["bottom"] <= price <= f["top"] and ((f["side"]=="bull")==is_long):
            setup["fvg"] = {"start": max(0,len(bars5)-30), "top":f["top"], "bottom":f["bottom"]}
            break
    if ocl: setup["ocl"] = {"idx": min(ocl["idx"], len(bars5)-1), "price": ocl["price"]}
    if idm: setup["idm"] = {"idx": idm["idx"], "price": idm["price"]}
    if qml: setup["qm"]  = {"idx": min(qml["idx"], len(bars5)-1), "price": qml["price"]}
    if smt:
        setup["smt"] = {
            "p1": {"idx": min(smt["p1"]["idx"], len(bars5)-1), "price": smt["p1"]["price"]},
            "p2": {"idx": min(smt["p2"]["idx"], len(bars5)-1), "price": smt["p2"]["price"]},
            "pair": smt.get("pair", ""),
        }

    extras = []
    if flips:
        for fl in flips[:2]:
            extras.append({"idx": min(fl["idx"], len(bars5)-1), "price": fl["price"],
                           "label": fl["kind"], "line": True,
                           "why": f"{fl['kind']}: role-flip"})
    if touch_idx is not None:
        extras.append({"idx": touch_idx, "price": lv["price"], "label": "ORDER 1",
                       "line": True, "why": f"ORDER 1 at {label}"})
    extras.append({"idx": len(bars5)-1, "price": plan["entry"], "label": "ENTRY ORDER 2",
                   "line": True, "why": "ENTRY ORDER 2 after SHIFT"})
    setup["extras"] = extras
    return setup


def send(title, short, png_path=None, full_text=None):
    print(title + " | " + short + "\n" + "-"*50)
    topic = os.getenv("NTFY_TOPIC")
    if not topic:
        print("NTFY_TOPIC not set → local only")
        return
    url = f"https://ntfy.sh/{topic}"
    try:
        if png_path and os.path.exists(png_path):
            with open(png_path, "rb") as f: data = f.read()
            req = urllib.request.Request(url, data=data, method="PUT",
                headers={"Filename":"chart.png", "Title":title, "Message":short})
        else:
            req = urllib.request.Request(url, data=(full_text or short).encode(),
                                         headers={"Title":title})
        req.add_header("User-Agent", "Mozilla/5.0")
        urllib.request.urlopen(req, timeout=30)
    except Exception as e:
        print("ntfy error:", e)


def deliver(title, short, df5, setup, reasons, footer):
    full = short + "\n\n" + "\n".join(f"{i}. {r}" for i,r in enumerate(reasons,1)) + "\n\n" + footer
    try:
        out = f"alert_{setup.get('symbol','chart')}.png"
        render_setup_chart(df5, setup, out, title=f"{setup.get('symbol')} {setup.get('tf')} | MSNR")
        send(title, short, png_path=out)
    except Exception as e:
        print("chart error:", e)
        send(title, short, full_text=full)


def load_state():
    try:
        with open(STATE_FILE) as f: return json.load(f)
    except: return {}


def save_state(s):
    keys = sorted(s)[-600:]
    with open(STATE_FILE, "w") as f:
        json.dump({k:s[k] for k in keys}, f)


FOOTER = ("MSNR map (BTC+GOLD+US100). Direction locked by 5m SHIFT. "
          "Opposite signals blocked in same zone. No proven edge. You decide.")


def scan_pair(cfg, state):
    symbol = cfg["symbol"]
    label  = cfg["label"]
    dec    = cfg["decimals"]
    sent   = 0
    now_ts = int(time.time())

    try:
        b1 = klines(symbol, "1h", LOOKBACK_1H+1)
        b5 = klines(symbol, "5m", 300)
    except Exception as e:
        print(f"[{label}] data error:", e)
        return 0

    # correlated pair for SMT
    bars_corr = None
    smt_sym = cfg.get("smt_pair")
    if smt_sym:
        try:
            bars_corr = klines(smt_sym, "5m", 300)
        except Exception:
            bars_corr = None

    if len(b5) < 50 or len(b1) < 50:
        print(f"[{label}] not enough bars")
        return 0

    price = b5[-1]["c"]
    df5 = bars_to_df(b5)

    classic = fresh_levels(b1, PIVOT_1H)
    ocls    = detect_ocl(b1, 100)
    pivs    = pivots_close(b1, PIVOT_1H)
    flips   = detect_sbr_rbs(b1, pivs)
    qmls    = detect_qml(b1, PIVOT_1H)
    direction, brk, shift_idx = shift_5m(b5)   # "bull" / "bear" / None
    smt_hit = None
    if bars_corr is not None and direction:
        smt_hit = detect_smt(b5, bars_corr, direction)
        if smt_hit:
            smt_hit["pair"] = f"{symbol} vs {smt_sym}"
    asia    = asia_status(b5)
    fvgs    = open_fvgs(b1)
    recent  = b5[-12:]
    day_hour = datetime.fromtimestamp(b5[-1]["t"], timezone.utc).strftime("%Y%m%d%H")

    # ----- DIRECTION LOCK -----
    # If we have a clear 5m shift, only allow that side.
    # If no shift, we still allow touch alerts but mark them weaker.
    locked_side = None
    if direction == "bull":
        locked_side = "LONG"
    elif direction == "bear":
        locked_side = "SHORT"

    levels = classic[:]
    for o in ocls[-8:]:
        levels.append({"price":o["price"], "kind":"OCL", "t":o["t"], "idx":o["idx"], "bull":o.get("bull")})
    for q in qmls[-5:]:
        levels.append({"price":q["price"], "kind":"QML", "t":q["t"], "idx":q["idx"], "side":q.get("side")})
    for f in flips[-5:]:
        levels.append({"price":f["price"], "kind":f["kind"], "t":f["t"], "idx":f["idx"]})

    touched = recent_touch(levels, b5, price)

    for lv in touched:
        kind = lv.get("kind", "V")
        is_v = kind in ("V","RBS") or (kind=="OCL" and lv.get("bull")) or (kind=="QML" and lv.get("side")=="bull")
        side_str = "LONG" if is_v else "SHORT"
        name = {"V":"Classic V","A":"Classic A","OCL":"OCL","SBR":"SBR","RBS":"RBS","QML":"QML"}.get(kind, kind)
        want = "bull" if is_v else "bear"
        extreme = min(b["l"] for b in recent) if is_v else max(b["h"] for b in recent)
        idm = detect_idm(b5, direction, brk)

        # ----- CONFLICT FILTERS -----
        # 1. Direction lock: if shift exists, only same side
        if locked_side and side_str != locked_side:
            continue

        # 2. Opposite zone block
        if is_opposite_blocked(state, symbol, side_str, lv["price"], now_ts):
            continue

        reasons = [
            f"{label} 1h {name} at {fmt(lv['price'],dec)} ({utc(lv.get('t', b5[-1]['t']))}).",
            f"5m wick reached ({'low' if is_v else 'high'} {fmt(extreme,dec)}).",
        ]
        if locked_side:
            reasons.append(f"Direction locked by 5m SHIFT → only {locked_side} allowed.")
        if smt_hit:
            reasons.append(f"SMT divergence: {smt_hit.get('pair', '')}.")
        if asia:
            if is_v and asia.get("swept_low"): reasons.append(f"Asia low {fmt(asia['low'],dec)} swept.")
            if not is_v and asia.get("swept_high"): reasons.append(f"Asia high {fmt(asia['high'],dec)} swept.")
        if kind == "OCL": reasons.append("OCL = body-to-body MSNR zone.")
        if kind in ("SBR","RBS"): reasons.append(f"{kind} role-flip.")
        if kind == "QML": reasons.append("QML structure.")
        for f in fvgs[:8]:
            if f["bottom"] <= price <= f["top"] and ((f["side"]=="bull")==is_v):
                reasons.append(f"Inside FVG {fmt(f['bottom'],dec)}-{fmt(f['top'],dec)}.")
                break

        # --- Touch alert (only if NO direction lock yet, to reduce noise) ---
        key = f"{symbol}:touch:{side_str}:{kind}:{round(lv['price'],dec)}:{day_hour}"
        if key not in state and locked_side is None:
            state[key] = now_ts
            r = reasons + ["No SHIFT yet — waiting for confirmation. Touch only."]
            setup = {"side": side_str, "symbol":symbol, "tf":cfg["tf_chart"],
                     "decimals":dec, "unit":cfg["unit"],
                     "entry":price, "sl":price*(0.997 if is_v else 1.003),
                     "tp":price*(1.006 if is_v else 0.994), "entry_idx":len(b5)-1,
                     "level":{"price":lv["price"], "label":name}}
            if asia and asia.get("sweep_idx") is not None:
                setup["asia"] = {"start":asia["start"],"end":asia["end"],
                                 "high":asia["high"],"low":asia["low"],
                                 "sweep_idx":asia["sweep_idx"]}
            deliver(f"LEVEL TOUCH {label} | {name}",
                    f"{name} {fmt(lv['price'],dec)} touched. Price {fmt(price,dec)}",
                    df5, setup, r, FOOTER)
            sent += 1

        # --- Full SHIFT alert (main signal) ---
        if direction == want:
            skey = f"{symbol}:shift:{side_str}:{kind}:{round(lv['price'],dec)}:{b5[-1]['t']}"
            if skey not in state:
                state[skey] = now_ts
                plan = make_plan("long" if is_v else "short", recent, price, levels, cfg)
                r = reasons + [f"5m closed {'above' if is_v else 'below'} {fmt(brk,dec)} → SHIFT confirmed.",
                               f"SL beyond {fmt(extreme,dec)}."]
                nearest_ocl = ocls[-1] if ocls else None
                nearest_qml = next((q for q in reversed(qmls) if (q.get("side")=="bull")==is_v), None)
                setup = build_setup("long" if is_v else "short", lv, price, plan, b5, asia, fvgs,
                                    shift_idx, extreme, cfg, ocl=nearest_ocl, idm=idm,
                                    qml=nearest_qml, flips=flips, smt=smt_hit)
                deliver(f"5M SHIFT {'UP' if is_v else 'DOWN'} {label} | {name}",
                        f"{plan['side'].upper()} entry {fmt(plan['entry'],dec)} SL {fmt(plan['sl'],dec)} "
                        f"TP1 {fmt(plan['tp1'],dec)} TP2 {fmt(plan['tp2'],dec)}",
                        df5, setup, r, FOOTER)
                sent += 1

    # Asia context (no direction, just info)
    if asia and datetime.now(timezone.utc).hour >= 8:
        d = datetime.now(timezone.utc).strftime("%Y%m%d")
        for flag, nm, p in ((asia.get("swept_high"),"HIGH",asia["high"]),
                            (asia.get("swept_low"),"LOW",asia["low"])):
            key = f"{symbol}:asia_{nm}:{d}"
            if flag and key not in state:
                state[key] = now_ts
                setup = {"side":"LONG","symbol":symbol,"tf":cfg["tf_chart"],
                         "decimals":dec, "unit":cfg["unit"],
                         "entry":price,"sl":price*0.997,"tp":price*1.006,
                         "entry_idx":len(b5)-1,
                         "asia":{"start":asia["start"],"end":asia["end"],
                                 "high":asia["high"],"low":asia["low"],
                                 "sweep_idx":asia.get("sweep_idx")}}
                deliver(f"ASIA {nm} SWEPT {label}",
                        f"Asia {nm.lower()} {fmt(p,dec)} taken. Price {fmt(price,dec)}",
                        df5, setup,
                        [f"Asia {nm.lower()} swept after 08:00 UTC.",
                         "Context only. Wait for level + SHIFT."], FOOTER)
                sent += 1

    return sent


def scan():
    state = load_state()
    total = 0
    for key, cfg in PAIRS.items():
        print(f"Scanning {cfg['label']}...")
        try:
            n = scan_pair(cfg, state)
            total += n
            print(f"  {cfg['label']}: {n} alerts")
        except Exception as e:
            print(f"  {cfg['label']} error:", e)
    save_state(state)
    return total


def demo():
    from chart_reasons import _demo
    path = _demo(out="demo_alert.png")
    send("DEMO FULL MSNR CHART",
         "Conflict-free | Direction locked by SHIFT | BTC+GOLD+US100",
         png_path=path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--every", type=int, default=60)
    a = ap.parse_args()
    if a.demo:
        demo()
        return
    while True:
        try:
            n = scan()
            print(datetime.now().strftime("%H:%M:%S"), "scan ok, total alerts:", n)
        except Exception as e:
            print("scan error:", e)
        if a.once:
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
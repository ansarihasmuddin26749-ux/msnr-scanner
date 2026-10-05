#!/usr/bin/env python3
"""
fetch_data.py - Binance se 5m candles download karke CSV banata hai
(columns: open_time, open, high, low, close, volume - UTC)

Chalao (har command alag line mein):
    python fetch_data.py ETHUSDT PAXGUSDT
    python fetch_data.py ETHUSDT PAXGUSDT --start 2022-10-05

Files banengi: eth_5m.csv, paxg_5m.csv (isi folder mein).
PAXGUSDT = gold-backed token (1 token = 1 oz gold), gold ka proxy hai.
"""
import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

HOSTS = ["https://api.binance.com", "https://data-api.binance.vision"]
STEP = 5 * 60 * 1000


def get(path, params):
    last = None
    for h in HOSTS:
        try:
            url = h + path + "?" + urllib.parse.urlencode(params)
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read())
        except Exception as e:  # try the next host
            last = e
    raise last


def fetch(symbol, start_ms, end_ms, out):
    cur, total = start_ms, 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["open_time", "open", "high", "low", "close", "volume"])
        while cur < end_ms:
            data = get("/api/v3/klines", dict(symbol=symbol, interval="5m", startTime=cur,
                                              endTime=end_ms, limit=1000))
            if not data:
                break
            for k in data:
                t = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S+00:00")
                w.writerow([t, k[1], k[2], k[3], k[4], k[5]])
            total += len(data)
            cur = data[-1][0] + STEP
            print(f"\r{symbol}: {total} candles  (till {t})", end="", flush=True)
            time.sleep(0.1)
    print(f"\n{symbol}: saved {total} candles -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--start", default="2022-10-05 18:30")
    args = ap.parse_args()
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(time.time() * 1000) // STEP * STEP - 1      # sirf closed candles
    for sym in args.symbols:
        sym = sym.upper()
        out = sym.replace("USDT", "").lower() + "_5m.csv"
        try:
            fetch(sym, start_ms, end_ms, out)
        except Exception as e:
            print(f"\n{sym}: fail - {e}\n(network/VPN check karo; api.binance.com block ho to data-api.binance.vision try hota hai)")
            sys.exit(1)


if __name__ == "__main__":
    main()
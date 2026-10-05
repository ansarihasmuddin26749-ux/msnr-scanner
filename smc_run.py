"""
Backtest Model A (sweep -> MSS -> OB/FVG retrace) on several timeframes and show which confluences matter.
Needs setup_labeler.py, mtf_run.py and smc.py in the same folder (uses the cached btc_5m.csv).

  python smc_run.py --fee 0
  python smc_run.py --fee 0.0002
  python smc_run.py --synthetic        # offline test
"""
import argparse
import os

import numpy as np
import pandas as pd

import setup_labeler as sl
from mtf_run import to_tf
from smc import find_smc_setups

TFS = ["5min", "15min", "1h", "4h"]


def stats(g):
    f = g[g["outcome"].isin(["win", "loss", "timeout"])]
    if f.empty:
        return pd.Series({"signals": len(g), "filled": 0})
    r, r2 = f["result_R"], f["result_2R"]
    se = r.std() / np.sqrt(len(f)) if len(f) > 1 else np.nan
    return pd.Series({
        "signals": len(g), "filled": len(f), "fill%": round(len(f) / len(g) * 100),
        "win%": round((f["outcome"] == "win").mean() * 100, 1), "avgR": round(r.mean(), 3),
        "t": round(r.mean() / se, 2) if se == se and se > 0 else np.nan,
        "win%_2R": round((f["outcome_2R"] == "win").mean() * 100, 1), "avgR_2R": round(r2.mean(), 3),
        "rr_plan": round(f["rr_planned"].mean(), 2),
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="smc_setups.csv")
    a = ap.parse_args()

    if a.synthetic:
        df5 = sl.synthetic_klines(n=105000)
    else:
        df5 = pd.read_csv("btc_5m.csv", index_col=0, parse_dates=True)
    print(f"5m candles: {len(df5)}  {df5.index[0]} -> {df5.index[-1]}   fee/side: {a.fee}")

    frames = []
    for tf in TFS:
        d = to_tf(df5, tf)
        r = pd.concat([find_smc_setups(d, "long", fee=a.fee), find_smc_setups(d, "short", fee=a.fee)],
                      ignore_index=True)
        if not r.empty:
            r.insert(0, "tf", tf)
            frames.append(r)
    if not frames:
        raise SystemExit("No setups found.")
    allr = pd.concat(frames, ignore_index=True).sort_values("time").reset_index(drop=True)
    allr.to_csv(a.out, index=False)
    pd.set_option("display.width", 220)

    print("\n=== Per timeframe (target = range high; _2R = fixed 2R target) ===")
    print(allr.groupby("tf").apply(stats, include_groups=False).reindex(TFS).to_string())
    print("\n=== Per timeframe and side ===")
    print(allr.groupby(["tf", "side"]).apply(stats, include_groups=False).to_string())
    for col in ("score", "sweep_kind", "zone", "session", "choch", "discount"):
        print(f"\n=== By {col} (all timeframes pooled) ===")
        print(allr.groupby(col).apply(stats, include_groups=False).to_string())
    print("\nRules of thumb: ignore any row with 'filled' < ~100 or |t| < 2. Pooled splits above test many ideas at once,")
    print("so a 'good' row can be luck. Re-check it on more years of data before trusting it.")
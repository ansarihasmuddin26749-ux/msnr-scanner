"""
Backtest the Malaysian SnR first-touch model on several timeframes (uses cached btc_5m.csv).
Needs msnr.py, smc.py, mtf_run.py, setup_labeler.py in the same folder.

  python msnr_run.py --fee 0
  python msnr_run.py --fee 0.0002
  python msnr_run.py --synthetic      # offline test
"""
import argparse

import numpy as np
import pandas as pd

import setup_labeler as sl
from mtf_run import to_tf
from msnr import find_msnr_setups

TFS = ["5min", "15min", "1h", "4h", "1D"]


def stats(g):
    if g.empty:
        return pd.Series({"n": 0})
    r1, r2 = g["result_1R"], g["result_2R"]

    def t(x):
        se = x.std() / np.sqrt(len(x)) if len(x) > 1 else np.nan
        return round(x.mean() / se, 2) if se == se and se > 0 else np.nan

    return pd.Series({
        "n": len(g),
        "win%_1R": round((g["outcome_1R"] == "win").mean() * 100, 1), "avgR_1R": round(r1.mean(), 3), "t_1R": t(r1),
        "win%_2R": round((g["outcome_2R"] == "win").mean() * 100, 1), "avgR_2R": round(r2.mean(), 3), "t_2R": t(r2),
        "risk%": round(g["risk_pct"].mean(), 2),
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="msnr_setups.csv")
    a = ap.parse_args()

    df5 = sl.synthetic_klines(n=105000) if a.synthetic else pd.read_csv("btc_5m.csv", index_col=0, parse_dates=True)
    print(f"5m candles: {len(df5)}  {df5.index[0]} -> {df5.index[-1]}   fee/side: {a.fee}")

    frames = []
    for tf in TFS:
        d = to_tf(df5, tf)
        r = pd.concat([find_msnr_setups(d, "long", fee=a.fee), find_msnr_setups(d, "short", fee=a.fee)],
                      ignore_index=True)
        print(f"  {tf}: {len(r)} setups")
        if not r.empty:
            r.insert(0, "tf", tf)
            frames.append(r)
    if not frames:
        raise SystemExit("No setups found.")
    allr = pd.concat(frames, ignore_index=True).sort_values("time").reset_index(drop=True)
    allr.to_csv(a.out, index=False)
    pd.set_option("display.width", 220)

    print("\n=== Per timeframe (1R and 2R targets; fees included) ===")
    print(allr.groupby("tf").apply(stats, include_groups=False).reindex(TFS).to_string())
    print("\n=== Per timeframe and side ===")
    print(allr.groupby(["tf", "side"]).apply(stats, include_groups=False).to_string())
    print("\n=== By level kind (pooled) ===")
    print(allr.groupby("level_kind").apply(stats, include_groups=False).to_string())
    allr["age_bucket"] = pd.cut(allr["level_age"], [0, 20, 100, 500, 2000, 1e9],
                                labels=["<=20", "21-100", "101-500", "501-2000", ">2000"])
    print("\n=== By level age in candles (pooled; older fresh level = higher significance?) ===")
    print(allr.groupby("age_bucket", observed=True).apply(stats, include_groups=False).to_string())
    print("\n=== By session (pooled) ===")
    print(allr.groupby("session").apply(stats, include_groups=False).to_string())
    print("\nRules of thumb: ignore rows with n < ~100 or |t| < 2. Many splits are shown, so one 'good' row can be luck.")
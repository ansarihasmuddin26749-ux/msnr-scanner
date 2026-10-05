import argparse
import os

import numpy as np
import pandas as pd

import setup_labeler as sl

# (HTF where the FVG lives, LTF where entry is confirmed)
PAIRS = [("1D", "1h"), ("4h", "1h"), ("4h", "15min"), ("1h", "15min"), ("1h", "5min")]


def to_tf(df5, rule):
    if rule == "5min":
        return df5
    return df5.resample(rule).agg({"open": "first", "high": "max", "low": "min",
                                   "close": "last", "volume": "sum"}).dropna()


def summarize(df, fee):
    r = df["result_R"]
    n = len(df)
    se = r.std() / np.sqrt(n) if n > 1 else np.nan
    cut = int(n * 0.7)
    fee_r = (2 * fee * 100 / df["risk_pct"]).mean()   # average fee cost per trade, in R
    return {
        "n": n,
        "win%": round((df["outcome"] == "win").mean() * 100, 1),
        "avgR": round(r.mean(), 3),
        "gross_R": round(r.mean() + fee_r, 3),
        "risk%": round(df["risk_pct"].mean(), 2),
        "feeR": round(fee_r, 2),
        "t": round(r.mean() / se, 2) if n > 1 else np.nan,
        "long_avgR": round(df.loc[df["side"] == "long", "result_R"].mean(), 3),
        "short_avgR": round(df.loc[df["side"] == "short", "result_R"].mean(), 3),
        "first70_avgR": round(r.iloc[:cut].mean(), 3),
        "last30_avgR": round(r.iloc[cut:].mean(), 3),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--fee", type=float, default=0.0005, help="per side, e.g. 0.0005 = 0.05%%; try 0 and 0.0002")
    ap.add_argument("--out", default="setups_mtf.csv")
    a = ap.parse_args()

    if a.synthetic:
        df5 = sl.synthetic_klines()
    elif os.path.exists("btc_5m.csv"):
        df5 = pd.read_csv("btc_5m.csv", index_col=0, parse_dates=True)
        print("Using cached btc_5m.csv (delete it to re-download)")
    else:
        df5 = sl.download_klines("BTCUSDT", "5m", a.days)
        df5.to_csv("btc_5m.csv")
    print(f"5m candles: {len(df5)}  {df5.index[0]} -> {df5.index[-1]}")

    frames, rows = [], []
    for htf, ltf in PAIRS:
        d = to_tf(df5, ltf)
        res = pd.concat([sl.find_setups(d, "long", htf=htf, fee=a.fee),
                         sl.find_setups(d, "short", htf=htf, fee=a.fee)], ignore_index=True)
        if res.empty:
            rows.append({"pair": f"{htf}->{ltf}", "n": 0})
            continue
        res = res.sort_values("time").reset_index(drop=True)
        res.insert(0, "ltf", ltf)
        res.insert(0, "htf", htf)
        frames.append(res)
        rows.append({"pair": f"{htf}->{ltf}", **summarize(res, a.fee)})

    if frames:
        pd.concat(frames, ignore_index=True).to_csv(a.out, index=False)
        print(f"\nSaved -> {a.out}\n")
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).set_index("pair").to_string())
    print("\nRead this carefully: with 5 pairs tested, one of them looking good by luck is likely.")
    print("Only trust a pair if: n >= ~100, t > 2, last30 not worse than first70, and both sides are not wildly different.")
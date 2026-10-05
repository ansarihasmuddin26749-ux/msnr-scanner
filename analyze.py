"""
Robustness check for setups.csv (output of setup_labeler.py).
Usage: python analyze.py            (reads setups.csv in the same folder)
Answers: is the +R real, or luck / one lucky month / a few outlier trades / just the market trend?
"""
import sys

import numpy as np
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "setups.csv"
df = pd.read_csv(path, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
r = df["result_R"]
n = len(df)

print(f"Trades: {n}")
print(f"Mean R: {r.mean():.3f} | Median R: {r.median():.3f} | Std: {r.std():.3f}")
se = r.std() / np.sqrt(n)
print(f"t-stat: {r.mean() / se:.2f}   (need roughly > 2 before it is even worth trusting)")
lo, hi = r.mean() - 1.96 * se, r.mean() + 1.96 * se
print(f"95% range for true avg R: {lo:.3f} to {hi:.3f}")

print("\n--- Outcomes ---")
print(df["outcome"].value_counts().to_string())
print(f"Biggest win: {r.max():.2f}R | Worst loss: {r.min():.2f}R")

print("\n--- Does it survive without the best trades? ---")
srt = r.sort_values(ascending=False)
for k in (3, 5, 10):
    print(f"Remove top {k:2d} trades -> mean R = {srt.iloc[k:].mean():.3f}")

print("\n--- Month by month (is it one lucky month?) ---")
m = df.groupby(df["time"].dt.strftime("%Y-%m")).agg(
    trades=("result_R", "size"),
    sumR=("result_R", "sum"),
    long_R=("result_R", lambda x: x[df.loc[x.index, "side"] == "long"].sum()),
    short_R=("result_R", lambda x: x[df.loc[x.index, "side"] == "short"].sum()),
)
print(m.round(1).to_string())
print(f"Profitable months: {(m['sumR'] > 0).sum()} / {len(m)}")

print("\n--- Long vs short (trend regime check) ---")
print(df.groupby("side")["result_R"].agg(["count", "mean", "sum"]).round(3).to_string())
print("If one side makes all the profit, the 'edge' is likely just the market's direction that year.")

print("\n--- Same-day clustering ---")
per_day = df.groupby(df["time"].dt.date)["result_R"].sum()
print(f"Trading days with setups: {len(per_day)} | share of days positive: {(per_day > 0).mean():.1%}")
print(f"Best single day: {per_day.max():.1f}R | Worst: {per_day.min():.1f}R")
"""
1-2, 1-3 のうちその日のオッズが低い方を100円・高い方を200円にする
「オッズ逆張り動的配分」が、固定(1-2:200円+1-3:100円)より優れているかを検証。

①②条件レースのうち、1-2と1-3の両方のオッズが記録されている races に限定。
・固定(1-2:200+1-3:100)
・固定の逆(1-2:100+1-3:200)
・動的(安い方100円・高い方200円)
・動的の逆(安い方200円・高い方100円)
の4パターンで回収率を比較する。
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "boatrace.db"
SAMPLE_SIZE_WARNING_THRESHOLD = 5
INN_NIGE_RATE_THRESHOLD = 0.8
NIGASHI_RATE_THRESHOLD = 0.5


def bb_recovery(df, ret, stake, n_resamples=2000, seed=42):
    block_key = df["race_date"].astype(str) + "_" + df["jcd"].astype(str)
    d = pd.DataFrame({"block": block_key, "return": np.asarray(ret)})
    blocks = d.groupby("block").agg(n=("return", "size"), return_sum=("return", "sum"))
    n_arr, ret_arr = blocks["n"].to_numpy(), blocks["return_sum"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(blocks), size=(n_resamples, len(blocks)))
    rs, rr = n_arr[idx].sum(axis=1) * stake, ret_arr[idx].sum(axis=1)
    rates = np.where(rs > 0, rr / rs * 100, 0.0)
    lo, hi = np.percentile(rates, [2.5, 97.5])
    point = ret_arr.sum() / (n_arr.sum() * stake) * 100
    return point, lo, hi


conn = sqlite3.connect(DB_PATH)
entries_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, toban FROM entries", conn)
results_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, rank FROM results", conn)
payouts_2tan = pd.read_sql_query("SELECT race_date, jcd, rno, combination, payout FROM payouts WHERE bet_type = '2連単'", conn)
odds_2tan = pd.read_sql_query("SELECT race_date, jcd, rno, combination, odds_low FROM odds WHERE bet_type = '2連単'", conn)

waku1_rank = results_all[results_all["waku"] == 1][["race_date", "jcd", "rno", "rank"]].rename(columns={"rank": "waku1_rank"})

c1 = entries_all[entries_all["waku"] == 1].merge(
    results_all[results_all["waku"] == 1][["race_date", "jcd", "rno", "rank"]],
    on=["race_date", "jcd", "rno"], how="inner")
c1_stats = c1.groupby("toban").agg(starts=("rank", "size"), wins=("rank", lambda s: (s == "1").sum()))
c1_stats["rate"] = c1_stats["wins"] / c1_stats["starts"]
qualified_toban1 = set(c1_stats[(c1_stats["rate"] >= INN_NIGE_RATE_THRESHOLD) & (c1_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)

c2 = entries_all[entries_all["waku"] == 2].merge(waku1_rank, on=["race_date", "jcd", "rno"], how="inner")
c2_stats = c2.groupby("toban").agg(starts=("waku1_rank", "size"), nigasare=("waku1_rank", lambda s: (s == "1").sum()))
c2_stats["rate"] = c2_stats["nigasare"] / c2_stats["starts"]
qualified_toban2 = set(c2_stats[(c2_stats["rate"] >= NIGASHI_RATE_THRESHOLD) & (c2_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)

entries1 = entries_all[entries_all["waku"] == 1][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban1"})
entries2 = entries_all[entries_all["waku"] == 2][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban2"})
race_pairs = entries1.merge(entries2, on=["race_date", "jcd", "rno"], how="inner")
cond = race_pairs[race_pairs["toban1"].isin(qualified_toban1) & race_pairs["toban2"].isin(qualified_toban2)][["race_date", "jcd", "rno"]].copy()

p12 = payouts_2tan[payouts_2tan["combination"] == "1-2"][["race_date", "jcd", "rno", "payout"]].rename(columns={"payout": "p12"})
p13 = payouts_2tan[payouts_2tan["combination"] == "1-3"][["race_date", "jcd", "rno", "payout"]].rename(columns={"payout": "p13"})
o12 = odds_2tan[odds_2tan["combination"] == "1-2"][["race_date", "jcd", "rno", "odds_low"]].rename(columns={"odds_low": "odds12"})
o13 = odds_2tan[odds_2tan["combination"] == "1-3"][["race_date", "jcd", "rno", "odds_low"]].rename(columns={"odds_low": "odds13"})

df = cond.merge(p12, on=["race_date", "jcd", "rno"], how="left").merge(p13, on=["race_date", "jcd", "rno"], how="left")
df = df.merge(o12, on=["race_date", "jcd", "rno"], how="left").merge(o13, on=["race_date", "jcd", "rno"], how="left")
df["p12"] = df["p12"].fillna(0)
df["p13"] = df["p13"].fillna(0)
df = df.dropna(subset=["odds12", "odds13"])  # 両方のオッズが揃っているレースのみ
df = df.sort_values(["race_date", "rno", "jcd"]).reset_index(drop=True)

print(f"①②条件レースのうち、1-2と1-3の両方のオッズが記録されているレース: n={len(df)}")
print(f"  1-2の方が低オッズ(=本命寄り): {(df['odds12'] < df['odds13']).sum()}件")
print(f"  1-3の方が低オッズ: {(df['odds13'] < df['odds12']).sum()}件")
print(f"  平均: 1-2オッズ{df['odds12'].mean():.2f}倍 / 1-3オッズ{df['odds13'].mean():.2f}倍\n")

df["low_is_12"] = df["odds12"] < df["odds13"]

df["fixed_ret"] = df["p12"] * 2 + df["p13"] * 1
df["fixed_rev_ret"] = df["p12"] * 1 + df["p13"] * 2
df["dyn_ret"] = np.where(df["low_is_12"], df["p12"] * 1 + df["p13"] * 2, df["p12"] * 2 + df["p13"] * 1)
df["dyn_rev_ret"] = np.where(df["low_is_12"], df["p12"] * 2 + df["p13"] * 1, df["p12"] * 1 + df["p13"] * 2)

for label, ret_col in [
    ("固定(1-2:200+1-3:100)", "fixed_ret"),
    ("固定の逆(1-2:100+1-3:200)", "fixed_rev_ret"),
    ("動的:オッズ低い方100円・高い方200円", "dyn_ret"),
    ("動的:オッズ低い方200円・高い方100円", "dyn_rev_ret"),
]:
    p, lo, hi = bb_recovery(df, df[ret_col], 300)
    print(f"  {label}: 回収率{p:.1f}%[{lo:.1f}-{hi:.1f}]")

print("\n=== 前半/後半での頑健性チェック ===")
mid = len(df) // 2
front, back = df.iloc[:mid], df.iloc[mid:]
for label, ret_col in [
    ("固定(1-2:200+1-3:100)", "fixed_ret"),
    ("動的:オッズ低い方100円・高い方200円", "dyn_ret"),
    ("動的:オッズ低い方200円・高い方100円", "dyn_rev_ret"),
]:
    pf, lof, hif = bb_recovery(front, front[ret_col], 300)
    pb, lob, hib = bb_recovery(back, back[ret_col], 300)
    print(f"  {label}: 前半n={len(front)} 回収率{pf:.1f}%[{lof:.1f}-{hif:.1f}] | 後半n={len(back)} 回収率{pb:.1f}%[{lob:.1f}-{hib:.1f}]")

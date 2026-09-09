"""
①②条件を満たすレースについて、1-2のオッズ水準によって
「買う/買わない」を変えるべきかを検証する。

オッズが低い(例: 1.5倍など)場合、当たってもトントン付近にしかならない
ため、回収率の観点でオッズ帯ごとにどう変わるかを見る。
・2連単1-2単体を、そのレースのオッズ帯別に賭けた場合の回収率
・現行ミックス(1-2:200+1-3:100)を、1-2のオッズ帯別に賭けた場合の回収率
・単勝1を、単勝1のオッズ帯別に賭けた場合の回収率(参考)
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "boatrace.db"
SAMPLE_SIZE_WARNING_THRESHOLD = 5
INN_NIGE_RATE_THRESHOLD = 0.8
NIGASHI_RATE_THRESHOLD = 0.5


def wilson_ci(hits, n, z=1.959963984540054):
    if n == 0:
        return 0.0, 0.0
    p = hits / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (center - half) * 100, (center + half) * 100


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
payouts_tan = pd.read_sql_query("SELECT race_date, jcd, rno, combination, payout FROM payouts WHERE bet_type = '単勝'", conn)
payouts_2tan = pd.read_sql_query("SELECT race_date, jcd, rno, combination, payout FROM payouts WHERE bet_type = '2連単'", conn)
odds_2tan = pd.read_sql_query("SELECT race_date, jcd, rno, combination, odds_low FROM odds WHERE bet_type = '2連単'", conn)
odds_tan = pd.read_sql_query("SELECT race_date, jcd, rno, combination, odds_low FROM odds WHERE bet_type = '単勝'", conn)

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
cond = race_pairs[race_pairs["toban1"].isin(qualified_toban1) & race_pairs["toban2"].isin(qualified_toban2)].copy()
cond = cond.merge(waku1_rank, on=["race_date", "jcd", "rno"], how="inner")
cond["escaped"] = (cond["waku1_rank"] == "1").astype(int)

# 1-2, 1-3の払戻(結果がそのコンビの時だけ値がある)
p12 = payouts_2tan[payouts_2tan["combination"] == "1-2"][["race_date", "jcd", "rno", "payout"]].rename(columns={"payout": "p12"})
p13 = payouts_2tan[payouts_2tan["combination"] == "1-3"][["race_date", "jcd", "rno", "payout"]].rename(columns={"payout": "p13"})
tan1 = payouts_tan[payouts_tan["combination"] == "1"][["race_date", "jcd", "rno", "payout"]].rename(columns={"payout": "tan1_payout"})
cond = cond.merge(p12, on=["race_date", "jcd", "rno"], how="left").merge(p13, on=["race_date", "jcd", "rno"], how="left")
cond = cond.merge(tan1, on=["race_date", "jcd", "rno"], how="left")
cond["mix_ret"] = cond["p12"].fillna(0) * 2 + cond["p13"].fillna(0) * 1
cond["p12_ret"] = cond["p12"].fillna(0)
cond["tan1_ret"] = np.where(cond["escaped"] == 1, cond["tan1_payout"].fillna(0), 0)

# オッズ(1-2, 単勝1)を付与
odds12 = odds_2tan[odds_2tan["combination"] == "1-2"][["race_date", "jcd", "rno", "odds_low"]].rename(columns={"odds_low": "odds_12"})
odds_t1 = odds_tan[odds_tan["combination"] == "1"][["race_date", "jcd", "rno", "odds_low"]].rename(columns={"odds_low": "odds_tan1"})
cond = cond.merge(odds12, on=["race_date", "jcd", "rno"], how="left").merge(odds_t1, on=["race_date", "jcd", "rno"], how="left")
cond = cond[cond["race_date"].notna()]

print(f"①②条件レース全体: n={len(cond)}")
print(f"  うち1-2オッズデータあり: n={cond['odds_12'].notna().sum()} ({cond['odds_12'].notna().mean()*100:.1f}%)")
print(f"  うち単勝1オッズデータあり: n={cond['odds_tan1'].notna().sum()} ({cond['odds_tan1'].notna().mean()*100:.1f}%)\n")

sub_odds = cond[cond["odds_12"].notna()].copy()
print(f"=== 1-2オッズ帯別: 2連単1-2単体の回収率・的中率 (n={len(sub_odds)}) ===")
bins = [0, 1.3, 1.5, 1.8, 2.2, 3.0, 100]
labels = ["~1.3", "1.3-1.5", "1.5-1.8", "1.8-2.2", "2.2-3.0", "3.0+"]
sub_odds["odds_bin"] = pd.cut(sub_odds["odds_12"], bins=bins, labels=labels, right=False)
for b in labels:
    part = sub_odds[sub_odds["odds_bin"] == b]
    n = len(part)
    if n < 15:
        print(f"  1-2オッズ{b}倍: n={n}(少なすぎるためスキップ)")
        continue
    hit = int((part["p12_ret"] > 0).sum())
    hr = hit / n * 100
    p, lo, hi = bb_recovery(part, part["p12_ret"], 100)
    print(f"  1-2オッズ{b}倍: n={n} 的中率{hr:.1f}% 回収率{p:.1f}%[{lo:.1f}-{hi:.1f}] 平均オッズ{part['odds_12'].mean():.2f}倍")

print(f"\n=== 1-2オッズ帯別: 現行ミックス(1-2:200+1-3:100)の回収率 (n={len(sub_odds)}) ===")
for b in labels:
    part = sub_odds[sub_odds["odds_bin"] == b]
    n = len(part)
    if n < 15:
        continue
    p, lo, hi = bb_recovery(part, part["mix_ret"], 300)
    print(f"  1-2オッズ{b}倍: n={n} ミックス回収率{p:.1f}%[{lo:.1f}-{hi:.1f}]")

print(f"\n=== 「1-2オッズがX倍未満なら見送る」場合の、残りレースでの回収率(2連単1-2単体) ===")
for thresh in [1.3, 1.5, 1.8, 2.0, 2.2]:
    part = sub_odds[sub_odds["odds_12"] >= thresh]
    n = len(part)
    skipped = len(sub_odds) - n
    if n < 30:
        continue
    p, lo, hi = bb_recovery(part, part["p12_ret"], 100)
    print(f"  1-2オッズ{thresh}倍以上のみ賭ける: 残りn={n}(見送り{skipped}件) 回収率{p:.1f}%[{lo:.1f}-{hi:.1f}]")

print(f"\n=== 参考: 単勝1のオッズ帯別回収率 (n={cond['odds_tan1'].notna().sum()}) ===")
sub_tan = cond[cond["odds_tan1"].notna()].copy()
tan_bins = [0, 1.3, 1.5, 1.8, 2.2, 3.0, 100]
sub_tan["odds_bin"] = pd.cut(sub_tan["odds_tan1"], bins=tan_bins, labels=labels, right=False)
for b in labels:
    part = sub_tan[sub_tan["odds_bin"] == b]
    n = len(part)
    if n < 15:
        continue
    p, lo, hi = bb_recovery(part, part["tan1_ret"], 100)
    hr = part["escaped"].mean() * 100
    print(f"  単勝1オッズ{b}倍: n={n} 逃げ率{hr:.1f}% 回収率{p:.1f}%[{lo:.1f}-{hi:.1f}]")

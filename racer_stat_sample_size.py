"""
①②条件そのものを判定する元になっている「選手ごとの過去成績」の
サンプル数(n)について調べる。

・①は「1号艇に乗った回数(starts)のうち何回1着になったか」
・②は「2号艇に乗った回数(starts)のうち何回1号艇に1着を譲ったか」
という選手ごとの過去実績で、現状は starts>=5 であれば対象に含めている。
このnが選手によってどれくらいバラついているか、n が少ない選手を
基準に含めることが実際の逃げ率にどう影響しているかを検証する。
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

waku1_rank = results_all[results_all["waku"] == 1][["race_date", "jcd", "rno", "rank"]].rename(columns={"rank": "waku1_rank"})

c1 = entries_all[entries_all["waku"] == 1].merge(
    results_all[results_all["waku"] == 1][["race_date", "jcd", "rno", "rank"]],
    on=["race_date", "jcd", "rno"], how="inner")
c1_stats = c1.groupby("toban").agg(starts=("rank", "size"), wins=("rank", lambda s: (s == "1").sum()))
c1_stats["rate"] = c1_stats["wins"] / c1_stats["starts"]

c2 = entries_all[entries_all["waku"] == 2].merge(waku1_rank, on=["race_date", "jcd", "rno"], how="inner")
c2_stats = c2.groupby("toban").agg(starts=("waku1_rank", "size"), nigasare=("waku1_rank", lambda s: (s == "1").sum()))
c2_stats["rate"] = c2_stats["nigasare"] / c2_stats["starts"]

print("=== 選手ごとのn(starts)の分布 ===")
print(f"1号艇starts全体: 選手数={len(c1_stats)}, "
      f"最小{c1_stats['starts'].min()} 中央値{c1_stats['starts'].median():.0f} "
      f"平均{c1_stats['starts'].mean():.1f} 最大{c1_stats['starts'].max()}")
print(f"2号艇starts全体: 選手数={len(c2_stats)}, "
      f"最小{c2_stats['starts'].min()} 中央値{c2_stats['starts'].median():.0f} "
      f"平均{c2_stats['starts'].mean():.1f} 最大{c2_stats['starts'].max()}\n")

bins = [5, 10, 20, 30, 50, 100, 10000]
labels = ["5-9", "10-19", "20-29", "30-49", "50-99", "100+"]

q1 = c1_stats[(c1_stats["rate"] >= INN_NIGE_RATE_THRESHOLD) & (c1_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].copy()
q1["bin"] = pd.cut(q1["starts"], bins=bins, labels=labels, right=False)
print(f"=== ①適格選手(逃げ率80%以上, starts>=5): {len(q1)}人 のstarts内訳 ===")
for b in labels:
    sub = q1[q1["bin"] == b]
    if len(sub) == 0:
        continue
    print(f"  starts {b}: {len(sub)}人 ({len(sub)/len(q1)*100:.1f}%) 平均逃げ率{sub['rate'].mean()*100:.1f}%")

q2 = c2_stats[(c2_stats["rate"] >= NIGASHI_RATE_THRESHOLD) & (c2_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].copy()
q2["bin"] = pd.cut(q2["starts"], bins=bins, labels=labels, right=False)
print(f"\n=== ②適格選手(逃がし率50%以上, starts>=5): {len(q2)}人 のstarts内訳 ===")
for b in labels:
    sub = q2[q2["bin"] == b]
    if len(sub) == 0:
        continue
    print(f"  starts {b}: {len(sub)}人 ({len(sub)/len(q2)*100:.1f}%) 平均逃がし率{sub['rate'].mean()*100:.1f}%")

# --- ①②条件のレースに、選手ごとのstartsを付与し、starts帯ごとに実際の逃げ率を見る ---
entries1 = entries_all[entries_all["waku"] == 1][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban1"})
entries2 = entries_all[entries_all["waku"] == 2][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban2"})
race_pairs = entries1.merge(entries2, on=["race_date", "jcd", "rno"], how="inner")

qualified_toban1 = set(q1.index)
qualified_toban2 = set(q2.index)
cond = race_pairs[race_pairs["toban1"].isin(qualified_toban1) & race_pairs["toban2"].isin(qualified_toban2)].copy()
cond = cond.merge(waku1_rank, on=["race_date", "jcd", "rno"], how="inner")
cond["escaped"] = (cond["waku1_rank"] == "1").astype(int)
cond = cond.merge(payouts_tan[payouts_tan["combination"] == "1"][["race_date", "jcd", "rno", "payout"]],
                   on=["race_date", "jcd", "rno"], how="left")
cond["tan_ret"] = np.where(cond["escaped"] == 1, cond["payout"].fillna(0), 0)
cond["starts1"] = cond["toban1"].map(c1_stats["starts"])
cond["starts2"] = cond["toban2"].map(c2_stats["starts"])
cond = cond[cond["race_date"].notna()]

total = len(cond)
h = int(cond["escaped"].sum())
lo, hi = wilson_ci(h, total)
print(f"\n①②条件レース全体: n={total} 逃げ率{h/total*100:.1f}%[{lo:.1f}-{hi:.1f}]\n")

print("=== 1号艇選手のstarts帯別(その選手が①適格と判定された時のn)の実際の逃げ率 ===")
cond["bin1"] = pd.cut(cond["starts1"], bins=bins, labels=labels, right=False)
for b in labels:
    sub = cond[cond["bin1"] == b]
    n = len(sub)
    if n < 20:
        continue
    hh = int(sub["escaped"].sum())
    r = hh / n * 100
    l, u = wilson_ci(hh, n)
    print(f"  starts {b}: n={n} 逃げ率{r:.1f}%[{l:.1f}-{u:.1f}]")

print("\n=== 2号艇選手のstarts帯別(その選手が②適格と判定された時のn)の実際の逃げ率 ===")
cond["bin2"] = pd.cut(cond["starts2"], bins=bins, labels=labels, right=False)
for b in labels:
    sub = cond[cond["bin2"] == b]
    n = len(sub)
    if n < 20:
        continue
    hh = int(sub["escaped"].sum())
    r = hh / n * 100
    l, u = wilson_ci(hh, n)
    print(f"  starts {b}: n={n} 逃げ率{r:.1f}%[{l:.1f}-{u:.1f}]")

print("\n=== 選手判定の最低starts閾値を上げた場合、対象レース数と逃げ率・単勝1回収率はどう変わるか ===")
for min_starts in [5, 10, 15, 20, 30, 50]:
    qt1 = set(c1_stats[(c1_stats["rate"] >= INN_NIGE_RATE_THRESHOLD) & (c1_stats["starts"] >= min_starts)].index)
    qt2 = set(c2_stats[(c2_stats["rate"] >= NIGASHI_RATE_THRESHOLD) & (c2_stats["starts"] >= min_starts)].index)
    sub_pairs = race_pairs[race_pairs["toban1"].isin(qt1) & race_pairs["toban2"].isin(qt2)].copy()
    sub_pairs = sub_pairs.merge(waku1_rank, on=["race_date", "jcd", "rno"], how="inner")
    sub_pairs["escaped"] = (sub_pairs["waku1_rank"] == "1").astype(int)
    sub_pairs = sub_pairs.merge(payouts_tan[payouts_tan["combination"] == "1"][["race_date", "jcd", "rno", "payout"]],
                                 on=["race_date", "jcd", "rno"], how="left")
    sub_pairs["tan_ret"] = np.where(sub_pairs["escaped"] == 1, sub_pairs["payout"].fillna(0), 0)
    sub_pairs = sub_pairs[sub_pairs["race_date"].notna()]
    n = len(sub_pairs)
    if n < 30:
        print(f"  選手判定の最低starts={min_starts}: n={n}(少なすぎるためスキップ)")
        continue
    hh = int(sub_pairs["escaped"].sum())
    r = hh / n * 100
    l, u = wilson_ci(hh, n)
    p, lo_r, hi_r = bb_recovery(sub_pairs, sub_pairs["tan_ret"], 100)
    print(f"  選手判定の最低starts={min_starts}: 対象レースn={n} 逃げ率{r:.1f}%[{l:.1f}-{u:.1f}] | 単勝1回収率{p:.1f}%[{lo_r:.1f}-{hi_r:.1f}]")

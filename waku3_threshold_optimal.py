"""
③条件(3号艇の逃がし率(個人傾向)がX%以上)の最適な閾値Xを探る。

①②条件(1号艇イン逃げ率80%以上 かつ 2号艇逃がし率50%以上)を満たす
現行の対象レース群の中で、3号艇の逃がし率(連続値)を閾値で絞り込んだ場合に
実際のイン逃げ率・単勝1回収率がどう変化するかを閾値ごとに調べる。
併せて前半/後半に分けた頑健性チェックも行う。
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

# ①: 1号艇の逃げ率
c1 = entries_all[entries_all["waku"] == 1].merge(
    results_all[results_all["waku"] == 1][["race_date", "jcd", "rno", "rank"]],
    on=["race_date", "jcd", "rno"], how="inner")
c1_stats = c1.groupby("toban").agg(starts=("rank", "size"), wins=("rank", lambda s: (s == "1").sum()))
c1_stats["rate"] = c1_stats["wins"] / c1_stats["starts"]
qualified_toban1 = set(c1_stats[(c1_stats["rate"] >= INN_NIGE_RATE_THRESHOLD) & (c1_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)

# ②: 2号艇の逃がし率
c2 = entries_all[entries_all["waku"] == 2].merge(waku1_rank, on=["race_date", "jcd", "rno"], how="inner")
c2_stats = c2.groupby("toban").agg(starts=("waku1_rank", "size"), nigasare=("waku1_rank", lambda s: (s == "1").sum()))
c2_stats["rate"] = c2_stats["nigasare"] / c2_stats["starts"]
qualified_toban2 = set(c2_stats[(c2_stats["rate"] >= NIGASHI_RATE_THRESHOLD) & (c2_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)

# ③: 3号艇の逃がし率(連続値)
c3 = entries_all[entries_all["waku"] == 3].merge(waku1_rank, on=["race_date", "jcd", "rno"], how="inner")
c3_stats = c3.groupby("toban").agg(starts=("waku1_rank", "size"), nigasare=("waku1_rank", lambda s: (s == "1").sum()))
c3_stats["rate"] = c3_stats["nigasare"] / c3_stats["starts"]
c3_reliable = c3_stats[c3_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD]

entries1 = entries_all[entries_all["waku"] == 1][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban1"})
entries2 = entries_all[entries_all["waku"] == 2][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban2"})
entries3 = entries_all[entries_all["waku"] == 3][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban3"})
race_pairs = entries1.merge(entries2, on=["race_date", "jcd", "rno"], how="inner").merge(entries3, on=["race_date", "jcd", "rno"], how="inner")

# ①②条件(現行の対象レース)を満たし、かつ3号艇の逃がし率データが十分にある選手のレースのみ
cond12 = race_pairs[
    race_pairs["toban1"].isin(qualified_toban1)
    & race_pairs["toban2"].isin(qualified_toban2)
    & race_pairs["toban3"].isin(c3_reliable.index)
].copy()
cond12 = cond12.merge(c3_reliable[["rate"]].rename(columns={"rate": "waku3_nigashi_rate"}), left_on="toban3", right_index=True, how="left")
cond12 = cond12.merge(waku1_rank, on=["race_date", "jcd", "rno"], how="inner")
cond12["escaped"] = (cond12["waku1_rank"] == "1").astype(int)
cond12 = cond12.merge(payouts_tan[payouts_tan["combination"] == "1"][["race_date", "jcd", "rno", "payout"]],
                       on=["race_date", "jcd", "rno"], how="left")
cond12["tan_ret"] = np.where(cond12["escaped"] == 1, cond12["payout"].fillna(0), 0)
cond12 = cond12.sort_values(["race_date", "rno", "jcd"]).reset_index(drop=True)

total = len(cond12)
base_hits = int(cond12["escaped"].sum())
lo, hi = wilson_ci(base_hits, total)
p0, lo0, hi0 = bb_recovery(cond12, cond12["tan_ret"], 100)
print(f"①②条件(現行)を満たすレース: {total}件、実際の逃げ率{base_hits/total*100:.1f}% [{lo:.1f}-{hi:.1f}] | 単勝1回収率{p0:.1f}%[{lo0:.1f}-{hi0:.1f}]\n")

corr = np.corrcoef(cond12["waku3_nigashi_rate"], cond12["escaped"])[0, 1]
print(f"①②条件下での3号艇逃がし率(連続値)と実際のイン逃げ成否との相関係数: r={corr:+.3f} (n={total})\n")

print("=== 3号艇逃がし率を10分位に区切った時の 実際のイン逃げ率 ===")
cond12["decile"] = pd.qcut(cond12["waku3_nigashi_rate"], q=10, duplicates="drop")
for b, sub in cond12.groupby("decile", observed=True):
    n = len(sub)
    h = int(sub["escaped"].sum())
    r = h / n * 100
    l, u = wilson_ci(h, n)
    print(f"  逃がし率{b}: n={n} 実際の逃げ率{r:.1f}% [{l:.1f}-{u:.1f}]")

print()
print("=== ③閾値(3号艇逃がし率x%以上)ごとの、対象レース数・逃げ率・単勝1回収率 ===")
for thresh in [0, 10, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]:
    sub = cond12[cond12["waku3_nigashi_rate"] >= thresh / 100]
    n = len(sub)
    if n < 30:
        print(f"  閾値{thresh}%以上: n={n}(少なすぎるためスキップ)")
        continue
    h = int(sub["escaped"].sum())
    r = h / n * 100
    l, u = wilson_ci(h, n)
    p, lo_r, hi_r = bb_recovery(sub, sub["tan_ret"], 100)
    print(f"  閾値{thresh}%以上: n={n} 逃げ率{r:.1f}%[{l:.1f}-{u:.1f}] | 単勝1回収率{p:.1f}%[{lo_r:.1f}-{hi_r:.1f}]")

print()
print("=== 前半/後半での頑健性チェック(代表的な閾値) ===")
mid = len(cond12) // 2
front = cond12.iloc[:mid]
back = cond12.iloc[mid:]
for thresh in [0, 30, 40, 50, 55, 60, 65, 70]:
    row = []
    for name, part in [("前半", front), ("後半", back)]:
        sub = part[part["waku3_nigashi_rate"] >= thresh / 100]
        n = len(sub)
        if n < 30:
            row.append(f"{name}n={n}(不足)")
            continue
        h = int(sub["escaped"].sum())
        r = h / n * 100
        p, lo_r, hi_r = bb_recovery(sub, sub["tan_ret"], 100)
        row.append(f"{name}n={n} 逃げ率{r:.1f}% 単勝1回収率{p:.1f}%[{lo_r:.1f}-{hi_r:.1f}]")
    print(f"  閾値{thresh}%以上: " + " | ".join(row))

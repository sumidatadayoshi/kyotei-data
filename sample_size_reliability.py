"""
「①イン逃げ80%以上 ②2号艇逃がし率50%以上」条件について、
n(対象レース数)と回収率・逃げ率の信頼区間の関係、および
どのくらいのnがあれば結果を信頼できるかを調べる。

1) 現状のn・逃げ率・単勝1回収率とその信頼区間
2) 時系列で見た「その時点までのレース数」ごとの累積逃げ率・回収率の推移
   (nが小さいうちはどれだけブレるか)
3) 目標とするCI幅を達成するのに必要なn の試算
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
cond = cond.merge(payouts_tan[payouts_tan["combination"] == "1"][["race_date", "jcd", "rno", "payout"]].rename(columns={"payout": "tan_payout"}),
                   on=["race_date", "jcd", "rno"], how="left")
cond["tan_ret"] = np.where(cond["escaped"] == 1, cond["tan_payout"].fillna(0), 0)

# 現行の2連単ミックス(1-2:200円 + 1-3:100円)のリターンも合わせて見る
p12 = payouts_2tan[payouts_2tan["combination"] == "1-2"][["race_date", "jcd", "rno", "payout"]].rename(columns={"payout": "p12"})
p13 = payouts_2tan[payouts_2tan["combination"] == "1-3"][["race_date", "jcd", "rno", "payout"]].rename(columns={"payout": "p13"})
cond = cond.merge(p12, on=["race_date", "jcd", "rno"], how="left").merge(p13, on=["race_date", "jcd", "rno"], how="left")
cond["mix_ret"] = cond["p12"].fillna(0) * 2 + cond["p13"].fillna(0) * 1  # 1-2は200円(=100円単位払戻の2倍), 1-3は100円

cond = cond.sort_values(["race_date", "rno", "jcd"]).reset_index(drop=True)
cond = cond[cond["race_date"].notna()]

total = len(cond)
base_hits = int(cond["escaped"].sum())
lo, hi = wilson_ci(base_hits, total)
p0, lo0, hi0 = bb_recovery(cond, cond["tan_ret"], 100)
pm, lom, him = bb_recovery(cond, cond["mix_ret"], 300)
print(f"①②条件を満たすレース: n={total}")
print(f"  実際の逃げ率: {base_hits/total*100:.1f}% [{lo:.1f}-{hi:.1f}] (半幅 ±{(hi-lo)/2:.1f}pt)")
print(f"  単勝1回収率: {p0:.1f}% [{lo0:.1f}-{hi0:.1f}] (半幅 ±{(hi0-lo0)/2:.1f}pt)")
print(f"  2連単ミックス(1-2:200+1-3:100)回収率: {pm:.1f}% [{lom:.1f}-{him:.1f}] (半幅 ±{(him-lom)/2:.1f}pt)\n")

print("=== 時系列でnが増えるにつれて、その時点までの累積逃げ率・単勝1回収率がどう変化したか ===")
checkpoints = [50, 100, 150, 200, 300, 400, 500, 700, 900, 1100, 1300, total]
for n in checkpoints:
    if n > total:
        continue
    sub = cond.iloc[:n]
    h = int(sub["escaped"].sum())
    r = h / n * 100
    l, u = wilson_ci(h, n)
    p, lo_r, hi_r = bb_recovery(sub, sub["tan_ret"], 100)
    print(f"  最初のn={n}件時点: 逃げ率{r:.1f}%[{l:.1f}-{u:.1f}](半幅±{(u-l)/2:.1f}) | 単勝1回収率{p:.1f}%[{lo_r:.1f}-{hi_r:.1f}](半幅±{(hi_r-lo_r)/2:.1f})")

print()
print("=== 目標CI半幅を達成するのに必要な概算n (現状のCI半幅からスケーリング, 半幅∝1/√n) ===")
cur_half_escape = (hi - lo) / 2
cur_half_recovery = (hi0 - lo0) / 2
print(f"  現状: n={total}, 逃げ率CI半幅=±{cur_half_escape:.2f}pt, 単勝1回収率CI半幅=±{cur_half_recovery:.2f}pt\n")
for target_half in [5, 3, 2, 1]:
    n_for_escape = total * (cur_half_escape / target_half) ** 2
    n_for_recovery = total * (cur_half_recovery / target_half) ** 2
    print(f"  逃げ率の半幅を±{target_half}pt以内にするには: 概算n≈{n_for_escape:,.0f}")
    print(f"  単勝1回収率の半幅を±{target_half}pt以内にするには: 概算n≈{n_for_recovery:,.0f}")

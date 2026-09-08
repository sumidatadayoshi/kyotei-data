"""
「2号艇の逃がし率(1号艇に1着を譲った過去割合)>=50%」という②条件が、
①条件(1号艇イン逃げ率80%以上)だけの場合と比べて、実際のイン逃げ率・回収率に
上乗せの効果を持っているかを検証する。

①条件のみを満たすレース群の中で、2号艇の逃がし率(連続値)を分位/閾値ごとに
区切り、実際のイン逃げ率・単勝1の回収率がどう変化するかを見る。
単調な右肩上がりが見えれば②条件には意味があると言える。
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "boatrace.db"
SAMPLE_SIZE_WARNING_THRESHOLD = 5
INN_NIGE_RATE_THRESHOLD = 0.8


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

# --- ①条件用: 1号艇の逃げ率(連続値) ---
c1 = entries_all[entries_all["waku"] == 1].merge(
    results_all[results_all["waku"] == 1][["race_date", "jcd", "rno", "rank"]],
    on=["race_date", "jcd", "rno"], how="inner")
c1_stats = c1.groupby("toban").agg(starts=("rank", "size"), wins=("rank", lambda s: (s == "1").sum()))
c1_stats["rate"] = c1_stats["wins"] / c1_stats["starts"]

# --- ②条件用: 2号艇の逃がし率(連続値、1号艇に1着を譲った過去割合) ---
waku1_rank = results_all[results_all["waku"] == 1][["race_date", "jcd", "rno", "rank"]].rename(columns={"rank": "waku1_rank"})
c2 = entries_all[entries_all["waku"] == 2].merge(waku1_rank, on=["race_date", "jcd", "rno"], how="inner")
c2_stats = c2.groupby("toban").agg(starts=("waku1_rank", "size"), nigasare=("waku1_rank", lambda s: (s == "1").sum()))
c2_stats["rate"] = c2_stats["nigasare"] / c2_stats["starts"]

qualified_toban1 = set(c1_stats[(c1_stats["rate"] >= INN_NIGE_RATE_THRESHOLD) & (c1_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)

entries1 = entries_all[entries_all["waku"] == 1][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban1"})
entries2 = entries_all[entries_all["waku"] == 2][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban2"})
race_pairs = entries1.merge(entries2, on=["race_date", "jcd", "rno"], how="inner")

# ①条件のみを満たすレース(②はまだ適用しない)。2号艇の逃がし率は十分なサンプル(starts>=5)がある選手のみ対象。
c2_reliable = c2_stats[c2_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD]
cond1_only = race_pairs[race_pairs["toban1"].isin(qualified_toban1) & race_pairs["toban2"].isin(c2_reliable.index)].copy()
cond1_only = cond1_only.merge(c2_reliable[["rate"]].rename(columns={"rate": "waku2_nigashi_rate"}), left_on="toban2", right_index=True, how="left")
cond1_only = cond1_only.merge(waku1_rank, on=["race_date", "jcd", "rno"], how="inner")
cond1_only["escaped"] = (cond1_only["waku1_rank"] == "1").astype(int)
cond1_only = cond1_only.merge(payouts_tan[payouts_tan["combination"] == "1"][["race_date", "jcd", "rno", "payout"]],
                               on=["race_date", "jcd", "rno"], how="left")
cond1_only["tan_ret"] = np.where(cond1_only["escaped"] == 1, cond1_only["payout"].fillna(0), 0)
cond1_only = cond1_only.sort_values(["race_date", "rno", "jcd"]).reset_index(drop=True)

total = len(cond1_only)
base_hits = int(cond1_only["escaped"].sum())
lo, hi = wilson_ci(base_hits, total)
print(f"①条件(1号艇イン逃げ率80%以上)のみを満たすレース: {total}件、実際の逃げ率{base_hits/total*100:.1f}% [{lo:.1f}-{hi:.1f}]\n")

corr = np.corrcoef(cond1_only["waku2_nigashi_rate"], cond1_only["escaped"])[0, 1]
print(f"2号艇の逃がし率(連続値)と実際のイン逃げ成否との相関係数: r={corr:+.3f} (n={total})\n")

print("=== 2号艇逃がし率(過去実績,連続値)を10分位に区切った時の 実際のイン逃げ率 ===")
cond1_only["decile"] = pd.qcut(cond1_only["waku2_nigashi_rate"], q=10, duplicates="drop")
for b, sub in cond1_only.groupby("decile", observed=True):
    n = len(sub)
    h = int(sub["escaped"].sum())
    r = h / n * 100
    l, u = wilson_ci(h, n)
    print(f"  逃がし率{b}: n={n} 実際の逃げ率{r:.1f}% [{l:.1f}-{u:.1f}]")

print()
print("=== 閾値(x%以上)ごとの、対象レース数・逃げ率・単勝1回収率 ===")
for thresh in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
    sub = cond1_only[cond1_only["waku2_nigashi_rate"] >= thresh / 100]
    n = len(sub)
    if n < 30:
        print(f"  閾値{thresh}%以上: n={n}(少なすぎるためスキップ)")
        continue
    h = int(sub["escaped"].sum())
    r = h / n * 100
    l, u = wilson_ci(h, n)
    p, lo_r, hi_r = bb_recovery(sub, sub["tan_ret"], 100)
    print(f"  閾値{thresh}%以上: n={n} 逃げ率{r:.1f}%[{l:.1f}-{u:.1f}] | 単勝1回収率{p:.1f}%[{lo_r:.1f}-{hi_r:.1f}]")

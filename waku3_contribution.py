"""
3号艇の成績がイン逃げ率に影響するかを検証する。
2号艇の逃がし率(nigashi_rate_contribution.py)と同じ発想で、
「3号艇として走った過去レースで1号艇に1着を譲った割合」を選手ごとに算出し、
①条件(1号艇イン逃げ率80%以上)のみを満たすレース群の中で、この値と実際の
イン逃げ成否との関係を見る。あわせて3号艇自身の一般的な指標(平均ST・
モーター2連率・全国勝率)も確認する。
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


conn = sqlite3.connect(DB_PATH)
entries_all = pd.read_sql_query("SELECT * FROM entries", conn)
results_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, rank FROM results", conn)

c1 = entries_all[entries_all["waku"] == 1].merge(
    results_all[results_all["waku"] == 1][["race_date", "jcd", "rno", "rank"]],
    on=["race_date", "jcd", "rno"], how="inner")
c1_stats = c1.groupby("toban").agg(starts=("rank", "size"), wins=("rank", lambda s: (s == "1").sum()))
c1_stats["rate"] = c1_stats["wins"] / c1_stats["starts"]
qualified_toban1 = set(c1_stats[(c1_stats["rate"] >= INN_NIGE_RATE_THRESHOLD) & (c1_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)

waku1_rank = results_all[results_all["waku"] == 1][["race_date", "jcd", "rno", "rank"]].rename(columns={"rank": "waku1_rank"})

# 3号艇版「逃がし率」: 3号艇として走ったレースで1号艇に1着を譲った割合
c3 = entries_all[entries_all["waku"] == 3].merge(waku1_rank, on=["race_date", "jcd", "rno"], how="inner")
c3_stats = c3.groupby("toban").agg(starts=("waku1_rank", "size"), nigasare=("waku1_rank", lambda s: (s == "1").sum()))
c3_stats["rate"] = c3_stats["nigasare"] / c3_stats["starts"]
c3_reliable = c3_stats[c3_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD]

entries1 = entries_all[entries_all["waku"] == 1][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban1"})
entries3 = entries_all[entries_all["waku"] == 3][["race_date", "jcd", "rno", "toban", "avg_st", "motor_2rate", "national_win_rate", "local_win_rate"]].rename(
    columns={"toban": "toban3", "avg_st": "waku3_avg_st", "motor_2rate": "waku3_motor_2rate",
             "national_win_rate": "waku3_national_win_rate", "local_win_rate": "waku3_local_win_rate"})
race_pairs = entries1.merge(entries3, on=["race_date", "jcd", "rno"], how="inner")

cond1_only = race_pairs[race_pairs["toban1"].isin(qualified_toban1) & race_pairs["toban3"].isin(c3_reliable.index)].copy()
cond1_only = cond1_only.merge(c3_reliable[["rate"]].rename(columns={"rate": "waku3_nigashi_rate"}), left_on="toban3", right_index=True, how="left")
cond1_only = cond1_only.merge(waku1_rank, on=["race_date", "jcd", "rno"], how="inner")
cond1_only["escaped"] = (cond1_only["waku1_rank"] == "1").astype(int)

total = len(cond1_only)
base_hits = int(cond1_only["escaped"].sum())
lo, hi = wilson_ci(base_hits, total)
print(f"①条件のみ(1号艇イン逃げ率80%以上)を満たすレース: {total}件、実際の逃げ率{base_hits/total*100:.1f}% [{lo:.1f}-{hi:.1f}]\n")

corr = np.corrcoef(cond1_only["waku3_nigashi_rate"], cond1_only["escaped"])[0, 1]
print(f"=== 3号艇版「逃がし率」(3号艇として1号艇に1着を譲った過去割合)と実際のイン逃げ成否の相関 ===")
print(f"r={corr:+.3f} (n={total})\n")

print("3号艇版逃がし率を10分位に区切った時の実際のイン逃げ率")
cond1_only["decile"] = pd.qcut(cond1_only["waku3_nigashi_rate"], q=10, duplicates="drop")
for b, sub in cond1_only.groupby("decile", observed=True):
    n = len(sub)
    h = int(sub["escaped"].sum())
    r = h / n * 100
    l, u = wilson_ci(h, n)
    print(f"  逃がし率{b}: n={n} 実際の逃げ率{r:.1f}% [{l:.1f}-{u:.1f}]")

print()
print("=== 参考: 3号艇自身の一般的な指標との相関 ===")
for col, label in [("waku3_avg_st", "3号艇 平均ST(小さいほど速い)"),
                    ("waku3_motor_2rate", "3号艇 モーター2連率"),
                    ("waku3_national_win_rate", "3号艇 全国勝率"),
                    ("waku3_local_win_rate", "3号艇 当地勝率")]:
    sub = cond1_only.dropna(subset=[col])
    corr_c = np.corrcoef(sub[col], sub["escaped"])[0, 1]
    print(f"  {label}: r={corr_c:+.3f} (n={len(sub)})")

print()
print("=== 3号艇 平均STを5分位に区切った時の実際のイン逃げ率(参考) ===")
cond1_only2 = cond1_only.dropna(subset=["waku3_avg_st"]).copy()
cond1_only2["st_bin"] = pd.qcut(cond1_only2["waku3_avg_st"], q=5, duplicates="drop")
for b, sub in cond1_only2.groupby("st_bin", observed=True):
    n = len(sub)
    h = int(sub["escaped"].sum())
    r = h / n * 100
    l, u = wilson_ci(h, n)
    print(f"  ST{b}: n={n} 実際の逃げ率{r:.1f}% [{l:.1f}-{u:.1f}]")

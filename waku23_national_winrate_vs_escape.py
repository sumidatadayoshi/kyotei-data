"""
「1号艇は逃げそうだが、2号艇の全国勝率が低すぎて買う気になれなかった」
→ 結果3号艇が優勝し、そもそも1号艇が逃げられなかった(本日鳴門3R)、
という実例を受けて、2号艇・3号艇の全国勝率が「1号艇が実際に逃げられるか」
(escaped)自体に関係しているかを検証する。

仮説: 2号艇は1号艇と3号艇の間に位置するため、2号艇が強い(全国勝率が高い)
ほど3号艇のまくり等を牽制でき、1号艇の逃げを助ける可能性がある。逆に
2号艇が弱いと3号艇が自由に攻めやすく、1号艇の逃げ失敗が増える可能性がある。
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


conn = sqlite3.connect(DB_PATH)
entries_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, toban, national_win_rate FROM entries", conn)
results_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, rank FROM results", conn)

waku1_rank = results_all[results_all["waku"] == 1][["race_date", "jcd", "rno", "rank"]].rename(columns={"rank": "waku1_rank"})

c1 = entries_all[entries_all["waku"] == 1].merge(
    results_all[results_all["waku"] == 1][["race_date", "jcd", "rno", "rank"]],
    on=["race_date", "jcd", "rno"], how="inner")
c1_stats = c1.groupby("toban").agg(starts=("rank", "size"), wins=("rank", lambda s: (s == "1").sum()))
c1_stats["rate"] = c1_stats["wins"] / c1_stats["starts"]
qualified_toban1 = set(c1_stats[(c1_stats["rate"] >= INN_NIGE_RATE_THRESHOLD) & (c1_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)

entries1 = entries_all[entries_all["waku"] == 1][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban1"})
e2 = entries_all[entries_all["waku"] == 2][["race_date", "jcd", "rno", "toban", "national_win_rate"]].rename(
    columns={"toban": "toban2", "national_win_rate": "nwr2"})
e3 = entries_all[entries_all["waku"] == 3][["race_date", "jcd", "rno", "toban", "national_win_rate"]].rename(
    columns={"toban": "toban3", "national_win_rate": "nwr3"})

race_pairs = entries1.merge(e2, on=["race_date", "jcd", "rno"], how="inner").merge(e3, on=["race_date", "jcd", "rno"], how="inner")
cond1_only = race_pairs[race_pairs["toban1"].isin(qualified_toban1)].copy()
cond1_only = cond1_only.merge(waku1_rank, on=["race_date", "jcd", "rno"], how="inner")
cond1_only["escaped"] = (cond1_only["waku1_rank"] == "1").astype(int)
cond1_only = cond1_only.dropna(subset=["nwr2", "nwr3"])

total = len(cond1_only)
base = cond1_only["escaped"].mean() * 100
print(f"①条件(1号艇イン逃げ率80%以上)のみのレース: n={total}, 実際の逃げ率{base:.1f}%\n")

r2 = np.corrcoef(cond1_only["nwr2"], cond1_only["escaped"])[0, 1]
r3 = np.corrcoef(cond1_only["nwr3"], cond1_only["escaped"])[0, 1]
diff = cond1_only["nwr3"] - cond1_only["nwr2"]
r_diff = np.corrcoef(diff, cond1_only["escaped"])[0, 1]
print(f"2号艇の全国勝率と逃げ成否の相関: r={r2:+.3f}")
print(f"3号艇の全国勝率と逃げ成否の相関: r={r3:+.3f}")
print(f"(3号艇全国勝率 - 2号艇全国勝率)と逃げ成否の相関: r={r_diff:+.3f}\n")

print("=== 2号艇の全国勝率を5分位に区切った時の実際の逃げ率 ===")
cond1_only["q2"] = pd.qcut(cond1_only["nwr2"], q=5, duplicates="drop")
for b, sub in cond1_only.groupby("q2", observed=True):
    n = len(sub)
    h = int(sub["escaped"].sum())
    l, u = wilson_ci(h, n)
    print(f"  2号艇全国勝率{b}: n={n} 逃げ率{h/n*100:.1f}%[{l:.1f}-{u:.1f}]")

print("\n=== 3号艇の全国勝率を5分位に区切った時の実際の逃げ率 ===")
cond1_only["q3"] = pd.qcut(cond1_only["nwr3"], q=5, duplicates="drop")
for b, sub in cond1_only.groupby("q3", observed=True):
    n = len(sub)
    h = int(sub["escaped"].sum())
    l, u = wilson_ci(h, n)
    print(f"  3号艇全国勝率{b}: n={n} 逃げ率{h/n*100:.1f}%[{l:.1f}-{u:.1f}]")

print("\n=== (3号艇全国勝率 - 2号艇全国勝率)を5分位に区切った時の実際の逃げ率 ===")
cond1_only["qdiff"] = pd.qcut(diff, q=5, duplicates="drop")
for b, sub in cond1_only.groupby("qdiff", observed=True):
    n = len(sub)
    h = int(sub["escaped"].sum())
    l, u = wilson_ci(h, n)
    print(f"  (3号艇-2号艇)全国勝率差{b}: n={n} 逃げ率{h/n*100:.1f}%[{l:.1f}-{u:.1f}]")

# 特に「今日の鳴門3Rのような、2号艇が極端に弱いケース」を切り出す
print("\n=== 2号艇の全国勝率が特に低い(下位10%)場合の逃げ率 ===")
p10 = cond1_only["nwr2"].quantile(0.1)
low2 = cond1_only[cond1_only["nwr2"] <= p10]
n = len(low2)
h = int(low2["escaped"].sum())
l, u = wilson_ci(h, n)
print(f"  2号艇全国勝率<= {p10:.2f}(下位10%): n={n} 逃げ率{h/n*100:.1f}%[{l:.1f}-{u:.1f}]")
rest = cond1_only[cond1_only["nwr2"] > p10]
n2 = len(rest)
h2 = int(rest["escaped"].sum())
l2, u2 = wilson_ci(h2, n2)
print(f"  それ以外: n={n2} 逃げ率{h2/n2*100:.1f}%[{l2:.1f}-{u2:.1f}]")

print("\n=== 前半/後半での頑健性チェック(2号艇全国勝率と逃げ成否の相関) ===")
cond1_sorted = cond1_only.sort_values(["race_date", "rno", "jcd"]).reset_index(drop=True)
mid = len(cond1_sorted) // 2
front, back = cond1_sorted.iloc[:mid], cond1_sorted.iloc[mid:]
for label, part in [("前半", front), ("後半", back)]:
    rr2 = np.corrcoef(part["nwr2"], part["escaped"])[0, 1]
    rr3 = np.corrcoef(part["nwr3"], part["escaped"])[0, 1]
    print(f"  {label}(n={len(part)}): 2号艇全国勝率 r={rr2:+.3f} | 3号艇全国勝率 r={rr3:+.3f}")

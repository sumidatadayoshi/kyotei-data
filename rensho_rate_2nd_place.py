"""
「当地2連率が低い(例:6.67%のような)選手が2着に来るとは思えない」を検証。
local_win_rate/national_win_rateではなく、2連率(2着以内に来る率)そのもの
で見るのが本題により近い指標のはず。

①②条件を満たし、実際に1号艇が逃げ、2着が2号艇/3号艇だったレースに絞り、
2号艇/3号艇それぞれの当地2連率・全国2連率(絶対水準)と、実際に2着に
来たかどうかの関係を見る。
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
entries_all = pd.read_sql_query(
    "SELECT race_date, jcd, rno, waku, toban, national_2rate, local_2rate FROM entries", conn)
results_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, rank FROM results", conn)

waku1_rank = results_all[results_all["waku"] == 1][["race_date", "jcd", "rno", "rank"]].rename(columns={"rank": "waku1_rank"})

c1 = entries_all[entries_all["waku"] == 1].merge(results_all[results_all["waku"] == 1][["race_date", "jcd", "rno", "rank"]], on=["race_date", "jcd", "rno"])
c1_stats = c1.groupby("toban").agg(starts=("rank", "size"), wins=("rank", lambda s: (s == "1").sum()))
c1_stats["rate"] = c1_stats["wins"] / c1_stats["starts"]
qualified_toban1 = set(c1_stats[(c1_stats["rate"] >= INN_NIGE_RATE_THRESHOLD) & (c1_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)

c2 = entries_all[entries_all["waku"] == 2].merge(waku1_rank, on=["race_date", "jcd", "rno"])
c2_stats = c2.groupby("toban").agg(starts=("waku1_rank", "size"), nigasare=("waku1_rank", lambda s: (s == "1").sum()))
c2_stats["rate"] = c2_stats["nigasare"] / c2_stats["starts"]
qualified_toban2 = set(c2_stats[(c2_stats["rate"] >= NIGASHI_RATE_THRESHOLD) & (c2_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)

entries1 = entries_all[entries_all["waku"] == 1][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban1"})
entries2 = entries_all[entries_all["waku"] == 2][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban2"})
race_pairs = entries1.merge(entries2, on=["race_date", "jcd", "rno"], how="inner")
cond = race_pairs[race_pairs["toban1"].isin(qualified_toban1) & race_pairs["toban2"].isin(qualified_toban2)][["race_date", "jcd", "rno"]]

r2 = results_all[results_all["waku"] == 2][["race_date", "jcd", "rno", "rank"]].rename(columns={"rank": "rank2"})
r3 = results_all[results_all["waku"] == 3][["race_date", "jcd", "rno", "rank"]].rename(columns={"rank": "rank3"})
df = cond.merge(waku1_rank, on=["race_date", "jcd", "rno"], how="inner")
df = df[df["waku1_rank"] == "1"]
df = df.merge(r2, on=["race_date", "jcd", "rno"], how="inner").merge(r3, on=["race_date", "jcd", "rno"], how="inner")
df = df[df["rank2"].isin(["1", "2", "3"]) & df["rank3"].isin(["1", "2", "3"])]
df["target"] = np.select([df["rank2"] == "2", df["rank3"] == "2"], [1, 0], default=np.nan)
df = df.dropna(subset=["target"])

e2 = entries_all[entries_all["waku"] == 2][["race_date", "jcd", "rno", "national_2rate", "local_2rate"]].rename(
    columns={"national_2rate": "n2r2", "local_2rate": "l2r2"})
e3 = entries_all[entries_all["waku"] == 3][["race_date", "jcd", "rno", "national_2rate", "local_2rate"]].rename(
    columns={"national_2rate": "n2r3", "local_2rate": "l2r3"})
df = df.merge(e2, on=["race_date", "jcd", "rno"], how="left").merge(e3, on=["race_date", "jcd", "rno"], how="left")

print(f"対象レース(①②を満たし、1号艇が逃げ、2着が2号艇/3号艇): n={len(df)}")
print(f"ベースライン(2号艇が2着に来る率) = {df['target'].mean()*100:.1f}%\n")

print("=== 2号艇の当地2連率(絶対水準)を4分位に区切った時、2号艇が実際に2着に来た率 ===")
d = df.dropna(subset=["l2r2"]).copy()
d["q"] = pd.qcut(d["l2r2"], q=4, duplicates="drop")
for b, sub in d.groupby("q", observed=True):
    n = len(sub)
    h = int(sub["target"].sum())
    l, u = wilson_ci(h, n)
    print(f"  2号艇当地2連率{b}: n={n} 2号艇が2着に来た率{h/n*100:.1f}%[{l:.1f}-{u:.1f}]")

print("\n=== 2号艇の全国2連率(絶対水準)を4分位に区切った時、2号艇が実際に2着に来た率 ===")
d = df.dropna(subset=["n2r2"]).copy()
d["q"] = pd.qcut(d["n2r2"], q=4, duplicates="drop")
for b, sub in d.groupby("q", observed=True):
    n = len(sub)
    h = int(sub["target"].sum())
    l, u = wilson_ci(h, n)
    print(f"  2号艇全国2連率{b}: n={n} 2号艇が2着に来た率{h/n*100:.1f}%[{l:.1f}-{u:.1f}]")

print("\n=== 2号艇の当地2連率が特に低いケース ===")
for cutoff in [5, 10, 15, 20]:
    sub = df[df["l2r2"] <= cutoff]
    n = len(sub)
    if n < 15:
        print(f"  当地2連率<= {cutoff}: n={n}(少なすぎるためスキップ)")
        continue
    h = int(sub["target"].sum())
    l, u = wilson_ci(h, n)
    print(f"  当地2連率<= {cutoff}: n={n} 2号艇が2着に来た率{h/n*100:.1f}%[{l:.1f}-{u:.1f}]")

print("\n=== 2号艇の全国2連率が特に低いケース ===")
for cutoff in [10, 15, 20, 25]:
    sub = df[df["n2r2"] <= cutoff]
    n = len(sub)
    if n < 15:
        print(f"  全国2連率<= {cutoff}: n={n}(少なすぎるためスキップ)")
        continue
    h = int(sub["target"].sum())
    l, u = wilson_ci(h, n)
    print(f"  全国2連率<= {cutoff}: n={n} 2号艇が2着に来た率{h/n*100:.1f}%[{l:.1f}-{u:.1f}]")

print("\n=== 相関係数まとめ(2号艇が2着=1として) ===")
for col, label in [("n2r2", "2号艇全国2連率"), ("l2r2", "2号艇当地2連率"), ("n2r3", "3号艇全国2連率"), ("l2r3", "3号艇当地2連率")]:
    dd = df.dropna(subset=[col])
    r = np.corrcoef(dd[col], dd["target"])[0, 1]
    print(f"  {label}: r={r:+.3f} (n={len(dd)})")

l2r2_all = df["l2r2"].dropna()
n2r2_all = df["n2r2"].dropna()
print(f"\n参考: 今日の加藤選手 当地2連率6.67 は下から{(l2r2_all < 6.67).mean()*100:.0f}パーセンタイル")
print(f"参考: 今日の加藤選手 全国2連率13.92 は下から{(n2r2_all < 13.92).mean()*100:.0f}パーセンタイル")

"""
「向かって左(=内側)の艇との比較」: 2号艇なら1号艇、3号艇なら2号艇との
スタート比較が、逃げ成否/2着争いに関係するかを調べる。

2種類の比較を行う:
(A) 実際のそのレースでのスタートタイミング比較(結果論・メカニズム確認用。
    ベット時点では分からない情報だが「なぜ」を理解するのに使う)
(B) 選手の平均ST(事前に分かる情報)同士の差(実際に賭け判断に使える指標)
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


def parse_st(v):
    if v is None:
        return np.nan
    v = str(v).strip()
    if v == "" or v.upper() == "L":
        return 9.99
    if v.upper().startswith("F"):
        try:
            return -float(v[1:])
        except ValueError:
            return np.nan
    try:
        return float(v)
    except ValueError:
        return np.nan


conn = sqlite3.connect(DB_PATH)
entries_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, toban, avg_st FROM entries", conn)
results_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, rank, toban, start_timing FROM results", conn)
results_all["st_val"] = results_all["start_timing"].map(parse_st)

waku1_res = results_all[results_all["waku"] == 1][["race_date", "jcd", "rno", "rank", "st_val"]].rename(
    columns={"rank": "waku1_rank", "st_val": "st1_actual"})
waku2_res = results_all[results_all["waku"] == 2][["race_date", "jcd", "rno", "rank", "st_val"]].rename(
    columns={"rank": "waku2_rank", "st_val": "st2_actual"})
waku3_res = results_all[results_all["waku"] == 3][["race_date", "jcd", "rno", "rank", "st_val"]].rename(
    columns={"rank": "waku3_rank", "st_val": "st3_actual"})

c1 = entries_all[entries_all["waku"] == 1].merge(waku1_res[["race_date", "jcd", "rno", "waku1_rank"]], on=["race_date", "jcd", "rno"])
c1_stats = c1.groupby("toban").agg(starts=("waku1_rank", "size"), wins=("waku1_rank", lambda s: (s == "1").sum()))
c1_stats["rate"] = c1_stats["wins"] / c1_stats["starts"]
qualified_toban1 = set(c1_stats[(c1_stats["rate"] >= INN_NIGE_RATE_THRESHOLD) & (c1_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)

entries1 = entries_all[entries_all["waku"] == 1][["race_date", "jcd", "rno", "toban", "avg_st"]].rename(columns={"toban": "toban1", "avg_st": "avg_st1"})
entries2 = entries_all[entries_all["waku"] == 2][["race_date", "jcd", "rno", "toban", "avg_st"]].rename(columns={"toban": "toban2", "avg_st": "avg_st2"})
entries3 = entries_all[entries_all["waku"] == 3][["race_date", "jcd", "rno", "toban", "avg_st"]].rename(columns={"toban": "toban3", "avg_st": "avg_st3"})

# ============ (A) 1号艇 vs 2号艇: 実際のST比較が逃げ成否に関係するか ============
cond1 = entries1[entries1["toban1"].isin(qualified_toban1)][["race_date", "jcd", "rno", "toban1", "avg_st1"]]
cond1 = cond1.merge(waku1_res, on=["race_date", "jcd", "rno"], how="inner")
cond1 = cond1.merge(waku2_res[["race_date", "jcd", "rno", "st2_actual"]], on=["race_date", "jcd", "rno"], how="inner")
cond1 = cond1.merge(entries2[["race_date", "jcd", "rno", "avg_st2"]], on=["race_date", "jcd", "rno"], how="left")
cond1["escaped"] = (cond1["waku1_rank"] == "1").astype(int)
cond1 = cond1.dropna(subset=["st1_actual", "st2_actual"])

total = len(cond1)
base = cond1["escaped"].mean() * 100
print(f"=== ①条件レース(n={total}, 逃げ率{base:.1f}%): 1号艇 vs 2号艇のスタート比較 ===\n")

cond1["st2_faster_than_st1_actual"] = cond1["st2_actual"] < cond1["st1_actual"]
sub = cond1[cond1["st2_faster_than_st1_actual"]]
n = len(sub); h = int(sub["escaped"].sum()); l, u = wilson_ci(h, n)
print(f"(A)実際のそのレースで2号艇の方がSTが早かった場合: n={n} 逃げ率{h/n*100:.1f}%[{l:.1f}-{u:.1f}]")
sub2 = cond1[~cond1["st2_faster_than_st1_actual"]]
n2 = len(sub2); h2 = int(sub2["escaped"].sum()); l2, u2 = wilson_ci(h2, n2)
print(f"           1号艇の方がSTが早かった(or同じ)場合: n={n2} 逃げ率{h2/n2*100:.1f}%[{l2:.1f}-{u2:.1f}]")

cond1b = cond1.dropna(subset=["avg_st1", "avg_st2"])
diff_pre = cond1b["avg_st2"] - cond1b["avg_st1"]  # 正なら2号艇の平均STの方が遅い(1号艇有利)
r_pre = np.corrcoef(diff_pre, cond1b["escaped"])[0, 1]
print(f"\n(B)事前情報: (2号艇の平均ST - 1号艇の平均ST)と逃げ成否の相関: r={r_pre:+.3f} (n={len(cond1b)})")
print("   ※値が小さい(2号艇の方が平均STが早い)ほど、1号艇の逃げに不利という仮説")

print("\n=== 前半/後半頑健性(B) ===")
cond1b_sorted = cond1b.sort_values(["race_date", "rno", "jcd"]).reset_index(drop=True)
mid = len(cond1b_sorted) // 2
for label, part in [("前半", cond1b_sorted.iloc[:mid]), ("後半", cond1b_sorted.iloc[mid:])]:
    d = part["avg_st2"] - part["avg_st1"]
    r = np.corrcoef(d, part["escaped"])[0, 1]
    print(f"  {label}(n={len(part)}): r={r:+.3f}")

# ============ 3号艇 vs 2号艇: 2着争いに関係するか ============
print("\n\n=== 2着争い(2号艇 vs 3号艇): 3号艇 vs 2号艇のスタート比較 ===\n")

c2 = entries_all[entries_all["waku"] == 2].merge(waku1_res[["race_date", "jcd", "rno", "waku1_rank"]], on=["race_date", "jcd", "rno"])
c2_stats = c2.groupby("toban").agg(starts=("waku1_rank", "size"), nigasare=("waku1_rank", lambda s: (s == "1").sum()))
c2_stats["rate"] = c2_stats["nigasare"] / c2_stats["starts"]
qualified_toban2 = set(c2_stats[(c2_stats["rate"] >= NIGASHI_RATE_THRESHOLD) & (c2_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)

pairs = entries1[entries1["toban1"].isin(qualified_toban1)][["race_date", "jcd", "rno"]].merge(
    entries2[entries2["toban2"].isin(qualified_toban2)][["race_date", "jcd", "rno", "avg_st2"]],
    on=["race_date", "jcd", "rno"], how="inner")
df = pairs.merge(waku1_res[["race_date", "jcd", "rno", "waku1_rank"]], on=["race_date", "jcd", "rno"], how="inner")
df = df[df["waku1_rank"] == "1"]
df = df.merge(waku2_res[["race_date", "jcd", "rno", "waku2_rank", "st2_actual"]], on=["race_date", "jcd", "rno"], how="inner")
df = df.merge(waku3_res[["race_date", "jcd", "rno", "waku3_rank", "st3_actual"]], on=["race_date", "jcd", "rno"], how="inner")
df = df[df["waku2_rank"].isin(["1", "2", "3"]) & df["waku3_rank"].isin(["1", "2", "3"])]
df["target"] = np.select([df["waku2_rank"] == "2", df["waku3_rank"] == "2"], [1, 0], default=np.nan)
df = df.dropna(subset=["target", "st2_actual", "st3_actual"])

total2 = len(df)
base2 = df["target"].mean() * 100
print(f"対象(n={total2}, 2号艇が2着に来る率{base2:.1f}%)\n")

df["st3_faster_actual"] = df["st3_actual"] < df["st2_actual"]
sub = df[df["st3_faster_actual"]]
n = len(sub); h = int(sub["target"].sum()); l, u = wilson_ci(h, n)
print(f"(A)実際のそのレースで3号艇の方がSTが早かった場合: n={n} 2号艇が2着に来た率{h/n*100:.1f}%[{l:.1f}-{u:.1f}]")
sub2 = df[~df["st3_faster_actual"]]
n2 = len(sub2); h2 = int(sub2["target"].sum()); l2, u2 = wilson_ci(h2, n2)
print(f"           2号艇の方がSTが早かった(or同じ)場合: n={n2} 2号艇が2着に来た率{h2/n2*100:.1f}%[{l2:.1f}-{u2:.1f}]")

df = df.merge(entries3[["race_date", "jcd", "rno", "avg_st3"]], on=["race_date", "jcd", "rno"], how="left")
dfb = df.dropna(subset=["avg_st2", "avg_st3"])
diff_pre2 = dfb["avg_st3"] - dfb["avg_st2"]  # 正なら3号艇の方が平均ST遅い(2号艇有利)
r_pre2 = np.corrcoef(diff_pre2, dfb["target"])[0, 1]
print(f"\n(B)事前情報: (3号艇の平均ST - 2号艇の平均ST)と「2号艇が2着」の相関: r={r_pre2:+.3f} (n={len(dfb)})")

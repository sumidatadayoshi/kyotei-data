"""
①②条件を満たし、かつ実際に1号艇が逃げたレースの中で、
2着に2号艇/3号艇のどちらが来るかを予想できる指標を探す
(second_place_predictor.py / second_place_dynamic_strategy.py の再検証、
DB増加後のデータで再実施。単一指標ごとの相関→頑健性チェックの順で進める)
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "boatrace.db"
SAMPLE_SIZE_WARNING_THRESHOLD = 5
INN_NIGE_RATE_THRESHOLD = 0.8
NIGASHI_RATE_THRESHOLD = 0.5

conn = sqlite3.connect(DB_PATH)
entries_all = pd.read_sql_query(
    "SELECT race_date, jcd, rno, waku, toban, racer_class, age, weight, f_count, l_count, "
    "avg_st, national_win_rate, national_2rate, local_win_rate, local_2rate, "
    "motor_2rate, motor_3rate, boat_2rate, boat_3rate FROM entries", conn)
results_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, rank FROM results", conn)

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
cond = race_pairs[race_pairs["toban1"].isin(qualified_toban1) & race_pairs["toban2"].isin(qualified_toban2)][["race_date", "jcd", "rno"]]

# ①②対象レースのうち、実際に1号艇が逃げて、かつ2着が2号艇 or 3号艇だったレースだけに絞る
r2 = results_all[results_all["waku"] == 2][["race_date", "jcd", "rno", "rank"]].rename(columns={"rank": "rank2"})
r3 = results_all[results_all["waku"] == 3][["race_date", "jcd", "rno", "rank"]].rename(columns={"rank": "rank3"})
df = cond.merge(waku1_rank, on=["race_date", "jcd", "rno"], how="inner")
df = df[df["waku1_rank"] == "1"]  # 実際に1号艇が逃げたレースのみ
df = df.merge(r2, on=["race_date", "jcd", "rno"], how="inner").merge(r3, on=["race_date", "jcd", "rno"], how="inner")
df = df[df["rank2"].isin(["1", "2", "3"]) & df["rank3"].isin(["1", "2", "3"])]
df["target"] = np.select([df["rank2"] == "2", df["rank3"] == "2"], [1, 0], default=np.nan)
df = df.dropna(subset=["target"])
print(f"対象レース(①②を満たし、実際に1号艇が逃げ、2着が2号艇/3号艇だったレース): n={len(df)}")
print(f"  2号艇が2着: {(df['target']==1).sum()}件 ({(df['target']==1).mean()*100:.1f}%)")
print(f"  3号艇が2着: {(df['target']==0).sum()}件 ({(df['target']==0).mean()*100:.1f}%)\n")

e2 = entries_all[entries_all["waku"] == 2].drop(columns=["waku"]).add_suffix("_2").rename(columns={"race_date_2": "race_date", "jcd_2": "jcd", "rno_2": "rno"})
e3 = entries_all[entries_all["waku"] == 3].drop(columns=["waku"]).add_suffix("_3").rename(columns={"race_date_3": "race_date", "jcd_3": "jcd", "rno_3": "rno"})
df = df.merge(e2, on=["race_date", "jcd", "rno"], how="left").merge(e3, on=["race_date", "jcd", "rno"], how="left")

class_map = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
df["class_2"] = df["racer_class_2"].map(class_map)
df["class_3"] = df["racer_class_3"].map(class_map)

feature_defs = [
    ("st_diff", "avg_st_3", "avg_st_2"),          # STは小さい方が有利 → 3の値-2の値 が正なら2号艇有利
    ("national_win_diff", "national_win_rate_2", "national_win_rate_3"),
    ("national_2rate_diff", "national_2rate_2", "national_2rate_3"),
    ("local_win_diff", "local_win_rate_2", "local_win_rate_3"),
    ("local_2rate_diff", "local_2rate_2", "local_2rate_3"),
    ("motor_2rate_diff", "motor_2rate_2", "motor_2rate_3"),
    ("motor_3rate_diff", "motor_3rate_2", "motor_3rate_3"),
    ("boat_2rate_diff", "boat_2rate_2", "boat_2rate_3"),
    ("class_diff", "class_2", "class_3"),
    ("age_diff", "age_3", "age_2"),
    ("weight_diff", "weight_3", "weight_2"),       # 軽い方が有利、という説があるので3-2
    ("f_count_diff", "f_count_3", "f_count_2"),
    ("l_count_diff", "l_count_3", "l_count_2"),
]

print("=== 各指標(2号艇側が有利なほど+になるよう設計)と「2着が2号艇か」の相関 ===")
rows = []
for name, a, b in feature_defs:
    d = df[[a, b, "target"]].dropna()
    if len(d) < 30:
        continue
    diff = d[a] - d[b]
    r = np.corrcoef(diff, d["target"])[0, 1]
    rows.append((name, r, len(d)))
rows.sort(key=lambda x: -abs(x[1]))
for name, r, n in rows:
    print(f"  {name}: r={r:+.3f} (n={n})")

print("\n=== 前半/後半で相関の符号・大きさが安定しているか(上位指標) ===")
df_sorted = df.sort_values(["race_date", "rno", "jcd"]).reset_index(drop=True)
mid = len(df_sorted) // 2
front, back = df_sorted.iloc[:mid], df_sorted.iloc[mid:]
for name, r_all, n_all in rows[:6]:
    a, b = [x for x in feature_defs if x[0] == name][0][1:]
    results = []
    for label, part in [("全体", df_sorted), ("前半", front), ("後半", back)]:
        d = part[[a, b, "target"]].dropna()
        if len(d) < 20:
            results.append(f"{label}n不足")
            continue
        diff = d[a] - d[b]
        r = np.corrcoef(diff, d["target"])[0, 1]
        results.append(f"{label}r={r:+.3f}(n={len(d)})")
    print(f"  {name}: " + " | ".join(results))

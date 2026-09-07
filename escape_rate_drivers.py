"""
探索その5-A: ①②条件(1号艇イン逃げ80%以上・2号艇逃し率50%以上)を満たすレースの中で、
実際のイン逃げ率(現状90.6%)をさらに引き上げられる追加指標を探す。

候補: 天候/風速/波高/水温、開催グレード、場(jcd)、1号艇自身の指標(平均ST・モーター
2連率・全国/当地勝率)、4号艇の指標(平均ST・モーター2連率=まくり力の代理)。
Wilson score 95%CIで区間を出し、ベースラインと重ならない(=統計的に見て意味がありそうな)
セグメントを探したうえで、前半/後半の時系列2分割で再現するかを検証する。
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "boatrace.db"
SAMPLE_SIZE_WARNING_THRESHOLD = 5
INN_NIGE_RATE_THRESHOLD = 0.8
NIGASHI_RATE_THRESHOLD = 0.5


def compute_racer_rate_stats(entries_all, results_all):
    c1 = entries_all[entries_all["waku"] == 1].merge(
        results_all[results_all["waku"] == 1][["race_date", "jcd", "rno", "rank"]],
        on=["race_date", "jcd", "rno"], how="inner")
    c1_stats = c1.groupby("toban").agg(starts=("rank", "size"), wins=("rank", lambda s: (s == "1").sum()))
    c1_stats["rate"] = c1_stats["wins"] / c1_stats["starts"]
    waku1_rank = results_all[results_all["waku"] == 1][["race_date", "jcd", "rno", "rank"]].rename(columns={"rank": "waku1_rank"})
    c2 = entries_all[entries_all["waku"] == 2].merge(waku1_rank, on=["race_date", "jcd", "rno"], how="inner")
    c2_stats = c2.groupby("toban").agg(starts=("waku1_rank", "size"), nigasare=("waku1_rank", lambda s: (s == "1").sum()))
    c2_stats["rate"] = c2_stats["nigasare"] / c2_stats["starts"]
    return c1_stats, c2_stats, waku1_rank


def find_qualifying_races(entries_df, c1_stats, c2_stats):
    q1 = set(c1_stats[(c1_stats["rate"] >= INN_NIGE_RATE_THRESHOLD) & (c1_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)
    q2 = set(c2_stats[(c2_stats["rate"] >= NIGASHI_RATE_THRESHOLD) & (c2_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)
    e1 = entries_df[entries_df["waku"] == 1][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban1"})
    e2 = entries_df[entries_df["waku"] == 2][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban2"})
    pairs = e1.merge(e2, on=["race_date", "jcd", "rno"], how="inner")
    return pairs[pairs["toban1"].isin(q1) & pairs["toban2"].isin(q2)].copy()


def wilson_ci(hits, n, z=1.959963984540054):
    if n == 0:
        return 0.0, 0.0
    p = hits / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (center - half) * 100, (center + half) * 100


def report_binary_segments(df, flag_col, group_col, label, min_n=30):
    print(f"--- {label} ---")
    for val, sub in df.groupby(group_col, dropna=False):
        n = len(sub)
        if n < min_n:
            continue
        hits = int(sub[flag_col].sum())
        rate = hits / n * 100
        lo, hi = wilson_ci(hits, n)
        print(f"  {val}: n={n} 逃げ率{rate:.1f}% [{lo:.1f}-{hi:.1f}]")
    print()


conn = sqlite3.connect(DB_PATH)
entries_all = pd.read_sql_query("SELECT * FROM entries", conn)
results_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, rank FROM results", conn)
races_meta = pd.read_sql_query(
    "SELECT race_date, jcd, rno, grade, weather, wind_speed, wave_height, water_temp FROM races", conn
)
payouts_2tan = pd.read_sql_query("SELECT race_date, jcd, rno, combination, payout FROM payouts WHERE bet_type = '2連単'", conn)

c1_stats, c2_stats, waku1_rank = compute_racer_rate_stats(entries_all, results_all)
candidates = find_qualifying_races(entries_all, c1_stats, c2_stats)
concluded = candidates.merge(payouts_2tan, on=["race_date", "jcd", "rno"], how="inner")
concluded = concluded.merge(waku1_rank, on=["race_date", "jcd", "rno"], how="inner")
concluded = concluded.merge(races_meta, on=["race_date", "jcd", "rno"], how="left")

feature_cols = ["avg_st", "motor_2rate", "motor_3rate", "national_win_rate", "local_win_rate", "weight", "age"]
wide = entries_all.pivot_table(index=["race_date", "jcd", "rno"], columns="waku", values=feature_cols)
wide.columns = [f"{col}_{int(waku)}" for col, waku in wide.columns]
wide = wide.reset_index()
concluded = concluded.merge(wide, on=["race_date", "jcd", "rno"], how="left")
concluded = concluded.sort_values(["race_date", "rno", "jcd"]).reset_index(drop=True)

concluded["escaped"] = (concluded["waku1_rank"] == "1").astype(int)

total = len(concluded)
base_hits = int(concluded["escaped"].sum())
base_rate = base_hits / total * 100
lo, hi = wilson_ci(base_hits, total)
print(f"対象レース数: {total}件 / 実際の逃げ回数: {base_hits}回 / 逃げ率{base_rate:.1f}% [95%CI {lo:.1f}-{hi:.1f}]\n")

print("=== 天候 ===")
report_binary_segments(concluded, "escaped", "weather", "天候別")

print("=== グレード ===")
report_binary_segments(concluded, "escaped", "grade", "グレード別")

concluded["wind_bin"] = pd.cut(concluded["wind_speed"], bins=[-0.1, 1, 2, 3, 4, 100], labels=["0-1m", "1-2m", "2-3m", "3-4m", "4m+"])
print("=== 風速 ===")
report_binary_segments(concluded, "escaped", "wind_bin", "風速帯別")

concluded["wave_bin"] = pd.cut(concluded["wave_height"], bins=[-0.1, 1, 2, 3, 100], labels=["0-1cm", "1-2cm", "2-3cm", "3cm+"])
print("=== 波高 ===")
report_binary_segments(concluded, "escaped", "wave_bin", "波高帯別")

print("=== 場(jcd)別、n>=30 ===")
report_binary_segments(concluded, "escaped", "jcd", "場別", min_n=30)

for col, label, bins in [
    ("avg_st_1", "1号艇 平均ST", [0, 0.13, 0.15, 0.17, 0.19, 1]),
    ("motor_2rate_1", "1号艇 モーター2連率", [0, 25, 30, 35, 40, 100]),
    ("national_win_rate_1", "1号艇 全国勝率", [0, 5.5, 6.0, 6.5, 7.0, 10]),
    ("local_win_rate_1", "1号艇 当地勝率", [0, 5.5, 6.0, 6.5, 7.0, 10]),
    ("avg_st_4", "4号艇 平均ST(まくり力の代理)", [0, 0.13, 0.15, 0.17, 0.19, 1]),
    ("motor_2rate_4", "4号艇 モーター2連率(まくり力の代理)", [0, 25, 30, 35, 40, 100]),
]:
    concluded[f"{col}_bin"] = pd.cut(concluded[col], bins=bins)
    print(f"=== {label} ===")
    report_binary_segments(concluded, "escaped", f"{col}_bin", label)

concluded["st_gap_1_4"] = concluded["avg_st_4"] - concluded["avg_st_1"]  # 正=1号艇の方がスタート速い
concluded["st_gap_bin"] = pd.qcut(concluded["st_gap_1_4"], q=5, duplicates="drop")
print("=== 1号艇と4号艇のST差(avg_st_4 - avg_st_1、正=1号艇が有利) ===")
report_binary_segments(concluded, "escaped", "st_gap_bin", "ST差(5分位)")

concluded.to_pickle("/tmp/concluded_escape.pkl")
print("saved concluded dataframe to /tmp/concluded_escape.pkl")

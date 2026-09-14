"""
①②条件(1号艇イン逃げ率80%以上・2号艇逃し率50%以上)に合致するレース数が、
期間を通じて減少傾向にあるかどうかを確認する。

判定ロジックはblock_bootstrap_weighted_12_13.pyと同一。
1日あたりの対象レース数・全レース数・該当率(%)を集計し、週単位の推移と
前半/後半の比較、単純な線形回帰の傾きを出す。
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "boatrace.db"

SAMPLE_SIZE_WARNING_THRESHOLD = 10
INN_NIGE_RATE_THRESHOLD = 0.8
NIGASHI_RATE_THRESHOLD = 0.5


def compute_racer_rate_stats(entries_all, results_all):
    c1 = entries_all[entries_all["waku"] == 1].merge(
        results_all[results_all["waku"] == 1][["race_date", "jcd", "rno", "rank"]],
        on=["race_date", "jcd", "rno"], how="inner",
    )
    c1_stats = c1.groupby("toban").agg(starts=("rank", "size"), wins=("rank", lambda s: (s == "1").sum()))
    c1_stats["rate"] = c1_stats["wins"] / c1_stats["starts"]

    waku1_rank = results_all[results_all["waku"] == 1][["race_date", "jcd", "rno", "rank"]].rename(
        columns={"rank": "waku1_rank"}
    )
    c2 = entries_all[entries_all["waku"] == 2].merge(
        waku1_rank, on=["race_date", "jcd", "rno"], how="inner",
    )
    c2_stats = c2.groupby("toban").agg(
        starts=("waku1_rank", "size"), nigasare=("waku1_rank", lambda s: (s == "1").sum())
    )
    c2_stats["rate"] = c2_stats["nigasare"] / c2_stats["starts"]

    return c1_stats, c2_stats


def find_qualifying_races(entries_df, c1_stats, c2_stats):
    qualified_toban1 = set(
        c1_stats[
            (c1_stats["rate"] >= INN_NIGE_RATE_THRESHOLD)
            & (c1_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)
        ].index
    )
    qualified_toban2 = set(
        c2_stats[
            (c2_stats["rate"] >= NIGASHI_RATE_THRESHOLD)
            & (c2_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)
        ].index
    )

    entries1 = entries_df[entries_df["waku"] == 1][
        ["race_date", "jcd", "rno", "toban"]
    ].rename(columns={"toban": "toban1"})
    entries2 = entries_df[entries_df["waku"] == 2][
        ["race_date", "jcd", "rno", "toban"]
    ].rename(columns={"toban": "toban2"})
    race_pairs = entries1.merge(entries2, on=["race_date", "jcd", "rno"], how="inner")

    return race_pairs[
        race_pairs["toban1"].isin(qualified_toban1) & race_pairs["toban2"].isin(qualified_toban2)
    ].copy()


conn = sqlite3.connect(DB_PATH)
entries_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, toban FROM entries", conn)
results_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, rank FROM results", conn)

c1_stats, c2_stats = compute_racer_rate_stats(entries_all, results_all)
candidates = find_qualifying_races(entries_all, c1_stats, c2_stats)

# 全レース数(1日あたり)は、1号艇・2号艇の出走情報が両方そろっているレース数を母数にする
all_races = entries_all[entries_all["waku"].isin([1, 2])].drop_duplicates(
    ["race_date", "jcd", "rno"]
)[["race_date", "jcd", "rno"]].drop_duplicates()

daily_all = all_races.groupby("race_date").size().rename("total_races")
daily_qual = candidates.groupby("race_date").size().rename("qualified_races")

daily = pd.concat([daily_all, daily_qual], axis=1).fillna(0)
daily["qualified_races"] = daily["qualified_races"].astype(int)
daily["total_races"] = daily["total_races"].astype(int)
daily["qual_rate"] = daily["qualified_races"] / daily["total_races"] * 100
daily = daily.sort_index()
daily.index = pd.to_datetime(daily.index, format="%Y%m%d")

out = []
out.append(f"対象期間: {daily.index.min().date()} 〜 {daily.index.max().date()} ({len(daily)}日)")
out.append(f"1日あたり対象レース数 平均: {daily['qualified_races'].mean():.2f}件, "
           f"該当率平均: {daily['qual_rate'].mean():.1f}%\n")

# 週単位集計(月曜始まり)
weekly = daily.resample("W-MON", label="left", closed="left").agg(
    {"qualified_races": "sum", "total_races": "sum"}
)
weekly = weekly[weekly["total_races"] > 0]
weekly["qual_rate"] = weekly["qualified_races"] / weekly["total_races"] * 100

out.append("■ 週ごとの対象レース数・該当率")
for idx, row in weekly.iterrows():
    out.append(f"  {idx.date()}週〜: 対象{int(row['qualified_races']):>3}件 / "
               f"全{int(row['total_races']):>4}件 / 該当率{row['qual_rate']:5.1f}%")
out.append("")

# 前半/後半比較
mid = len(daily) // 2
first_half = daily.iloc[:mid]
second_half = daily.iloc[mid:]
out.append("■ 前半 vs 後半 比較")
out.append(f"  前半({first_half.index.min().date()}〜{first_half.index.max().date()}, {len(first_half)}日): "
           f"1日平均{first_half['qualified_races'].mean():.2f}件 / 該当率平均{first_half['qual_rate'].mean():.1f}%")
out.append(f"  後半({second_half.index.min().date()}〜{second_half.index.max().date()}, {len(second_half)}日): "
           f"1日平均{second_half['qualified_races'].mean():.2f}件 / 該当率平均{second_half['qual_rate'].mean():.1f}%")
out.append("")

# 線形回帰の傾き(該当率 vs 経過日数)
x = np.arange(len(daily))
y_rate = daily["qual_rate"].to_numpy()
y_n = daily["qualified_races"].to_numpy()
slope_rate, intercept_rate = np.polyfit(x, y_rate, 1)
slope_n, intercept_n = np.polyfit(x, y_n, 1)
out.append("■ 線形回帰(全期間, x=経過日数)")
out.append(f"  該当率の傾き: {slope_rate:+.3f} ポイント/日 (期間全体で{slope_rate*len(daily):+.1f}ポイント相当)")
out.append(f"  対象レース数の傾き: {slope_n:+.3f} 件/日 (期間全体で{slope_n*len(daily):+.1f}件相当)")

# 直近7日 vs それ以前
recent = daily.tail(7)
earlier = daily.iloc[:-7]
out.append("")
out.append("■ 直近7日 vs それ以前")
out.append(f"  直近7日({recent.index.min().date()}〜{recent.index.max().date()}): "
           f"1日平均{recent['qualified_races'].mean():.2f}件 / 該当率平均{recent['qual_rate'].mean():.1f}%")
out.append(f"  それ以前({earlier.index.min().date()}〜{earlier.index.max().date()}, {len(earlier)}日): "
           f"1日平均{earlier['qualified_races'].mean():.2f}件 / 該当率平均{earlier['qual_rate'].mean():.1f}%")

text = "\n".join(out)
print(text)
Path(__file__).parent.joinpath("qualifying_races_trend_output.txt").write_text(text, encoding="utf-8")

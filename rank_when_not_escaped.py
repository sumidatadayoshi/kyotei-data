"""
①②条件(1号艇イン逃げ率80%以上・2号艇逃し率50%以上)に合致した過去レースのうち、
実際には1号艇が1着に来なかったレース（＝逃げ切れなかったレース）だけを対象に、
実際の2連単の決着パターン（1着-2着）の出現頻度を件数・割合とともにランキングする。

判定ロジックはblock_bootstrap_1_2.py等と同一内容を複製。
"""
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "boatrace.db"

SAMPLE_SIZE_WARNING_THRESHOLD = 5
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

    return c1_stats, c2_stats, waku1_rank


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
        ["race_date", "jcd", "rno", "toban", "racer_name", "venue_name"]
    ].rename(columns={"toban": "toban1", "racer_name": "racer1_name"})
    entries2 = entries_df[entries_df["waku"] == 2][
        ["race_date", "jcd", "rno", "toban", "racer_name"]
    ].rename(columns={"toban": "toban2", "racer_name": "racer2_name"})
    race_pairs = entries1.merge(entries2, on=["race_date", "jcd", "rno"], how="inner")

    return race_pairs[
        race_pairs["toban1"].isin(qualified_toban1) & race_pairs["toban2"].isin(qualified_toban2)
    ].copy()


conn = sqlite3.connect(DB_PATH)
entries_all = pd.read_sql_query(
    "SELECT race_date, jcd, rno, waku, toban, racer_name, gender, venue_name FROM entries", conn
)
results_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, rank FROM results", conn)
payouts_2tan = pd.read_sql_query(
    "SELECT race_date, jcd, rno, combination, payout FROM payouts WHERE bet_type = '2連単'", conn
)

c1_stats, c2_stats, waku1_rank = compute_racer_rate_stats(entries_all, results_all)
candidates = find_qualifying_races(entries_all, c1_stats, c2_stats)

# ①②条件に合致し、かつ結果(2連単payout)が判明しているレース
concluded = candidates.merge(payouts_2tan, on=["race_date", "jcd", "rno"], how="inner")

# 実際に1号艇の着順を突き合わせる
concluded = concluded.merge(waku1_rank, on=["race_date", "jcd", "rno"], how="inner")

# 1号艇が1着に来なかったレースだけに絞る
not_escaped = concluded[concluded["waku1_rank"] != "1"].copy()

total = len(not_escaped)
ranking = (
    not_escaped["combination"]
    .value_counts()
    .reset_index()
)
ranking.columns = ["combination", "count"]
ranking["share"] = ranking["count"] / total * 100

print(f"①②条件に合致し、かつ1号艇が1着に来なかったレース: {total}件\n")
print("実際の2連単決着パターン 上位10（件数・割合）")
for i, row in ranking.head(10).iterrows():
    print(f"{i+1:2d}位 {row['combination']}: {int(row['count'])}件 ({row['share']:.1f}%)")

print(f"\n（参考）1号艇が1着に来たレース: {len(concluded) - total}件 / "
      f"対象全体: {len(concluded)}件 "
      f"（1号艇の実際の逃げ率 = {(len(concluded) - total) / len(concluded) * 100:.1f}%）")

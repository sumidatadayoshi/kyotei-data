"""
①②条件(1号艇イン逃げ率80%以上・2号艇逃し率50%以上)に合致した過去レースを
対象に、2連単「1-2」に500円、「1-3」に400円、「1-4」に300円、「1-5」に200円
（1-6は買わない、1レースあたり合計1,400円）を賭け続けた場合について、
1日ごとの対象レース数・的中回数・回収率を表にする。

block_bootstrap_weighted_no16.pyと同一の判定ロジック・買い目。
"""
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "boatrace.db"

SAMPLE_SIZE_WARNING_THRESHOLD = 5
INN_NIGE_RATE_THRESHOLD = 0.8
NIGASHI_RATE_THRESHOLD = 0.5

BET_WEIGHTS = {"1-2": 500, "1-3": 400, "1-4": 300, "1-5": 200}
TOTAL_BET_PER_RACE = sum(BET_WEIGHTS.values())  # 1,400円


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

c1_stats, c2_stats, _ = compute_racer_rate_stats(entries_all, results_all)
candidates = find_qualifying_races(entries_all, c1_stats, c2_stats)
concluded = candidates.merge(payouts_2tan, on=["race_date", "jcd", "rno"], how="inner")


def race_return(row):
    weight = BET_WEIGHTS.get(row["combination"])
    if weight is None:
        return 0
    return row["payout"] * (weight / 100)


concluded["return"] = concluded.apply(race_return, axis=1)
concluded["hit"] = concluded["combination"].isin(BET_WEIGHTS).astype(int)

daily = concluded.groupby("race_date").agg(n=("hit", "size"), hits=("hit", "sum"), ret=("return", "sum"))
daily["stake"] = daily["n"] * TOTAL_BET_PER_RACE
daily["recovery_rate"] = daily["ret"] / daily["stake"] * 100
daily["hit_rate"] = daily["hits"] / daily["n"] * 100
daily = daily.sort_index()

print("買い目: 2連単 1-2(500円) + 1-3(400円) + 1-4(300円) + 1-5(200円) 、"
      "1レースあたり計1,400円（1-6は買わない）")
print("1日ごとの内訳\n")
for race_date, row in daily.iterrows():
    date_fmt = f"{race_date[0:4]}-{race_date[4:6]}-{race_date[6:8]}"
    print(f"{date_fmt}: {int(row['n'])}件 / 的中{int(row['hits'])}回 / "
          f"的中率{row['hit_rate']:.1f}% / 回収率{row['recovery_rate']:.1f}%")

total_n = len(concluded)
total_hits = int(concluded["hit"].sum())
total_stake = total_n * TOTAL_BET_PER_RACE
total_ret = int(concluded["return"].sum())
print(f"\n合計: {total_n}件 / 的中{total_hits}回 / 回収率{total_ret/total_stake*100:.1f}%")
print(f"対象日数: {len(daily)}日")

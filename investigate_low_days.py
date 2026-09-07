"""
2026-07-07・2026-07-08の回収率が低かった理由を調べるため、①②条件に合致した
対象レースのレース単位の明細（実際の決着・払戻・このレースでの損益）を出力する。

daily_breakdown_weighted_no16.pyと同一の判定ロジック・買い目
（1-2:500円/1-3:400円/1-4:300円/1-5:200円、1-6は買わない）。
"""
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "boatrace.db"

SAMPLE_SIZE_WARNING_THRESHOLD = 5
INN_NIGE_RATE_THRESHOLD = 0.8
NIGASHI_RATE_THRESHOLD = 0.5

BET_WEIGHTS = {"1-2": 500, "1-3": 400, "1-4": 300, "1-5": 200}
TOTAL_BET_PER_RACE = sum(BET_WEIGHTS.values())

TARGET_DATES = ["20260707", "20260708"]


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
races_kimarite = pd.read_sql_query("SELECT race_date, jcd, rno, kimarite FROM races", conn)

c1_stats, c2_stats, waku1_rank = compute_racer_rate_stats(entries_all, results_all)
candidates = find_qualifying_races(entries_all, c1_stats, c2_stats)
concluded = candidates.merge(payouts_2tan, on=["race_date", "jcd", "rno"], how="inner")
concluded = concluded.merge(waku1_rank, on=["race_date", "jcd", "rno"], how="inner")
concluded = concluded.merge(races_kimarite, on=["race_date", "jcd", "rno"], how="left")


def race_return(row):
    weight = BET_WEIGHTS.get(row["combination"])
    if weight is None:
        return 0
    return row["payout"] * (weight / 100)


concluded["return"] = concluded.apply(race_return, axis=1)
concluded["profit"] = concluded["return"] - TOTAL_BET_PER_RACE

for d in TARGET_DATES:
    day_df = concluded[concluded["race_date"] == d].sort_values(["jcd", "rno"])
    n = len(day_df)
    hits = int(day_df["combination"].isin(BET_WEIGHTS).sum())
    stake = n * TOTAL_BET_PER_RACE
    ret = int(day_df["return"].sum())
    rate = ret / stake * 100 if stake > 0 else 0.0

    print(f"===== {d[0:4]}-{d[4:6]}-{d[6:8]} (対象{n}件 / 的中{hits}回 / 回収率{rate:.1f}%) =====")
    for _, r in day_df.iterrows():
        hit_mark = "○的中" if r["combination"] in BET_WEIGHTS else "×"
        waku1_note = "(1号艇1着)" if r["waku1_rank"] == "1" else f"(1号艇{r['waku1_rank']}着)"
        print(f"  {r['venue_name']}(jcd{r['jcd']}) R{r['rno']}: "
              f"結果={r['combination']} 決まり手={r['kimarite']} {waku1_note} "
              f"{hit_mark} 払戻={int(r['payout']):,}円 損益={int(r['profit']):+,}円")
    print(f"  --> 合計: 賭け金{stake:,}円 / 払戻{ret:,}円 / 損益{ret - stake:+,}円\n")

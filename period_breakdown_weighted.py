"""
①②条件(1号艇イン逃げ率80%以上・2号艇逃し率50%以上)に合致した過去レースのみを
対象に、2連単「1-2」に200円、「1-3」「1-4」にそれぞれ100円（1レースあたり合計
400円）を賭け続けた場合について、対象データの最古日から5日ごとの期間に区切り、
期間ごとの対象レース数・的中回数・回収率を表にする。

判定ロジック・重み付けはblock_bootstrap_weighted.pyと同一内容を複製。
"""
import sqlite3
from pathlib import Path
from datetime import timedelta

import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "boatrace.db"
PERIOD_DAYS = 5

SAMPLE_SIZE_WARNING_THRESHOLD = 5
INN_NIGE_RATE_THRESHOLD = 0.8
NIGASHI_RATE_THRESHOLD = 0.5

BET_WEIGHTS = {"1-2": 200, "1-3": 100, "1-4": 100}
TOTAL_BET_PER_RACE = sum(BET_WEIGHTS.values())  # 400円


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

concluded["date_dt"] = pd.to_datetime(concluded["race_date"], format="%Y%m%d")
min_date = concluded["date_dt"].min()
max_date = concluded["date_dt"].max()

concluded["days_from_start"] = (concluded["date_dt"] - min_date).dt.days
concluded["period_idx"] = concluded["days_from_start"] // PERIOD_DAYS

rows = []
for period_idx, g in concluded.groupby("period_idx"):
    period_start = min_date + timedelta(days=int(period_idx) * PERIOD_DAYS)
    period_end = period_start + timedelta(days=PERIOD_DAYS - 1)
    n = len(g)
    hits = int(g["hit"].sum())
    stake = n * TOTAL_BET_PER_RACE
    ret = int(g["return"].sum())
    rate = ret / stake * 100 if stake > 0 else 0.0
    hit_rate = hits / n * 100 if n > 0 else 0.0
    rows.append({
        "期間": f"{period_start.strftime('%Y-%m-%d')}〜{min(period_end, max_date).strftime('%Y-%m-%d')}",
        "対象レース数": n,
        "的中回数": hits,
        "的中率": f"{hit_rate:.1f}%",
        "回収率": f"{rate:.1f}%",
    })

result_df = pd.DataFrame(rows)

print("買い目: 2連単 1-2(200円) + 1-3(100円) + 1-4(100円) 、1レースあたり計400円")
print(f"対象データの最古日: {min_date.strftime('%Y-%m-%d')} / 最新日: {max_date.strftime('%Y-%m-%d')}")
print(f"{PERIOD_DAYS}日ごとの期間区切り\n")
print(result_df.to_string(index=False))

total_n = len(concluded)
total_hits = int(concluded["hit"].sum())
total_stake = total_n * TOTAL_BET_PER_RACE
total_ret = int(concluded["return"].sum())
print(f"\n合計: {total_n}件 / 的中{total_hits}回 / 回収率{total_ret/total_stake*100:.1f}%")

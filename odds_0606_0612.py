"""
①②条件合致レースのうち、2連単「1-2」「1-3」がそれぞれ的中した場合の
払戻オッズ(倍率 = payout/100)について、2026-06-06〜06-12の期間と
それ以外の期間(通常期間)とで平均値・中央値を比較する。

判定ロジックはblock_bootstrap_weighted_12_13.pyと同一。
"""
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "boatrace.db"

SAMPLE_SIZE_WARNING_THRESHOLD = 10
INN_NIGE_RATE_THRESHOLD = 0.8
NIGASHI_RATE_THRESHOLD = 0.5

PERIOD_START = "20260606"
PERIOD_END = "20260612"


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
payouts_2tan = pd.read_sql_query(
    "SELECT race_date, jcd, rno, combination, payout FROM payouts WHERE bet_type = '2連単'", conn
)

c1_stats, c2_stats = compute_racer_rate_stats(entries_all, results_all)
candidates = find_qualifying_races(entries_all, c1_stats, c2_stats)
concluded = candidates.merge(payouts_2tan, on=["race_date", "jcd", "rno"], how="inner")
concluded["odds"] = concluded["payout"] / 100

period_mask = (concluded["race_date"] >= PERIOD_START) & (concluded["race_date"] <= PERIOD_END)

out = []
for combo in ["1-2", "1-3"]:
    sub = concluded[concluded["combination"] == combo]
    period_odds = sub[period_mask]["odds"]
    normal_odds = sub[~period_mask]["odds"]
    out.append(f"■ 2連単「{combo}」的中時オッズ(倍率)")
    out.append(f"  {PERIOD_START[:4]}-{PERIOD_START[4:6]}-{PERIOD_START[6:8]}〜"
               f"{PERIOD_END[:4]}-{PERIOD_END[4:6]}-{PERIOD_END[6:8]}: "
               f"的中{len(period_odds)}回 / 平均{period_odds.mean():.2f}倍 / 中央値{period_odds.median():.2f}倍")
    out.append(f"  通常期間: 的中{len(normal_odds)}回 / 平均{normal_odds.mean():.2f}倍 / "
               f"中央値{normal_odds.median():.2f}倍")
    out.append("")

text = "\n".join(out)
print(text)
Path(__file__).parent.joinpath("odds_0606_0612_output.txt").write_text(text, encoding="utf-8")

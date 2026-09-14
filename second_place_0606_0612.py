"""
①②条件(1号艇イン逃げ率80%以上・2号艇逃し率50%以上)に合致し、かつ1号艇が
1着だったレースについて、2着が何号艇だったかの内訳を、2026-06-06〜06-12の
期間とそれ以外の期間(通常期間)とで比較する。

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
entries_all = pd.read_sql_query(
    "SELECT race_date, jcd, rno, waku, toban FROM entries", conn
)
results_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, rank FROM results", conn)

c1_stats, c2_stats = compute_racer_rate_stats(entries_all, results_all)
candidates = find_qualifying_races(entries_all, c1_stats, c2_stats)

waku1_rank = results_all[results_all["waku"] == 1][["race_date", "jcd", "rno", "rank"]].rename(
    columns={"rank": "waku1_rank"}
)
qualified = candidates.merge(waku1_rank, on=["race_date", "jcd", "rno"], how="inner")

# 1号艇が1着だったレースのみ
waku1_win = qualified[qualified["waku1_rank"] == "1"].copy()

# 2着の艇番を付与
rank2 = results_all[results_all["rank"] == "2"][["race_date", "jcd", "rno", "waku"]].rename(
    columns={"waku": "second_waku"}
)
waku1_win = waku1_win.merge(rank2, on=["race_date", "jcd", "rno"], how="left")

period_mask = (waku1_win["race_date"] >= PERIOD_START) & (waku1_win["race_date"] <= PERIOD_END)
period_df = waku1_win[period_mask]
normal_df = waku1_win[~period_mask]


def breakdown(df, label):
    total = len(df)
    print(f"■ {label} (1号艇1着レース {total}件)")
    counts = df["second_waku"].value_counts().sort_index()
    for waku in range(2, 7):
        n = int(counts.get(waku, 0))
        print(f"  2着={waku}号艇: {n:>4}件 ({n/total*100:5.1f}%)")
    n456 = int(counts.get(4, 0) + counts.get(5, 0) + counts.get(6, 0))
    print(f"  (4・5・6号艇合計: {n456}件 / {n456/total*100:5.1f}%)")
    print()


breakdown(period_df, f"{PERIOD_START[:4]}-{PERIOD_START[4:6]}-{PERIOD_START[6:8]}〜"
                       f"{PERIOD_END[:4]}-{PERIOD_END[4:6]}-{PERIOD_END[6:8]}")
breakdown(normal_df, "それ以外の期間(通常期間)")

"""
①②条件(1号艇イン逃げ率80%以上・2号艇逃し率50%以上)に合致した過去レースのうち、
実際に1号艇が1着だったレースだけを対象に、2〜6号艇それぞれの着順(2着〜6着・
失格等)の分布を件数・割合で比較する。

rank_distribution_when_escaped.pyと同一の判定ロジックに、5号艇・6号艇を追加。
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

concluded = candidates.merge(payouts_2tan, on=["race_date", "jcd", "rno"], how="inner")
concluded = concluded.merge(waku1_rank, on=["race_date", "jcd", "rno"], how="inner")

escaped = concluded[concluded["waku1_rank"] == "1"][["race_date", "jcd", "rno"]].drop_duplicates()
total = len(escaped)

print(f"①②条件に合致し、かつ1号艇が実際に1着だったレース: {total}件\n")

TARGET_RANKS = ["2", "3", "4", "5", "6"]
WAKUS = [2, 3, 4, 5, 6]

waku_counts = {}
for waku in WAKUS:
    waku_results = results_all[results_all["waku"] == waku].merge(
        escaped, on=["race_date", "jcd", "rno"], how="inner"
    )
    waku_counts[waku] = waku_results["rank"].value_counts()

col_width = 16
header = f"{'着順':<6}" + "".join(f"{str(w) + '号艇':>{col_width}}" for w in WAKUS)
print(header)
for r in TARGET_RANKS:
    row = f"{r + '着':<6}"
    for waku in WAKUS:
        n = int(waku_counts[waku].get(r, 0))
        pct = n / total * 100 if total > 0 else 0.0
        row += f"{f'{n}件({pct:.1f}%)':>{col_width}}"
    print(row)

row = f"{'失格等':<6}"
for waku in WAKUS:
    known = sum(int(waku_counts[waku].get(r, 0)) for r in TARGET_RANKS)
    other = total - known
    row += f"{f'{other}件({other / total * 100:.1f}%)':>{col_width}}"
print(row)

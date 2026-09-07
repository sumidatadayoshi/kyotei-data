"""
①②条件(1号艇イン逃げ率80%以上・2号艇逃し率50%以上)に合致した過去レースのうち、
実際に1号艇が1着に来なかったレースについて、決まり手(kimarite)の内訳を
件数・割合の多い順に集計する。あわせて「まくり」を決めた艇番の内訳も見る。

判定ロジックはrank_when_not_escaped.py等と同一内容を複製。
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
races_kimarite = pd.read_sql_query("SELECT race_date, jcd, rno, kimarite FROM races", conn)

c1_stats, c2_stats, waku1_rank = compute_racer_rate_stats(entries_all, results_all)
candidates = find_qualifying_races(entries_all, c1_stats, c2_stats)

concluded = candidates.merge(payouts_2tan, on=["race_date", "jcd", "rno"], how="inner")
concluded = concluded.merge(waku1_rank, on=["race_date", "jcd", "rno"], how="inner")

not_escaped = concluded[concluded["waku1_rank"] != "1"].copy()
not_escaped = not_escaped.merge(races_kimarite, on=["race_date", "jcd", "rno"], how="left")

total = len(not_escaped)

print(f"①②条件に合致し、かつ1号艇が1着に来なかったレース: {total}件\n")

kimarite_counts = not_escaped["kimarite"].value_counts(dropna=False)
print("決まり手の内訳（多い順）")
for kimarite, cnt in kimarite_counts.items():
    label = kimarite if pd.notna(kimarite) else "(不明/未取得)"
    print(f"  {label}: {cnt}件 ({cnt / total * 100:.1f}%)")

# --- 「まくり」を決めた艇番の内訳 ---
makuri_races = not_escaped[not_escaped["kimarite"] == "まくり"]
makuri_count = len(makuri_races)

print(f"\n『まくり』が占める割合: {makuri_count}件 / {total}件 = {makuri_count / total * 100:.1f}%")

if makuri_count > 0:
    # まくりを決めた艇＝そのレースで1着だった艇番
    winners = results_all.merge(
        makuri_races[["race_date", "jcd", "rno"]], on=["race_date", "jcd", "rno"], how="inner"
    )
    winners = winners[winners["rank"] == "1"]
    waku_counts = winners["waku"].value_counts().sort_index()

    print("\n『まくり』を決めた艇番の内訳")
    for waku, cnt in waku_counts.items():
        print(f"  {int(waku)}号艇: {cnt}件 ({cnt / makuri_count * 100:.1f}%)")

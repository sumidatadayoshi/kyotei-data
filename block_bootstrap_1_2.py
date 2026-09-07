"""
dashboard.pyの①②条件(1号艇イン逃げ率80%以上・2号艇逃し率50%以上)に合致した
過去レースのうち、2連単「1-2」を100円ずつ買い続けた場合の回収率について、
開催日×競艇場(jcd)単位のブロックブートストラップで95%信頼区間を計算する。

dashboard.pyをそのままimportするとStreamlit UIコードが全部実行されてしまうため、
判定ロジック(compute_racer_rate_stats / find_qualifying_races)の中身だけを
dashboard.pyの定義と同一の内容で複製している(定数もdashboard.pyと同値)。
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "boatrace.db"
BET_AMOUNT = 100
FIXED_COMBO = "1-2"
N_RESAMPLES = 2000
SEED = 42

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

c1_stats, c2_stats, _ = compute_racer_rate_stats(entries_all, results_all)
candidates = find_qualifying_races(entries_all, c1_stats, c2_stats)

concluded = candidates.merge(payouts_2tan, on=["race_date", "jcd", "rno"], how="inner")

total_races = len(concluded)
hits = concluded[concluded["combination"] == FIXED_COMBO]
hit_count = len(hits)
total_return = int(hits["payout"].sum())
total_stake = total_races * BET_AMOUNT
recovery_rate = (total_return / total_stake * 100) if total_stake > 0 else 0.0

print(f"対象レース数: {total_races}件")
print(f"的中回数: {hit_count}回")
print(f"回収率: {recovery_rate:.1f}%")
print()

# --- 開催日×競艇場 単位のブロックブートストラップ ---
race_level_hit = (concluded["combination"] == FIXED_COMBO).astype(int).to_numpy()
race_level_payout = np.where(race_level_hit == 1, concluded["payout"].to_numpy(), 0)
block_key = concluded["race_date"].astype(str) + "_" + concluded["jcd"].astype(str)

df = pd.DataFrame({"block": block_key, "hit": race_level_hit, "payout": race_level_payout})
blocks = df.groupby("block").agg(n=("payout", "size"), payout_sum=("payout", "sum"))
n_blocks = len(blocks)

n_arr = blocks["n"].to_numpy()
payout_arr = blocks["payout_sum"].to_numpy()

rng = np.random.default_rng(SEED)
idx = rng.integers(0, n_blocks, size=(N_RESAMPLES, n_blocks))
resample_stake = n_arr[idx].sum(axis=1) * BET_AMOUNT
resample_return = payout_arr[idx].sum(axis=1)
rates = np.where(resample_stake > 0, resample_return / resample_stake * 100, 0.0)

lower, upper = np.percentile(rates, [2.5, 97.5])

print(f"ブロック数(開催日×競艇場): {n_blocks}")
print(f"リサンプル回数: {N_RESAMPLES}, シード: {SEED}")
print(f"95%信頼区間: {lower:.1f}% 〜 {upper:.1f}%")
print(f"ブートストラップ分布の中央値: {np.median(rates):.1f}%")

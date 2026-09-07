"""
①②条件(1号艇イン逃げ率80%以上・2号艇逃し率50%以上)に合致した過去レースのみを
対象に、2連単「1-2」に200円、「1-3」に100円（1レースあたり合計300円、1-4は
買わない）を賭け続けた場合の回収率と、開催日×競艇場(jcd)単位のブロック
ブートストラップによる95%信頼区間を計算する。

判定ロジックはblock_bootstrap_1_2.py等と同一内容を複製。
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "boatrace.db"
N_RESAMPLES = 2000
SEED = 42

SAMPLE_SIZE_WARNING_THRESHOLD = 5
INN_NIGE_RATE_THRESHOLD = 0.8
NIGASHI_RATE_THRESHOLD = 0.5

# 買い目ごとの賭け金（円）。1-4は買わない。
BET_WEIGHTS = {"1-2": 200, "1-3": 100}
TOTAL_BET_PER_RACE = sum(BET_WEIGHTS.values())  # 300円


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

# ①②の条件に合致し、かつ結果（2連単payout）が判明しているレースのみが対象
concluded = candidates.merge(payouts_2tan, on=["race_date", "jcd", "rno"], how="inner")
total_races = len(concluded)


def race_return(row):
    """結果が1-2または1-3ならその点の払戻を、それ以外は0円を返す。
    payoutテーブルの値は100円賭けた場合の払戻金なので、賭け金の倍率をかける。"""
    weight = BET_WEIGHTS.get(row["combination"])
    if weight is None:
        return 0
    return row["payout"] * (weight / 100)


concluded["return"] = concluded.apply(race_return, axis=1)
concluded["hit"] = concluded["combination"].isin(BET_WEIGHTS).astype(int)

hit_count = int(concluded["hit"].sum())
total_return = int(concluded["return"].sum())
total_stake = total_races * TOTAL_BET_PER_RACE
recovery_rate = (total_return / total_stake * 100) if total_stake > 0 else 0.0
hit_rate = (hit_count / total_races * 100) if total_races > 0 else 0.0

print("買い目: 2連単 1-2(200円) + 1-3(100円) 、1レースあたり計300円（1-4は買わない）")
print(f"対象レース数（①②条件に合致し結果判明済み）: {total_races}件")
print(f"的中回数（1-2または1-3）: {hit_count}回")
print(f"的中率: {hit_rate:.1f}%")
print(f"回収率: {recovery_rate:.1f}%")
print(f"賭け金合計: {total_stake:,}円 / 払戻金合計: {total_return:,}円 / "
      f"通算損益: {total_return - total_stake:,}円")

for combo, w in BET_WEIGHTS.items():
    sub = concluded[concluded["combination"] == combo]
    n_hit = len(sub)
    ret = int(sub["payout"].sum() * (w / 100))
    print(f"  内訳 {combo}（賭け金{w}円）: 的中{n_hit}回、払戻合計{ret:,}円")

print()

# --- 開催日×競艇場 単位のブロックブートストラップ ---
# レースごとに賭け金合計(300円固定)・払戻金合計(上のreturn列)を求めたうえで、
# race_date × jcd のブロック単位に集計してからリサンプリングする。
block_key = concluded["race_date"].astype(str) + "_" + concluded["jcd"].astype(str)
df = pd.DataFrame({
    "block": block_key,
    "return": concluded["return"].to_numpy(),
})
blocks = df.groupby("block").agg(n=("return", "size"), return_sum=("return", "sum"))
n_blocks = len(blocks)

n_arr = blocks["n"].to_numpy()
return_arr = blocks["return_sum"].to_numpy()

rng = np.random.default_rng(SEED)
idx = rng.integers(0, n_blocks, size=(N_RESAMPLES, n_blocks))
resample_stake = n_arr[idx].sum(axis=1) * TOTAL_BET_PER_RACE
resample_return = return_arr[idx].sum(axis=1)
rates = np.where(resample_stake > 0, resample_return / resample_stake * 100, 0.0)

lower, upper = np.percentile(rates, [2.5, 97.5])

print(f"ブロック数（開催日×競艇場）: {n_blocks}")
print(f"リサンプル回数: {N_RESAMPLES}, シード: {SEED}")
print(f"95%信頼区間: {lower:.1f}% 〜 {upper:.1f}%")
print(f"ブートストラップ分布の中央値: {np.median(rates):.1f}%")

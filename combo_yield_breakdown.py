"""
新しい買い方を探索するための調査スクリプト(その1)。
①②条件(1号艇イン逃げ率80%以上・2号艇逃し率50%以上)を満たすレースの中で:
  A) 実際の決着(2連単)の分布・組み合わせ別の平均払戻
  B) 単一組み合わせ(1-2単独 / 1-3単独 / 1-4単独)の回収率とブロックブートストラップCI
  C) 現行の1-2:200円+1-3:100円との比較
を計算する。
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
        c1_stats[(c1_stats["rate"] >= INN_NIGE_RATE_THRESHOLD) & (c1_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index
    )
    qualified_toban2 = set(
        c2_stats[(c2_stats["rate"] >= NIGASHI_RATE_THRESHOLD) & (c2_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index
    )
    entries1 = entries_df[entries_df["waku"] == 1][["race_date", "jcd", "rno", "toban", "racer_name", "venue_name"]].rename(
        columns={"toban": "toban1", "racer_name": "racer1_name"})
    entries2 = entries_df[entries_df["waku"] == 2][["race_date", "jcd", "rno", "toban", "racer_name"]].rename(
        columns={"toban": "toban2", "racer_name": "racer2_name"})
    race_pairs = entries1.merge(entries2, on=["race_date", "jcd", "rno"], how="inner")
    return race_pairs[race_pairs["toban1"].isin(qualified_toban1) & race_pairs["toban2"].isin(qualified_toban2)].copy()


def block_bootstrap_recovery(concluded, weight_fn, total_bet_per_race, n_resamples=N_RESAMPLES, seed=SEED):
    ret = concluded.apply(weight_fn, axis=1)
    block_key = concluded["race_date"].astype(str) + "_" + concluded["jcd"].astype(str)
    df = pd.DataFrame({"block": block_key, "return": ret.to_numpy()})
    blocks = df.groupby("block").agg(n=("return", "size"), return_sum=("return", "sum"))
    n_arr = blocks["n"].to_numpy()
    return_arr = blocks["return_sum"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(blocks), size=(n_resamples, len(blocks)))
    resample_stake = n_arr[idx].sum(axis=1) * total_bet_per_race
    resample_return = return_arr[idx].sum(axis=1)
    rates = np.where(resample_stake > 0, resample_return / resample_stake * 100, 0.0)
    lower, upper = np.percentile(rates, [2.5, 97.5])
    total_stake = len(concluded) * total_bet_per_race
    total_return = ret.sum()
    point = total_return / total_stake * 100 if total_stake > 0 else 0.0
    return point, lower, upper, len(blocks)


conn = sqlite3.connect(DB_PATH)
entries_all = pd.read_sql_query("SELECT * FROM entries", conn)
results_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, rank FROM results", conn)
payouts_2tan = pd.read_sql_query("SELECT race_date, jcd, rno, combination, payout FROM payouts WHERE bet_type = '2連単'", conn)

c1_stats, c2_stats, waku1_rank = compute_racer_rate_stats(entries_all, results_all)
candidates = find_qualifying_races(entries_all, c1_stats, c2_stats)
concluded = candidates.merge(payouts_2tan, on=["race_date", "jcd", "rno"], how="inner")
concluded = concluded.sort_values(["race_date", "rno", "jcd"]).reset_index(drop=True)
total_races = len(concluded)
print(f"対象レース数(①②条件×結果判明済み): {total_races}件\n")

print("=== A) 決着(2連単)の分布・組み合わせ別 平均払戻/回収寄与 ===")
dist = concluded["combination"].value_counts()
for combo, n in dist.head(15).items():
    sub = concluded[concluded["combination"] == combo]
    avg_payout = sub["payout"].mean()
    yield_per_100 = sub["payout"].sum() / total_races  # 100円均等仮想時の1レースあたり期待払戻
    print(f"  {combo}: {n}件({n/total_races*100:.1f}%) 平均払戻{avg_payout:.0f}円 "
          f"/ 100円均等換算の期待払戻{yield_per_100:.1f}円(回収率{yield_per_100:.1f}%)")

print()
print("=== B) 単一組み合わせの回収率(100円均等)とブロックブートストラップ95%CI ===")
for combo in ["1-2", "1-3", "1-4", "1-5", "1-6"]:
    def wfn(row, combo=combo):
        return row["payout"] if row["combination"] == combo else 0
    point, lo, hi, nblk = block_bootstrap_recovery(concluded, wfn, 100)
    print(f"  {combo}単独(100円均等): 回収率{point:.1f}% [95%CI {lo:.1f}%〜{hi:.1f}%] (blocks={nblk})")

print()
print("=== C) 現行 1-2:200円+1-3:100円 (再掲) ===")
def wfn_current(row):
    if row["combination"] == "1-2":
        return row["payout"] * 2
    if row["combination"] == "1-3":
        return row["payout"] * 1
    return 0
point, lo, hi, nblk = block_bootstrap_recovery(concluded, wfn_current, 300)
print(f"  回収率{point:.1f}% [95%CI {lo:.1f}%〜{hi:.1f}%] (blocks={nblk})")

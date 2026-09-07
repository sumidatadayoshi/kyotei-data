"""
①②条件(1号艇イン逃げ80%以上・2号艇逃し率50%以上)を満たすレースを対象に、
2連単以外の券種(単勝・複勝・拡連複)の実際の回収率を検証する。
判定ロジックは他のスクリプトと同一。
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
        on=["race_date", "jcd", "rno"], how="inner")
    c1_stats = c1.groupby("toban").agg(starts=("rank", "size"), wins=("rank", lambda s: (s == "1").sum()))
    c1_stats["rate"] = c1_stats["wins"] / c1_stats["starts"]
    waku1_rank = results_all[results_all["waku"] == 1][["race_date", "jcd", "rno", "rank"]].rename(columns={"rank": "waku1_rank"})
    c2 = entries_all[entries_all["waku"] == 2].merge(waku1_rank, on=["race_date", "jcd", "rno"], how="inner")
    c2_stats = c2.groupby("toban").agg(starts=("waku1_rank", "size"), nigasare=("waku1_rank", lambda s: (s == "1").sum()))
    c2_stats["rate"] = c2_stats["nigasare"] / c2_stats["starts"]
    return c1_stats, c2_stats


def find_qualifying_races(entries_df, c1_stats, c2_stats):
    q1 = set(c1_stats[(c1_stats["rate"] >= INN_NIGE_RATE_THRESHOLD) & (c1_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)
    q2 = set(c2_stats[(c2_stats["rate"] >= NIGASHI_RATE_THRESHOLD) & (c2_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)
    e1 = entries_df[entries_df["waku"] == 1][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban1"})
    e2 = entries_df[entries_df["waku"] == 2][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban2"})
    pairs = e1.merge(e2, on=["race_date", "jcd", "rno"], how="inner")
    return pairs[pairs["toban1"].isin(q1) & pairs["toban2"].isin(q2)].copy()


def block_bootstrap_recovery(df, ret_series, stake_per_race, n_resamples=N_RESAMPLES, seed=SEED):
    block_key = df["race_date"].astype(str) + "_" + df["jcd"].astype(str)
    d = pd.DataFrame({"block": block_key, "return": np.asarray(ret_series)})
    blocks = d.groupby("block").agg(n=("return", "size"), return_sum=("return", "sum"))
    n_arr = blocks["n"].to_numpy()
    ret_arr = blocks["return_sum"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(blocks), size=(n_resamples, len(blocks)))
    resample_stake = n_arr[idx].sum(axis=1) * stake_per_race
    resample_return = ret_arr[idx].sum(axis=1)
    rates = np.where(resample_stake > 0, resample_return / resample_stake * 100, 0.0)
    lo, hi = np.percentile(rates, [2.5, 97.5])
    point = ret_arr.sum() / (n_arr.sum() * stake_per_race) * 100
    return point, lo, hi


conn = sqlite3.connect(DB_PATH)
entries_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, toban FROM entries", conn)
results_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, rank FROM results", conn)

c1_stats, c2_stats = compute_racer_rate_stats(entries_all, results_all)
candidates = find_qualifying_races(entries_all, c1_stats, c2_stats)
print(f"①②条件に合致するレース数(現時点のDB全量): {len(candidates)}件\n")

bet_defs = [
    ("単勝1(1号艇の単勝)", "単勝", "1", 100),
    ("複勝1(1号艇の複勝)", "複勝", "1", 100),
    ("拡連複1=2(1号艇and2号艇が3着以内)", "拡連複", "1=2", 100),
    ("拡連複1=3(1号艇and3号艇が3着以内)", "拡連複", "1=3", 100),
]

results_summary = []
for label, bet_type, combo, stake in bet_defs:
    payouts = pd.read_sql_query(
        f"SELECT race_date, jcd, rno, combination, payout FROM payouts WHERE bet_type = ?", conn, params=(bet_type,)
    )
    merged = candidates.merge(payouts[payouts["combination"] == combo],
                               on=["race_date", "jcd", "rno"], how="left")
    # 結果が判明していない(まだ確定していない)レースは除外。判明済みかどうかは、
    # そのbet_typeの全payout行が存在するレース数で判定する必要があるため、
    # 「そのレースのpayoutsに1行でも存在するか」で判定する。
    all_payouts_for_races = candidates.merge(payouts, on=["race_date", "jcd", "rno"], how="inner")
    concluded_keys = all_payouts_for_races[["race_date", "jcd", "rno"]].drop_duplicates()
    merged = concluded_keys.merge(merged, on=["race_date", "jcd", "rno"], how="left")
    merged["payout"] = merged["payout"].fillna(0)
    total = len(merged)
    hit = int((merged["payout"] > 0).sum())
    point, lo, hi = block_bootstrap_recovery(merged, merged["payout"], stake)
    print(f"{label}: n={total} 的中{hit}回({hit/total*100:.1f}%) 回収率{point:.1f}% [95%CI {lo:.1f}-{hi:.1f}]")
    results_summary.append((label, total, hit, point, lo, hi))

print()
print("=== 参考: 現行の2連単ミックス(1-2:200+1-3:100) ===")
payouts_2tan = pd.read_sql_query("SELECT race_date, jcd, rno, combination, payout FROM payouts WHERE bet_type = '2連単'", conn)
conc = candidates.merge(payouts_2tan, on=["race_date", "jcd", "rno"], how="inner")
ret_mix = np.where(conc["combination"] == "1-2", conc["payout"] * 2, np.where(conc["combination"] == "1-3", conc["payout"], 0))
p, lo, hi = block_bootstrap_recovery(conc, ret_mix, 300)
print(f"n={len(conc)} 回収率{p:.1f}% [95%CI {lo:.1f}-{hi:.1f}]")

"""
探索その3: 2号艇motor_2rateフィルタの頑健性検証。
  I) 1-2単独/1-3単独をmotor_2rate別に分解(メカニズムの確認)
  J) motor_2rate<40フィルタを前半/後半(時系列2分割)で別々に検証(過学習チェック)
  K) motor_2rate<40 かつ 現行ミックス を、フィルタ無しの母集団と比較(実務上のサイズ感)
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


def block_bootstrap_recovery(concluded, ret_series, total_bet_per_race, n_resamples=N_RESAMPLES, seed=SEED):
    block_key = concluded["race_date"].astype(str) + "_" + concluded["jcd"].astype(str)
    df = pd.DataFrame({"block": block_key, "return": np.asarray(ret_series)})
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
    total_return = np.asarray(ret_series).sum()
    point = total_return / total_stake * 100 if total_stake > 0 else 0.0
    return point, lower, upper, len(blocks)


conn = sqlite3.connect(DB_PATH)
entries_all = pd.read_sql_query("SELECT * FROM entries", conn)
results_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, rank FROM results", conn)
payouts_2tan = pd.read_sql_query("SELECT race_date, jcd, rno, combination, payout FROM payouts WHERE bet_type = '2連単'", conn)

c1_stats, c2_stats, waku1_rank = compute_racer_rate_stats(entries_all, results_all)
candidates = find_qualifying_races(entries_all, c1_stats, c2_stats)
concluded = candidates.merge(payouts_2tan, on=["race_date", "jcd", "rno"], how="inner")
waku2_motor = entries_all[entries_all["waku"] == 2][["race_date", "jcd", "rno", "motor_2rate"]].rename(columns={"motor_2rate": "waku2_motor_2rate"})
concluded = concluded.merge(waku2_motor, on=["race_date", "jcd", "rno"], how="left")
concluded = concluded.sort_values(["race_date", "rno", "jcd"]).reset_index(drop=True)

concluded["ret_mix"] = np.where(concluded["combination"] == "1-2", concluded["payout"] * 2,
                         np.where(concluded["combination"] == "1-3", concluded["payout"] * 1, 0))
concluded["ret_12"] = np.where(concluded["combination"] == "1-2", concluded["payout"], 0)
concluded["ret_13"] = np.where(concluded["combination"] == "1-3", concluded["payout"], 0)

print(f"対象レース数: {len(concluded)}件\n")

print("=== I) motor_2rate帯別: 1-2単独 / 1-3単独 の回収率 ===")
bins = [(-1, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 100)]
for lo_b, hi_b in bins:
    sub = concluded[(concluded["waku2_motor_2rate"] > lo_b) & (concluded["waku2_motor_2rate"] <= hi_b)]
    if len(sub) < 15:
        print(f"  motor_2rate({lo_b},{hi_b}]: n={len(sub)} (少なすぎるためスキップ)")
        continue
    p12, l12, h12, _ = block_bootstrap_recovery(sub, sub["ret_12"], 100)
    p13, l13, h13, _ = block_bootstrap_recovery(sub, sub["ret_13"], 100)
    hit12 = (sub["combination"] == "1-2").mean() * 100
    hit13 = (sub["combination"] == "1-3").mean() * 100
    print(f"  motor_2rate({lo_b},{hi_b}]: n={len(sub)} | 1-2的中率{hit12:.1f}% 回収率{p12:.1f}%[{l12:.1f}-{h12:.1f}] "
          f"| 1-3的中率{hit13:.1f}% 回収率{p13:.1f}%[{l13:.1f}-{h13:.1f}]")

print()
print("=== J) motor_2rate<40フィルタの前半/後半 時系列検証(過学習チェック) ===")
half = len(concluded) // 2
first = concluded.iloc[:half]
second = concluded.iloc[half:]
for label, df in [("前半", first), ("後半", second)]:
    base_p, base_l, base_h, _ = block_bootstrap_recovery(df, df["ret_mix"], 300)
    filt = df[df["waku2_motor_2rate"] < 40]
    filt_p, filt_l, filt_h, _ = block_bootstrap_recovery(filt, filt["ret_mix"], 300)
    print(f"  [{label}] 全体n={len(df)} 回収率{base_p:.1f}%[{base_l:.1f}-{base_h:.1f}] -> "
          f"motor<40適用後 n={len(filt)} 回収率{filt_p:.1f}%[{filt_l:.1f}-{filt_h:.1f}]")

print()
print("=== K) motor_2rate<40フィルタ 全期間まとめ ===")
filt_all = concluded[concluded["waku2_motor_2rate"] < 40]
p, l, h, nblk = block_bootstrap_recovery(filt_all, filt_all["ret_mix"], 300)
print(f"  n={len(filt_all)}件 ({len(filt_all)/len(concluded)*100:.1f}%が対象として残る)")
print(f"  回収率{p:.1f}% [95%CI {l:.1f}%〜{h:.1f}%] (blocks={nblk})")
print(f"  (フィルタ無し全体: n={len(concluded)}, 回収率113.0% [105.1%-120.9%])")

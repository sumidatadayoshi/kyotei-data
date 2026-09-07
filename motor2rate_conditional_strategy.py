"""
探索その4: 条件付き買い方(2号艇motor_2rateが高い時は1-3を見送る)の検証。
  L) 「1-2は常に200円、1-3はmotor_2rate<40の時だけ100円」という条件付き戦略
  M) 上記を前半/後半で別々に検証(過学習チェック)
  N) 参考: 閾値を35/45に変えた場合の感度
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


def block_bootstrap_recovery_varstake(concluded, ret_series, stake_series, n_resamples=N_RESAMPLES, seed=SEED):
    """レースごとに賭け金が変わる(条件付き)場合用。"""
    block_key = concluded["race_date"].astype(str) + "_" + concluded["jcd"].astype(str)
    df = pd.DataFrame({"block": block_key, "return": np.asarray(ret_series), "stake": np.asarray(stake_series)})
    blocks = df.groupby("block").agg(return_sum=("return", "sum"), stake_sum=("stake", "sum"))
    return_arr = blocks["return_sum"].to_numpy()
    stake_arr = blocks["stake_sum"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(blocks), size=(n_resamples, len(blocks)))
    resample_stake = stake_arr[idx].sum(axis=1)
    resample_return = return_arr[idx].sum(axis=1)
    rates = np.where(resample_stake > 0, resample_return / resample_stake * 100, 0.0)
    lower, upper = np.percentile(rates, [2.5, 97.5])
    total_stake = stake_arr.sum()
    total_return = return_arr.sum()
    point = total_return / total_stake * 100 if total_stake > 0 else 0.0
    return point, lower, upper, len(blocks), total_stake, total_return


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


def conditional_strategy(df, threshold, w12=200, w13=100):
    bet_13 = (df["waku2_motor_2rate"] < threshold).astype(int) * w13
    stake = w12 + bet_13
    ret = np.where(df["combination"] == "1-2", df["payout"] * (w12 / 100), 0.0)
    ret = ret + np.where((df["combination"] == "1-3") & (bet_13 > 0), df["payout"] * (w13 / 100), 0.0)
    return ret, stake


print(f"対象レース数: {len(concluded)}件\n")

print("=== L) 条件付き戦略: 1-2は常に200円 / 1-3はmotor_2rate(2号艇)<40の時だけ100円 ===")
ret, stake = conditional_strategy(concluded, 40)
p, l, h, nblk, tot_stake, tot_ret = block_bootstrap_recovery_varstake(concluded, ret, stake)
print(f"  回収率{p:.1f}% [95%CI {l:.1f}%〜{h:.1f}%] (blocks={nblk})")
print(f"  賭け金合計{int(tot_stake):,}円 / 払戻合計{int(tot_ret):,}円 / 通算損益{int(tot_ret-tot_stake):+,}円")
print(f"  (1-3を見送ったレース数: {(concluded['waku2_motor_2rate']>=40).sum()}件 / 全{len(concluded)}件)")

print()
print("=== M) 上記条件付き戦略を前半/後半で検証(過学習チェック) ===")
half = len(concluded) // 2
for label, df in [("前半", concluded.iloc[:half]), ("後半", concluded.iloc[half:])]:
    ret_f, stake_f = conditional_strategy(df, 40)
    p_f, l_f, h_f, _, ts_f, tr_f = block_bootstrap_recovery_varstake(df, ret_f, stake_f)
    # 比較用: 常時1-2+1-3(200/100)固定
    ret_fix = np.where(df["combination"] == "1-2", df["payout"] * 2, np.where(df["combination"] == "1-3", df["payout"], 0))
    stake_fix = np.full(len(df), 300)
    p_x, l_x, h_x, _, _, _ = block_bootstrap_recovery_varstake(df, ret_fix, stake_fix)
    print(f"  [{label}] n={len(df)} 固定ミックス回収率{p_x:.1f}%[{l_x:.1f}-{h_x:.1f}] -> "
          f"条件付き戦略回収率{p_f:.1f}%[{l_f:.1f}-{h_f:.1f}]")

print()
print("=== N) 閾値感度(30/35/40/45) ===")
for th in [30, 35, 40, 45]:
    ret_t, stake_t = conditional_strategy(concluded, th)
    p_t, l_t, h_t, _, ts_t, tr_t = block_bootstrap_recovery_varstake(concluded, ret_t, stake_t)
    skip_n = (concluded['waku2_motor_2rate'] >= th).sum()
    print(f"  閾値{th}: 回収率{p_t:.1f}% [{l_t:.1f}-{h_t:.1f}] (1-3見送り{skip_n}件/{len(concluded)}件)")

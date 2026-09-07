"""
新しい買い方の探索(その2)。
①②条件に合致したレースを対象に、
  D) レースグレード別の1-2/1-3回収率
  E) 4号艇の平均ST(avg_st)による追加絞り込みの効果
  F) 2号艇のモーター2連率(motor_2rate)による追加絞り込みの効果
  G) 3連単「1-2-3」「1-3-2」ボックス(計200円)と現行2連単ミックス(300円)の比較
  H) 直前オッズが取得できているレース(部分データ)での市場評価と実際の回収率の比較
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
    total_return = ret_series.sum()
    point = total_return / total_stake * 100 if total_stake > 0 else 0.0
    return point, lower, upper, len(blocks)


conn = sqlite3.connect(DB_PATH)
entries_all = pd.read_sql_query("SELECT * FROM entries", conn)
results_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, rank FROM results", conn)
payouts_2tan = pd.read_sql_query("SELECT race_date, jcd, rno, combination, payout FROM payouts WHERE bet_type = '2連単'", conn)
payouts_3tan = pd.read_sql_query("SELECT race_date, jcd, rno, combination, payout FROM payouts WHERE bet_type = '3連単'", conn)
races_meta = pd.read_sql_query("SELECT race_date, jcd, rno, grade, weather, wind_speed FROM races", conn)
odds_2tan = pd.read_sql_query("SELECT race_date, jcd, rno, combination, odds_low FROM odds WHERE bet_type = '2連単'", conn)

c1_stats, c2_stats, waku1_rank = compute_racer_rate_stats(entries_all, results_all)
candidates = find_qualifying_races(entries_all, c1_stats, c2_stats)
concluded = candidates.merge(payouts_2tan, on=["race_date", "jcd", "rno"], how="inner")
concluded = concluded.merge(races_meta, on=["race_date", "jcd", "rno"], how="left")

# 4号艇avg_st, 2号艇motor_2rateを付与
waku4_st = entries_all[entries_all["waku"] == 4][["race_date", "jcd", "rno", "avg_st"]].rename(columns={"avg_st": "waku4_avg_st"})
waku2_motor = entries_all[entries_all["waku"] == 2][["race_date", "jcd", "rno", "motor_2rate"]].rename(columns={"motor_2rate": "waku2_motor_2rate"})
concluded = concluded.merge(waku4_st, on=["race_date", "jcd", "rno"], how="left")
concluded = concluded.merge(waku2_motor, on=["race_date", "jcd", "rno"], how="left")

concluded["ret_mix"] = np.where(concluded["combination"] == "1-2", concluded["payout"] * 2,
                         np.where(concluded["combination"] == "1-3", concluded["payout"] * 1, 0))
concluded["ret_12"] = np.where(concluded["combination"] == "1-2", concluded["payout"], 0)

total_races = len(concluded)
print(f"対象レース数: {total_races}件\n")

print("=== D) グレード別 (現行ミックス 1-2:200+1-3:100) ===")
for grade, sub in concluded.groupby("grade"):
    if len(sub) < 10:
        continue
    point, lo, hi, nblk = block_bootstrap_recovery(sub, sub["ret_mix"], 300)
    print(f"  {grade}: n={len(sub)} 回収率{point:.1f}% [{lo:.1f}%〜{hi:.1f}%]")

print()
print("=== E) 4号艇avg_stによる追加絞り込み(現行ミックス) ===")
print(f"  waku4_avg_st 欠損: {concluded['waku4_avg_st'].isna().sum()}件")
valid_st = concluded.dropna(subset=["waku4_avg_st"])
print(f"  有効データ: {len(valid_st)}件, 分布: min={valid_st['waku4_avg_st'].min():.2f} "
      f"25%={valid_st['waku4_avg_st'].quantile(.25):.2f} 中央値={valid_st['waku4_avg_st'].median():.2f} "
      f"75%={valid_st['waku4_avg_st'].quantile(.75):.2f} max={valid_st['waku4_avg_st'].max():.2f}")
for thresh in [0.13, 0.15, 0.17, 0.19, 0.21]:
    slow = valid_st[valid_st["waku4_avg_st"] >= thresh]  # STが大きい=スタートが遅い(まくりにくい)
    fast = valid_st[valid_st["waku4_avg_st"] < thresh]
    if len(slow) >= 20:
        p_s, l_s, h_s, _ = block_bootstrap_recovery(slow, slow["ret_mix"], 300)
        print(f"  4号艇avg_st>={thresh}(遅め,n={len(slow)}): 回収率{p_s:.1f}% [{l_s:.1f}%〜{h_s:.1f}%]")
    if len(fast) >= 20:
        p_f, l_f, h_f, _ = block_bootstrap_recovery(fast, fast["ret_mix"], 300)
        print(f"  4号艇avg_st<{thresh}(速め,n={len(fast)}): 回収率{p_f:.1f}% [{l_f:.1f}%〜{h_f:.1f}%]")

print()
print("=== F) 2号艇motor_2rateによる追加絞り込み(現行ミックス) ===")
print(f"  motor_2rate 欠損: {concluded['waku2_motor_2rate'].isna().sum()}件")
valid_mo = concluded.dropna(subset=["waku2_motor_2rate"])
print(f"  有効データ: {len(valid_mo)}件, 分布: min={valid_mo['waku2_motor_2rate'].min():.1f} "
      f"中央値={valid_mo['waku2_motor_2rate'].median():.1f} max={valid_mo['waku2_motor_2rate'].max():.1f}")
for thresh in [30, 35, 40, 45]:
    high = valid_mo[valid_mo["waku2_motor_2rate"] >= thresh]
    low = valid_mo[valid_mo["waku2_motor_2rate"] < thresh]
    if len(high) >= 20:
        p_h, l_h, h_h, _ = block_bootstrap_recovery(high, high["ret_mix"], 300)
        print(f"  2号艇motor_2rate>={thresh}(n={len(high)}): 回収率{p_h:.1f}% [{l_h:.1f}%〜{h_h:.1f}%]")
    if len(low) >= 20:
        p_l, l_l, h_l, _ = block_bootstrap_recovery(low, low["ret_mix"], 300)
        print(f"  2号艇motor_2rate<{thresh}(n={len(low)}): 回収率{p_l:.1f}% [{l_l:.1f}%〜{h_l:.1f}%]")

print()
print("=== G) 3連単「1-2-3」+「1-3-2」ボックス(各100円,計200円) vs 現行2連単ミックス(300円) ===")
c3 = concluded.merge(payouts_3tan, on=["race_date", "jcd", "rno"], how="left", suffixes=("", "_3tan"))
c3["ret_3tan_box"] = np.where(c3["combination_3tan"].isin(["1-2-3", "1-3-2"]), c3["payout_3tan"], 0)
# 同一レースに複数の3連単行がmergeされている可能性があるので、レース単位で再集計
c3_race = c3.groupby(["race_date", "jcd", "rno"], as_index=False).agg(
    ret_3tan_box=("ret_3tan_box", "sum"), ret_mix=("ret_mix", "first")
)
p3, l3, h3, n3 = block_bootstrap_recovery(c3_race, c3_race["ret_3tan_box"], 200)
pmix, lmix, hmix, _ = block_bootstrap_recovery(c3_race, c3_race["ret_mix"], 300)
print(f"  3連単1-2-3&1-3-2ボックス(200円): 回収率{p3:.1f}% [{l3:.1f}%〜{h3:.1f}%] (n={len(c3_race)})")
print(f"  現行2連単ミックス(300円,同母集団で再計算): 回収率{pmix:.1f}% [{lmix:.1f}%〜{hmix:.1f}%]")

print()
print("=== H) 直前オッズが取得できているレースでの市場評価 vs 実際の回収率 ===")
odds_12 = odds_2tan[odds_2tan["combination"] == "1-2"][["race_date", "jcd", "rno", "odds_low"]].rename(columns={"odds_low": "odds_12"})
odds_13 = odds_2tan[odds_2tan["combination"] == "1-3"][["race_date", "jcd", "rno", "odds_low"]].rename(columns={"odds_low": "odds_13"})
c_odds = concluded.merge(odds_12, on=["race_date", "jcd", "rno"], how="inner").merge(odds_13, on=["race_date", "jcd", "rno"], how="inner")
print(f"  オッズ取得済みで①②条件に合致するレース: {len(c_odds)}件")
if len(c_odds) >= 10:
    print(f"  1-2オッズ平均: {c_odds['odds_12'].mean():.2f}倍 / 1-3オッズ平均: {c_odds['odds_13'].mean():.2f}倍")
    hit12 = (c_odds["combination"] == "1-2").mean() * 100
    hit13 = (c_odds["combination"] == "1-3").mean() * 100
    print(f"  この母集団での1-2的中率: {hit12:.1f}% (単純換算の損益分岐オッズ={100/hit12:.2f}倍)")
    print(f"  この母集団での1-3的中率: {hit13:.1f}% (単純換算の損益分岐オッズ={100/hit13:.2f}倍)")
    p_o12, l_o12, h_o12, _ = block_bootstrap_recovery(c_odds, np.where(c_odds["combination"] == "1-2", c_odds["payout"], 0), 100)
    p_o13, l_o13, h_o13, _ = block_bootstrap_recovery(c_odds, np.where(c_odds["combination"] == "1-3", c_odds["payout"], 0), 100)
    print(f"  同母集団: 1-2単独回収率{p_o12:.1f}% [{l_o12:.1f}%〜{h_o12:.1f}%]")
    print(f"  同母集団: 1-3単独回収率{p_o13:.1f}% [{l_o13:.1f}%〜{h_o13:.1f}%]")

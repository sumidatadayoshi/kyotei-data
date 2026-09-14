"""
①②honest版(時系列安全版、trailing閾値5走)を1号艇選手の級別(A1/A2/B1/B2)に
分けたレースについて、各級の2連単「1-2」「1-3」それぞれの的中時の平均オッズ・
中央値オッズを算出する。あわせてA1のROI・95%信頼区間上限を小数点第2位まで表示する。

判定ロジックはclass_breakdown_roi_noleak.pyと同一。
オッズ = payout(100円あたりの払戻金) / 100。
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "boatrace.db"
N_RESAMPLES = 2000
SEED = 42

SAMPLE_SIZE_WARNING_THRESHOLD = 10
INN_NIGE_RATE_THRESHOLD = 0.8
NIGASHI_RATE_THRESHOLD = 0.5
CLASS_ORDER = ["A1", "A2", "B1", "B2"]

BET_WEIGHTS = {"1-2": 200, "1-3": 100}
TOTAL_BET_PER_RACE = sum(BET_WEIGHTS.values())


def asof_prior_rate(base, role_waku, event_col, race_date_dt_col="race_date_dt"):
    role_df = base[base["waku"] == role_waku]
    day_stats = role_df.groupby(["toban", race_date_dt_col]).agg(
        day_starts=(event_col, "size"), day_wins=(event_col, "sum")
    ).reset_index().sort_values(["toban", race_date_dt_col])
    day_stats["cum_starts_incl"] = day_stats.groupby("toban")["day_starts"].cumsum()
    day_stats["cum_wins_incl"] = day_stats.groupby("toban")["day_wins"].cumsum()

    left = base[["toban", race_date_dt_col]].copy()
    left["search_date"] = left[race_date_dt_col] - pd.Timedelta(days=1)
    left = left.reset_index().sort_values("search_date")

    right = day_stats[["toban", race_date_dt_col, "cum_starts_incl", "cum_wins_incl"]].sort_values(
        race_date_dt_col
    )

    merged = pd.merge_asof(
        left, right, left_on="search_date", right_on=race_date_dt_col, by="toban", direction="backward",
        suffixes=("", "_event"),
    ).set_index("index").sort_index()

    rate = merged["cum_wins_incl"] / merged["cum_starts_incl"]
    starts = merged["cum_starts_incl"].fillna(0)
    return rate, starts


def race_return(row):
    weight = BET_WEIGHTS.get(row["combination"])
    if weight is None:
        return 0
    return row["payout"] * (weight / 100)


def block_bootstrap_ci(concluded):
    block_key = concluded["race_date"].astype(str) + "_" + concluded["jcd"].astype(str)
    df = pd.DataFrame({"block": block_key, "return": concluded["return"].to_numpy()})
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
    return n_blocks, lower, upper, np.median(rates)


conn = sqlite3.connect(DB_PATH)
entries_all = pd.read_sql_query(
    "SELECT race_date, jcd, rno, waku, toban, racer_class FROM entries", conn
)
results_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, rank FROM results", conn)
payouts_2tan = pd.read_sql_query(
    "SELECT race_date, jcd, rno, combination, payout FROM payouts WHERE bet_type = '2連単'", conn
)

base = entries_all.merge(results_all, on=["race_date", "jcd", "rno", "waku"], how="inner")
base["is_win"] = (base["rank"] == "1").astype(int)
base["race_date_dt"] = pd.to_datetime(base["race_date"], format="%Y%m%d")

waku1_win = base[base["waku"] == 1][["race_date", "jcd", "rno", "is_win"]].rename(
    columns={"is_win": "waku1_is_win"}
)
base = base.merge(waku1_win, on=["race_date", "jcd", "rno"], how="left")

base["inn_nige_rate_c1_prior"], base["c1_starts_prior"] = asof_prior_rate(base, 1, "is_win")
base["nigashi_rate_c2_prior"], base["c2_starts_prior"] = asof_prior_rate(base, 2, "waku1_is_win")

c1_prior = base[base["waku"] == 1][
    ["race_date", "jcd", "rno", "toban", "racer_class", "inn_nige_rate_c1_prior", "c1_starts_prior"]
].rename(columns={"toban": "toban1", "racer_class": "racer1_class"})
c2_prior = base[base["waku"] == 2][
    ["race_date", "jcd", "rno", "toban", "nigashi_rate_c2_prior", "c2_starts_prior"]
].rename(columns={"toban": "toban2"})
race_pairs = c1_prior.merge(c2_prior, on=["race_date", "jcd", "rno"], how="inner")

candidates = race_pairs[
    (race_pairs["inn_nige_rate_c1_prior"] >= INN_NIGE_RATE_THRESHOLD)
    & (race_pairs["c1_starts_prior"] >= SAMPLE_SIZE_WARNING_THRESHOLD)
    & (race_pairs["nigashi_rate_c2_prior"] >= NIGASHI_RATE_THRESHOLD)
    & (race_pairs["c2_starts_prior"] >= SAMPLE_SIZE_WARNING_THRESHOLD)
][["race_date", "jcd", "rno", "racer1_class"]].copy()

concluded_all = candidates.merge(payouts_2tan, on=["race_date", "jcd", "rno"], how="inner").copy()
concluded_all["return"] = concluded_all.apply(race_return, axis=1)
concluded_all["hit"] = concluded_all["combination"].isin(BET_WEIGHTS)
concluded_all["odds"] = concluded_all["payout"] / 100.0

print("=== 級別 × 2連単1-2/1-3 的中時オッズ(払戻金/100) ===")
print()
for cls in CLASS_ORDER:
    sub = concluded_all[concluded_all["racer1_class"] == cls]
    print(f"----- 1号艇級 {cls}(対象{len(sub)}件) -----")
    for combo in ["1-2", "1-3"]:
        hits = sub[sub["combination"] == combo]
        n_hit = len(hits)
        if n_hit == 0:
            print(f"  {combo}: 的中0件")
            continue
        mean_odds = hits["odds"].mean()
        median_odds = hits["odds"].median()
        print(f"  {combo}: 的中{n_hit}件 / 平均オッズ {mean_odds:.2f}倍 / 中央値オッズ {median_odds:.2f}倍 "
              f"(最小{hits['odds'].min():.1f}倍・最大{hits['odds'].max():.1f}倍)")
    both_hits = sub[sub["hit"]]
    if len(both_hits) > 0:
        print(f"  1-2・1-3合計: 的中{len(both_hits)}件 / 平均オッズ {both_hits['odds'].mean():.2f}倍 / "
              f"中央値オッズ {both_hits['odds'].median():.2f}倍")
    print()

print("=== A1のROI・95%信頼区間(小数点第2位まで) ===")
sub_a1 = concluded_all[concluded_all["racer1_class"] == "A1"]
total_races = len(sub_a1)
total_return = int(sub_a1["return"].sum())
total_stake = total_races * TOTAL_BET_PER_RACE
recovery_rate = (total_return / total_stake * 100) if total_stake > 0 else 0.0
n_blocks, lower, upper, median = block_bootstrap_ci(sub_a1)

print(f"対象レース数: {total_races}件 / ブロック数: {n_blocks}")
print(f"賭け金合計: {total_stake:,}円 / 払戻金合計: {total_return:,}円")
print(f"ROI: {recovery_rate:.2f}%")
print(f"95%信頼区間: {lower:.2f}% 〜 {upper:.2f}%(中央値 {median:.2f}%)")

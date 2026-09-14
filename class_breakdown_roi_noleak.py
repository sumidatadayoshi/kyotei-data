"""
時系列安全版(trailing最低成績数閾値=5走)の①②条件を満たすレースを、
1号艇選手の級別(A1/A2/B1/B2)ごとに分け、それぞれについて
2連単「1-2」に200円・「1-3」に100円(1レースあたり計300円、1-4は買わない)の
ROIと、開催日×競艇場(jcd)単位のブロックブートストラップ(2000回、seed=42)による
95%信頼区間、レース数・ブロック数を算出する。

判定ロジックはthreshold_sensitivity_noleak.py(閾値=5のケース)と同一。
級別は1号艇選手(waku=1)のentries.racer_classを使う。
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
TOTAL_BET_PER_RACE = sum(BET_WEIGHTS.values())  # 300円


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

print(f"買い目: 2連単 1-2({BET_WEIGHTS['1-2']}円) + 1-3({BET_WEIGHTS['1-3']}円) "
      f"、1レースあたり計{TOTAL_BET_PER_RACE}円(1-4は買わない)")
print(f"判定: ①②とも時系列安全版(前日までの累積成績)、trailing最低成績数閾値={SAMPLE_SIZE_WARNING_THRESHOLD}走")
print(f"級別: 1号艇選手(waku=1)のracer_class")
print()

results_summary = []
for cls in CLASS_ORDER + ["(級不明)"]:
    if cls == "(級不明)":
        sub = concluded_all[~concluded_all["racer1_class"].isin(CLASS_ORDER)]
    else:
        sub = concluded_all[concluded_all["racer1_class"] == cls]

    total_races = len(sub)
    hit_count = int(sub["hit"].sum())
    total_return = int(sub["return"].sum())
    total_stake = total_races * TOTAL_BET_PER_RACE
    recovery_rate = (total_return / total_stake * 100) if total_stake > 0 else 0.0
    hit_rate = (hit_count / total_races * 100) if total_races > 0 else 0.0

    if total_races > 0:
        n_blocks, lower, upper, median = block_bootstrap_ci(sub)
    else:
        n_blocks, lower, upper, median = 0, float("nan"), float("nan"), float("nan")

    print(f"===== 1号艇級別 = {cls} =====")
    print(f"対象レース数: {total_races}件 / 的中回数: {hit_count}回({hit_rate:.1f}%)")
    print(f"賭け金合計: {total_stake:,}円 / 払戻金合計: {total_return:,}円 / "
          f"通算損益: {total_return - total_stake:,}円")
    print(f"ROI: {recovery_rate:.1f}%")
    print(f"ブロック数(開催日×競艇場): {n_blocks}")
    if total_races > 0:
        print(f"95%信頼区間: {lower:.1f}% 〜 {upper:.1f}%(中央値 {median:.1f}%)")
    else:
        print("95%信頼区間: 算出不可(対象レースなし)")
    print()

    results_summary.append({
        "class": cls, "total_races": total_races, "hit_count": hit_count,
        "recovery_rate": recovery_rate, "n_blocks": n_blocks,
        "ci_lower": lower, "ci_upper": upper, "median": median,
    })

print("===== 級別比較サマリー =====")
header = f"{'級':>8}{'レース数':>10}{'ブロック数':>12}{'的中回数':>10}{'ROI':>10}{'95%CI':>22}"
print(header)
for r in results_summary:
    if r["total_races"] > 0:
        ci_str = f"{r['ci_lower']:.1f}%~{r['ci_upper']:.1f}%"
        roi_str = f"{r['recovery_rate']:.1f}%"
    else:
        ci_str = "N/A"
        roi_str = "N/A"
    print(f"{r['class']:>8}{r['total_races']:>10}{r['n_blocks']:>12}"
          f"{r['hit_count']:>10}{roi_str:>10}{ci_str:>22}")

"""
①②条件(1号艇イン逃げ率80%以上・2号艇逃し率50%以上)の判定を、従来の
「全期間(未来含む)の通算成績」ではなく、model_with_ab_features_noleak.py と
同じ時系列安全な方式(そのレースの前日までの累積成績のみ)に修正したうえで、
2連単「1-2」に10,000円・「1-3」に5,000円(1-4は買わない)を賭け続けた場合の
回収率と、開催日×競艇場(jcd)単位のブロックブートストラップによる95%信頼区間を
計算する。

従来版(block_bootstrap_weighted_12_13_10000_5000.py, 全期間統計)と並べて
表示し、リーク修正前後でどう変わるかを比較する。
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

BET_WEIGHTS = {"1-2": 10000, "1-3": 5000}
TOTAL_BET_PER_RACE = sum(BET_WEIGHTS.values())


def asof_prior_rate(base, role_waku, event_col, race_date_dt_col="race_date_dt"):
    """役割(waku==role_waku)で走ったレースだけを対象に、選手(toban)ごとの
    累積成績(当日より前のみ)を、merge_asofで各行に逆引きして付与する。
    model_with_ab_features_noleak.py の同名関数と同一ロジック。"""
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


def race_return(row):
    weight = BET_WEIGHTS.get(row["combination"])
    if weight is None:
        return 0
    return row["payout"] * (weight / 100)


def summarize(label, concluded):
    total_races = len(concluded)
    hit_count = int(concluded["hit"].sum())
    total_return = int(concluded["return"].sum())
    total_stake = total_races * TOTAL_BET_PER_RACE
    recovery_rate = (total_return / total_stake * 100) if total_stake > 0 else 0.0
    hit_rate = (hit_count / total_races * 100) if total_races > 0 else 0.0
    n_blocks, lower, upper, median = block_bootstrap_ci(concluded)

    print(f"===== {label} =====")
    print(f"対象レース数: {total_races}件 / 的中回数: {hit_count}回 ({hit_rate:.1f}%)")
    print(f"賭け金合計: {total_stake:,}円 / 払戻金合計: {total_return:,}円 / "
          f"通算損益: {total_return - total_stake:,}円")
    print(f"回収率: {recovery_rate:.1f}%")
    print(f"ブロック数(開催日×競艇場): {n_blocks}")
    print(f"95%信頼区間: {lower:.1f}% 〜 {upper:.1f}% (中央値 {median:.1f}%)")
    print()
    return {
        "label": label, "total_races": total_races, "hit_count": hit_count,
        "recovery_rate": recovery_rate, "n_blocks": n_blocks,
        "ci_lower": lower, "ci_upper": upper, "median": median,
    }


conn = sqlite3.connect(DB_PATH)
entries_all = pd.read_sql_query(
    "SELECT race_date, jcd, rno, waku, toban, racer_name, gender, venue_name FROM entries", conn
)
results_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, rank FROM results", conn)
payouts_2tan = pd.read_sql_query(
    "SELECT race_date, jcd, rno, combination, payout FROM payouts WHERE bet_type = '2連単'", conn
)

# ============================================================
# 旧版: 全期間(未来含む)の通算成績で①②を判定(従来ロジック、比較用に再掲)
# ============================================================
def compute_racer_rate_stats_fullperiod(entries_all, results_all):
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
    return c1_stats, c2_stats


def find_qualifying_races_fullperiod(entries_df, c1_stats, c2_stats):
    qualified_toban1 = set(
        c1_stats[(c1_stats["rate"] >= INN_NIGE_RATE_THRESHOLD)
                 & (c1_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index
    )
    qualified_toban2 = set(
        c2_stats[(c2_stats["rate"] >= NIGASHI_RATE_THRESHOLD)
                 & (c2_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index
    )
    entries1 = entries_df[entries_df["waku"] == 1][["race_date", "jcd", "rno", "toban"]].rename(
        columns={"toban": "toban1"}
    )
    entries2 = entries_df[entries_df["waku"] == 2][["race_date", "jcd", "rno", "toban"]].rename(
        columns={"toban": "toban2"}
    )
    race_pairs = entries1.merge(entries2, on=["race_date", "jcd", "rno"], how="inner")
    return race_pairs[
        race_pairs["toban1"].isin(qualified_toban1) & race_pairs["toban2"].isin(qualified_toban2)
    ][["race_date", "jcd", "rno"]].copy()


c1_stats_full, c2_stats_full = compute_racer_rate_stats_fullperiod(entries_all, results_all)
candidates_full = find_qualifying_races_fullperiod(entries_all, c1_stats_full, c2_stats_full)
concluded_full = candidates_full.merge(payouts_2tan, on=["race_date", "jcd", "rno"], how="inner").copy()
concluded_full["return"] = concluded_full.apply(race_return, axis=1)
concluded_full["hit"] = concluded_full["combination"].isin(BET_WEIGHTS)

# ============================================================
# 新版: 時系列安全(そのレースの前日までの累積成績のみ)で①②を判定
# ============================================================
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
    ["race_date", "jcd", "rno", "toban", "inn_nige_rate_c1_prior", "c1_starts_prior"]
].rename(columns={"toban": "toban1"})
c2_prior = base[base["waku"] == 2][
    ["race_date", "jcd", "rno", "toban", "nigashi_rate_c2_prior", "c2_starts_prior"]
].rename(columns={"toban": "toban2"})

race_pairs_noleak = c1_prior.merge(c2_prior, on=["race_date", "jcd", "rno"], how="inner")

candidates_noleak = race_pairs_noleak[
    (race_pairs_noleak["inn_nige_rate_c1_prior"] >= INN_NIGE_RATE_THRESHOLD)
    & (race_pairs_noleak["c1_starts_prior"] >= SAMPLE_SIZE_WARNING_THRESHOLD)
    & (race_pairs_noleak["nigashi_rate_c2_prior"] >= NIGASHI_RATE_THRESHOLD)
    & (race_pairs_noleak["c2_starts_prior"] >= SAMPLE_SIZE_WARNING_THRESHOLD)
][["race_date", "jcd", "rno"]].copy()

concluded_noleak = candidates_noleak.merge(
    payouts_2tan, on=["race_date", "jcd", "rno"], how="inner"
).copy()
concluded_noleak["return"] = concluded_noleak.apply(race_return, axis=1)
concluded_noleak["hit"] = concluded_noleak["combination"].isin(BET_WEIGHTS)

# ============================================================
# 比較表示
# ============================================================
print(f"判定方式: ①1号艇イン逃げ率{INN_NIGE_RATE_THRESHOLD*100:.0f}%以上 "
      f"②2号艇逃し率{NIGASHI_RATE_THRESHOLD*100:.0f}%以上 "
      f"(いずれもサンプル数{SAMPLE_SIZE_WARNING_THRESHOLD}走以上)")
print(f"買い目: 2連単 1-2({BET_WEIGHTS['1-2']:,}円) + 1-3({BET_WEIGHTS['1-3']:,}円) "
      f"、1レースあたり計{TOTAL_BET_PER_RACE:,}円(1-4は買わない)")
print()

result_full = summarize("旧版(全期間統計・未来データ含む/リークあり)", concluded_full)
result_noleak = summarize("新版(前日までの累積成績のみ/時系列安全)", concluded_noleak)

print("===== 比較サマリー =====")
print(f"{'':30}{'旧版(リークあり)':>18}{'新版(時系列安全)':>18}")
print(f"{'対象レース数':30}{result_full['total_races']:>18}{result_noleak['total_races']:>18}")
print(f"{'的中回数':30}{result_full['hit_count']:>18}{result_noleak['hit_count']:>18}")
print(f"{'回収率':30}{result_full['recovery_rate']:>17.1f}%{result_noleak['recovery_rate']:>17.1f}%")
print(f"{'ブロック数':30}{result_full['n_blocks']:>18}{result_noleak['n_blocks']:>18}")
ci_full_str = f"{result_full['ci_lower']:.1f}%~{result_full['ci_upper']:.1f}%"
ci_noleak_str = f"{result_noleak['ci_lower']:.1f}%~{result_noleak['ci_upper']:.1f}%"
print(f"{'95%信頼区間':30}{ci_full_str:>18}{ci_noleak_str:>18}")

"""
時系列安全版(trailing最低成績数閾値5走)の①②条件を満たすレースについて、
単勝「1」(1号艇単勝)を100円ずつ買い続けた場合のROIと、開催日×競艇場(jcd)
単位のブロックブートストラップ(2000回、seed=42)による95%信頼区間、
的中時の平均オッズ・中央値オッズを算出する。

判定ロジックはclass_breakdown_roi_noleak.py等と同一。
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

BET_AMOUNT = 100  # 単勝1に100円


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


def block_bootstrap_ci(concluded, return_col="return"):
    block_key = concluded["race_date"].astype(str) + "_" + concluded["jcd"].astype(str)
    df = pd.DataFrame({"block": block_key, "return": concluded[return_col].to_numpy()})
    blocks = df.groupby("block").agg(n=("return", "size"), return_sum=("return", "sum"))
    n_blocks = len(blocks)

    n_arr = blocks["n"].to_numpy()
    return_arr = blocks["return_sum"].to_numpy()

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n_blocks, size=(N_RESAMPLES, n_blocks))
    resample_stake = n_arr[idx].sum(axis=1) * BET_AMOUNT
    resample_return = return_arr[idx].sum(axis=1)
    rates = np.where(resample_stake > 0, resample_return / resample_stake * 100, 0.0)

    lower, upper = np.percentile(rates, [2.5, 97.5])
    return n_blocks, lower, upper, np.median(rates)


conn = sqlite3.connect(DB_PATH)
entries_all = pd.read_sql_query(
    "SELECT race_date, jcd, rno, waku, toban FROM entries", conn
)
results_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, rank FROM results", conn)
payouts_tansho = pd.read_sql_query(
    "SELECT race_date, jcd, rno, combination, payout FROM payouts WHERE bet_type = '単勝'", conn
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
    ["race_date", "jcd", "rno", "toban", "inn_nige_rate_c1_prior", "c1_starts_prior"]
].rename(columns={"toban": "toban1"})
c2_prior = base[base["waku"] == 2][
    ["race_date", "jcd", "rno", "toban", "nigashi_rate_c2_prior", "c2_starts_prior"]
].rename(columns={"toban": "toban2"})
race_pairs = c1_prior.merge(c2_prior, on=["race_date", "jcd", "rno"], how="inner")

candidates = race_pairs[
    (race_pairs["inn_nige_rate_c1_prior"] >= INN_NIGE_RATE_THRESHOLD)
    & (race_pairs["c1_starts_prior"] >= SAMPLE_SIZE_WARNING_THRESHOLD)
    & (race_pairs["nigashi_rate_c2_prior"] >= NIGASHI_RATE_THRESHOLD)
    & (race_pairs["c2_starts_prior"] >= SAMPLE_SIZE_WARNING_THRESHOLD)
][["race_date", "jcd", "rno"]].copy()

ODDS_CAP = 1.3  # このオッズ以上になる場合は見送る(賭けない)

concluded = candidates.merge(payouts_tansho, on=["race_date", "jcd", "rno"], how="inner").copy()
concluded["hit"] = concluded["combination"] == "1"
concluded["return"] = np.where(concluded["hit"], concluded["payout"] * (BET_AMOUNT / 100), 0)
concluded["odds"] = concluded["payout"] / 100.0

# 単勝1の発走前オッズは「oddsテーブル」(結果とは別に、発走前スナップショットを
# 記録したテーブル)から取得する必要がある。ただしこのテーブルは日次スクレイプ
# ジョブが実装されて以降のレースにしか存在せず、対象期間全体はカバーしていない。
odds_tansho1 = pd.read_sql_query(
    "SELECT race_date, jcd, rno, odds_low AS pre_race_odds FROM odds "
    "WHERE bet_type = '単勝' AND combination = '1'", conn
)
covered_dates = sorted(odds_tansho1["race_date"].unique())
print(f"oddsテーブルに発走前オッズが存在する日数: {len(covered_dates)}日"
      f"(範囲: {covered_dates[0]}〜{covered_dates[-1]}, 対象期間全体のごく一部)")

before_merge = len(concluded)
concluded_with_odds = concluded.merge(
    odds_tansho1, on=["race_date", "jcd", "rno"], how="inner"
)
print(f"①②条件該当レース{before_merge}件のうち、発走前オッズが判明しているのは"
      f"{len(concluded_with_odds)}件のみ(この一部データのみで以下を計算するため、"
      f"直近期間に偏ったサンプルである点に注意)")
print()
filtered = concluded_with_odds[concluded_with_odds["pre_race_odds"] < ODDS_CAP].copy()
print(f"オッズ{ODDS_CAP}倍未満のみ賭ける絞り込み: {len(concluded_with_odds)}件 -> {len(filtered)}件"
      f"({ODDS_CAP}倍以上のため見送り: {len(concluded_with_odds) - len(filtered)}件)")
print()


def summarize(label, df):
    total_races = len(df)
    hit_count = int(df["hit"].sum())
    total_return = int(df["return"].sum())
    total_stake = total_races * BET_AMOUNT
    recovery_rate = (total_return / total_stake * 100) if total_stake > 0 else 0.0
    hit_rate = (hit_count / total_races * 100) if total_races > 0 else 0.0

    print(f"===== {label} =====")
    if total_races == 0:
        print("対象レース数: 0件")
        print()
        return
    n_blocks, lower, upper, median = block_bootstrap_ci(df)
    hits = df[df["hit"]]
    mean_odds = hits["odds"].mean() if len(hits) > 0 else float("nan")
    median_odds = hits["odds"].median() if len(hits) > 0 else float("nan")

    print(f"対象レース数: {total_races}件 / 的中回数: {hit_count}回({hit_rate:.2f}%)")
    print(f"賭け金合計: {total_stake:,}円 / 払戻金合計: {total_return:,}円 / "
          f"通算損益: {total_return - total_stake:,}円")
    print(f"ROI: {recovery_rate:.2f}%")
    print(f"ブロック数(開催日×競艇場): {n_blocks}")
    print(f"95%信頼区間: {lower:.2f}% 〜 {upper:.2f}%(中央値 {median:.2f}%)")
    if len(hits) > 0:
        print(f"的中時オッズ(結果payoutベース): 平均 {mean_odds:.2f}倍 / 中央値 {median_odds:.2f}倍 "
              f"(最小{hits['odds'].min():.1f}倍・最大{hits['odds'].max():.1f}倍)")
    print()


print("買い目: 単勝「1」(1号艇単勝) 100円")
print("判定: ①②とも時系列安全版(前日までの累積成績)、trailing最低成績数閾値=5走")
print()
summarize(f"oddsデータありサブセット全体(絞り込みなし、n={len(concluded_with_odds)})", concluded_with_odds)
summarize(f"オッズ{ODDS_CAP}倍未満のみ賭ける(n={len(filtered)})", filtered)

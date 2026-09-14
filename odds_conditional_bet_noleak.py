"""
①②honest版(時系列安全版、trailing最低成績数閾値5走)に該当する全レースについて、
発走前オッズによる条件付き買い方を検証する:
  - 2連単「1-2」: 直前オッズ2.0倍以上の場合のみ200円
  - 2連単「1-3」: 直前オッズ3.0倍以上の場合のみ100円
  - どちらの条件も満たさない場合はそのレースを見送る(賭けない)

実際に賭けたレースのみを対象に、賭け金合計に対するROIと、開催日×競艇場(jcd)
単位のブロックブートストラップ(2000回、seed=42)による95%信頼区間を算出する。

あわせて、
  ①oddsデータの有無別レース数
  ②フィルタ後に実際に賭けたレース数・的中回数
  ③フィルタ前後での1号艇級別構成比の変化
も出す。
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

BET_12_AMOUNT = 200
BET_13_AMOUNT = 100
ODDS_CAP_12 = 2.0
ODDS_CAP_13 = 3.0


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


def block_bootstrap_ci(df, return_col, stake_col):
    block_key = df["race_date"].astype(str) + "_" + df["jcd"].astype(str)
    g = pd.DataFrame({
        "block": block_key, "return": df[return_col].to_numpy(), "stake": df[stake_col].to_numpy(),
    })
    blocks = g.groupby("block").agg(stake_sum=("stake", "sum"), return_sum=("return", "sum"))
    n_blocks = len(blocks)

    stake_arr = blocks["stake_sum"].to_numpy()
    return_arr = blocks["return_sum"].to_numpy()

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n_blocks, size=(N_RESAMPLES, n_blocks))
    resample_stake = stake_arr[idx].sum(axis=1)
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
odds_2tan = pd.read_sql_query(
    "SELECT race_date, jcd, rno, combination, odds_low FROM odds "
    "WHERE bet_type = '2連単' AND combination IN ('1-2', '1-3')", conn
)

# --- ①②honest版(時系列安全、閾値5走)の判定 ---
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

print(f"①②honest版(時系列安全・閾値{SAMPLE_SIZE_WARNING_THRESHOLD}走)該当レース数: {len(candidates)}件")
print()

# ============================================================
# ① オッズデータの有無別レース数
# ============================================================
odds_12 = odds_2tan[odds_2tan["combination"] == "1-2"][["race_date", "jcd", "rno", "odds_low"]].rename(
    columns={"odds_low": "odds_12"}
)
odds_13 = odds_2tan[odds_2tan["combination"] == "1-3"][["race_date", "jcd", "rno", "odds_low"]].rename(
    columns={"odds_low": "odds_13"}
)

merged = candidates.merge(odds_12, on=["race_date", "jcd", "rno"], how="left")
merged = merged.merge(odds_13, on=["race_date", "jcd", "rno"], how="left")

has_odds_12 = merged["odds_12"].notna()
has_odds_13 = merged["odds_13"].notna()
has_any_odds = has_odds_12 | has_odds_13
has_both_odds = has_odds_12 & has_odds_13

print("=== ① オッズデータの有無別レース数(①②該当レース全体に対して) ===")
print(f"全体: {len(merged)}件")
print(f"  1-2オッズあり: {has_odds_12.sum()}件 / 1-3オッズあり: {has_odds_13.sum()}件")
print(f"  1-2・1-3どちらかオッズあり: {has_any_odds.sum()}件")
print(f"  1-2・1-3両方オッズあり: {has_both_odds.sum()}件")
print(f"  1-2・1-3どちらもオッズなし: {(~has_any_odds).sum()}件"
      f"(→この分は条件を確認できないため自動的に見送り扱い)")
print()

# ============================================================
# 条件付き賭け金の決定
#   1-2: オッズ2.0倍以上のときだけ200円
#   1-3: オッズ3.0倍以上のときだけ100円
#   どちらも賭けない場合はそのレースを見送る
# ============================================================
merged["bet_12"] = np.where(has_odds_12 & (merged["odds_12"] >= ODDS_CAP_12), BET_12_AMOUNT, 0)
merged["bet_13"] = np.where(has_odds_13 & (merged["odds_13"] >= ODDS_CAP_13), BET_13_AMOUNT, 0)
merged["stake"] = merged["bet_12"] + merged["bet_13"]

played = merged[merged["stake"] > 0].copy()

print("=== ② フィルタ後に実際に賭けたレース数 ===")
print(f"①②該当{len(merged)}件のうち、条件を満たして実際に賭けたレース: {len(played)}件"
      f"(見送り: {len(merged) - len(played)}件)")
print(f"  うち1-2のみ賭けた: {((played['bet_12'] > 0) & (played['bet_13'] == 0)).sum()}件")
print(f"  うち1-3のみ賭けた: {((played['bet_12'] == 0) & (played['bet_13'] > 0)).sum()}件")
print(f"  うち1-2・1-3両方賭けた: {((played['bet_12'] > 0) & (played['bet_13'] > 0)).sum()}件")
print()

payouts_pivot = payouts_2tan[payouts_2tan["combination"].isin(["1-2", "1-3"])].pivot_table(
    index=["race_date", "jcd", "rno"], columns="combination", values="payout", aggfunc="first"
).reset_index()
payouts_pivot.columns.name = None
for col in ["1-2", "1-3"]:
    if col not in payouts_pivot.columns:
        payouts_pivot[col] = np.nan

played = played.merge(payouts_pivot, on=["race_date", "jcd", "rno"], how="left")
played["return_12"] = np.where(played["1-2"].notna(), played["1-2"] * (played["bet_12"] / 100), 0)
played["return_13"] = np.where(played["1-3"].notna(), played["1-3"] * (played["bet_13"] / 100), 0)
played["return"] = played["return_12"] + played["return_13"]
played["hit"] = (played["return"] > 0)

total_races = len(played)
hit_count = int(played["hit"].sum())
total_stake = int(played["stake"].sum())
total_return = int(played["return"].sum())
recovery_rate = (total_return / total_stake * 100) if total_stake > 0 else 0.0
hit_rate = (hit_count / total_races * 100) if total_races > 0 else 0.0

print("=== ROI(実際の賭け金合計ベース) ===")
print(f"対象レース数(実際に賭けた): {total_races}件 / 的中回数: {hit_count}回({hit_rate:.2f}%)")
print(f"賭け金合計: {total_stake:,}円 / 払戻金合計: {total_return:,}円 / "
      f"通算損益: {total_return - total_stake:,}円")
print(f"ROI: {recovery_rate:.2f}%")

if total_races > 0:
    n_blocks, lower, upper, median = block_bootstrap_ci(played, "return", "stake")
    print(f"ブロック数(開催日×競艇場): {n_blocks}")
    print(f"95%信頼区間: {lower:.2f}% 〜 {upper:.2f}%(中央値 {median:.2f}%)")
else:
    print("対象レースが0件のためブートストラップ不可")
print()

# ============================================================
# ③ フィルタ前後での1号艇級別構成比の変化
# ============================================================
print("=== ③ フィルタ前後での1号艇級別構成比の変化 ===")


def class_ratio(df, label):
    valid = df[df["racer1_class"].isin(CLASS_ORDER)]
    counts = valid["racer1_class"].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int)
    ratios = (counts / len(valid) * 100).round(1) if len(valid) > 0 else counts * 0.0
    print(f"[{label}] 総数(級判明分): {len(valid)}件")
    for cls in CLASS_ORDER:
        print(f"  {cls}: {counts[cls]}件 ({ratios[cls]:.1f}%)")
    return ratios


ratio_before = class_ratio(candidates, "フィルタ前(①②該当レース全体)")
print()
ratio_after = class_ratio(played, "フィルタ後(実際に賭けたレース)")
print()

print("[変化(フィルタ後-フィルタ前, ポイント)]")
for cls in CLASS_ORDER:
    diff = ratio_after[cls] - ratio_before[cls]
    print(f"  {cls}: {diff:+.1f}pt")

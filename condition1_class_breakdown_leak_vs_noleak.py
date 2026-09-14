"""
①(1号艇イン逃げ率80%以上)の判定について、
- leak版(全期間統計、未来データ含む)
- 時系列安全版(そのレースの前日までの累積成績のみ)
のそれぞれで、1号艇選手の級別(A1/A2/B1/B2)ごとの該当率(waku=1の出走に占める、
①条件を満たす出走の割合)を比較する。

あわせて、leak版と時系列安全版で①の判定(該当/非該当)が食い違った出走について、
その時点での該当選手のコース1試行回数(時系列安全版のカウント=c1_starts_prior)
の分布を見る。

ロジックはblock_bootstrap_weighted_12_13_noleak.pyのasof_prior_rate等と同一。
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "boatrace.db"

SAMPLE_SIZE_WARNING_THRESHOLD = 10
INN_NIGE_RATE_THRESHOLD = 0.8
CLASS_ORDER = ["A1", "A2", "B1", "B2"]


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


conn = sqlite3.connect(DB_PATH)
entries_all = pd.read_sql_query(
    "SELECT race_date, jcd, rno, waku, toban, racer_class FROM entries", conn
)
results_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, rank FROM results", conn)

# --- leak版: 全期間(未来含む)統計で①(1号艇イン逃げ率)を選手(toban)ごとに算出 ---
c1 = entries_all[entries_all["waku"] == 1].merge(
    results_all[results_all["waku"] == 1][["race_date", "jcd", "rno", "rank"]],
    on=["race_date", "jcd", "rno"], how="inner",
)
c1_stats_full = c1.groupby("toban").agg(starts=("rank", "size"), wins=("rank", lambda s: (s == "1").sum()))
c1_stats_full["rate"] = c1_stats_full["wins"] / c1_stats_full["starts"]
qualified_toban1_leak = set(
    c1_stats_full[
        (c1_stats_full["rate"] >= INN_NIGE_RATE_THRESHOLD)
        & (c1_stats_full["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)
    ].index
)

# --- 時系列安全版: そのレースの前日までの累積成績のみで①を出走ごとに算出 ---
base = entries_all.merge(results_all, on=["race_date", "jcd", "rno", "waku"], how="inner")
base["is_win"] = (base["rank"] == "1").astype(int)
base["race_date_dt"] = pd.to_datetime(base["race_date"], format="%Y%m%d")

base["inn_nige_rate_c1_prior"], base["c1_starts_prior"] = asof_prior_rate(base, 1, "is_win")

waku1 = base[base["waku"] == 1].copy()
waku1["leak_qualified"] = waku1["toban"].isin(qualified_toban1_leak)
waku1["noleak_qualified"] = (
    (waku1["inn_nige_rate_c1_prior"] >= INN_NIGE_RATE_THRESHOLD)
    & (waku1["c1_starts_prior"] >= SAMPLE_SIZE_WARNING_THRESHOLD)
)

waku1_valid_class = waku1[waku1["racer_class"].isin(CLASS_ORDER)].copy()

print(f"対象: 1号艇としての出走 全{len(waku1)}件(うち級別が判明: {len(waku1_valid_class)}件)")
print()
print("=== ①(イン逃げ率80%以上)の該当率: 級別 × leak版/時系列安全版 ===")
summary = waku1_valid_class.groupby("racer_class").agg(
    出走数=("leak_qualified", "size"),
    leak版該当率=("leak_qualified", "mean"),
    時系列安全版該当率=("noleak_qualified", "mean"),
).reindex(CLASS_ORDER)
summary["leak版該当率"] = (summary["leak版該当率"] * 100).round(1)
summary["時系列安全版該当率"] = (summary["時系列安全版該当率"] * 100).round(1)
summary["差分(leak-時系列安全)pt"] = (summary["leak版該当率"] - summary["時系列安全版該当率"]).round(1)
print(summary.to_string())
print()

total_row = pd.Series({
    "出走数": len(waku1_valid_class),
    "leak版該当率": round(waku1_valid_class["leak_qualified"].mean() * 100, 1),
    "時系列安全版該当率": round(waku1_valid_class["noleak_qualified"].mean() * 100, 1),
})
total_row["差分(leak-時系列安全)pt"] = round(total_row["leak版該当率"] - total_row["時系列安全版該当率"], 1)
print("全体:")
print(total_row.to_string())
print()

# ============================================================
# leak版と時系列安全版で①の判定が食い違った出走
# ============================================================
mismatch = waku1[waku1["leak_qualified"] != waku1["noleak_qualified"]].copy()
mismatch_leak_only = mismatch[mismatch["leak_qualified"] & ~mismatch["noleak_qualified"]]
mismatch_noleak_only = mismatch[~mismatch["leak_qualified"] & mismatch["noleak_qualified"]]

print("=== leak版と時系列安全版で①の判定が食い違った出走 ===")
print(f"食い違い合計: {len(mismatch)}件")
print(f"  leak版のみ該当(時系列安全版では非該当): {len(mismatch_leak_only)}件")
print(f"  時系列安全版のみ該当(leak版では非該当): {len(mismatch_noleak_only)}件")
print()

print("--- 食い違い出走における c1_starts_prior(時系列安全版のコース1試行回数)の分布 ---")
print("[食い違い全体]")
print(mismatch["c1_starts_prior"].describe().round(1).to_string())
print()
print("[内訳: leak版のみ該当(時系列安全版では非該当。主に前日までの試行数不足=リークで水増しされていたケース)]")
print(mismatch_leak_only["c1_starts_prior"].describe().round(1).to_string())
print()
print("[内訳: 時系列安全版のみ該当(leak版では非該当。直近は好調でも通算では基準未満のケースなど)]")
print(mismatch_noleak_only["c1_starts_prior"].describe().round(1).to_string())
print()

bins = [0, 4, 9, 19, 49, 99, np.inf]
bin_labels = ["0-4(基準未満)", "5-9", "10-19", "20-49", "50-99", "100+"]
mismatch["c1_starts_prior_bucket"] = pd.cut(
    mismatch["c1_starts_prior"], bins=bins, labels=bin_labels, right=True
)
print("--- 食い違い出走の c1_starts_prior 区間別件数(全体) ---")
bucket_counts = mismatch.groupby("c1_starts_prior_bucket", observed=True).size()
print(bucket_counts.to_string())

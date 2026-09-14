"""
win_prob_model.pyと同じ手順でモデルを学習・評価したうえで、テスト期間の予測結果から
①②条件(1号艇イン逃げ率80%以上・2号艇逃し率50%以上)に合致したレースだけを抜き出し、
モデルが1号艇に予測した勝率と、実際の的中率を比較する。

さらに、モデルが1号艇に80%以上の勝率を予測したレースだけに絞った場合の実際の的中率も見る。

①②の判定ロジックはblock_bootstrap_1_2.py等と同一内容を複製（学習用データとは
独立に、全期間のentries/resultsから選手ごとのイン逃げ率・逃し率を計算する、
これまでの分析と同じ方法）。
"""
import re
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

DB_PATH = Path(__file__).parent / "data" / "boatrace.db"
TEST_DAYS = 21
RANDOM_STATE = 42

SAMPLE_SIZE_WARNING_THRESHOLD = 10
INN_NIGE_RATE_THRESHOLD = 0.8
NIGASHI_RATE_THRESHOLD = 0.5

CATEGORICAL_COLS = ["racer_class", "gender", "venue_name", "weather", "grade"]
NUMERIC_COLS = [
    "waku", "age", "weight", "f_count", "l_count", "avg_st",
    "national_win_rate", "national_2rate", "national_3rate",
    "local_win_rate", "local_2rate", "local_3rate",
    "motor_2rate", "motor_3rate", "boat_2rate", "boat_3rate",
    "temperature", "wind_speed", "water_temp", "wave_height", "distance_m",
    "course_win_rate_prior", "course_starts_prior",
]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


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

    return c1_stats, c2_stats


def find_qualifying_races(entries_df, c1_stats, c2_stats):
    qualified_toban1 = set(
        c1_stats[
            (c1_stats["rate"] >= INN_NIGE_RATE_THRESHOLD)
            & (c1_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)
        ].index
    )
    qualified_toban2 = set(
        c2_stats[
            (c2_stats["rate"] >= NIGASHI_RATE_THRESHOLD)
            & (c2_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)
        ].index
    )

    entries1 = entries_df[entries_df["waku"] == 1][
        ["race_date", "jcd", "rno", "toban", "racer_name"]
    ].rename(columns={"toban": "toban1"})
    entries2 = entries_df[entries_df["waku"] == 2][
        ["race_date", "jcd", "rno", "toban"]
    ].rename(columns={"toban": "toban2"})
    race_pairs = entries1.merge(entries2, on=["race_date", "jcd", "rno"], how="inner")

    return race_pairs[
        race_pairs["toban1"].isin(qualified_toban1) & race_pairs["toban2"].isin(qualified_toban2)
    ][["race_date", "jcd", "rno"]].copy()


conn = sqlite3.connect(DB_PATH)

races = pd.read_sql_query(
    "SELECT race_date, jcd, rno, venue_name, weather, temperature, wind_speed, "
    "water_temp, wave_height, distance, grade FROM races", conn
)
entries = pd.read_sql_query(
    "SELECT race_date, jcd, rno, waku, toban, racer_name, racer_class, gender, age, weight, "
    "f_count, l_count, avg_st, national_win_rate, national_2rate, national_3rate, "
    "local_win_rate, local_2rate, local_3rate, motor_2rate, motor_3rate, "
    "boat_2rate, boat_3rate FROM entries", conn
)
results = pd.read_sql_query("SELECT race_date, jcd, rno, waku, rank FROM results", conn)

races["distance_m"] = races["distance"].apply(
    lambda s: float(re.sub(r"[^0-9.]", "", s)) if isinstance(s, str) and re.search(r"[0-9]", s) else np.nan
)

base = entries.merge(races, on=["race_date", "jcd", "rno"], how="inner")
base = base.merge(results, on=["race_date", "jcd", "rno", "waku"], how="inner")

base["is_win"] = (base["rank"] == "1").astype(int)

day_stats = base.groupby(["toban", "waku", "race_date"]).agg(
    day_starts=("is_win", "size"), day_wins=("is_win", "sum")
).reset_index().sort_values(["toban", "waku", "race_date"])
day_stats["cum_starts_incl"] = day_stats.groupby(["toban", "waku"])["day_starts"].cumsum()
day_stats["cum_wins_incl"] = day_stats.groupby(["toban", "waku"])["day_wins"].cumsum()
day_stats["course_starts_prior"] = day_stats["cum_starts_incl"] - day_stats["day_starts"]
day_stats["course_wins_prior"] = day_stats["cum_wins_incl"] - day_stats["day_wins"]
day_stats["course_win_rate_prior"] = np.where(
    day_stats["course_starts_prior"] > 0,
    day_stats["course_wins_prior"] / day_stats["course_starts_prior"],
    np.nan,
)
base = base.merge(
    day_stats[["toban", "waku", "race_date", "course_win_rate_prior", "course_starts_prior"]],
    on=["toban", "waku", "race_date"], how="left",
)

for col in NUMERIC_COLS:
    if col not in ("course_win_rate_prior", "course_starts_prior"):
        base[col] = pd.to_numeric(base[col], errors="coerce")
for col in CATEGORICAL_COLS:
    base[col] = base[col].astype("category")

X = base[FEATURE_COLS]
y = base["is_win"]

base["race_date_dt"] = pd.to_datetime(base["race_date"], format="%Y%m%d")
max_date = base["race_date_dt"].max()
cutoff = max_date - pd.Timedelta(days=TEST_DAYS)
train_mask = base["race_date_dt"] < cutoff
test_mask = ~train_mask

X_train, y_train = X[train_mask], y[train_mask]
X_test = X[test_mask]
meta_test = base.loc[test_mask, ["race_date", "jcd", "rno", "waku", "rank"]].reset_index(drop=True)

model = HistGradientBoostingClassifier(
    categorical_features="from_dtype",
    max_iter=300, learning_rate=0.05,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
    random_state=RANDOM_STATE,
)
t0 = time.time()
model.fit(X_train, y_train)
print(f"モデル学習完了({time.time() - t0:.1f}秒)")

meta_test["pred_proba"] = model.predict_proba(X_test)[:, 1]

# --- ①②条件に合致するレースを、全期間データから判定 ---
c1_stats, c2_stats = compute_racer_rate_stats(entries, results)
qualifying_races = find_qualifying_races(entries, c1_stats, c2_stats)

waku1_test = meta_test[meta_test["waku"] == 1].copy()
waku1_test["actual_win"] = (waku1_test["rank"] == "1").astype(int)

# テスト期間×①②条件に合致するレースのみに絞る
waku1_qualified = waku1_test.merge(
    qualifying_races, on=["race_date", "jcd", "rno"], how="inner"
)

n_all_test = len(waku1_test)
n_qualified = len(waku1_qualified)

print()
print("=== ①②条件に合致したレース(テスト期間内)での1号艇 予測 vs 実際 ===")
print(f"テスト期間の全レース数: {n_all_test}件")
print(f"うち①②条件に合致したレース数: {n_qualified}件")
print()
print(f"モデルが1号艇に予測した勝率(平均): {waku1_qualified['pred_proba'].mean() * 100:.1f}%")
print(f"実際に1号艇が1着だった割合(実測的中率): "
      f"{waku1_qualified['actual_win'].mean() * 100:.1f}% "
      f"({int(waku1_qualified['actual_win'].sum())}/{n_qualified}件)")

print()
print("(参考)テスト期間の全レースでの1号艇 予測 vs 実際")
print(f"モデルが1号艇に予測した勝率(平均): {waku1_test['pred_proba'].mean() * 100:.1f}%")
print(f"実際に1号艇が1着だった割合: {waku1_test['actual_win'].mean() * 100:.1f}% "
      f"({int(waku1_test['actual_win'].sum())}/{n_all_test}件)")

# --- モデルが1号艇に80%以上の勝率を予測したレースだけに絞る ---
high_conf_all = waku1_test[waku1_test["pred_proba"] >= 0.8]
high_conf_qualified = waku1_qualified[waku1_qualified["pred_proba"] >= 0.8]

print()
print("=== モデルが1号艇に80%以上の勝率を予測したレース ===")
print(f"[テスト期間の全レースのうち] 該当レース数: {len(high_conf_all)}件 "
      f"(平均予測勝率{high_conf_all['pred_proba'].mean() * 100:.1f}%)")
if len(high_conf_all) > 0:
    print(f"  実際の的中率: {high_conf_all['actual_win'].mean() * 100:.1f}% "
          f"({int(high_conf_all['actual_win'].sum())}/{len(high_conf_all)}件)")

print(f"[さらに①②条件にも合致するレースのうち] 該当レース数: {len(high_conf_qualified)}件")
if len(high_conf_qualified) > 0:
    print(f"  平均予測勝率: {high_conf_qualified['pred_proba'].mean() * 100:.1f}%")
    print(f"  実際の的中率: {high_conf_qualified['actual_win'].mean() * 100:.1f}% "
          f"({int(high_conf_qualified['actual_win'].sum())}/{len(high_conf_qualified)}件)")
else:
    print("  該当レースなし")

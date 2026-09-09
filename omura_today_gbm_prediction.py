"""
本日(2026-09-09)の大村全レースのうち①②条件(1号艇イン逃げ率80%以上・
2号艇逃がし率50%以上)に合致するレースについて、GBM(HistGradientBoosting)
で2着・3着になりそうな艇を予測し、1着=1号艇固定・2-3着を上位3艇で
ボックスにした3連単6点を提案する。
"""
import re
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

DB_PATH = Path(__file__).parent / "data" / "boatrace.db"
RANDOM_STATE = 42
SAMPLE_SIZE_WARNING_THRESHOLD = 5
INN_NIGE_RATE_THRESHOLD = 0.8
NIGASHI_RATE_THRESHOLD = 0.5
TARGET_DATE = "20260909"
TARGET_JCD = "24"  # 大村

CATEGORICAL_COLS = ["racer_class", "gender", "venue_name", "weather", "grade"]
BASE_NUMERIC_COLS = [
    "waku", "age", "weight", "f_count", "l_count", "avg_st",
    "national_win_rate", "national_2rate", "national_3rate",
    "local_win_rate", "local_2rate", "local_3rate",
    "motor_2rate", "motor_3rate", "boat_2rate", "boat_3rate",
    "temperature", "wind_speed", "water_temp", "wave_height", "distance_m",
    "course_win_rate_prior", "course_starts_prior",
]
FEATURE_COLS = BASE_NUMERIC_COLS + CATEGORICAL_COLS

conn = sqlite3.connect(DB_PATH)
races = pd.read_sql_query(
    "SELECT race_date, jcd, rno, venue_name, weather, temperature, wind_speed, "
    "water_temp, wave_height, distance, grade FROM races", conn)
entries = pd.read_sql_query(
    "SELECT race_date, jcd, rno, waku, toban, racer_class, gender, age, weight, "
    "f_count, l_count, avg_st, national_win_rate, national_2rate, national_3rate, "
    "local_win_rate, local_2rate, local_3rate, motor_2rate, motor_3rate, "
    "boat_2rate, boat_3rate FROM entries", conn)
results = pd.read_sql_query("SELECT race_date, jcd, rno, waku, rank FROM results", conn)

races["distance_m"] = races["distance"].apply(
    lambda s: float(re.sub(r"[^0-9.]", "", s)) if isinstance(s, str) and re.search(r"[0-9]", s) else np.nan)

# ============ ①②条件の判定(全履歴ベース、従来通り) ============
waku1_rank_res = results[results["waku"] == 1][["race_date", "jcd", "rno", "rank"]].rename(columns={"rank": "waku1_rank"})
c1 = entries[entries["waku"] == 1].merge(results[results["waku"] == 1][["race_date", "jcd", "rno", "rank"]], on=["race_date", "jcd", "rno"])
c1_stats = c1.groupby("toban").agg(starts=("rank", "size"), wins=("rank", lambda s: (s == "1").sum()))
c1_stats["rate"] = c1_stats["wins"] / c1_stats["starts"]
qualified_toban1 = set(c1_stats[(c1_stats["rate"] >= INN_NIGE_RATE_THRESHOLD) & (c1_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)

c2 = entries[entries["waku"] == 2].merge(waku1_rank_res, on=["race_date", "jcd", "rno"])
c2_stats = c2.groupby("toban").agg(starts=("waku1_rank", "size"), nigasare=("waku1_rank", lambda s: (s == "1").sum()))
c2_stats["rate"] = c2_stats["nigasare"] / c2_stats["starts"]
qualified_toban2 = set(c2_stats[(c2_stats["rate"] >= NIGASHI_RATE_THRESHOLD) & (c2_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)

# 本日・大村の各レースの1号艇/2号艇
today_e1 = entries[(entries["race_date"] == TARGET_DATE) & (entries["jcd"] == TARGET_JCD) & (entries["waku"] == 1)][["rno", "toban"]].rename(columns={"toban": "toban1"})
today_e2 = entries[(entries["race_date"] == TARGET_DATE) & (entries["jcd"] == TARGET_JCD) & (entries["waku"] == 2)][["rno", "toban"]].rename(columns={"toban": "toban2"})
today_pairs = today_e1.merge(today_e2, on="rno", how="inner")
today_pairs["c1_ok"] = today_pairs["toban1"].isin(qualified_toban1)
today_pairs["c2_ok"] = today_pairs["toban2"].isin(qualified_toban2)
today_pairs["qualified"] = today_pairs["c1_ok"] & today_pairs["c2_ok"]

print("=== 本日(2026-09-09) 大村 各レースの①②判定 ===")
c1r = c1_stats["rate"].reindex(today_pairs["toban1"]).values
c2r = c2_stats["rate"].reindex(today_pairs["toban2"]).values
for i, row in today_pairs.reset_index(drop=True).iterrows():
    r1 = c1_stats.loc[row["toban1"], "rate"] if row["toban1"] in c1_stats.index else float("nan")
    r2 = c2_stats.loc[row["toban2"], "rate"] if row["toban2"] in c2_stats.index else float("nan")
    mark = "◎対象" if row["qualified"] else "  -"
    print(f"  {row['rno']}R: 1号艇逃げ率{r1*100:.1f}% 2号艇逃がし率{r2*100:.1f}% {mark}")

qualified_rnos = today_pairs.loc[today_pairs["qualified"], "rno"].tolist()
print(f"\n①②対象レース: {qualified_rnos}\n")

if not qualified_rnos:
    print("本日大村に①②対象レースはありません。")
    raise SystemExit

# ============ GBMモデルの学習(全レース・履歴全体を使用) ============
base = entries.merge(races, on=["race_date", "jcd", "rno"], how="inner")
base = base.merge(results, on=["race_date", "jcd", "rno", "waku"], how="left")  # 本日分はrankがNaNになる
base["race_date_dt"] = pd.to_datetime(base["race_date"], format="%Y%m%d")
base["is_win"] = (base["rank"] == "1").astype(float)
base["is_2nd"] = (base["rank"] == "2").astype(float)
base["is_3rd"] = (base["rank"] == "3").astype(float)

# コース別勝率(当日より前のみ、時系列安全)
hist = base[base["rank"].notna()].copy()
day_stats = hist.groupby(["toban", "waku", "race_date"]).agg(
    day_starts=("is_win", "size"), day_wins=("is_win", "sum")
).reset_index().sort_values(["toban", "waku", "race_date"])
day_stats["cum_starts_incl"] = day_stats.groupby(["toban", "waku"])["day_starts"].cumsum()
day_stats["cum_wins_incl"] = day_stats.groupby(["toban", "waku"])["day_wins"].cumsum()

# 学習用(履歴): 当日より前の累積(=自分の行を除く)
day_stats["course_starts_prior"] = day_stats["cum_starts_incl"] - day_stats["day_starts"]
day_stats["course_wins_prior"] = day_stats["cum_wins_incl"] - day_stats["day_wins"]
day_stats["course_win_rate_prior"] = np.where(
    day_stats["course_starts_prior"] > 0, day_stats["course_wins_prior"] / day_stats["course_starts_prior"], np.nan)
hist = hist.merge(day_stats[["toban", "waku", "race_date", "course_win_rate_prior", "course_starts_prior"]],
                   on=["toban", "waku", "race_date"], how="left")

# 本日分: これまでの全履歴の累積(cum_*_incl の最新値)をそのまま「事前情報」として使う
latest_cum = day_stats.sort_values("race_date").groupby(["toban", "waku"]).tail(1)
latest_cum = latest_cum.rename(columns={"cum_starts_incl": "course_starts_prior_today", "cum_wins_incl": "course_wins_prior_today"})
latest_cum["course_win_rate_prior_today"] = np.where(
    latest_cum["course_starts_prior_today"] > 0,
    latest_cum["course_wins_prior_today"] / latest_cum["course_starts_prior_today"], np.nan)

for col in BASE_NUMERIC_COLS:
    if col not in ("course_win_rate_prior", "course_starts_prior"):
        hist[col] = pd.to_numeric(hist[col], errors="coerce")
for col in CATEGORICAL_COLS:
    hist[col] = hist[col].astype("category")

t0 = time.time()
models = {}
for target in ["is_2nd", "is_3rd"]:
    X_train, y_train = hist[FEATURE_COLS], hist[target]
    model = HistGradientBoostingClassifier(
        categorical_features="from_dtype", max_iter=300, learning_rate=0.05,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    models[target] = model
print(f"モデル学習完了 ({time.time()-t0:.1f}秒, 学習データn={len(hist)})\n")

# ============ 本日・大村・①②対象レースの予測 ============
today_rows = base[(base["race_date"] == TARGET_DATE) & (base["jcd"] == TARGET_JCD) & (base["waku"].isin([2, 3, 4, 5, 6]))].copy()
today_rows = today_rows.merge(
    latest_cum[["toban", "waku", "course_win_rate_prior_today", "course_starts_prior_today"]],
    on=["toban", "waku"], how="left")
today_rows["course_win_rate_prior"] = today_rows["course_win_rate_prior_today"]
today_rows["course_starts_prior"] = today_rows["course_starts_prior_today"]

for col in BASE_NUMERIC_COLS:
    if col not in ("course_win_rate_prior", "course_starts_prior"):
        today_rows[col] = pd.to_numeric(today_rows[col], errors="coerce")
for col in CATEGORICAL_COLS:
    today_rows[col] = today_rows[col].astype("category").cat.set_categories(hist[col].cat.categories)

X_today = today_rows[FEATURE_COLS]
today_rows["p2"] = models["is_2nd"].predict_proba(X_today)[:, 1]
today_rows["p3"] = models["is_3rd"].predict_proba(X_today)[:, 1]
today_rows["score"] = today_rows["p2"] + today_rows["p3"]

print("=== ①②対象レースごとの3連単6点(1号艇固定・2-3着上位3艇ボックス)提案 ===\n")
for rno in qualified_rnos:
    sub = today_rows[today_rows["rno"] == rno].sort_values("score", ascending=False)
    print(f"--- 大村 {rno}R ---")
    for _, r in sub.iterrows():
        print(f"  {r['waku']}号艇(登録{r['toban']}): P(2着)={r['p2']*100:.1f}% P(3着)={r['p3']*100:.1f}% score={r['score']*100:.1f}")
    top3 = sub.head(3)["waku"].astype(int).tolist()
    combos = []
    for a in top3:
        for b in top3:
            if a != b:
                combos.append(f"1-{a}-{b}")
    print(f"  → 推奨6点: {', '.join(combos)}\n")

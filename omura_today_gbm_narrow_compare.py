"""
①②+実際に1号艇が逃げた母集団だけに絞ってGBMを学習し直し、全レース学習版
(omura_today_gbm_prediction.py)との予測の違いを比較する。
既存ファイルは変更しない。本日(2026-09-09)大村の①②対象レース(4R/9R/12R)が対象。
"""
import re
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

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

# ============ ①②条件判定(全履歴ベース、従来通り) ============
waku1_rank_res = results[results["waku"] == 1][["race_date", "jcd", "rno", "rank"]].rename(columns={"rank": "waku1_rank"})
c1 = entries[entries["waku"] == 1].merge(results[results["waku"] == 1][["race_date", "jcd", "rno", "rank"]], on=["race_date", "jcd", "rno"])
c1_stats = c1.groupby("toban").agg(starts=("rank", "size"), wins=("rank", lambda s: (s == "1").sum()))
c1_stats["rate"] = c1_stats["wins"] / c1_stats["starts"]
qualified_toban1 = set(c1_stats[(c1_stats["rate"] >= INN_NIGE_RATE_THRESHOLD) & (c1_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)

c2 = entries[entries["waku"] == 2].merge(waku1_rank_res, on=["race_date", "jcd", "rno"])
c2_stats = c2.groupby("toban").agg(starts=("waku1_rank", "size"), nigasare=("waku1_rank", lambda s: (s == "1").sum()))
c2_stats["rate"] = c2_stats["nigasare"] / c2_stats["starts"]
qualified_toban2 = set(c2_stats[(c2_stats["rate"] >= NIGASHI_RATE_THRESHOLD) & (c2_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)

# ①②+実際に1号艇が逃げた過去レースのキー(=②③今回の狭い学習母集団)
e1 = entries[entries["waku"] == 1][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban1"})
e2 = entries[entries["waku"] == 2][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban2"})
pairs_hist = e1.merge(e2, on=["race_date", "jcd", "rno"]).merge(waku1_rank_res, on=["race_date", "jcd", "rno"])
pairs_hist["qualified"] = pairs_hist["toban1"].isin(qualified_toban1) & pairs_hist["toban2"].isin(qualified_toban2)
narrow_keys = pairs_hist[pairs_hist["qualified"] & (pairs_hist["waku1_rank"] == "1")][["race_date", "jcd", "rno"]].reset_index(drop=True)
print(f"①②+実際escaped 母集団: {len(narrow_keys)}レース\n")

# 本日・大村の①②対象レース
today_e1 = entries[(entries["race_date"] == TARGET_DATE) & (entries["jcd"] == TARGET_JCD) & (entries["waku"] == 1)][["rno", "toban"]].rename(columns={"toban": "toban1"})
today_e2 = entries[(entries["race_date"] == TARGET_DATE) & (entries["jcd"] == TARGET_JCD) & (entries["waku"] == 2)][["rno", "toban"]].rename(columns={"toban": "toban2"})
today_pairs = today_e1.merge(today_e2, on="rno", how="inner")
today_pairs["qualified"] = today_pairs["toban1"].isin(qualified_toban1) & today_pairs["toban2"].isin(qualified_toban2)
qualified_rnos = today_pairs.loc[today_pairs["qualified"], "rno"].tolist()
print(f"本日大村①②対象レース: {qualified_rnos}\n")

# ============ 特徴量準備(共通) ============
base = entries.merge(races, on=["race_date", "jcd", "rno"], how="inner")
base = base.merge(results, on=["race_date", "jcd", "rno", "waku"], how="left")
base["is_2nd"] = (base["rank"] == "2").astype(float)
base["is_3rd"] = (base["rank"] == "3").astype(float)

hist_all = base[base["rank"].notna()].copy()
day_stats = hist_all.groupby(["toban", "waku", "race_date"]).agg(
    day_starts=("is_2nd", "size"), day_wins=("rank", lambda s: (s == "1").sum())
).reset_index().sort_values(["toban", "waku", "race_date"])
day_stats["cum_starts_incl"] = day_stats.groupby(["toban", "waku"])["day_starts"].cumsum()
day_stats["cum_wins_incl"] = day_stats.groupby(["toban", "waku"])["day_wins"].cumsum()
day_stats["course_starts_prior"] = day_stats["cum_starts_incl"] - day_stats["day_starts"]
day_stats["course_wins_prior"] = day_stats["cum_wins_incl"] - day_stats["day_wins"]
day_stats["course_win_rate_prior"] = np.where(
    day_stats["course_starts_prior"] > 0, day_stats["course_wins_prior"] / day_stats["course_starts_prior"], np.nan)
hist_all = hist_all.merge(day_stats[["toban", "waku", "race_date", "course_win_rate_prior", "course_starts_prior"]],
                           on=["toban", "waku", "race_date"], how="left")

latest_cum = day_stats.sort_values("race_date").groupby(["toban", "waku"]).tail(1)
latest_cum = latest_cum.rename(columns={"cum_starts_incl": "course_starts_prior_today", "cum_wins_incl": "course_wins_prior_today"})
latest_cum["course_win_rate_prior_today"] = np.where(
    latest_cum["course_starts_prior_today"] > 0,
    latest_cum["course_wins_prior_today"] / latest_cum["course_starts_prior_today"], np.nan)

for col in BASE_NUMERIC_COLS:
    if col not in ("course_win_rate_prior", "course_starts_prior"):
        hist_all[col] = pd.to_numeric(hist_all[col], errors="coerce")
for col in CATEGORICAL_COLS:
    hist_all[col] = hist_all[col].astype("category")

# 狭い母集団(①②+escaped、2-6号艇のみ)
narrow_hist = hist_all.merge(narrow_keys, on=["race_date", "jcd", "rno"], how="inner")
narrow_hist = narrow_hist[narrow_hist["waku"].isin([2, 3, 4, 5, 6])].reset_index(drop=True)
n_races_narrow = narrow_hist[["race_date", "jcd", "rno"]].drop_duplicates().shape[0]
print(f"①②限定モデルの学習データ: n={len(narrow_hist)}行 ({n_races_narrow}レース×最大5艇)\n")


def make_model():
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype", max_iter=300, learning_rate=0.05,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=RANDOM_STATE)


# ---- 前半/後半の簡易頑健性チェック(過学習していないか) ----
races_sorted = narrow_hist.sort_values(["race_date", "jcd", "rno"])[["race_date", "jcd", "rno"]].drop_duplicates().reset_index(drop=True)
mid = len(races_sorted) // 2
front_keys, back_keys = races_sorted.iloc[:mid], races_sorted.iloc[mid:]

print("=== ①②限定モデルの前半/後半 相互検証 ===")
for target in ["is_2nd", "is_3rd"]:
    for train_keys, test_keys, label in [(front_keys, back_keys, "前半学習→後半検証"), (back_keys, front_keys, "後半学習→前半検証")]:
        tr = narrow_hist.merge(train_keys, on=["race_date", "jcd", "rno"])
        te = narrow_hist.merge(test_keys, on=["race_date", "jcd", "rno"])
        m = make_model()
        m.fit(tr[FEATURE_COLS], tr[target])
        proba = m.predict_proba(te[FEATURE_COLS])[:, 1]
        auc = roc_auc_score(te[target], proba)
        print(f"  {target} {label}: train_n={len(tr)} test_n={len(te)} AUC={auc:.3f}")
print()

# ============ 本番: 全期間の狭い母集団で学習 ============
t0 = time.time()
models_broad, models_narrow = {}, {}
for target in ["is_2nd", "is_3rd"]:
    m_broad = make_model()
    m_broad.fit(hist_all[FEATURE_COLS], hist_all[target])
    models_broad[target] = m_broad

    m_narrow = make_model()
    m_narrow.fit(narrow_hist[FEATURE_COLS], narrow_hist[target])
    models_narrow[target] = m_narrow
print(f"広域モデル(n={len(hist_all)})・①②限定モデル(n={len(narrow_hist)}) 学習完了 ({time.time()-t0:.1f}秒)\n")

# 特徴量重要度(①②限定モデル, permutation importance)
from sklearn.inspection import permutation_importance
print("=== ①②限定モデルの特徴量重要度(is_2nd, permutation importance) ===")
pi = permutation_importance(models_narrow["is_2nd"], narrow_hist[FEATURE_COLS], narrow_hist["is_2nd"],
                             n_repeats=5, random_state=RANDOM_STATE, scoring="roc_auc")
order = np.argsort(pi.importances_mean)[::-1]
for i in order[:8]:
    print(f"  {FEATURE_COLS[i]}: {pi.importances_mean[i]:+.4f}")
print()

# ============ 本日・大村①②対象レースへの適用と比較 ============
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
    today_rows[col] = today_rows[col].astype("category").cat.set_categories(hist_all[col].cat.categories)

X_today = today_rows[FEATURE_COLS]
today_rows["p2_broad"] = models_broad["is_2nd"].predict_proba(X_today)[:, 1]
today_rows["p3_broad"] = models_broad["is_3rd"].predict_proba(X_today)[:, 1]
today_rows["score_broad"] = today_rows["p2_broad"] + today_rows["p3_broad"]
today_rows["p2_narrow"] = models_narrow["is_2nd"].predict_proba(X_today)[:, 1]
today_rows["p3_narrow"] = models_narrow["is_3rd"].predict_proba(X_today)[:, 1]
today_rows["score_narrow"] = today_rows["p2_narrow"] + today_rows["p3_narrow"]

print("=== 広域モデル vs ①②限定モデル: 本日大村①②対象レースの比較 ===\n")
for rno in qualified_rnos:
    sub = today_rows[today_rows["rno"] == rno].sort_values("score_narrow", ascending=False)
    print(f"--- 大村 {rno}R ---")
    for _, r in sub.iterrows():
        print(f"  {int(r['waku'])}号艇: [広域]P2={r['p2_broad']*100:.1f}% P3={r['p3_broad']*100:.1f}% score={r['score_broad']*100:.1f}"
              f"  →  [①②限定]P2={r['p2_narrow']*100:.1f}% P3={r['p3_narrow']*100:.1f}% score={r['score_narrow']*100:.1f}")
    top3_broad = sub.sort_values("score_broad", ascending=False).head(3)["waku"].astype(int).tolist()
    top3_narrow = sub.sort_values("score_narrow", ascending=False).head(3)["waku"].astype(int).tolist()
    print(f"  上位3艇: 広域={top3_broad}  ①②限定={top3_narrow}\n")

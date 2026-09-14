"""
①条件(選手ごとの1コースでのイン逃げ率)・②条件(選手ごとの2コースでの逃し率)と
全く同じ定義の数値を特徴量として追加し、追加前(v1)・追加後(v2)のモデルを
同じ条件で学習・比較する。

【重要な注意】①②の統計(c1_stats/c2_stats)はこれまでの分析と同じく、
entries/results全期間（学習期間だけでなくテスト期間の結果も含む）から計算した
選手ごとの通算値。そのためこの2特徴量には一部リーク（未来のテスト期間の結果が
統計に混ざる）が含まれる。これは①②ルール自体がこれまで全期間統計として
使われてきたことに合わせた設定であり、「①②ルールと全く同じ情報を渡したら
モデルはどう変わるか」を見るための意図的な比較。真に厳密な汎化性能を見たい
場合は、course_win_rate_prior（当日より前のデータのみを使う既存特徴量）を
使うべき。

v1: win_prob_model.pyと同じ特徴量セット
v2: v1 + inn_nige_rate_c1（①と同一定義）+ nigashi_rate_c2（②と同一定義）
"""
import re
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, log_loss

DB_PATH = Path(__file__).parent / "data" / "boatrace.db"
TEST_DAYS = 21
RANDOM_STATE = 42

SAMPLE_SIZE_WARNING_THRESHOLD = 10
INN_NIGE_RATE_THRESHOLD = 0.8
NIGASHI_RATE_THRESHOLD = 0.5

CATEGORICAL_COLS = ["racer_class", "gender", "venue_name", "weather", "grade"]
BASE_NUMERIC_COLS = [
    "waku", "age", "weight", "f_count", "l_count", "avg_st",
    "national_win_rate", "national_2rate", "national_3rate",
    "local_win_rate", "local_2rate", "local_3rate",
    "motor_2rate", "motor_3rate", "boat_2rate", "boat_3rate",
    "temperature", "wind_speed", "water_temp", "wave_height", "distance_m",
    "course_win_rate_prior", "course_starts_prior",
]
AB_FEATURE_COLS = ["inn_nige_rate_c1", "nigashi_rate_c2"]


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


conn = sqlite3.connect(DB_PATH)
races = pd.read_sql_query(
    "SELECT race_date, jcd, rno, venue_name, weather, temperature, wind_speed, "
    "water_temp, wave_height, distance, grade FROM races", conn
)
entries = pd.read_sql_query(
    "SELECT race_date, jcd, rno, waku, toban, racer_class, gender, age, weight, "
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

# 時系列安全なコース別勝率（既存特徴量、v1・v2共通）
day_stats = base.groupby(["toban", "waku", "race_date"]).agg(
    day_starts=("is_win", "size"), day_wins=("is_win", "sum")
).reset_index().sort_values(["toban", "waku", "race_date"])
day_stats["cum_starts_incl"] = day_stats.groupby(["toban", "waku"])["day_starts"].cumsum()
day_stats["cum_wins_incl"] = day_stats.groupby(["toban", "waku"])["day_wins"].cumsum()
day_stats["course_starts_prior"] = day_stats["cum_starts_incl"] - day_stats["day_starts"]
day_stats["course_wins_prior"] = day_stats["cum_wins_incl"] - day_stats["day_wins"]
day_stats["course_win_rate_prior"] = np.where(
    day_stats["course_starts_prior"] > 0,
    day_stats["course_wins_prior"] / day_stats["course_starts_prior"], np.nan,
)
base = base.merge(
    day_stats[["toban", "waku", "race_date", "course_win_rate_prior", "course_starts_prior"]],
    on=["toban", "waku", "race_date"], how="left",
)

# --- ①②と全く同じ定義の統計を、選手(toban)ごとに全レースへ付与 ---
c1_stats, c2_stats = compute_racer_rate_stats(entries, results)
base["inn_nige_rate_c1"] = base["toban"].map(c1_stats["rate"])
base["nigashi_rate_c2"] = base["toban"].map(c2_stats["rate"])

for col in BASE_NUMERIC_COLS:
    if col not in ("course_win_rate_prior", "course_starts_prior"):
        base[col] = pd.to_numeric(base[col], errors="coerce")
for col in CATEGORICAL_COLS:
    base[col] = base[col].astype("category")

base["race_date_dt"] = pd.to_datetime(base["race_date"], format="%Y%m%d")
max_date = base["race_date_dt"].max()
cutoff = max_date - pd.Timedelta(days=TEST_DAYS)
train_mask = base["race_date_dt"] < cutoff
test_mask = ~train_mask

qualifying_races = find_qualifying_races(entries, c1_stats, c2_stats)
meta_test_base = base.loc[test_mask, ["race_date", "jcd", "rno", "waku", "rank"]].reset_index(drop=True)
meta_test_base = meta_test_base.merge(
    qualifying_races.assign(qualified=True), on=["race_date", "jcd", "rno"], how="left"
)
meta_test_base["qualified"] = meta_test_base["qualified"].fillna(False)


def train_and_eval(feature_cols, label):
    X = base[feature_cols]
    y = base["is_win"]
    X_train, y_train = X[train_mask], y[train_mask]
    X_test = X[test_mask]

    model = HistGradientBoostingClassifier(
        categorical_features="from_dtype",
        max_iter=300, learning_rate=0.05,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    meta_test = meta_test_base.copy()
    meta_test["pred_proba"] = model.predict_proba(X_test)[:, 1]

    y_test = base.loc[test_mask, "is_win"]
    auc = roc_auc_score(y_test, meta_test["pred_proba"])
    ll = log_loss(y_test, meta_test["pred_proba"])

    n_races = n_model_correct = n_baseline_correct = 0
    for _, g in meta_test.groupby(["race_date", "jcd", "rno"]):
        if (g["rank"] == "1").sum() != 1:
            continue
        n_races += 1
        actual_winner_waku = g.loc[g["rank"] == "1", "waku"].iloc[0]
        model_pick = g.loc[g["pred_proba"].idxmax(), "waku"]
        n_model_correct += int(model_pick == actual_winner_waku)
        n_baseline_correct += int(actual_winner_waku == 1)

    waku1 = meta_test[meta_test["waku"] == 1].copy()
    waku1["actual_win"] = (waku1["rank"] == "1").astype(int)
    waku1_q = waku1[waku1["qualified"]]
    high_conf = waku1[waku1["pred_proba"] >= 0.8]
    high_conf_q = waku1_q[waku1_q["pred_proba"] >= 0.8]

    print(f"===== {label} =====")
    print(f"特徴量数: {len(feature_cols)}")
    print(f"ROC-AUC: {auc:.3f} / Log Loss: {ll:.3f}")
    print(f"レース単位1着的中率: モデル {n_model_correct}/{n_races} ({n_model_correct / n_races * 100:.1f}%)"
          f" / ベースライン(常に1号艇) {n_baseline_correct}/{n_races} ({n_baseline_correct / n_races * 100:.1f}%)"
          f" / 差分 {(n_model_correct - n_baseline_correct) / n_races * 100:+.1f}pt")
    print(f"①②合致レースでの1号艇: モデル平均予測勝率 {waku1_q['pred_proba'].mean() * 100:.1f}%"
          f" / 実際の的中率 {waku1_q['actual_win'].mean() * 100:.1f}%"
          f" ({int(waku1_q['actual_win'].sum())}/{len(waku1_q)}件)")
    print(f"モデルが1号艇に80%以上の勝率を予測したレース: {len(high_conf)}件"
          f"（平均予測{high_conf['pred_proba'].mean() * 100:.1f}%、実際の的中率"
          f"{high_conf['actual_win'].mean() * 100:.1f}%）")
    if len(high_conf_q) > 0:
        print(f"  うち①②条件にも合致: {len(high_conf_q)}件"
              f"（平均予測{high_conf_q['pred_proba'].mean() * 100:.1f}%、実際の的中率"
              f"{high_conf_q['actual_win'].mean() * 100:.1f}%）")
    else:
        print("  うち①②条件にも合致: 0件")

    rng = np.random.default_rng(RANDOM_STATE)
    sample_size = min(6000, len(X_test))
    sample_idx = rng.choice(len(X_test), size=sample_size, replace=False)
    perm = permutation_importance(
        model, X_test.iloc[sample_idx], y_test.iloc[sample_idx],
        scoring="roc_auc", n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1,
    )
    importance_df = pd.DataFrame({
        "feature": feature_cols, "importance_mean": perm.importances_mean,
    }).sort_values("importance_mean", ascending=False)
    print("上位10特徴量:")
    for _, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:<24} {row['importance_mean']:+.4f}")
    print()
    return importance_df


v1_cols = BASE_NUMERIC_COLS + CATEGORICAL_COLS
v2_cols = BASE_NUMERIC_COLS + AB_FEATURE_COLS + CATEGORICAL_COLS

t0 = time.time()
imp_v1 = train_and_eval(v1_cols, "v1: ①②特徴量なし(従来モデル)")
imp_v2 = train_and_eval(v2_cols, "v2: ①②と同一定義の特徴量あり")
print(f"(全体の実行時間: {time.time() - t0:.1f}秒)")

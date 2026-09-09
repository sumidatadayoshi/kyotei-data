"""
①②条件を前提として、2着(・3着)を予測する勾配ブースティングは組めるか。

方針: model_with_ab_features_noleak.py と同じデータ準備パイプラインを流用し、
目的変数だけ「1着か」から「2着か」に変更する。学習は①②に絞らず全レースを
使う(train broadの方針、データ量を確保するため)。評価だけを①②条件かつ
実際に1号艇が逃げたレースに絞り、2号艇・3号艇それぞれについて
「モデルの予測確率」と「実際に2着に来た率」を比較する。
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

SAMPLE_SIZE_WARNING_THRESHOLD = 5
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
    c2 = entries_all[entries_all["waku"] == 2].merge(waku1_rank, on=["race_date", "jcd", "rno"], how="inner")
    c2_stats = c2.groupby("toban").agg(starts=("waku1_rank", "size"), nigasare=("waku1_rank", lambda s: (s == "1").sum()))
    c2_stats["rate"] = c2_stats["nigasare"] / c2_stats["starts"]
    return c1_stats, c2_stats


def find_qualifying_races(entries_df, c1_stats, c2_stats):
    qualified_toban1 = set(c1_stats[(c1_stats["rate"] >= INN_NIGE_RATE_THRESHOLD) & (c1_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)
    qualified_toban2 = set(c2_stats[(c2_stats["rate"] >= NIGASHI_RATE_THRESHOLD) & (c2_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)
    entries1 = entries_df[entries_df["waku"] == 1][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban1"})
    entries2 = entries_df[entries_df["waku"] == 2][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban2"})
    race_pairs = entries1.merge(entries2, on=["race_date", "jcd", "rno"], how="inner")
    return race_pairs[race_pairs["toban1"].isin(qualified_toban1) & race_pairs["toban2"].isin(qualified_toban2)][
        ["race_date", "jcd", "rno"]].copy()


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

base = entries.merge(races, on=["race_date", "jcd", "rno"], how="inner")
base = base.merge(results, on=["race_date", "jcd", "rno", "waku"], how="inner")
base["is_win"] = (base["rank"] == "1").astype(int)
base["is_2nd"] = (base["rank"] == "2").astype(int)
base["race_date_dt"] = pd.to_datetime(base["race_date"], format="%Y%m%d")

day_stats = base.groupby(["toban", "waku", "race_date"]).agg(
    day_starts=("is_win", "size"), day_wins=("is_win", "sum")
).reset_index().sort_values(["toban", "waku", "race_date"])
day_stats["cum_starts_incl"] = day_stats.groupby(["toban", "waku"])["day_starts"].cumsum()
day_stats["cum_wins_incl"] = day_stats.groupby(["toban", "waku"])["day_wins"].cumsum()
day_stats["course_starts_prior"] = day_stats["cum_starts_incl"] - day_stats["day_starts"]
day_stats["course_wins_prior"] = day_stats["cum_wins_incl"] - day_stats["day_wins"]
day_stats["course_win_rate_prior"] = np.where(
    day_stats["course_starts_prior"] > 0, day_stats["course_wins_prior"] / day_stats["course_starts_prior"], np.nan)
base = base.merge(day_stats[["toban", "waku", "race_date", "course_win_rate_prior", "course_starts_prior"]],
                   on=["toban", "waku", "race_date"], how="left")

for col in BASE_NUMERIC_COLS:
    if col not in ("course_win_rate_prior", "course_starts_prior"):
        base[col] = pd.to_numeric(base[col], errors="coerce")
for col in CATEGORICAL_COLS:
    base[col] = base[col].astype("category")

max_date = base["race_date_dt"].max()
cutoff = max_date - pd.Timedelta(days=TEST_DAYS)
train_mask = base["race_date_dt"] < cutoff
test_mask = ~train_mask

c1_stats_full, c2_stats_full = compute_racer_rate_stats(entries, results)
qualifying_races = find_qualifying_races(entries, c1_stats_full, c2_stats_full)

waku1_rank_res = results[results["waku"] == 1][["race_date", "jcd", "rno", "rank"]].rename(columns={"rank": "waku1_rank"})
escaped_races = waku1_rank_res[waku1_rank_res["waku1_rank"] == "1"][["race_date", "jcd", "rno"]]
qual_and_escaped = qualifying_races.merge(escaped_races, on=["race_date", "jcd", "rno"], how="inner")

feature_cols = BASE_NUMERIC_COLS + CATEGORICAL_COLS
X = base[feature_cols]
y = base["is_2nd"]
X_train, y_train = X[train_mask], y[train_mask]
X_test = X[test_mask]

t0 = time.time()
model = HistGradientBoostingClassifier(
    categorical_features="from_dtype", max_iter=300, learning_rate=0.05,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=RANDOM_STATE)
model.fit(X_train, y_train)
print(f"学習時間: {time.time()-t0:.1f}秒 (train n={len(X_train)}, test n={len(X_test)})\n")

meta_test = base.loc[test_mask, ["race_date", "jcd", "rno", "waku", "rank", "is_2nd"]].reset_index(drop=True)
meta_test["pred_proba"] = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(meta_test["is_2nd"], meta_test["pred_proba"])
ll = log_loss(meta_test["is_2nd"], meta_test["pred_proba"])
print(f"=== 「2着になるか」を予測するモデル(全レースで学習, n_train={len(X_train)}) ===")
print(f"ROC-AUC: {auc:.3f} / Log Loss: {ll:.3f}\n")

meta_test = meta_test.merge(qual_and_escaped.assign(qualified_escaped=True),
                             on=["race_date", "jcd", "rno"], how="left")
meta_test["qualified_escaped"] = meta_test["qualified_escaped"].fillna(False)

sub23 = meta_test[meta_test["qualified_escaped"] & meta_test["waku"].isin([2, 3])].copy()
print(f"=== ①②条件+1号艇が実際に逃げたレースでの、2号艇/3号艇のモデル予測 vs 実際 (n={len(sub23)}レース分艇) ===")
for w in [2, 3]:
    d = sub23[sub23["waku"] == w]
    n = len(d)
    if n == 0:
        continue
    print(f"  {w}号艇: n={n} モデル平均予測確率{d['pred_proba'].mean()*100:.1f}% 実際の2着率{d['is_2nd'].mean()*100:.1f}%")

# レースごとに2号艇/3号艇の予測確率を比較し、「モデルがどちらを2着と予想したか」の的中率
pivot = sub23.pivot_table(index=["race_date", "jcd", "rno"], columns="waku", values=["pred_proba", "is_2nd"])
pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
pivot = pivot.dropna(subset=["pred_proba_2", "pred_proba_3", "is_2nd_2", "is_2nd_3"])
pivot = pivot[(pivot["is_2nd_2"] + pivot["is_2nd_3"]) == 1]  # 2着が2号艇か3号艇のどちらかだったレースのみ
n_races = len(pivot)
model_picks_2 = pivot["pred_proba_2"] > pivot["pred_proba_3"]
correct = np.where(model_picks_2, pivot["is_2nd_2"] == 1, pivot["is_2nd_3"] == 1)
baseline_picks_2_rate = pivot["is_2nd_2"].mean()
print(f"\n=== レース単位: モデルが2号艇/3号艇どちらを2着と予想したかの的中率 (n={n_races}) ===")
print(f"  モデルの的中率: {correct.mean()*100:.1f}%")
print(f"  ベースライン(常に2号艇を選ぶ): {baseline_picks_2_rate*100:.1f}%")
print(f"  ベースライン(常に確率高い方=多数派側を選ぶ場合と同じ)")

sample_size = min(6000, len(X_test))
rng = np.random.default_rng(RANDOM_STATE)
sample_idx = rng.choice(len(X_test), size=sample_size, replace=False)
perm = permutation_importance(model, X_test.iloc[sample_idx], meta_test["is_2nd"].iloc[sample_idx],
                               scoring="roc_auc", n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
importance_df = pd.DataFrame({"feature": feature_cols, "importance_mean": perm.importances_mean}).sort_values(
    "importance_mean", ascending=False)
print("\n上位10特徴量:")
for _, row in importance_df.head(10).iterrows():
    print(f"  {row['feature']:<24} {row['importance_mean']:+.4f}")


# ============ 回収率換算: モデルの予測を使った動的配分 vs 固定配分 ============
def bb_recovery_varstake(df, ret, stake_series, n_resamples=2000, seed=42):
    block_key = df["race_date"].astype(str) + "_" + df["jcd"].astype(str)
    d = pd.DataFrame({"block": block_key, "return": np.asarray(ret), "stake": np.asarray(stake_series)})
    blocks = d.groupby("block").agg(stake_sum=("stake", "sum"), return_sum=("return", "sum"))
    s_arr, ret_arr = blocks["stake_sum"].to_numpy(), blocks["return_sum"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(blocks), size=(n_resamples, len(blocks)))
    rs, rr = s_arr[idx].sum(axis=1), ret_arr[idx].sum(axis=1)
    rates = np.where(rs > 0, rr / rs * 100, 0.0)
    lo, hi = np.percentile(rates, [2.5, 97.5])
    point = ret_arr.sum() / s_arr.sum() * 100
    return point, lo, hi


payouts_2tan = pd.read_sql_query("SELECT race_date, jcd, rno, combination, payout FROM payouts WHERE bet_type = '2連単'", conn)
p12 = payouts_2tan[payouts_2tan["combination"] == "1-2"][["race_date", "jcd", "rno", "payout"]].rename(columns={"payout": "p12"})
p13 = payouts_2tan[payouts_2tan["combination"] == "1-3"][["race_date", "jcd", "rno", "payout"]].rename(columns={"payout": "p13"})

bt = qual_and_escaped.merge(p12, on=["race_date", "jcd", "rno"], how="left").merge(p13, on=["race_date", "jcd", "rno"], how="left")
bt["p12"] = bt["p12"].fillna(0)
bt["p13"] = bt["p13"].fillna(0)
# テスト期間のレースのみ(モデルの予測が使えるのはtest_maskの範囲だけ)
bt["race_date_dt"] = pd.to_datetime(bt["race_date"], format="%Y%m%d")
bt = bt[bt["race_date_dt"] >= cutoff]

pred2 = sub23[sub23["waku"] == 2][["race_date", "jcd", "rno", "pred_proba"]].rename(columns={"pred_proba": "proba2"})
pred3 = sub23[sub23["waku"] == 3][["race_date", "jcd", "rno", "pred_proba"]].rename(columns={"pred_proba": "proba3"})
bt = bt.merge(pred2, on=["race_date", "jcd", "rno"], how="inner").merge(pred3, on=["race_date", "jcd", "rno"], how="inner")

bt["fixed_ret"] = bt["p12"] * 2 + bt["p13"] * 1
bt["dyn_ret"] = np.where(bt["proba2"] >= bt["proba3"], bt["p12"] * 2 + bt["p13"] * 1, bt["p12"] * 1 + bt["p13"] * 2)

print(f"\n=== テスト期間の①②+逃げレースで回収率換算 (n={len(bt)}) ===")
p, lo, hi = bb_recovery_varstake(bt, bt["fixed_ret"], pd.Series(300, index=bt.index))
print(f"  固定(1-2:200+1-3:100): 回収率{p:.1f}%[{lo:.1f}-{hi:.1f}]")
p, lo, hi = bb_recovery_varstake(bt, bt["dyn_ret"], pd.Series(300, index=bt.index))
print(f"  動的(GBM予測に基づき厚い方を200円): 回収率{p:.1f}%[{lo:.1f}-{hi:.1f}]")
print(f"  (モデルが2号艇favor: {(bt['proba2']>=bt['proba3']).sum()}件 / 3号艇favor: {(bt['proba2']<bt['proba3']).sum()}件)")

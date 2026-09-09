"""
3着を諦めて「2着のみ」を当てにいく場合の予想。①②+実際に1号艇が逃げたレースで、
2-6号艇のうち誰が2着に来るかをGBMでランク付けし、
  (a) モデル1位を1点勝負(2連単 1-X に300円)
  (b) モデル1位・2位に200円/100円を配分(2連単ナガシ、既存の固定1-2:200+1-3:100を一般化)
の2通りを、既存の固定(1-2:200+1-3:100)と回収率で比較する。
モデルは「広域学習(全レース)」と「①②+escaped限定学習」の両方を試す。
second_place_gbm.py と同じ TEST_DAYS=21 の時系列ホールドアウトで評価。
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
payouts_2tan = pd.read_sql_query("SELECT race_date, jcd, rno, combination, payout FROM payouts WHERE bet_type = '2連単'", conn)

races["distance_m"] = races["distance"].apply(
    lambda s: float(re.sub(r"[^0-9.]", "", s)) if isinstance(s, str) and re.search(r"[0-9]", s) else np.nan)

# ============ ①②条件 + 実escape 母集団(全履歴ベース、従来通り) ============
waku1_rank_res = results[results["waku"] == 1][["race_date", "jcd", "rno", "rank"]].rename(columns={"rank": "waku1_rank"})
c1 = entries[entries["waku"] == 1].merge(results[results["waku"] == 1][["race_date", "jcd", "rno", "rank"]], on=["race_date", "jcd", "rno"])
c1_stats = c1.groupby("toban").agg(starts=("rank", "size"), wins=("rank", lambda s: (s == "1").sum()))
c1_stats["rate"] = c1_stats["wins"] / c1_stats["starts"]
qualified_toban1 = set(c1_stats[(c1_stats["rate"] >= INN_NIGE_RATE_THRESHOLD) & (c1_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)

c2 = entries[entries["waku"] == 2].merge(waku1_rank_res, on=["race_date", "jcd", "rno"])
c2_stats = c2.groupby("toban").agg(starts=("waku1_rank", "size"), nigasare=("waku1_rank", lambda s: (s == "1").sum()))
c2_stats["rate"] = c2_stats["nigasare"] / c2_stats["starts"]
qualified_toban2 = set(c2_stats[(c2_stats["rate"] >= NIGASHI_RATE_THRESHOLD) & (c2_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)

e1 = entries[entries["waku"] == 1][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban1"})
e2 = entries[entries["waku"] == 2][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban2"})
pairs_hist = e1.merge(e2, on=["race_date", "jcd", "rno"]).merge(waku1_rank_res, on=["race_date", "jcd", "rno"])
pairs_hist["qualified"] = pairs_hist["toban1"].isin(qualified_toban1) & pairs_hist["toban2"].isin(qualified_toban2)
narrow_keys = pairs_hist[pairs_hist["qualified"] & (pairs_hist["waku1_rank"] == "1")][["race_date", "jcd", "rno"]].reset_index(drop=True)
print(f"①②+実際escaped 母集団: {len(narrow_keys)}レース\n")

# ============ 特徴量準備 ============
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
print(f"最終日={max_date.date()} / テスト期間カットオフ={cutoff.date()} (直近{TEST_DAYS}日)\n")

feature_cols = FEATURE_COLS


def make_model():
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype", max_iter=300, learning_rate=0.05,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=RANDOM_STATE)


# ---- 広域モデル: 全レースで学習(train期間) ----
t0 = time.time()
model_broad = make_model()
model_broad.fit(base.loc[train_mask, feature_cols], base.loc[train_mask, "is_2nd"])
print(f"広域モデル学習完了 ({time.time()-t0:.1f}秒, train_n={train_mask.sum()})")

# ---- ①②限定モデル: narrow_keysのレースのみ・かつtrain期間で学習 ----
narrow_mask = base.merge(narrow_keys.assign(_n=1), on=["race_date", "jcd", "rno"], how="left")["_n"].notna().values
narrow_train_mask = narrow_mask & train_mask.values & base["waku"].isin([2, 3, 4, 5, 6]).values
t0 = time.time()
model_narrow = make_model()
model_narrow.fit(base.loc[narrow_train_mask, feature_cols], base.loc[narrow_train_mask, "is_2nd"])
print(f"①②限定モデル学習完了 ({time.time()-t0:.1f}秒, train_n={narrow_train_mask.sum()})\n")

# ============ テスト期間: ①②+escapedレースの2-6号艇に予測を付与 ============
test_sub = base[test_mask & narrow_mask & base["waku"].isin([2, 3, 4, 5, 6])].copy()
test_sub["proba_broad"] = model_broad.predict_proba(test_sub[feature_cols])[:, 1]
test_sub["proba_narrow"] = model_narrow.predict_proba(test_sub[feature_cols])[:, 1]

n_test_races = test_sub[["race_date", "jcd", "rno"]].drop_duplicates().shape[0]
print(f"テスト期間の①②+escapedレース: {n_test_races}レース (5艇×{n_test_races}={len(test_sub)}行)\n")


def evaluate_picks(df, proba_col, label):
    picks = df.loc[df.groupby(["race_date", "jcd", "rno"])[proba_col].idxmax()]
    top1_acc = picks["is_2nd"].mean()
    baseline_acc = df[df["waku"] == 2].set_index(["race_date", "jcd", "rno"])["is_2nd"].mean()

    def top2(g):
        g2 = g.sort_values(proba_col, ascending=False)
        return pd.Series({"waku1": g2.iloc[0]["waku"], "waku2": g2.iloc[1]["waku"]})
    top2_keys = df.groupby(["race_date", "jcd", "rno"]).apply(top2).reset_index()
    print(f"[{label}] 1位ピック的中率={top1_acc*100:.1f}% (n={len(picks)})  ベースライン(常に2号艇)={baseline_acc*100:.1f}%")
    return picks, top2_keys


picks_broad, top2_broad = evaluate_picks(test_sub, "proba_broad", "広域モデル")
picks_narrow, top2_narrow = evaluate_picks(test_sub, "proba_narrow", "①②限定モデル")

# ============ 回収率換算 ============
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


pay = {}
for w in [2, 3, 4, 5, 6]:
    pay[w] = payouts_2tan[payouts_2tan["combination"] == f"1-{w}"][["race_date", "jcd", "rno", "payout"]].rename(columns={"payout": f"pay_{w}"})

bt = narrow_keys.merge(pd.DataFrame({"race_date": base["race_date"], "jcd": base["jcd"], "rno": base["rno"]}).drop_duplicates(),
                        on=["race_date", "jcd", "rno"], how="inner")
bt["race_date_dt"] = pd.to_datetime(bt["race_date"], format="%Y%m%d")
bt = bt[bt["race_date_dt"] >= cutoff]
for w in [2, 3, 4, 5, 6]:
    bt = bt.merge(pay[w], on=["race_date", "jcd", "rno"], how="left")
    bt[f"pay_{w}"] = bt[f"pay_{w}"].fillna(0)

bt["fixed_ret"] = bt["pay_2"] * 2 + bt["pay_3"] * 1

for label, picks, top2_keys in [("広域モデル", picks_broad, top2_broad), ("①②限定モデル", picks_narrow, top2_narrow)]:
    key = ["race_date", "jcd", "rno"]
    t2 = top2_keys.rename(columns={"waku1": "pick1", "waku2": "pick2"})
    b = bt.merge(t2, on=key, how="inner")

    def pay_of(row, w):
        w = int(w)
        return row[f"pay_{w}"]

    b["single_ret"] = b.apply(lambda r: pay_of(r, r["pick1"]) * 3, axis=1)  # 300円1点
    b["nagashi_ret"] = b.apply(lambda r: pay_of(r, r["pick1"]) * 2 + pay_of(r, r["pick2"]) * 1, axis=1)  # 200/100配分

    print(f"\n=== {label}: テスト期間①②+escapedレースでの回収率換算 (n={len(b)}) ===")
    p, lo, hi = bb_recovery_varstake(b, b["fixed_ret"], pd.Series(300, index=b.index))
    print(f"  固定(1-2:200+1-3:100)          : 回収率{p:.1f}%[{lo:.1f}-{hi:.1f}]")
    p, lo, hi = bb_recovery_varstake(b, b["single_ret"], pd.Series(300, index=b.index))
    print(f"  モデル1位に300円1点勝負        : 回収率{p:.1f}%[{lo:.1f}-{hi:.1f}]")
    p, lo, hi = bb_recovery_varstake(b, b["nagashi_ret"], pd.Series(300, index=b.index))
    print(f"  モデル上位2点(200円/100円配分) : 回収率{p:.1f}%[{lo:.1f}-{hi:.1f}]")
    print(f"  (1位ピックの内訳: {b['pick1'].value_counts().sort_index().to_dict()})")

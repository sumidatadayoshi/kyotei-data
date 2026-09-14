"""
second_place_only_pick.py の続き。①②限定モデルのランキングに従って、
上位3点(2連単)に500円/300円/100円を配分した場合の回収率を検証する。
既存の「固定(1-2:200+1-3:100)」「上位2点(200/100)」と比較する。
合計賭け金が900円と300円で異なるので、bb_recovery_varstakeは
「賭けた金額に対する回収率(%)」なので金額差は正規化されて比較可能。
"""
import re
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

DB_PATH = "/mnt/user-data/uploads/race-info/data/boatrace.db"
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
print(f"最終日={max_date.date()} / カットオフ={cutoff.date()}")

narrow_mask = base.merge(narrow_keys.assign(_n=1), on=["race_date", "jcd", "rno"], how="left")["_n"].notna().values
narrow_train_mask = narrow_mask & train_mask.values & base["waku"].isin([2, 3, 4, 5, 6]).values

model_narrow = HistGradientBoostingClassifier(
    categorical_features="from_dtype", max_iter=300, learning_rate=0.05,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=RANDOM_STATE)
model_narrow.fit(base.loc[narrow_train_mask, FEATURE_COLS], base.loc[narrow_train_mask, "is_2nd"])
print(f"①②限定モデル学習完了 (train_n={narrow_train_mask.sum()})")

test_sub = base[test_mask & narrow_mask & base["waku"].isin([2, 3, 4, 5, 6])].copy()
test_sub["proba_narrow"] = model_narrow.predict_proba(test_sub[FEATURE_COLS])[:, 1]
n_test_races = test_sub[["race_date", "jcd", "rno"]].drop_duplicates().shape[0]
print(f"テスト期間の①②+escapedレース: {n_test_races}レース\n")


def top_n(g, n, proba_col="proba_narrow"):
    g2 = g.sort_values(proba_col, ascending=False)
    return g2["waku"].astype(int).tolist()[:n]


top3_keys = test_sub.groupby(["race_date", "jcd", "rno"]).apply(
    lambda g: pd.Series({"pick1": top_n(g, 3)[0], "pick2": top_n(g, 3)[1], "pick3": top_n(g, 3)[2]}),
    include_groups=False,
).reset_index()


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

bt["fixed_ret"] = bt["pay_2"] * 2 + bt["pay_3"] * 1  # 1-2:200円 + 1-3:100円 (合計300円)

b = bt.merge(top3_keys, on=["race_date", "jcd", "rno"], how="inner")


def pay_of(row, w):
    return row[f"pay_{int(w)}"]


# 上位2点(200/100) 既存戦略(合計300円)
b["top2_200_100"] = b.apply(lambda r: pay_of(r, r["pick1"]) * 2 + pay_of(r, r["pick2"]) * 1, axis=1)
# 上位3点(500/300/100) 新戦略(合計900円)
b["top3_500_300_100"] = b.apply(lambda r: pay_of(r, r["pick1"]) * 5 + pay_of(r, r["pick2"]) * 3 + pay_of(r, r["pick3"]) * 1, axis=1)
# 参考: 上位3点を均等(300/300/300、合計900円)にした場合
b["top3_equal_300"] = b.apply(lambda r: (pay_of(r, r["pick1"]) + pay_of(r, r["pick2"]) + pay_of(r, r["pick3"])) * 3, axis=1)

print(f"=== ①②限定モデル: テスト期間①②+escapedレースでの回収率比較 (n={len(b)}) ===")
p, lo, hi = bb_recovery_varstake(b, b["fixed_ret"], pd.Series(300, index=b.index))
print(f"  固定(1-2:200+1-3:100, 合計300円)          : 回収率{p:.1f}%[{lo:.1f}-{hi:.1f}]")
p, lo, hi = bb_recovery_varstake(b, b["top2_200_100"], pd.Series(300, index=b.index))
print(f"  上位2点(200/100, 合計300円)                : 回収率{p:.1f}%[{lo:.1f}-{hi:.1f}]")
p, lo, hi = bb_recovery_varstake(b, b["top3_500_300_100"], pd.Series(900, index=b.index))
print(f"  上位3点(500/300/100, 合計900円)            : 回収率{p:.1f}%[{lo:.1f}-{hi:.1f}]")
p, lo, hi = bb_recovery_varstake(b, b["top3_equal_300"], pd.Series(900, index=b.index))
print(f"  上位3点均等(300/300/300, 合計900円, 参考)  : 回収率{p:.1f}%[{lo:.1f}-{hi:.1f}]")
print(f"\n  (pick1内訳: {b['pick1'].value_counts().sort_index().to_dict()})")
print(f"  (pick3内訳: {b['pick3'].value_counts().sort_index().to_dict()})")

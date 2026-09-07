"""
探索その5-B: 逃げが決まったレースで、2着が2号艇になるか3号艇になるかを
事前情報(平均ST・モーター成績・勝率・体重など)から予測できるか検証する。
現状は1-2:200円/1-3:100円で常に2号艇を優先しているが、これが逆転する
条件があれば、レースごとに重みを動的に変える余地がある。
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "boatrace.db"
SAMPLE_SIZE_WARNING_THRESHOLD = 5
INN_NIGE_RATE_THRESHOLD = 0.8
NIGASHI_RATE_THRESHOLD = 0.5


def compute_racer_rate_stats(entries_all, results_all):
    c1 = entries_all[entries_all["waku"] == 1].merge(
        results_all[results_all["waku"] == 1][["race_date", "jcd", "rno", "rank"]],
        on=["race_date", "jcd", "rno"], how="inner")
    c1_stats = c1.groupby("toban").agg(starts=("rank", "size"), wins=("rank", lambda s: (s == "1").sum()))
    c1_stats["rate"] = c1_stats["wins"] / c1_stats["starts"]
    waku1_rank = results_all[results_all["waku"] == 1][["race_date", "jcd", "rno", "rank"]].rename(columns={"rank": "waku1_rank"})
    c2 = entries_all[entries_all["waku"] == 2].merge(waku1_rank, on=["race_date", "jcd", "rno"], how="inner")
    c2_stats = c2.groupby("toban").agg(starts=("waku1_rank", "size"), nigasare=("waku1_rank", lambda s: (s == "1").sum()))
    c2_stats["rate"] = c2_stats["nigasare"] / c2_stats["starts"]
    return c1_stats, c2_stats, waku1_rank


def find_qualifying_races(entries_df, c1_stats, c2_stats):
    q1 = set(c1_stats[(c1_stats["rate"] >= INN_NIGE_RATE_THRESHOLD) & (c1_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)
    q2 = set(c2_stats[(c2_stats["rate"] >= NIGASHI_RATE_THRESHOLD) & (c2_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)
    e1 = entries_df[entries_df["waku"] == 1][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban1"})
    e2 = entries_df[entries_df["waku"] == 2][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban2"})
    pairs = e1.merge(e2, on=["race_date", "jcd", "rno"], how="inner")
    return pairs[pairs["toban1"].isin(q1) & pairs["toban2"].isin(q2)].copy()


conn = sqlite3.connect(DB_PATH)
entries_all = pd.read_sql_query("SELECT * FROM entries", conn)
results_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, rank FROM results", conn)
payouts_2tan = pd.read_sql_query("SELECT race_date, jcd, rno, combination, payout FROM payouts WHERE bet_type = '2連単'", conn)

c1_stats, c2_stats, waku1_rank = compute_racer_rate_stats(entries_all, results_all)
candidates = find_qualifying_races(entries_all, c1_stats, c2_stats)
concluded = candidates.merge(payouts_2tan, on=["race_date", "jcd", "rno"], how="inner")

feature_cols = ["avg_st", "motor_2rate", "motor_3rate", "national_win_rate", "local_win_rate", "weight", "age"]
wide = entries_all.pivot_table(index=["race_date", "jcd", "rno"], columns="waku", values=feature_cols)
wide.columns = [f"{col}_{int(waku)}" for col, waku in wide.columns]
wide = wide.reset_index()
concluded = concluded.merge(wide, on=["race_date", "jcd", "rno"], how="left")
concluded = concluded.sort_values(["race_date", "rno", "jcd"]).reset_index(drop=True)

duel = concluded[concluded["combination"].isin(["1-2", "1-3"])].copy()
duel["is2"] = (duel["combination"] == "1-2").astype(int)
total = len(duel)
base_rate = duel["is2"].mean() * 100
print(f"1-2 or 1-3 で決着したレース: {total}件 (うち1-2: {int(duel['is2'].sum())}件={base_rate:.1f}%, "
      f"1-3: {total-int(duel['is2'].sum())}件={100-base_rate:.1f}%)\n")

duel["st_diff"] = duel["avg_st_3"] - duel["avg_st_2"]        # 正=2号艇の方がスタート速い
duel["motor2_diff"] = duel["motor_2rate_2"] - duel["motor_2rate_3"]
duel["motor3_diff"] = duel["motor_3rate_2"] - duel["motor_3rate_3"]
duel["national_diff"] = duel["national_win_rate_2"] - duel["national_win_rate_3"]
duel["local_diff"] = duel["local_win_rate_2"] - duel["local_win_rate_3"]
duel["weight_diff"] = duel["weight_3"] - duel["weight_2"]     # 正=2号艇の方が軽い

print("=== 各特徴量とis2(2号艇が2着)の相関係数 ===")
for col in ["st_diff", "motor2_diff", "motor3_diff", "national_diff", "local_diff", "weight_diff"]:
    sub = duel.dropna(subset=[col, "is2"])
    corr = np.corrcoef(sub[col], sub["is2"])[0, 1]
    print(f"  {col}: r={corr:+.3f} (n={len(sub)})")

print()
print("=== st_diff(正=2号艇の方がスタート速い) 5分位ごとのis2率 ===")
duel_st = duel.dropna(subset=["st_diff"]).copy()
duel_st["bin"] = pd.qcut(duel_st["st_diff"], q=5, duplicates="drop")
for b, sub in duel_st.groupby("bin", observed=True):
    print(f"  {b}: n={len(sub)} is2率={sub['is2'].mean()*100:.1f}%")

print()
print("=== motor2_diff(正=2号艇のモーターの方が良い) 5分位ごとのis2率 ===")
duel_mo = duel.dropna(subset=["motor2_diff"]).copy()
duel_mo["bin"] = pd.qcut(duel_mo["motor2_diff"], q=5, duplicates="drop")
for b, sub in duel_mo.groupby("bin", observed=True):
    print(f"  {b}: n={len(sub)} is2率={sub['is2'].mean()*100:.1f}%")

print()
print("=== ロジスティック回帰(時系列train/test分割で検証) ===")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

feat_list = ["st_diff", "motor2_diff", "motor3_diff", "national_diff", "local_diff", "weight_diff"]
model_df = duel.dropna(subset=feat_list + ["is2"]).reset_index(drop=True)
half = len(model_df) // 2
train, test = model_df.iloc[:half], model_df.iloc[half:]
print(f"train n={len(train)}, test n={len(test)}")

X_train, y_train = train[feat_list], train["is2"]
X_test, y_test = test[feat_list], test["is2"]

clf = LogisticRegression()
clf.fit(X_train, y_train)
pred_proba = clf.predict_proba(X_test)[:, 1]
pred = (pred_proba >= 0.5).astype(int)

baseline_acc_test = max(y_test.mean(), 1 - y_test.mean()) * 100
model_acc_test = accuracy_score(y_test, pred) * 100
try:
    auc = roc_auc_score(y_test, pred_proba) * 100
except ValueError:
    auc = float("nan")
print(f"  testでの単純ベースライン正解率(多数派に賭け続けた場合): {baseline_acc_test:.1f}%")
print(f"  testでのロジスティック回帰モデル正解率: {model_acc_test:.1f}%")
print(f"  AUC: {auc:.1f}")
print(f"  係数: " + ", ".join(f"{f}={c:+.3f}" for f, c in zip(feat_list, clf.coef_[0])))

# --- モデルの予測に従って賭け方を変えた場合の回収率シミュレーション ---
test = test.copy()
test["pred_is2"] = pred
test["bet_combo"] = np.where(test["pred_is2"] == 1, "1-2", "1-3")
test["hit"] = (test["combination"] == test["bet_combo"]).astype(int)
test["ret"] = np.where(test["hit"] == 1, test["payout"], 0)
n_test = len(test)
stake_test = n_test * 100
ret_test = test["ret"].sum()
print(f"\n  [参考] test期間でモデルの予測に従い毎回200円を的中側1点に賭けた場合(単純化のため片方のみ100円換算):")
print(f"    回収率(モデル追従): {ret_test/stake_test*100:.1f}% (n={n_test})")
fixed_ret = np.where(test['combination'] == '1-2', test['payout']*2, np.where(test['combination']=='1-3', test['payout'], 0)).sum()
print(f"    参考: 同期間で固定1-2:200/1-3:100ミックスの回収率: {fixed_ret/(n_test*300)*100:.1f}%")

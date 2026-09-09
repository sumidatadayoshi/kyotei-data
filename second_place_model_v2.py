"""
2着に2号艇/3号艇のどちらが来るかを、少数の頑健な指標(st_diff, national_win_diff,
motor_2rate_diff, local_2rate_diff)だけで予想するロジスティック回帰モデルの
前半/後半 相互検証(second_place_indicators_v2.py の相関チェックの続き)。

①②条件を満たし、実際に1号艇が逃げ、2着が2号艇/3号艇だったレースに限定。
DBが小さい(n=313)ため、過去の analysis(second_place_predictor.py, n=758,
①②条件で絞っていない集団)より相関・精度とも大きく弱いことに注意。
"""
import sqlite3

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

DB_PATH = "data/boatrace.db"
SAMPLE_SIZE_WARNING_THRESHOLD = 5
INN_NIGE_RATE_THRESHOLD = 0.8
NIGASHI_RATE_THRESHOLD = 0.5

conn = sqlite3.connect(DB_PATH)
entries_all = pd.read_sql_query(
    "SELECT race_date, jcd, rno, waku, toban, avg_st, national_win_rate, motor_2rate, motor_3rate, local_2rate FROM entries", conn)
results_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, rank FROM results", conn)
waku1_rank = results_all[results_all["waku"] == 1][["race_date", "jcd", "rno", "rank"]].rename(columns={"rank": "waku1_rank"})

c1 = entries_all[entries_all["waku"] == 1].merge(results_all[results_all["waku"] == 1][["race_date", "jcd", "rno", "rank"]], on=["race_date", "jcd", "rno"])
c1_stats = c1.groupby("toban").agg(starts=("rank", "size"), wins=("rank", lambda s: (s == "1").sum()))
c1_stats["rate"] = c1_stats["wins"] / c1_stats["starts"]
qualified_toban1 = set(c1_stats[(c1_stats["rate"] >= INN_NIGE_RATE_THRESHOLD) & (c1_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)

c2 = entries_all[entries_all["waku"] == 2].merge(waku1_rank, on=["race_date", "jcd", "rno"])
c2_stats = c2.groupby("toban").agg(starts=("waku1_rank", "size"), nigasare=("waku1_rank", lambda s: (s == "1").sum()))
c2_stats["rate"] = c2_stats["nigasare"] / c2_stats["starts"]
qualified_toban2 = set(c2_stats[(c2_stats["rate"] >= NIGASHI_RATE_THRESHOLD) & (c2_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)

entries1 = entries_all[entries_all["waku"] == 1][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban1"})
entries2 = entries_all[entries_all["waku"] == 2][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban2"})
race_pairs = entries1.merge(entries2, on=["race_date", "jcd", "rno"])
cond = race_pairs[race_pairs["toban1"].isin(qualified_toban1) & race_pairs["toban2"].isin(qualified_toban2)][["race_date", "jcd", "rno"]]

r2 = results_all[results_all["waku"] == 2][["race_date", "jcd", "rno", "rank"]].rename(columns={"rank": "rank2"})
r3 = results_all[results_all["waku"] == 3][["race_date", "jcd", "rno", "rank"]].rename(columns={"rank": "rank3"})
df = cond.merge(waku1_rank, on=["race_date", "jcd", "rno"])
df = df[df["waku1_rank"] == "1"]
df = df.merge(r2, on=["race_date", "jcd", "rno"]).merge(r3, on=["race_date", "jcd", "rno"])
df = df[df["rank2"].isin(["1", "2", "3"]) & df["rank3"].isin(["1", "2", "3"])]
df["target"] = np.select([df["rank2"] == "2", df["rank3"] == "2"], [1, 0], default=np.nan)
df = df.dropna(subset=["target"])

e2 = entries_all[entries_all["waku"] == 2].drop(columns=["waku"]).add_suffix("_2").rename(columns={"race_date_2": "race_date", "jcd_2": "jcd", "rno_2": "rno"})
e3 = entries_all[entries_all["waku"] == 3].drop(columns=["waku"]).add_suffix("_3").rename(columns={"race_date_3": "race_date", "jcd_3": "jcd", "rno_3": "rno"})
df = df.merge(e2, on=["race_date", "jcd", "rno"], how="left").merge(e3, on=["race_date", "jcd", "rno"], how="left")

df["st_diff"] = df["avg_st_3"] - df["avg_st_2"]
df["national_win_diff"] = df["national_win_rate_2"] - df["national_win_rate_3"]
df["motor_2rate_diff"] = df["motor_2rate_2"] - df["motor_2rate_3"]
df["local_2rate_diff"] = df["local_2rate_2"] - df["local_2rate_3"]

feats = ["st_diff", "national_win_diff", "motor_2rate_diff", "local_2rate_diff"]
df = df.dropna(subset=feats + ["target"]).sort_values(["race_date", "rno", "jcd"]).reset_index(drop=True)
n = len(df)
mid = n // 2
front, back = df.iloc[:mid], df.iloc[mid:]
print(f"n={n} (ベースライン: 2号艇が2着になる率={df['target'].mean()*100:.1f}%)")


def run(train, test, label):
    X_train, y_train = train[feats].to_numpy(), train["target"].to_numpy()
    X_test, y_test = test[feats].to_numpy(), test["target"].to_numpy()
    model = LogisticRegression()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    baseline_acc = max(y_test.mean(), 1 - y_test.mean())
    print(f"{label}: test_n={len(test)} baseline_acc={baseline_acc*100:.1f}% model_acc={acc*100:.1f}% AUC={auc:.3f}")
    print("   係数:", dict(zip(feats, np.round(model.coef_[0], 3))))


run(front, back, "前半で学習→後半で検証")
run(back, front, "後半で学習→前半で検証")

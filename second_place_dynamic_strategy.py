"""
探索その5-C: 2着が2号艇/3号艇どちらになりやすいかをロジスティック回帰で予測し、
「基本は1-2:200円+1-3:100円だが、モデルが3号艇有利と判定した時だけ1-3:200円+
1-2:100円に入れ替える」という動的戦略を、全レース(①②条件を満たす全件)に対して
時系列train/testで検証する。
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

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


def block_bootstrap_recovery_varstake(df, ret_series, stake_series, n_resamples=2000, seed=42):
    block_key = df["race_date"].astype(str) + "_" + df["jcd"].astype(str)
    d = pd.DataFrame({"block": block_key, "return": np.asarray(ret_series), "stake": np.asarray(stake_series)})
    blocks = d.groupby("block").agg(return_sum=("return", "sum"), stake_sum=("stake", "sum"))
    return_arr = blocks["return_sum"].to_numpy()
    stake_arr = blocks["stake_sum"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(blocks), size=(n_resamples, len(blocks)))
    resample_stake = stake_arr[idx].sum(axis=1)
    resample_return = return_arr[idx].sum(axis=1)
    rates = np.where(resample_stake > 0, resample_return / resample_stake * 100, 0.0)
    lower, upper = np.percentile(rates, [2.5, 97.5])
    point = return_arr.sum() / stake_arr.sum() * 100 if stake_arr.sum() > 0 else 0.0
    return point, lower, upper


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

concluded["st_diff"] = concluded["avg_st_3"] - concluded["avg_st_2"]
concluded["motor2_diff"] = concluded["motor_2rate_2"] - concluded["motor_2rate_3"]
concluded["motor3_diff"] = concluded["motor_3rate_2"] - concluded["motor_3rate_3"]
concluded["national_diff"] = concluded["national_win_rate_2"] - concluded["national_win_rate_3"]
concluded["local_diff"] = concluded["local_win_rate_2"] - concluded["local_win_rate_3"]
concluded["weight_diff"] = concluded["weight_3"] - concluded["weight_2"]

feat_list = ["st_diff", "motor2_diff", "motor3_diff", "national_diff", "local_diff", "weight_diff"]

half = len(concluded) // 2
train_all, test_all = concluded.iloc[:half].copy(), concluded.iloc[half:].copy()

# 学習はtrain期間の中で実際に1-2/1-3で決着したレースのみ(ラベルがあるので)
train_duel = train_all[train_all["combination"].isin(["1-2", "1-3"])].dropna(subset=feat_list).copy()
train_duel["is2"] = (train_duel["combination"] == "1-2").astype(int)
clf = LogisticRegression()
clf.fit(train_duel[feat_list], train_duel["is2"])
print(f"train(前半)の決着レースで学習: n={len(train_duel)}")
print("係数: " + ", ".join(f"{f}={c:+.3f}" for f, c in zip(feat_list, clf.coef_[0])))

# test期間の「全レース」(決着に関わらず)にモデルを適用し、動的戦略の実際の回収率を見る
test_valid = test_all.dropna(subset=feat_list).copy()
proba_is2 = clf.predict_proba(test_valid[feat_list])[:, 1]
test_valid["pred_is2"] = (proba_is2 >= 0.5)

# 戦略A(固定): 1-2:200円 + 1-3:100円 常時
retA = np.where(test_valid["combination"] == "1-2", test_valid["payout"] * 2,
        np.where(test_valid["combination"] == "1-3", test_valid["payout"], 0))
stakeA = np.full(len(test_valid), 300)

# 戦略B(動的): モデルが2号艇有利なら1-2:200/1-3:100、3号艇有利なら逆に1-3:200/1-2:100
w12 = np.where(test_valid["pred_is2"], 200, 100)
w13 = np.where(test_valid["pred_is2"], 100, 200)
retB = np.where(test_valid["combination"] == "1-2", test_valid["payout"] * (w12 / 100), 0) + \
       np.where(test_valid["combination"] == "1-3", test_valid["payout"] * (w13 / 100), 0)
stakeB = w12 + w13

pA, lA, hA = block_bootstrap_recovery_varstake(test_valid, retA, stakeA)
pB, lB, hB = block_bootstrap_recovery_varstake(test_valid, retB, stakeB)

n_flip = int((~test_valid["pred_is2"]).sum())
print(f"\ntest(後半)の全対象レース: n={len(test_valid)}件 (うちモデルが3号艇有利と判定=入れ替え: {n_flip}件)")
print(f"  戦略A(固定1-2:200/1-3:100): 回収率{pA:.1f}% [{lA:.1f}-{hA:.1f}]")
print(f"  戦略B(モデルに応じて入替): 回収率{pB:.1f}% [{lB:.1f}-{hB:.1f}]")

# 参考: test期間で学習しなおして同じtest期間を評価する「答え合わせ用」上振れ参考値(過学習の参考)
print("\n(参考: test期間データそのもので学習した場合の同一期間再評価は掲載しません。過学習の参照にしかならないため)")

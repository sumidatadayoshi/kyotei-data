"""
①(1号艇イン逃げ率80%以上)②(2号艇逃がし率50%以上)に、
③(3号艇の"逃がし率"、3号艇が1号艇を勝たせてしまう率)50%以上 を追加した場合、
①②のみと比べて対象レース数・逃げ率・回収率がどう変わるかを検証する。
"""
import sqlite3

import numpy as np
import pandas as pd

DB_PATH = "/mnt/user-data/uploads/race-info/data/boatrace.db"
SAMPLE_SIZE_WARNING_THRESHOLD = 10
INN_NIGE_RATE_THRESHOLD = 0.8
NIGASHI_RATE_THRESHOLD = 0.5
NIGASHI3_RATE_THRESHOLD = 0.5

conn = sqlite3.connect(DB_PATH)
entries = pd.read_sql_query("SELECT race_date, jcd, rno, waku, toban FROM entries", conn)
results = pd.read_sql_query("SELECT race_date, jcd, rno, waku, rank FROM results", conn)
payouts_2tan = pd.read_sql_query("SELECT race_date, jcd, rno, combination, payout FROM payouts WHERE bet_type='2連単'", conn)


def wilson_ci(hits, n, z=1.959963984540054):
    if n == 0:
        return (np.nan, np.nan)
    p = hits / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((center - margin) / denom, (center + margin) / denom)


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


waku1_rank = results[results["waku"] == 1][["race_date", "jcd", "rno", "rank"]].rename(columns={"rank": "waku1_rank"})

# ① 1号艇イン逃げ率
c1 = entries[entries["waku"] == 1].merge(results[results["waku"] == 1][["race_date", "jcd", "rno", "rank"]], on=["race_date", "jcd", "rno"])
c1_stats = c1.groupby("toban").agg(starts=("rank", "size"), wins=("rank", lambda s: (s == "1").sum()))
c1_stats["rate"] = c1_stats["wins"] / c1_stats["starts"]
qualified_toban1 = set(c1_stats[(c1_stats["rate"] >= INN_NIGE_RATE_THRESHOLD) & (c1_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)

# ② 2号艇 逃がし率
c2 = entries[entries["waku"] == 2].merge(waku1_rank, on=["race_date", "jcd", "rno"])
c2_stats = c2.groupby("toban").agg(starts=("waku1_rank", "size"), nigasare=("waku1_rank", lambda s: (s == "1").sum()))
c2_stats["rate"] = c2_stats["nigasare"] / c2_stats["starts"]
qualified_toban2 = set(c2_stats[(c2_stats["rate"] >= NIGASHI_RATE_THRESHOLD) & (c2_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)

# ③ 3号艇 逃がし率(3号艇版)
c3 = entries[entries["waku"] == 3].merge(waku1_rank, on=["race_date", "jcd", "rno"])
c3_stats = c3.groupby("toban").agg(starts=("waku1_rank", "size"), nigasare=("waku1_rank", lambda s: (s == "1").sum()))
c3_stats["rate"] = c3_stats["nigasare"] / c3_stats["starts"]
qualified_toban3 = set(c3_stats[(c3_stats["rate"] >= NIGASHI3_RATE_THRESHOLD) & (c3_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)

e1 = entries[entries["waku"] == 1][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban1"})
e2 = entries[entries["waku"] == 2][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban2"})
e3 = entries[entries["waku"] == 3][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban3"})
pairs = e1.merge(e2, on=["race_date", "jcd", "rno"]).merge(e3, on=["race_date", "jcd", "rno"]).merge(waku1_rank, on=["race_date", "jcd", "rno"])

pairs["c1_ok"] = pairs["toban1"].isin(qualified_toban1)
pairs["c2_ok"] = pairs["toban2"].isin(qualified_toban2)
pairs["c3_ok"] = pairs["toban3"].isin(qualified_toban3)
pairs["escaped"] = pairs["waku1_rank"] == "1"

cond_12 = pairs[pairs["c1_ok"] & pairs["c2_ok"]]
cond_123 = pairs[pairs["c1_ok"] & pairs["c2_ok"] & pairs["c3_ok"]]

print(f"③(3号艇逃がし率>=50%, starts>=5)を満たす選手数: {len(qualified_toban3)} / 全{c3_stats.shape[0]}選手\n")

for label, cond in [("①②のみ", cond_12), ("①②③(3号艇も追加)", cond_123)]:
    n = len(cond)
    escaped_n = cond["escaped"].sum()
    esc_rate = escaped_n / n if n else np.nan
    lo, hi = wilson_ci(escaped_n, n)
    print(f"=== {label}: n={n} ===")
    print(f"  1号艇の逃げ率: {esc_rate*100:.1f}%[{lo*100:.1f}-{hi*100:.1f}]")

    # 単勝1点(1号艇)の回収率換算 + 固定2連単(1-2:200+1-3:100)の回収率
    bt = cond[["race_date", "jcd", "rno", "escaped"]].copy()
    p12 = payouts_2tan[payouts_2tan["combination"] == "1-2"][["race_date", "jcd", "rno", "payout"]].rename(columns={"payout": "p12"})
    p13 = payouts_2tan[payouts_2tan["combination"] == "1-3"][["race_date", "jcd", "rno", "payout"]].rename(columns={"payout": "p13"})
    bt = bt.merge(p12, on=["race_date", "jcd", "rno"], how="left").merge(p13, on=["race_date", "jcd", "rno"], how="left")
    bt["p12"] = bt["p12"].fillna(0)
    bt["p13"] = bt["p13"].fillna(0)
    bt["ret"] = bt["p12"] * 2 + bt["p13"] * 1
    p, lo2, hi2 = bb_recovery_varstake(bt, bt["ret"], pd.Series(300, index=bt.index))
    print(f"  固定2連単(1-2:200+1-3:100)回収率: {p:.1f}%[{lo2:.1f}-{hi2:.1f}]")
    print()

"""
「平均スタート順位」(そのレースで6艇中何番目に良いスタートを切ったか、を
選手ごとに平均したもの)が、①②戦略の逃げ率・単勝1回収率に関係するかを
調べる。avg_st(平均STの秒数そのもの)とは違う指標として、順位ベースで見る。

start_timingは ".15"=0.15秒遅れ, "F.02"=0.02秒フライング(-0.02扱い),
"L"=出遅れ(大きく遅れたものとして扱う) という形式なので、まずレースごとに
数値化してランク付けし、そのランクを選手ごとに平均する。
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "boatrace.db"
SAMPLE_SIZE_WARNING_THRESHOLD = 5
INN_NIGE_RATE_THRESHOLD = 0.8
NIGASHI_RATE_THRESHOLD = 0.5


def wilson_ci(hits, n, z=1.959963984540054):
    if n == 0:
        return 0.0, 0.0
    p = hits / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (center - half) * 100, (center + half) * 100


def bb_recovery(df, ret, stake, n_resamples=2000, seed=42):
    block_key = df["race_date"].astype(str) + "_" + df["jcd"].astype(str)
    d = pd.DataFrame({"block": block_key, "return": np.asarray(ret)})
    blocks = d.groupby("block").agg(n=("return", "size"), return_sum=("return", "sum"))
    n_arr, ret_arr = blocks["n"].to_numpy(), blocks["return_sum"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(blocks), size=(n_resamples, len(blocks)))
    rs, rr = n_arr[idx].sum(axis=1) * stake, ret_arr[idx].sum(axis=1)
    rates = np.where(rs > 0, rr / rs * 100, 0.0)
    lo, hi = np.percentile(rates, [2.5, 97.5])
    point = ret_arr.sum() / (n_arr.sum() * stake) * 100
    return point, lo, hi


def parse_st(v):
    if v is None:
        return np.nan
    v = str(v).strip()
    if v == "" or v.upper() == "L":
        return 9.99
    if v.upper().startswith("F"):
        try:
            return -float(v[1:])
        except ValueError:
            return np.nan
    try:
        return float(v)
    except ValueError:
        return np.nan


conn = sqlite3.connect(DB_PATH)
entries_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, toban FROM entries", conn)
results_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, rank, toban, start_timing FROM results", conn)
payouts_tan = pd.read_sql_query("SELECT race_date, jcd, rno, combination, payout FROM payouts WHERE bet_type = '単勝'", conn)

results_all["st_val"] = results_all["start_timing"].map(parse_st)
results_all = results_all.dropna(subset=["st_val"])
results_all["st_rank"] = results_all.groupby(["race_date", "jcd", "rno"])["st_val"].rank(method="min")

# 選手ごとの平均スタート順位(全waku通算)
rank_stats = results_all.groupby("toban").agg(starts=("st_rank", "size"), avg_rank=("st_rank", "mean"))
rank_reliable = rank_stats[rank_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD]
print(f"平均スタート順位を算出できた選手: {len(rank_reliable)}人 (starts>={SAMPLE_SIZE_WARNING_THRESHOLD})")
print(f"  平均スタート順位の分布: 最小{rank_reliable['avg_rank'].min():.2f} 中央値{rank_reliable['avg_rank'].median():.2f} 最大{rank_reliable['avg_rank'].max():.2f}\n")

waku1_rank_res = results_all[results_all["waku"] == 1][["race_date", "jcd", "rno", "rank"]].rename(columns={"rank": "waku1_rank"})

c1 = entries_all[entries_all["waku"] == 1].merge(
    results_all[results_all["waku"] == 1][["race_date", "jcd", "rno", "rank"]],
    on=["race_date", "jcd", "rno"], how="inner")
c1_stats = c1.groupby("toban").agg(starts=("rank", "size"), wins=("rank", lambda s: (s == "1").sum()))
c1_stats["rate"] = c1_stats["wins"] / c1_stats["starts"]
qualified_toban1 = set(c1_stats[(c1_stats["rate"] >= INN_NIGE_RATE_THRESHOLD) & (c1_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)

entries1 = entries_all[entries_all["waku"] == 1][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban1"})
cond1_only = entries1[entries1["toban1"].isin(qualified_toban1)].copy()
cond1_only = cond1_only.merge(waku1_rank_res, on=["race_date", "jcd", "rno"], how="inner")
cond1_only["escaped"] = (cond1_only["waku1_rank"] == "1").astype(int)
cond1_only = cond1_only.merge(payouts_tan[payouts_tan["combination"] == "1"][["race_date", "jcd", "rno", "payout"]],
                               on=["race_date", "jcd", "rno"], how="left")
cond1_only["tan_ret"] = np.where(cond1_only["escaped"] == 1, cond1_only["payout"].fillna(0), 0)
cond1_only = cond1_only.merge(rank_reliable[["avg_rank"]].rename(columns={"avg_rank": "avg_rank1"}),
                               left_on="toban1", right_index=True, how="inner")

total = len(cond1_only)
base = cond1_only["escaped"].mean() * 100
print(f"①条件のみのレース(1号艇の平均スタート順位データあり): n={total}, 逃げ率{base:.1f}%\n")

corr = np.corrcoef(cond1_only["avg_rank1"], cond1_only["escaped"])[0, 1]
print(f"1号艇の平均スタート順位(数字が小さいほどスタートが良い)と逃げ成否の相関: r={corr:+.3f}\n")

print("=== 1号艇の平均スタート順位を4分位に区切った時の逃げ率・単勝1回収率 ===")
cond1_only["q"] = pd.qcut(cond1_only["avg_rank1"], q=4, duplicates="drop")
for b, sub in cond1_only.groupby("q", observed=True):
    n = len(sub)
    h = int(sub["escaped"].sum())
    l, u = wilson_ci(h, n)
    p, lo, hi = bb_recovery(sub, sub["tan_ret"], 100)
    print(f"  1号艇平均ST順位{b}: n={n} 逃げ率{h/n*100:.1f}%[{l:.1f}-{u:.1f}] 単勝1回収率{p:.1f}%[{lo:.1f}-{hi:.1f}]")

print("\n=== 前半/後半での頑健性チェック ===")
cond1_sorted = cond1_only.sort_values(["race_date", "rno", "jcd"]).reset_index(drop=True)
mid = len(cond1_sorted) // 2
front, back = cond1_sorted.iloc[:mid], cond1_sorted.iloc[mid:]
for label, part in [("前半", front), ("後半", back)]:
    r = np.corrcoef(part["avg_rank1"], part["escaped"])[0, 1]
    print(f"  {label}(n={len(part)}): r={r:+.3f}")

# 4号艇の平均スタート順位も(1号艇との差)チェック
e4 = entries_all[entries_all["waku"] == 4][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban4"})
cond_with4 = cond1_only.merge(e4, on=["race_date", "jcd", "rno"], how="inner")
cond_with4 = cond_with4.merge(rank_reliable[["avg_rank"]].rename(columns={"avg_rank": "avg_rank4"}),
                               left_on="toban4", right_index=True, how="inner")
cond_with4["rank_gap_1_4"] = cond_with4["avg_rank4"] - cond_with4["avg_rank1"]
r_gap = np.corrcoef(cond_with4["rank_gap_1_4"], cond_with4["escaped"])[0, 1]
print(f"\n参考: (4号艇平均ST順位 - 1号艇平均ST順位)と逃げ成否の相関: r={r_gap:+.3f} (n={len(cond_with4)})")

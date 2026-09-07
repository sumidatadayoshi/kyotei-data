"""
①②条件(1号艇イン逃げ率80%以上・2号艇逃し率50%以上)に合致した過去レースで
2連単「1-2」を100円ずつ買い続けた場合の、詳細な検証指標をまとめて計算する。

判定ロジック(compute_racer_rate_stats / find_qualifying_races)はdashboard.pyの
定義と同一内容を複製している(block_bootstrap_1_2.pyと同じ)。

時系列の並び順について:
DBには発走時刻が保存されていないため、race_date(日付)→rno(レース番号)→jcd(場)の
順を「時系列の代理」として使う。同じ日・同じレース番号は各競艇場でおおむね近い時間帯に
発走するため、連敗・ドローダウンの近似としては妥当だが、厳密な発走順ではない点に注意。
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "boatrace.db"
BET_AMOUNT = 100
FIXED_COMBO = "1-2"
N_RESAMPLES = 2000
SEED = 42

SAMPLE_SIZE_WARNING_THRESHOLD = 5
INN_NIGE_RATE_THRESHOLD = 0.8
NIGASHI_RATE_THRESHOLD = 0.5


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

    return c1_stats, c2_stats, waku1_rank


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

    entries1 = entries_df[entries_df["waku"] == 1][
        ["race_date", "jcd", "rno", "toban", "racer_name", "venue_name"]
    ].rename(columns={"toban": "toban1", "racer_name": "racer1_name"})
    entries2 = entries_df[entries_df["waku"] == 2][
        ["race_date", "jcd", "rno", "toban", "racer_name"]
    ].rename(columns={"toban": "toban2", "racer_name": "racer2_name"})
    race_pairs = entries1.merge(entries2, on=["race_date", "jcd", "rno"], how="inner")

    return race_pairs[
        race_pairs["toban1"].isin(qualified_toban1) & race_pairs["toban2"].isin(qualified_toban2)
    ].copy()


def block_bootstrap(block_key, n_arr, payout_arr, bet_amount, n_resamples, seed, metric="rate"):
    """block_key単位（既にblockごとに集計済みのn_arr/payout_arrを渡す想定）で
    ブロックブートストラップし、metric='rate'なら回収率(%)の配列、
    metric='hitrate'なら的中率(%)の配列を返す（呼び出し側でhit集計をpayout_arrに渡す）。"""
    n_blocks = len(n_arr)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_blocks, size=(n_resamples, n_blocks))
    resample_n = n_arr[idx].sum(axis=1)
    resample_payout = payout_arr[idx].sum(axis=1)
    if metric == "rate":
        stake = resample_n * bet_amount
        return np.where(stake > 0, resample_payout / stake * 100, 0.0)
    else:  # hitrate
        return np.where(resample_n > 0, resample_payout / resample_n * 100, 0.0)


def wilson_ci(hits, n, z=1.959963984540054):
    if n == 0:
        return 0.0, 0.0
    p = hits / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (center - half) * 100, (center + half) * 100


conn = sqlite3.connect(DB_PATH)
entries_all = pd.read_sql_query(
    "SELECT race_date, jcd, rno, waku, toban, racer_name, gender, venue_name FROM entries", conn
)
results_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, rank FROM results", conn)
payouts_2tan = pd.read_sql_query(
    "SELECT race_date, jcd, rno, combination, payout FROM payouts WHERE bet_type = '2連単'", conn
)

c1_stats, c2_stats, _ = compute_racer_rate_stats(entries_all, results_all)
candidates = find_qualifying_races(entries_all, c1_stats, c2_stats)
concluded = candidates.merge(payouts_2tan, on=["race_date", "jcd", "rno"], how="inner")

# 時系列の代理順（日付→レース番号→場コード）で並べ替え
concluded = concluded.sort_values(["race_date", "rno", "jcd"]).reset_index(drop=True)

concluded["hit"] = (concluded["combination"] == FIXED_COMBO).astype(int)
concluded["net_payout"] = np.where(concluded["hit"] == 1, concluded["payout"], 0)
concluded["profit"] = concluded["net_payout"] - BET_AMOUNT  # 1レースあたり損益(円)

total_races = len(concluded)
hit_count = int(concluded["hit"].sum())
total_stake = total_races * BET_AMOUNT
total_return = int(concluded["net_payout"].sum())
total_profit = total_return - total_stake
recovery_rate = total_return / total_stake * 100
hit_rate = hit_count / total_races * 100

print("=" * 60)
print("① 基本指標")
print("=" * 60)
print(f"対象レース数: {total_races}件")
print(f"的中回数: {hit_count}回")
print(f"的中率: {hit_rate:.1f}%")
print(f"回収率: {recovery_rate:.1f}%")
print(f"通算損益: {total_profit:,}円（賭け金合計{total_stake:,}円 / 払戻金合計{total_return:,}円）")

# --- 95%信頼区間（レース単位・クラスタ構造を無視した参考値） ---
rng = np.random.default_rng(SEED)
race_idx = rng.integers(0, total_races, size=(N_RESAMPLES, total_races))
net_arr = concluded["net_payout"].to_numpy()
race_level_rates = net_arr[race_idx].sum(axis=1) / (total_races * BET_AMOUNT) * 100
naive_lower, naive_upper = np.percentile(race_level_rates, [2.5, 97.5])

# --- ブロックブートストラップ95%CI（開催日×競艇場） ---
block_key = concluded["race_date"].astype(str) + "_" + concluded["jcd"].astype(str)
blk = pd.DataFrame({"block": block_key, "hit": concluded["hit"], "net": concluded["net_payout"]})
blocks = blk.groupby("block").agg(n=("net", "size"), net_sum=("net", "sum"), hit_sum=("hit", "sum"))
n_blocks = len(blocks)
n_arr = blocks["n"].to_numpy()
net_arr_blk = blocks["net_sum"].to_numpy()
hit_arr_blk = blocks["hit_sum"].to_numpy()

boot_rates = block_bootstrap(blocks.index, n_arr, net_arr_blk, BET_AMOUNT, N_RESAMPLES, SEED, metric="rate")
block_lower, block_upper = np.percentile(boot_rates, [2.5, 97.5])

boot_hitrates = block_bootstrap(blocks.index, n_arr, hit_arr_blk, BET_AMOUNT, N_RESAMPLES, SEED, metric="hitrate")
hit_block_lower, hit_block_upper = np.percentile(boot_hitrates, [2.5, 97.5])

wilson_lower, wilson_upper = wilson_ci(hit_count, total_races)

print()
print("=" * 60)
print("② 信頼区間")
print("=" * 60)
print(f"[回収率] 95%信頼区間（レース単位ブートストラップ・クラスタ無視/参考）: "
      f"{naive_lower:.1f}% 〜 {naive_upper:.1f}%")
print(f"[回収率] ブートストラップ95%CI（開催日×競艇場ブロック, n_blocks={n_blocks}）: "
      f"{block_lower:.1f}% 〜 {block_upper:.1f}%")
print(f"[的中率] Wilson score 95%CI（レース単位・独立仮定）: "
      f"{wilson_lower:.1f}% 〜 {wilson_upper:.1f}%")
print(f"[的中率] ブロックブートストラップ95%CI（開催日×競艇場）: "
      f"{hit_block_lower:.1f}% 〜 {hit_block_upper:.1f}%")

# --- 最大連敗 ---
is_miss = (concluded["hit"] == 0).to_numpy()
max_losing_streak = 0
cur = 0
for m in is_miss:
    if m:
        cur += 1
        max_losing_streak = max(max_losing_streak, cur)
    else:
        cur = 0

# --- 最大ドローダウン（累計損益ベース） ---
cum_profit = concluded["profit"].cumsum().to_numpy()
running_peak = np.maximum.accumulate(np.concatenate(([0], cum_profit)))[1:]
drawdown = cum_profit - running_peak
max_drawdown = drawdown.min()  # 負の値
max_dd_idx = int(np.argmin(drawdown))

print()
print("=" * 60)
print("③ 連敗・ドローダウン")
print("=" * 60)
print(f"最大連敗: {max_losing_streak}回")
print(f"最大ドローダウン: {max_drawdown:,.0f}円"
      f"（{concluded.iloc[max_dd_idx]['race_date']} 時点、対象レース通算{max_dd_idx+1}件目）")

# --- 月別回収率 ---
concluded["year_month"] = concluded["race_date"].str[0:6]
monthly = concluded.groupby("year_month").agg(
    n=("hit", "size"), hits=("hit", "sum"), net_return=("net_payout", "sum")
)
monthly["stake"] = monthly["n"] * BET_AMOUNT
monthly["recovery_rate"] = monthly["net_return"] / monthly["stake"] * 100
monthly["hit_rate"] = monthly["hits"] / monthly["n"] * 100

print()
print("=" * 60)
print("④ 月別回収率")
print("=" * 60)
for ym, row in monthly.iterrows():
    ym_fmt = f"{ym[0:4]}-{ym[4:6]}"
    print(f"{ym_fmt}: {int(row['n'])}件 / 的中{int(row['hits'])}回 / "
          f"的中率{row['hit_rate']:.1f}% / 回収率{row['recovery_rate']:.1f}%")

# --- 前半/後半 分割（約550R/550R相当の均等2分割） ---
half = total_races // 2
first_half = concluded.iloc[:half]
second_half = concluded.iloc[half:]

def summarize(df, label):
    n = len(df)
    h = int(df["hit"].sum())
    ret = int(df["net_payout"].sum())
    stake = n * BET_AMOUNT
    rate = ret / stake * 100 if stake > 0 else 0.0
    hr = h / n * 100 if n > 0 else 0.0
    print(f"{label}: {n}件 / 的中{h}回 / 的中率{hr:.1f}% / 回収率{rate:.1f}%")

print()
print("=" * 60)
print(f"⑤ 前半/後半 分割（{half}R / {total_races - half}R）")
print("=" * 60)
summarize(first_half, "前半")
summarize(second_half, "後半")

# --- 高配当の寄与度分析 ---
hit_rows = concluded[concluded["hit"] == 1].sort_values("payout", ascending=False)
top10 = hit_rows.head(10)
top10_profit = int((top10["payout"] - BET_AMOUNT).sum())
top10_share = top10_profit / total_profit * 100 if total_profit != 0 else float("nan")

excl_races = total_races - len(top10)
excl_return = total_return - int(top10["payout"].sum())
excl_stake = excl_races * BET_AMOUNT
excl_rate = excl_return / excl_stake * 100 if excl_stake > 0 else 0.0

print()
print("=" * 60)
print("⑥ 高配当の寄与度")
print("=" * 60)
print("的中の中で払戻金が高い順トップ10:")
for _, r in top10.iterrows():
    print(f"  {r['race_date']} {r['jcd']} R{r['rno']}: 払戻{int(r['payout']):,}円"
          f"（損益+{int(r['payout']) - BET_AMOUNT:,}円）")
print(f"通算損益に占めるトップ10の割合: {top10_share:.1f}%"
      f"（トップ10損益合計 {top10_profit:,}円 / 通算損益 {total_profit:,}円）")
print(f"トップ10（高配当10レース）を除外した場合の回収率: {excl_rate:.1f}%"
      f"（対象{excl_races}件、賭け金合計{excl_stake:,}円、払戻金合計{excl_return:,}円）")

"""
①②条件を満たすレース vs 満たさないレースで、レース前に発表されている実際の
オッズ(oddsテーブル、odds_low)にどれくらい差があるかを検証する。

odds テーブルは 2026-08-21〜2026-09-08 の期間でほぼ全レース分揃っている
(それ以前は散発的なテスト取得のみ)ため、この期間に絞って比較する。

対象:
  - 単勝(1号艇の単勝オッズ)
  - 2連単 1-2 / 1-3 のオッズ
"""
import sqlite3

import numpy as np
import pandas as pd

DB_PATH = "/mnt/user-data/uploads/race-info/data/boatrace.db"
SAMPLE_SIZE_WARNING_THRESHOLD = 10
INN_NIGE_RATE_THRESHOLD = 0.8
NIGASHI_RATE_THRESHOLD = 0.5
WINDOW_START = "20260821"
WINDOW_END = "20260908"

conn = sqlite3.connect(DB_PATH)
entries_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, toban FROM entries", conn)
results_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, rank FROM results", conn)
odds = pd.read_sql_query(
    "SELECT race_date, jcd, rno, bet_type, combination, odds_low FROM odds "
    f"WHERE race_date >= '{WINDOW_START}' AND race_date <= '{WINDOW_END}'", conn)

# ============ ①②条件判定(全履歴ベース、従来通り) ============
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
race_pairs["qualified"] = race_pairs["toban1"].isin(qualified_toban1) & race_pairs["toban2"].isin(qualified_toban2)

# ウィンドウ内だけに限定
race_pairs_w = race_pairs[(race_pairs["race_date"] >= WINDOW_START) & (race_pairs["race_date"] <= WINDOW_END)].copy()
print(f"検証期間: {WINDOW_START}〜{WINDOW_END}")
print(f"対象レース数: {len(race_pairs_w)}  (①②対象: {race_pairs_w['qualified'].sum()}件, {race_pairs_w['qualified'].mean()*100:.1f}%)\n")

# ============ 単勝オッズ(1号艇) ============
tansho1 = odds[(odds["bet_type"] == "単勝") & (odds["combination"] == "1")][["race_date", "jcd", "rno", "odds_low"]].rename(columns={"odds_low": "tansho1_odds"})
d = race_pairs_w.merge(tansho1, on=["race_date", "jcd", "rno"], how="inner")
d["implied_prob"] = 1 / d["tansho1_odds"]

print("=== 1号艇 単勝オッズ: ①②の有無で比較 ===")
for q, label in [(True, "①②を満たす"), (False, "①②を満たさない")]:
    sub = d[d["qualified"] == q]
    print(f"  {label}: n={len(sub)}  平均オッズ={sub['tansho1_odds'].mean():.2f}倍  "
          f"中央値={sub['tansho1_odds'].median():.2f}倍  平均インプライド確率={sub['implied_prob'].mean()*100:.1f}%")

# 参考: ①②を満たすレースでの1号艇の「実際の」逃げ率(このウィンドウ内・全履歴基準の推定値)との比較
qualified_toban1_rate_avg = np.mean([c1_stats.loc[t, "rate"] for t in d.loc[d["qualified"], "toban1"].unique() if t in c1_stats.index])
print(f"\n  (参考)①②対象レースの1号艇・全履歴での平均逃げ率: {qualified_toban1_rate_avg*100:.1f}%")
print(f"  → オッズが示す平均インプライド確率({d[d['qualified']]['implied_prob'].mean()*100:.1f}%)との差が、")
print(f"     『市場がどこまで織り込めていないか』の目安になる\n")

# ============ 2連単 1-2 / 1-3 オッズ ============
print("=== 2連単オッズ: ①②の有無で比較 ===")
for combo in ["1-2", "1-3"]:
    nt = odds[(odds["bet_type"] == "2連単") & (odds["combination"] == combo)][["race_date", "jcd", "rno", "odds_low"]].rename(columns={"odds_low": "odds"})
    dd = race_pairs_w.merge(nt, on=["race_date", "jcd", "rno"], how="inner")
    print(f"  [{combo}]")
    for q, label in [(True, "①②を満たす"), (False, "①②を満たさない")]:
        sub = dd[dd["qualified"] == q]
        if len(sub) == 0:
            continue
        print(f"    {label}: n={len(sub)}  平均オッズ={sub['odds'].mean():.2f}倍  中央値={sub['odds'].median():.2f}倍")

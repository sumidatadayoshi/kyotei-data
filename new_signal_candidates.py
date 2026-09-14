"""
①②のような「強いシグナル条件」の探索(単独/複合、他艇も含む)。

進め方:
  1. 各枠番ごとに、自艇の各種指標(4分位/カテゴリ)で「3着以内に来られない率」の
     ばらつきをスキャン(national_win_rate, racer_class, motor_2rate等)。
     → national_win_rateやracer_classは効果が非常に大きい(gap 20〜40pt)が、
       これらはレース表に載ってる最も基本的な指標そのものなので、①②ほど
       「隠れて」おらず、既にオッズにかなり織り込まれている可能性が高い。

  2. ①②の発想を拡張し、「この選手がこの枠番の時だけ、3着以内に来られない率」
     という個人×枠番の条件付き指標を作ってみる。全期間データで見ると全国3連対率
     で実力をコントロールしてもgap 30〜40pt という非常に強い効果に見えるが、
     これは選手ごとの出走数が少ない(starts>=5程度)ことによる循環参照
     (自分の悪い結果で作った指標で、その悪い結果自体を当てているだけ)の疑いが
     強い。前半データで指標を作り、後半データで検証すると、gapはほぼ0〜10pt
     まで縮小し、①②のような頑健な信号ではないことが分かった
     (このアプローチは"デバンク"=不採用とする)。

  3. 個人の小標本に依存しない、レース内の相対比較・複合条件に絞って再探索。
     見つかった2つの候補(前半/後半検証済み):

     H1) 1号艇が①条件(逃げ率80%以上)を満たさない時、6号艇の全国勝率が高いほど
         6号艇のtop3率が上乗せされる(弱い1号艇 x 強い6号艇の相互作用)。
     H2) レース内で自艇のモーター2連率が全6艇中トップの場合、top3率が
         枠番を問わず約+3〜8pt上乗せされる。
"""
import sqlite3

import numpy as np
import pandas as pd

DB_PATH = "data/boatrace.db"
SAMPLE_SIZE_WARNING_THRESHOLD = 10
INN_NIGE_RATE_THRESHOLD = 0.8


def wilson_ci(hits, n, z=1.959963984540054):
    if n == 0:
        return (np.nan, np.nan)
    p = hits / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((center - margin) / denom, (center + margin) / denom)


conn = sqlite3.connect(DB_PATH)
entries = pd.read_sql_query(
    "SELECT race_date, jcd, rno, waku, toban, national_win_rate, motor_2rate FROM entries", conn)
results = pd.read_sql_query("SELECT race_date, jcd, rno, waku, rank FROM results", conn)

df = entries.merge(results, on=["race_date", "jcd", "rno", "waku"], how="inner")
df["top3"] = df["rank"].isin(["1", "2", "3"]).astype(int)
df["race_date_dt"] = pd.to_datetime(df["race_date"], format="%Y%m%d")
max_date, min_date = df["race_date_dt"].max(), df["race_date_dt"].min()
mid_date = min_date + (max_date - min_date) / 2
print(f"データ期間: {min_date.date()}〜{max_date.date()} / 前半後半分割点: {mid_date.date()}\n")

# ============ H1: 1号艇が弱い時、6号艇の全国勝率でtop3率はどう変わるか ============
c1 = entries[entries["waku"] == 1].merge(results[results["waku"] == 1][["race_date", "jcd", "rno", "rank"]], on=["race_date", "jcd", "rno"])
c1_stats = c1.groupby("toban").agg(starts=("rank", "size"), wins=("rank", lambda s: (s == "1").sum()))
c1_stats["rate"] = c1_stats["wins"] / c1_stats["starts"]
weak_toban1 = set(c1_stats[(c1_stats["rate"] < INN_NIGE_RATE_THRESHOLD) & (c1_stats["starts"] >= SAMPLE_SIZE_WARNING_THRESHOLD)].index)

e1 = entries[entries["waku"] == 1][["race_date", "jcd", "rno", "toban"]].rename(columns={"toban": "toban1"})
race_flag = e1.copy()
race_flag["weak1"] = race_flag["toban1"].isin(weak_toban1)

w6 = df[df["waku"] == 6].merge(race_flag[["race_date", "jcd", "rno", "weak1"]], on=["race_date", "jcd", "rno"], how="inner")
w6 = w6.dropna(subset=["national_win_rate"])

print("=== H1: 1号艇が①未達成(弱い)時、6号艇の全国勝率とtop3率の関係 (前半/後半検証) ===")
for label, sub in [("前半", w6[w6["race_date_dt"] < mid_date]), ("後半", w6[w6["race_date_dt"] >= mid_date])]:
    print(f"  [{label}] n={len(sub)}")
    for weak1_val, wlabel in [(True, "1号艇弱い"), (False, "1号艇強い(①達成)")]:
        s = sub[sub["weak1"] == weak1_val].copy()
        try:
            s["nwq"] = pd.qcut(s["national_win_rate"], 3, duplicates="drop")
        except ValueError:
            continue
        g = s.groupby("nwq", observed=True)["top3"].agg(["mean", "size"])
        vals = "  ".join([f"{idx}:{row['mean']*100:.1f}%(n={int(row['size'])})" for idx, row in g.iterrows()])
        print(f"    {wlabel}: {vals}")
print()

# ============ H2: レース内モーター2連率トップの艇のtop3率 ============
print("=== H2: レース内で自艇のモーター2連率が最高の場合のtop3率 (前半/後半検証) ===")
tmp = df.dropna(subset=["motor_2rate"]).copy()
tmp["motor_rank_in_race"] = tmp.groupby(["race_date", "jcd", "rno"])["motor_2rate"].rank(ascending=False, method="min")
tmp["motor_best"] = tmp["motor_rank_in_race"] == 1

for label, sub in [("前半", tmp[tmp["race_date_dt"] < mid_date]), ("後半", tmp[tmp["race_date_dt"] >= mid_date])]:
    print(f"  [{label}] n={len(sub)}")
    for w in [1, 2, 3, 4, 5, 6]:
        s = sub[sub["waku"] == w]
        g = s.groupby("motor_best")["top3"].agg(["mean", "size"])
        if True not in g.index or False not in g.index:
            continue
        rt, rf = g.loc[True], g.loc[False]
        gap = rt["mean"] - rf["mean"]
        lo, hi = wilson_ci(rt["mean"] * rt["size"], rt["size"])
        print(f"    waku={w}: モーター最良(n={int(rt['size'])})={rt['mean']*100:.1f}%[{lo*100:.1f}-{hi*100:.1f}] "
              f"vs 非最良(n={int(rf['size'])})={rf['mean']*100:.1f}%  gap={gap*100:+.1f}pt")
    print()

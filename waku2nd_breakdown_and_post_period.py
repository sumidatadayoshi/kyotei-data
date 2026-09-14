"""
①②条件(1号艇イン逃げ率80%以上・2号艇逃し率50%以上)合致レースのうち、
1号艇が1着だったレースについて、2着艇番(2/3/4/5/6号艇)ごとの発生率を
複数期間で比較する。

  1. 5/26〜6/5・6/13〜6/15(187件)における2/3/4/5/6号艇それぞれの2着率を、
     6/6〜6/12(中日)・5/26より前(通常期間前半)・6/16以降(通常期間後半)と
     比較し、「4号艇の台頭」なのか「2・3号艇の不振の裏返し」なのかを切り分ける
  2. 6/16以降、4号艇2着率が通常の16〜17%に戻っているかを、月ごとの推移で確認

判定ロジック(①②条件の抽出)はblock_bootstrap_weighted_12_13.pyと同一。
"""
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "boatrace.db"

SAMPLE_SIZE_WARNING_THRESHOLD = 10
INN_NIGE_RATE_THRESHOLD = 0.8
NIGASHI_RATE_THRESHOLD = 0.5

FULL_START = "20260526"
FULL_END = "20260615"
EXCLUDE_START = "20260606"
EXCLUDE_END = "20260612"
POST_START = "20260616"


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
        ["race_date", "jcd", "rno", "toban"]
    ].rename(columns={"toban": "toban1"})
    entries2 = entries_df[entries_df["waku"] == 2][
        ["race_date", "jcd", "rno", "toban"]
    ].rename(columns={"toban": "toban2"})
    race_pairs = entries1.merge(entries2, on=["race_date", "jcd", "rno"], how="inner")

    return race_pairs[
        race_pairs["toban1"].isin(qualified_toban1) & race_pairs["toban2"].isin(qualified_toban2)
    ].copy()


conn = sqlite3.connect(DB_PATH)
entries_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, toban FROM entries", conn)
results_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, rank FROM results", conn)

c1_stats, c2_stats, waku1_rank = compute_racer_rate_stats(entries_all, results_all)
candidates = find_qualifying_races(entries_all, c1_stats, c2_stats)

rank2 = results_all[results_all["rank"] == "2"][["race_date", "jcd", "rno", "waku"]].rename(
    columns={"waku": "second_waku"}
)
qualified = candidates.merge(waku1_rank, on=["race_date", "jcd", "rno"], how="inner")
qualified = qualified.merge(rank2, on=["race_date", "jcd", "rno"], how="left")
qualified["tan1_hit"] = qualified["waku1_rank"] == "1"

win_df = qualified[qualified["tan1_hit"]].copy()  # 1号艇1着レースのみ(2着分布の母数)

in_full = (win_df["race_date"] >= FULL_START) & (win_df["race_date"] <= FULL_END)
in_excluded_mid = (win_df["race_date"] >= EXCLUDE_START) & (win_df["race_date"] <= EXCLUDE_END)
sub_mask = in_full & ~in_excluded_mid
mid_mask = in_full & in_excluded_mid
pre_mask = win_df["race_date"] < FULL_START
post_mask = win_df["race_date"] >= POST_START
normal_mask = pre_mask | post_mask

SUB_LABEL = "5/26〜6/5・6/13〜6/15"
MID_LABEL = "6/6〜6/12(中日)"
PRE_LABEL = f"〜5/25(通常期間・前)"
POST_LABEL = "6/16〜(通常期間・後)"
NORMAL_LABEL = "通常期間 全体(前+後)"

out = []
out.append("対象: ①②条件合致レースのうち1号艇1着レース(2着艇番の分布を見る母数)")
out.append("")

# ---------------------------------------------------------------------------
# 1. 期間別 2着艇番(2/3/4/5/6)の発生率
# ---------------------------------------------------------------------------
out.append("=" * 78)
out.append("1. 期間別 2着艇番の発生率(1号艇1着レース中)")
out.append("=" * 78)


def second_breakdown(df, label):
    total = len(df)
    counts = df["second_waku"].value_counts()
    row = {"label": label, "n": total}
    for w in range(2, 7):
        n = int(counts.get(w, 0))
        row[f"w{w}_n"] = n
        row[f"w{w}_pct"] = n / total * 100 if total else float("nan")
    return row


rows = [
    second_breakdown(win_df[sub_mask], SUB_LABEL),
    second_breakdown(win_df[mid_mask], MID_LABEL),
    second_breakdown(win_df[pre_mask], PRE_LABEL),
    second_breakdown(win_df[post_mask], POST_LABEL),
    second_breakdown(win_df[normal_mask], NORMAL_LABEL),
]

out.append(f"\n{'期間':28} {'n':>5}  {'2号艇':>12} {'3号艇':>12} {'4号艇':>12} {'5号艇':>12} {'6号艇':>12}")
for r in rows:
    out.append(
        f"{r['label']:28} {r['n']:>5}  "
        + " ".join(f"{r[f'w{w}_n']:>3}件({r[f'w{w}_pct']:5.1f}%)" for w in range(2, 7))
    )

out.append("")
out.append("■ 通常期間(前+後)平均を基準にした各期間の差分(ポイント)")
base = rows[-1]
for r in rows[:-1]:
    diffs = " / ".join(f"{w}号艇{r[f'w{w}_pct'] - base[f'w{w}_pct']:+5.1f}pt" for w in range(2, 7))
    out.append(f"  {r['label']:28}: {diffs}")
out.append("")

# ---------------------------------------------------------------------------
# 2. 6/16以降の4号艇2着率の月次推移
# ---------------------------------------------------------------------------
out.append("=" * 78)
out.append("2. 6/16以降の4号艇2着率 月次推移(通常水準16〜17%への回帰確認)")
out.append("=" * 78)

post_df = win_df[post_mask].copy()
post_df["month"] = post_df["race_date"].str[:6]

monthly = post_df.groupby("month").apply(
    lambda g: pd.Series({"n": len(g), "n4": int((g["second_waku"] == 4).sum())})
)
monthly["rate4"] = monthly["n4"] / monthly["n"] * 100

out.append("")
for month, row in monthly.iterrows():
    m_disp = f"{month[:4]}-{month[4:6]}"
    out.append(f"  {m_disp}: {int(row['n']):>4}件中 4号艇2着{int(row['n4']):>3}件 ({row['rate4']:5.1f}%)")

total_post_n = len(post_df)
total_post_n4 = int((post_df["second_waku"] == 4).sum())
out.append(
    f"\n  6/16〜データ末尾({post_df['race_date'].max()})の合計: "
    f"{total_post_n}件中 4号艇2着{total_post_n4}件 ({total_post_n4 / total_post_n * 100:.1f}%)"
)
out.append(f"  (参考) 5/26〜6/5・6/13〜6/15: {rows[0]['w4_pct']:.1f}% / "
           f"通常期間全体平均: {base['w4_pct']:.1f}%")

# 直近4週の週次推移も参考として出す
out.append("\n■ (参考) 6/16以降の週次推移")
post_df_w = post_df.copy()
post_df_w["race_date_dt"] = pd.to_datetime(post_df_w["race_date"], format="%Y%m%d")
start_dt = pd.to_datetime(POST_START, format="%Y%m%d")
post_df_w["week"] = ((post_df_w["race_date_dt"] - start_dt).dt.days // 7)
weekly = post_df_w.groupby("week").apply(
    lambda g: pd.Series({"n": len(g), "n4": int((g["second_waku"] == 4).sum())})
)
weekly["rate4"] = weekly["n4"] / weekly["n"] * 100
for wk, row in weekly.iterrows():
    w_start = start_dt + pd.Timedelta(days=int(wk) * 7)
    w_end = w_start + pd.Timedelta(days=6)
    out.append(f"  {w_start.date()}〜{w_end.date()}: {int(row['n']):>3}件中 4号艇2着{int(row['n4']):>2}件 ({row['rate4']:5.1f}%)")

text = "\n".join(out)
print(text)
out_path = Path(__file__).parent / "waku2nd_breakdown_and_post_period_output.txt"
out_path.write_text(text, encoding="utf-8")
print(f"\n(全文を {out_path.name} に保存しました)")

"""
2026-05-26〜06-15のうち、6/6〜6/12を除いた部分(=5/26〜6/5と6/13〜6/15)に
絞って、①②条件(1号艇イン逃げ率80%以上・2号艇逃し率50%以上)合致レースを
対象に、4号艇に着目した分析を行う。

  1. 4号艇2着率(1号艇1着時に限定/限定なしの両方)を、
     この部分期間・除外した中日(6/6〜6/12)・通常期間で比較
  2. 4号艇のまくり・まくり差し発生率(4号艇が1着かつ決まり手がまくり/
     まくり差しだった割合)と、4号艇1着時の決まり手内訳
  3. 開催場別(宮島・浜名湖 등)の4号艇2着率・まくり発生率

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
        ["race_date", "jcd", "rno", "toban", "venue_name"]
    ].rename(columns={"toban": "toban1"})
    entries2 = entries_df[entries_df["waku"] == 2][
        ["race_date", "jcd", "rno", "toban"]
    ].rename(columns={"toban": "toban2"})
    race_pairs = entries1.merge(entries2, on=["race_date", "jcd", "rno"], how="inner")

    return race_pairs[
        race_pairs["toban1"].isin(qualified_toban1) & race_pairs["toban2"].isin(qualified_toban2)
    ].copy()


conn = sqlite3.connect(DB_PATH)
entries_all = pd.read_sql_query(
    "SELECT race_date, jcd, rno, waku, toban, venue_name FROM entries", conn
)
results_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, rank FROM results", conn)
races_all = pd.read_sql_query("SELECT race_date, jcd, rno, kimarite FROM races", conn)

c1_stats, c2_stats, waku1_rank = compute_racer_rate_stats(entries_all, results_all)
candidates = find_qualifying_races(entries_all, c1_stats, c2_stats)

# 1号艇の着順・2着艇番・勝者(rank=1)の艇番+決まり手 を付与
rank1 = results_all[results_all["rank"] == "1"][["race_date", "jcd", "rno", "waku"]].rename(
    columns={"waku": "winner_waku"}
)
rank2 = results_all[results_all["rank"] == "2"][["race_date", "jcd", "rno", "waku"]].rename(
    columns={"waku": "second_waku"}
)
qualified = candidates.merge(waku1_rank, on=["race_date", "jcd", "rno"], how="inner")
qualified = qualified.merge(rank1, on=["race_date", "jcd", "rno"], how="left")
qualified = qualified.merge(rank2, on=["race_date", "jcd", "rno"], how="left")
qualified = qualified.merge(races_all, on=["race_date", "jcd", "rno"], how="left")
qualified["tan1_hit"] = qualified["waku1_rank"] == "1"

in_full = (qualified["race_date"] >= FULL_START) & (qualified["race_date"] <= FULL_END)
in_excluded_mid = (qualified["race_date"] >= EXCLUDE_START) & (qualified["race_date"] <= EXCLUDE_END)
sub_mask = in_full & ~in_excluded_mid  # 5/26〜6/5 と 6/13〜6/15
mid_mask = in_full & in_excluded_mid   # 6/6〜6/12(除外した中日、参考用)
normal_mask = ~in_full                 # 通常期間

sub_df = qualified[sub_mask].copy()
mid_df = qualified[mid_mask].copy()
normal_df = qualified[normal_mask].copy()

SUB_LABEL = "5/26〜6/5・6/13〜6/15(6/6〜6/12を除く)"
MID_LABEL = "6/6〜6/12(除外した中日、参考)"
NORMAL_LABEL = "通常期間(参考)"

out = []
out.append(f"対象: ①②条件合致レース")
out.append(f"{SUB_LABEL}: {len(sub_df)}件 / {MID_LABEL}: {len(mid_df)}件 / {NORMAL_LABEL}: {len(normal_df)}件")
out.append("")

# ---------------------------------------------------------------------------
# 1. 4号艇2着率
# ---------------------------------------------------------------------------
out.append("=" * 70)
out.append("1. 4号艇2着率")
out.append("=" * 70)


def waku4_second_rate(df, label):
    total = len(df)
    win_df = df[df["tan1_hit"]]
    n_win = len(win_df)
    n4_cond = int((win_df["second_waku"] == 4).sum())
    n4_uncond = int((df["second_waku"] == 4).sum())
    out.append(f"\n■ {label}")
    if n_win:
        out.append(f"  1号艇1着時に限定: 対象{n_win}件中、4号艇2着 {n4_cond}件 ({n4_cond / n_win * 100:5.1f}%)")
    else:
        out.append("  1号艇1着時に限定: 該当なし")
    if total:
        out.append(f"  限定なし(全体): 対象{total}件中、4号艇2着 {n4_uncond}件 ({n4_uncond / total * 100:5.1f}%)")
    else:
        out.append("  限定なし(全体): 該当なし")


waku4_second_rate(sub_df, SUB_LABEL)
waku4_second_rate(mid_df, MID_LABEL)
waku4_second_rate(normal_df, NORMAL_LABEL)
out.append("")

# ---------------------------------------------------------------------------
# 2. 4号艇のまくり・まくり差し発生率
# ---------------------------------------------------------------------------
out.append("=" * 70)
out.append("2. 4号艇のまくり・まくり差し発生率(4号艇1着時の決まり手)")
out.append("=" * 70)


def waku4_kimarite(df, label):
    total = len(df)
    w4win = df[df["winner_waku"] == 4]
    n_w4win = len(w4win)
    n_makuri = int(w4win["kimarite"].isin(["まくり", "まくり差し"]).sum())
    out.append(f"\n■ {label} (対象{total}件)")
    out.append(f"  4号艇1着: {n_w4win}件 ({n_w4win / total * 100:5.1f}%)")
    out.append(f"  うち まくり・まくり差し: {n_makuri}件 (全体対比 {n_makuri / total * 100:5.1f}%)")
    if n_w4win:
        out.append("  4号艇1着時の決まり手内訳:")
        counts = w4win["kimarite"].value_counts(dropna=False)
        for k, n in counts.items():
            k_disp = k if pd.notna(k) else "(不明)"
            out.append(f"    {k_disp:8}: {n:>3}件 ({n / n_w4win * 100:5.1f}%)")


waku4_kimarite(sub_df, SUB_LABEL)
waku4_kimarite(mid_df, MID_LABEL)
waku4_kimarite(normal_df, NORMAL_LABEL)
out.append("")

# ---------------------------------------------------------------------------
# 3. 開催場別: 4号艇2着率・まくり発生率
# ---------------------------------------------------------------------------
out.append("=" * 70)
out.append(f"3. 開催場別 4号艇2着率・まくり発生率 ({SUB_LABEL})")
out.append("=" * 70)

sub_win = sub_df[sub_df["tan1_hit"]].copy()
venue_2nd = sub_win.groupby("venue_name").apply(
    lambda g: pd.Series({"n": len(g), "n4": int((g["second_waku"] == 4).sum())})
)
venue_2nd["rate4_2nd"] = venue_2nd["n4"] / venue_2nd["n"] * 100

venue_makuri = sub_df.groupby("venue_name").apply(
    lambda g: pd.Series({
        "total": len(g),
        "n4win": int((g["winner_waku"] == 4).sum()),
        "n4makuri": int(((g["winner_waku"] == 4) & (g["kimarite"].isin(["まくり", "まくり差し"]))).sum()),
    })
)
venue_makuri["makuri_rate"] = venue_makuri["n4makuri"] / venue_makuri["total"] * 100

venue_table = venue_2nd.join(venue_makuri[["total", "n4win", "n4makuri", "makuri_rate"]], how="outer").fillna(0)
venue_table = venue_table.sort_values("n", ascending=False)

out.append(f"\n{'開催場':6} {'1号艇1着n':>9} {'4号艇2着':>8} {'2着率':>7}  |  {'対象n':>6} {'4号艇1着':>8} {'まくり系':>8} {'まくり率':>8}")
for venue, row in venue_table.iterrows():
    n = int(row["n"])
    n4 = int(row["n4"])
    rate4 = row["rate4_2nd"] if n else float("nan")
    total = int(row["total"])
    n4win = int(row["n4win"])
    n4makuri = int(row["n4makuri"])
    makuri_rate = row["makuri_rate"] if total else float("nan")
    out.append(
        f"{venue:6} {n:>9} {n4:>8} {rate4:6.1f}%  |  {total:>6} {n4win:>8} {n4makuri:>8} {makuri_rate:7.1f}%"
    )

out.append("")
out.append("(参考) 宮島・浜名湖 vs その他 の単純平均比較")
highlight = ["宮島", "浜名湖"]
hi = venue_table.loc[venue_table.index.isin(highlight)]
others = venue_table.loc[~venue_table.index.isin(highlight)]


def weighted_avg(df, num_col, den_col):
    den = df[den_col].sum()
    return (df[num_col].sum() / den * 100) if den else float("nan")


out.append(
    f"  宮島・浜名湖 合算: 1号艇1着{int(hi['n'].sum())}件中 4号艇2着{int(hi['n4'].sum())}件 "
    f"({weighted_avg(hi, 'n4', 'n'):.1f}%) / まくり系 対象{int(hi['total'].sum())}件中{int(hi['n4makuri'].sum())}件 "
    f"({weighted_avg(hi, 'n4makuri', 'total'):.1f}%)"
)
out.append(
    f"  その他会場 合算: 1号艇1着{int(others['n'].sum())}件中 4号艇2着{int(others['n4'].sum())}件 "
    f"({weighted_avg(others, 'n4', 'n'):.1f}%) / まくり系 対象{int(others['total'].sum())}件中{int(others['n4makuri'].sum())}件 "
    f"({weighted_avg(others, 'n4makuri', 'total'):.1f}%)"
)

text = "\n".join(out)
print(text)
out_path = Path(__file__).parent / "waku4_analysis_sub_period_output.txt"
out_path.write_text(text, encoding="utf-8")
print(f"\n(全文を {out_path.name} に保存しました)")

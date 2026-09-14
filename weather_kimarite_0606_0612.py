"""
①②条件(1号艇イン逃げ率80%以上・2号艇逃し率50%以上)に合致したレースについて、
2026-06-06〜2026-06-12の期間の一覧(開催場・天候・風速・波高・決まり手)を出力する。
あわせて、単勝1(1号艇)が外れたレースの決まり手内訳を、対象期間と
それ以外の期間(通常期間)とで比較する。

判定ロジックはblock_bootstrap_weighted_12_13.pyと同一。
"""
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "boatrace.db"

SAMPLE_SIZE_WARNING_THRESHOLD = 10
INN_NIGE_RATE_THRESHOLD = 0.8
NIGASHI_RATE_THRESHOLD = 0.5

PERIOD_START = "20260606"
PERIOD_END = "20260612"


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

    return c1_stats, c2_stats


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


conn = sqlite3.connect(DB_PATH)
entries_all = pd.read_sql_query(
    "SELECT race_date, jcd, rno, waku, toban, racer_name, gender, venue_name FROM entries", conn
)
results_all = pd.read_sql_query("SELECT race_date, jcd, rno, waku, rank FROM results", conn)
races_all = pd.read_sql_query(
    "SELECT race_date, jcd, rno, weather, wind_speed, wave_height, kimarite FROM races", conn
)

c1_stats, c2_stats = compute_racer_rate_stats(entries_all, results_all)
candidates = find_qualifying_races(entries_all, c1_stats, c2_stats)

# レース結果(races.kimarite等)が判明しているものに限定
qualified = candidates.merge(races_all, on=["race_date", "jcd", "rno"], how="inner")

# 1号艇の着順(単勝1が的中したか)を付与
waku1_rank = results_all[results_all["waku"] == 1][["race_date", "jcd", "rno", "rank"]].rename(
    columns={"rank": "waku1_rank"}
)
qualified = qualified.merge(waku1_rank, on=["race_date", "jcd", "rno"], how="left")
qualified["tan1_hit"] = qualified["waku1_rank"] == "1"

out_lines = []

# --- 6/6〜6/12の一覧 ---
period = qualified[
    (qualified["race_date"] >= PERIOD_START) & (qualified["race_date"] <= PERIOD_END)
].sort_values(["race_date", "jcd", "rno"])

out_lines.append(f"■ {PERIOD_START[:4]}-{PERIOD_START[4:6]}-{PERIOD_START[6:8]}〜"
                  f"{PERIOD_END[:4]}-{PERIOD_END[4:6]}-{PERIOD_END[6:8]} ①②条件合致レース一覧 "
                  f"({len(period)}件)\n")
out_lines.append(
    f"{'日付':10} {'場':6} {'R':>2} {'天候':4} {'風速':>5} {'波高':>5} "
    f"{'単勝1':5} {'決まり手':8} 1号艇/2号艇"
)
for _, row in period.iterrows():
    date_fmt = f"{row['race_date'][0:4]}-{row['race_date'][4:6]}-{row['race_date'][6:8]}"
    hit = "○的中" if row["tan1_hit"] else "×外れ"
    kimarite = row["kimarite"] if pd.notna(row["kimarite"]) else "-"
    weather = row["weather"] if pd.notna(row["weather"]) else "-"
    wind = f"{row['wind_speed']:.1f}m" if pd.notna(row["wind_speed"]) else "-"
    wave = f"{row['wave_height']:.1f}cm" if pd.notna(row["wave_height"]) else "-"
    out_lines.append(
        f"{date_fmt:10} {row['venue_name']:6} {row['rno']:>2} {weather:4} {wind:>5} {wave:>5} "
        f"{hit:5} {kimarite:8} {row['racer1_name']}/{row['racer2_name']}"
    )

out_lines.append("")

# --- 単勝1が外れたレースの決まり手内訳: 対象期間 vs 通常期間 ---
missed_period = period[~period["tan1_hit"]]
missed_normal = qualified[
    ~((qualified["race_date"] >= PERIOD_START) & (qualified["race_date"] <= PERIOD_END))
    & (~qualified["tan1_hit"])
]

def kimarite_breakdown(df, label):
    total = len(df)
    out_lines.append(f"■ 単勝1が外れたレースの決まり手内訳: {label} (対象{total}件)")
    if total == 0:
        out_lines.append("  該当なし\n")
        return
    counts = df["kimarite"].value_counts(dropna=False)
    for k, n in counts.items():
        k_disp = k if pd.notna(k) else "(不明)"
        out_lines.append(f"  {k_disp:8}: {n:>4}件 ({n/total*100:5.1f}%)")
    out_lines.append("")

kimarite_breakdown(missed_period, f"{PERIOD_START[:4]}-{PERIOD_START[4:6]}-{PERIOD_START[6:8]}〜"
                                    f"{PERIOD_END[:4]}-{PERIOD_END[4:6]}-{PERIOD_END[6:8]}")
kimarite_breakdown(missed_normal, "それ以外の期間(通常期間)")

text = "\n".join(out_lines)
print(text)

out_path = Path(__file__).parent / "weather_kimarite_0606_0612_output.txt"
out_path.write_text(text, encoding="utf-8")
print(f"\n(全文を {out_path.name} に保存しました)")

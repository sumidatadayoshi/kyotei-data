"""
2026-05-26〜2026-06-15の期間について、①②条件(1号艇イン逃げ率80%以上・
2号艇逃し率50%以上)に合致したレースを対象に、以下を集計する。

  1. 1号艇勝率・2着艇番分布(2/3/4/5/6号艇の割合、4/5/6号艇合計の取りこぼし率)
     を通常期間(それ以外の全期間)と比較
  2. 2連単「1-2」「1-3」それぞれの的中時オッズ(平均・中央値)を通常期間と比較
  3. 5日ごとの投資額・払戻額・回収率の推移(買い目はダッシュボードの
     「①②条件×重み付け買い」と同じ、2連単1-2に10,000円/1-3に5,000円)
  4. 開催場ごとの内訳(対象レース数・的中率・回収率)
  5. 天候(晴/曇り/雨)・風速・波高の内訳。特に①1号艇が1着にならなかった
     レース(外れ)、②1-2/1-3が的中したがオッズが低かったレース(配当が
     薄かったレース、期間内オッズの下位25%を薄い配当とみなす)での傾向

判定ロジック(①②条件の抽出)はblock_bootstrap_weighted_12_13.pyと同一
(全期間の実績から選手ごとの通算成績を求め、閾値を満たす1号艇・2号艇の
組み合わせのレースを対象とする)。
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "boatrace.db"

SAMPLE_SIZE_WARNING_THRESHOLD = 10
INN_NIGE_RATE_THRESHOLD = 0.8
NIGASHI_RATE_THRESHOLD = 0.5

PERIOD_START = "20260526"
PERIOD_END = "20260615"

BET_WEIGHTS = {"1-2": 10_000, "1-3": 5_000}
TOTAL_BET_PER_RACE = sum(BET_WEIGHTS.values())  # 15,000円


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
races_all = pd.read_sql_query(
    "SELECT race_date, jcd, rno, weather, wind_speed, wave_height FROM races", conn
)
payouts_2tan = pd.read_sql_query(
    "SELECT race_date, jcd, rno, combination, payout FROM payouts WHERE bet_type = '2連単'", conn
)

c1_stats, c2_stats, waku1_rank = compute_racer_rate_stats(entries_all, results_all)
candidates = find_qualifying_races(entries_all, c1_stats, c2_stats)

# 1号艇の着順・2着艇番を付与
rank2 = results_all[results_all["rank"] == "2"][["race_date", "jcd", "rno", "waku"]].rename(
    columns={"waku": "second_waku"}
)
qualified = candidates.merge(waku1_rank, on=["race_date", "jcd", "rno"], how="inner")
qualified = qualified.merge(rank2, on=["race_date", "jcd", "rno"], how="left")
qualified["tan1_hit"] = qualified["waku1_rank"] == "1"

period_mask = (qualified["race_date"] >= PERIOD_START) & (qualified["race_date"] <= PERIOD_END)
period_df = qualified[period_mask].copy()
normal_df = qualified[~period_mask].copy()

PERIOD_LABEL = f"{PERIOD_START[:4]}-{PERIOD_START[4:6]}-{PERIOD_START[6:8]}〜{PERIOD_END[:4]}-{PERIOD_END[4:6]}-{PERIOD_END[6:8]}"

out = []
out.append(f"対象: ①②条件(1号艇イン逃げ率80%以上・2号艇逃し率50%以上)合致レース")
out.append(f"対象期間: {PERIOD_LABEL} ({period_mask.sum()}件) / 通常期間: それ以外全体 ({(~period_mask).sum()}件)")
out.append(f"(データベース収録範囲: {entries_all['race_date'].min()}〜{entries_all['race_date'].max()})")
out.append("")

# ---------------------------------------------------------------------------
# 1. 1号艇勝率・2着艇番分布
# ---------------------------------------------------------------------------
out.append("=" * 70)
out.append("1. 1号艇勝率・2着艇番分布(1号艇1着レースのみ対象)")
out.append("=" * 70)


def winrate_and_second_breakdown(df, label):
    total = len(df)
    wins = int(df["tan1_hit"].sum())
    winrate = wins / total * 100 if total else float("nan")
    out.append(f"\n■ {label} (対象{total}件)")
    out.append(f"  1号艇勝率: {winrate:.1f}% ({wins}/{total})")

    win_df = df[df["tan1_hit"]]
    n_win = len(win_df)
    if n_win == 0:
        out.append("  1号艇1着レースがないため、2着分布は集計できません。")
        return
    counts = win_df["second_waku"].value_counts().sort_index()
    for waku in range(2, 7):
        n = int(counts.get(waku, 0))
        out.append(f"  2着={waku}号艇: {n:>4}件 ({n / n_win * 100:5.1f}%)")
    n456 = int(counts.get(4, 0) + counts.get(5, 0) + counts.get(6, 0))
    out.append(f"  (4・5・6号艇合計「取りこぼし」: {n456}件 / {n456 / n_win * 100:5.1f}%)")


winrate_and_second_breakdown(period_df, PERIOD_LABEL)
winrate_and_second_breakdown(normal_df, "通常期間")
out.append("")

# ---------------------------------------------------------------------------
# 2. 1-2/1-3 的中時オッズ
# ---------------------------------------------------------------------------
out.append("=" * 70)
out.append("2. 2連単「1-2」「1-3」的中時オッズ(倍率)")
out.append("=" * 70)

concluded = candidates.merge(payouts_2tan, on=["race_date", "jcd", "rno"], how="inner")
concluded["odds"] = concluded["payout"] / 100
concluded_period_mask = (concluded["race_date"] >= PERIOD_START) & (concluded["race_date"] <= PERIOD_END)

for combo in ["1-2", "1-3"]:
    sub = concluded[concluded["combination"] == combo]
    sub_period_mask = (sub["race_date"] >= PERIOD_START) & (sub["race_date"] <= PERIOD_END)
    p_odds = sub[sub_period_mask]["odds"]
    n_odds = sub[~sub_period_mask]["odds"]
    out.append(f"\n■ 2連単「{combo}」")
    out.append(f"  {PERIOD_LABEL}: 的中{len(p_odds)}回 / 平均{p_odds.mean():.2f}倍 / 中央値{p_odds.median():.2f}倍")
    out.append(f"  通常期間: 的中{len(n_odds)}回 / 平均{n_odds.mean():.2f}倍 / 中央値{n_odds.median():.2f}倍")
out.append("")

# ---------------------------------------------------------------------------
# 3. 5日ごとの投資額・払戻額・回収率推移(対象期間のみ)
# ---------------------------------------------------------------------------
out.append("=" * 70)
out.append(f"3. 5日ごとの投資額・払戻額・回収率推移(対象期間のみ、買い目: 2連単1-2に10,000円+1-3に5,000円=1レースあたり計15,000円)")
out.append("=" * 70)

period_concluded = concluded[concluded_period_mask].copy()


def race_return(row):
    weight = BET_WEIGHTS.get(row["combination"])
    if weight is None:
        return 0
    return row["payout"] * (weight / 100)


period_concluded["return"] = period_concluded.apply(race_return, axis=1)
period_concluded["hit"] = period_concluded["combination"].isin(BET_WEIGHTS).astype(int)

daily = period_concluded.groupby("race_date").agg(n=("hit", "size"), hits=("hit", "sum"), ret=("return", "sum"))
daily["stake"] = daily["n"] * TOTAL_BET_PER_RACE
daily = daily.sort_index()
daily.index = pd.to_datetime(daily.index, format="%Y%m%d")

start_dt = pd.to_datetime(PERIOD_START, format="%Y%m%d")
daily["bucket"] = ((daily.index - start_dt).days // 5)

out.append("\n■ 5日ごと")
bucketed = daily.groupby("bucket").agg(n=("n", "sum"), hits=("hits", "sum"), stake=("stake", "sum"), ret=("ret", "sum"))
for bucket, row in bucketed.iterrows():
    b_start = start_dt + pd.Timedelta(days=int(bucket) * 5)
    b_end = min(b_start + pd.Timedelta(days=4), pd.to_datetime(PERIOD_END, format="%Y%m%d"))
    stake = row["stake"]
    ret = row["ret"]
    rr = ret / stake * 100 if stake else float("nan")
    out.append(
        f"  {b_start.date()}〜{b_end.date()}: {int(row['n']):>3}件 / 的中{int(row['hits']):>2}回 / "
        f"投資額{int(stake):,} / 払戻額{int(ret):,} / 回収率{rr:5.1f}%"
    )

out.append("\n■ 日ごと")
for date, row in daily.iterrows():
    stake = row["stake"]
    ret = row["ret"]
    rr = ret / stake * 100 if stake else float("nan")
    out.append(
        f"  {date.date()}: {int(row['n']):>2}件 / 的中{int(row['hits']):>2}回 / "
        f"投資額{int(stake):,} / 払戻額{int(ret):,} / 回収率{rr:5.1f}%"
    )

total_stake = int(daily["stake"].sum())
total_ret = int(daily["ret"].sum())
out.append(f"\n  合計: 投資額{total_stake:,} / 払戻額{total_ret:,} / 回収率{total_ret / total_stake * 100:.1f}%")
out.append("")

# ---------------------------------------------------------------------------
# 4. 開催場ごとの内訳(対象期間のみ)
# ---------------------------------------------------------------------------
out.append("=" * 70)
out.append("4. 開催場ごとの内訳(対象期間のみ)")
out.append("=" * 70)

period_qualified = period_df.copy()
venue_all = period_qualified.groupby("venue_name").agg(
    n=("tan1_hit", "size"), wins=("tan1_hit", "sum")
)
venue_bet = period_concluded.groupby("venue_name").agg(
    n=("hit", "size"), hits=("hit", "sum"), ret=("return", "sum")
)
venue_bet["stake"] = venue_bet["n"] * TOTAL_BET_PER_RACE

venue_all = venue_all.sort_values("n", ascending=False)
out.append("")
for venue, row in venue_all.iterrows():
    n = int(row["n"])
    wins = int(row["wins"])
    winrate = wins / n * 100 if n else float("nan")
    if venue in venue_bet.index:
        b = venue_bet.loc[venue]
        stake = b["stake"]
        ret = b["ret"]
        rr = ret / stake * 100 if stake else float("nan")
        bet_str = f" / 決着済み{int(b['n'])}件 投資額{int(stake):,} 払戻額{int(ret):,} 回収率{rr:5.1f}%"
    else:
        bet_str = ""
    out.append(f"  {venue:6}: {n:>3}件 / 1号艇勝率{winrate:5.1f}%{bet_str}")
out.append("")

# ---------------------------------------------------------------------------
# 5. 天候・風速・波高の内訳
# ---------------------------------------------------------------------------
out.append("=" * 70)
out.append("5. 天候(晴/曇り/雨)・風速・波高の内訳(対象期間)")
out.append("=" * 70)

period_weather = period_df.merge(races_all, on=["race_date", "jcd", "rno"], how="left")


def wind_bucket(v):
    if pd.isna(v):
        return "不明"
    if v < 3:
        return "0-3m未満"
    if v < 5:
        return "3-5m未満"
    if v < 7:
        return "5-7m未満"
    return "7m以上"


def wave_bucket(v):
    if pd.isna(v):
        return "不明"
    if v < 3:
        return "0-3cm未満"
    if v < 6:
        return "3-6cm未満"
    if v < 10:
        return "6-10cm未満"
    return "10cm以上"


period_weather["wind_bucket"] = period_weather["wind_speed"].apply(wind_bucket)
period_weather["wave_bucket"] = period_weather["wave_height"].apply(wave_bucket)


def dist_breakdown(df, col, label, order=None):
    total = len(df)
    out.append(f"\n■ {label} (対象{total}件)")
    if total == 0:
        out.append("  該当なし")
        return
    counts = df[col].value_counts(dropna=False)
    keys = order if order is not None else counts.index
    for k in keys:
        n = int(counts.get(k, 0))
        if n == 0 and order is not None:
            continue
        k_disp = k if pd.notna(k) else "不明"
        out.append(f"  {k_disp:10}: {n:>4}件 ({n / total * 100:5.1f}%)")


out.append("\n--- 全対象レース(①②条件合致・期間内) ---")
dist_breakdown(period_weather, "weather", "天候内訳", order=["晴", "曇り", "雨", None])
dist_breakdown(period_weather, "wind_bucket", "風速内訳", order=["0-3m未満", "3-5m未満", "5-7m未満", "7m以上", "不明"])
dist_breakdown(period_weather, "wave_bucket", "波高内訳", order=["0-3cm未満", "3-6cm未満", "6-10cm未満", "10cm以上", "不明"])

out.append("\n--- ①1号艇が1着にならなかったレース(外れ) ---")
missed = period_weather[~period_weather["tan1_hit"]]
dist_breakdown(missed, "weather", "天候内訳(外れ)", order=["晴", "曇り", "雨", None])
dist_breakdown(missed, "wind_bucket", "風速内訳(外れ)", order=["0-3m未満", "3-5m未満", "5-7m未満", "7m以上", "不明"])
dist_breakdown(missed, "wave_bucket", "波高内訳(外れ)", order=["0-3cm未満", "3-6cm未満", "6-10cm未満", "10cm以上", "不明"])

out.append(
    f"\n  参考: 全対象レースに占める「雨」の割合 {(period_weather['weather'] == '雨').mean() * 100:.1f}% に対し、"
    f"外れレースに占める「雨」の割合 {(missed['weather'] == '雨').mean() * 100:.1f}%"
    if len(missed) else ""
)

# 配当が薄かったレース: 1-2/1-3が的中したレースのうち、オッズが期間内下位25%
hit_period = period_concluded[period_concluded["hit"] == 1].copy()
hit_period["odds"] = hit_period["payout"] / 100
if len(hit_period) >= 4:
    q1 = hit_period["odds"].quantile(0.25)
    thin = hit_period[hit_period["odds"] <= q1]
    thin_weather = thin.merge(races_all, on=["race_date", "jcd", "rno"], how="left")
    thin_weather["wind_bucket"] = thin_weather["wind_speed"].apply(wind_bucket)
    thin_weather["wave_bucket"] = thin_weather["wave_height"].apply(wave_bucket)
    hit_weather = hit_period.merge(races_all, on=["race_date", "jcd", "rno"], how="left")

    out.append(f"\n--- ②配当が薄かったレース(1-2/1-3的中かつオッズが期間内下位25%以下、閾値{q1:.2f}倍) ---")
    dist_breakdown(thin_weather, "weather", "天候内訳(配当薄)", order=["晴", "曇り", "雨", None])
    dist_breakdown(thin_weather, "wind_bucket", "風速内訳(配当薄)", order=["0-3m未満", "3-5m未満", "5-7m未満", "7m以上", "不明"])
    dist_breakdown(thin_weather, "wave_bucket", "波高内訳(配当薄)", order=["0-3cm未満", "3-6cm未満", "6-10cm未満", "10cm以上", "不明"])
    out.append(
        f"\n  参考: 的中レース全体({len(hit_period)}件)に占める「雨」の割合 "
        f"{(hit_weather['weather'] == '雨').mean() * 100:.1f}% に対し、"
        f"配当薄レース({len(thin)}件)に占める「雨」の割合 {(thin_weather['weather'] == '雨').mean() * 100:.1f}%"
    )
else:
    out.append("\n--- ②配当が薄かったレース ---\n  的中件数が少なく(4件未満)、下位25%の判定ができません。")

text = "\n".join(out)
print(text)
out_path = Path(__file__).parent / "period_report_0526_0615_output.txt"
out_path.write_text(text, encoding="utf-8")
print(f"\n(全文を {out_path.name} に保存しました)")

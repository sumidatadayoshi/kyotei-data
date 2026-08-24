"""
BOATRACEデータの動作確認用ダッシュボード。

取得したデータ(出走表・結果・払戻金)が正しく保存されているかを
目視確認することが目的。

「🎯 今日のおすすめ」と「🎯 イン逃げ狙い目レース分析」は、
1号艇のイン逃げ率・2号艇の逃し率という簡易な統計に基づく実験的機能。
データ量が少ないうちはサンプル数が少なく参考にならない点に注意。

使い方:
    streamlit run dashboard.py
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

DB_PATH = Path(__file__).parent / "data" / "boatrace.db"
JST = ZoneInfo("Asia/Tokyo")

SAMPLE_SIZE_WARNING_THRESHOLD = 5
INN_NIGE_RATE_THRESHOLD = 0.8
NIGASHI_RATE_THRESHOLD = 0.5

st.set_page_config(page_title="BOATRACE データ確認ダッシュボード", layout="wide")


@st.cache_resource
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def load_df(query, params=()):
    return pd.read_sql_query(query, get_connection(), params=params)


def format_yen(v):
    if pd.isna(v):
        return "-"
    return f"¥{int(v):,}"


def fmt_date(d):
    return f"{d[0:4]}-{d[4:6]}-{d[6:8]}"


def compute_racer_rate_stats(entries_all, results_all):
    """全期間のentries/resultsから、選手(登番)ごとの
    ①1号艇イン逃げ率、②2号艇逃し率を計算する。
    戻り値: (c1_stats, c2_stats, waku1_rank)"""
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


def rate_label(stats_df, toban):
    row = stats_df.loc[toban]
    starts = int(row["starts"])
    label = f"{row['rate'] * 100:.1f}% (n={starts})"
    if starts < SAMPLE_SIZE_WARNING_THRESHOLD:
        label += " ⚠️参考データ不足"
    return label


def find_qualifying_races(entries_df, c1_stats, c2_stats):
    """①1号艇イン逃げ率・②2号艇逃し率の両条件を満たすレース(1号艇・2号艇の
    出走表ペア)を抽出する。entries_dfは対象を絞ったentries(例: 当日分のみ、
    または全期間)を渡す。"""
    qualified_toban1 = set(c1_stats[c1_stats["rate"] >= INN_NIGE_RATE_THRESHOLD].index)
    qualified_toban2 = set(c2_stats[c2_stats["rate"] >= NIGASHI_RATE_THRESHOLD].index)

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


def add_rate_labels(candidates, c1_stats, c2_stats):
    candidates = candidates.copy()
    candidates["1号艇イン逃げ率"] = candidates["toban1"].apply(lambda t: rate_label(c1_stats, t))
    candidates["2号艇逃し率"] = candidates["toban2"].apply(lambda t: rate_label(c2_stats, t))
    return candidates


st.title("🚤 BOATRACE データ確認ダッシュボード")
st.caption("取得済みデータの中身をレース単位で目視確認するための簡易ツールです。")

conn = get_connection()

entries_all = load_df("SELECT race_date, jcd, rno, waku, toban, racer_name, venue_name FROM entries")
results_all = load_df("SELECT race_date, jcd, rno, waku, rank FROM results")

if not entries_all.empty and not results_all.empty:
    c1_stats, c2_stats, waku1_rank = compute_racer_rate_stats(entries_all, results_all)
else:
    c1_stats = c2_stats = waku1_rank = None

# ---------------------------------------------------------------------------
# 🎯 今日のおすすめ
#
# 当日の出走表(まだ結果が出ていないレース)について、1号艇・2号艇の選手を
# これまでの実績(過去のentries/results全体)と照らし合わせ、
# ①1号艇イン逃げ率80%以上 かつ ②2号艇逃し率50%以上 を満たすレースを表示する。
# 当日の出走表データは、GitHub Actionsの日次ジョブが
# `--when today --entries-only` で毎朝取得する(結果はまだ存在しないため
# 取得しない)。
# ---------------------------------------------------------------------------
st.header("🎯 今日のおすすめ")

today_str = datetime.now(JST).strftime("%Y%m%d")
today_entries = entries_all[entries_all["race_date"] == today_str] if not entries_all.empty else entries_all

if today_entries.empty:
    st.info(
        f"{fmt_date(today_str)} の出走表データがまだありません。"
        "日次ジョブの実行後、または手動で "
        "`python boatrace_scraper.py --when today --entries-only` を実行すると表示されます。"
    )
elif c1_stats is None or c1_stats.empty or c2_stats.empty:
    st.info("判定に使う過去の実績データがまだ十分にありません。")
else:
    today_candidates = find_qualifying_races(today_entries, c1_stats, c2_stats)
    if today_candidates.empty:
        st.info("本日は条件に合うレースがありません。")
    else:
        today_candidates = add_rate_labels(today_candidates, c1_stats, c2_stats)
        display_df = today_candidates.rename(
            columns={"venue_name": "場", "rno": "R", "racer1_name": "1号艇選手", "racer2_name": "2号艇選手"}
        )[["場", "R", "1号艇選手", "1号艇イン逃げ率", "2号艇選手", "2号艇逃し率"]]
        st.dataframe(display_df, hide_index=True, use_container_width=True)
        st.caption(
            "①1号艇イン逃げ率80%以上 かつ ②2号艇逃し率50%以上 の条件に合致したレースです。"
            "サンプル数(n)が少ない選手は参考データ不足である旨を併記しています。"
            "レース前の情報に基づく参考表示であり、結果を保証するものではありません。"
        )

st.divider()

# ---------------------------------------------------------------------------
# 日付選択
# ---------------------------------------------------------------------------
dates = load_df("SELECT DISTINCT race_date FROM races ORDER BY race_date DESC")["race_date"].tolist()

if not dates:
    st.warning("データがまだありません。boatrace_scraper.py を実行してデータを取得してください。")
    st.stop()

st.sidebar.header("レース選択")
selected_date = st.sidebar.selectbox("日付", dates, format_func=fmt_date)

# ---------------------------------------------------------------------------
# 場・レース一覧
# ---------------------------------------------------------------------------
venues_df = load_df(
    """SELECT DISTINCT jcd, venue_name FROM races
       WHERE race_date = ? ORDER BY jcd""",
    (selected_date,),
)

if venues_df.empty:
    st.info(f"{fmt_date(selected_date)} の開催データがありません。")
    st.stop()

venue_options = {f"{row.venue_name} ({row.jcd})": row.jcd for row in venues_df.itertuples()}
selected_venue_label = st.sidebar.selectbox("場", list(venue_options.keys()))
selected_jcd = venue_options[selected_venue_label]

races_df = load_df(
    """SELECT rno, title, race_type, distance, weather, temperature,
              wind_speed, water_temp, wave_height, kimarite
       FROM races WHERE race_date = ? AND jcd = ? ORDER BY rno""",
    (selected_date, selected_jcd),
)

race_options = {f"{row.rno}R": row.rno for row in races_df.itertuples()}
selected_race_label = st.sidebar.selectbox("レース", list(race_options.keys()))
selected_rno = race_options[selected_race_label]

# ---------------------------------------------------------------------------
# その日・その場のレース一覧(概要表)
# ---------------------------------------------------------------------------
st.subheader(f"📅 {fmt_date(selected_date)} 開催場一覧")
venue_summary = load_df(
    """SELECT jcd AS 場コード, venue_name AS 場名, COUNT(*) AS レース数
       FROM races WHERE race_date = ? GROUP BY jcd, venue_name ORDER BY jcd""",
    (selected_date,),
)
st.dataframe(venue_summary, hide_index=True, use_container_width=True)

st.subheader(f"🏁 {selected_venue_label} レース一覧")
race_list_display = races_df.rename(columns={
    "rno": "R", "title": "タイトル", "race_type": "種別", "distance": "距離",
    "weather": "天候", "temperature": "気温", "wind_speed": "風速",
    "water_temp": "水温", "wave_height": "波高", "kimarite": "決まり手",
})
st.dataframe(race_list_display, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# 選択レースの詳細
# ---------------------------------------------------------------------------
st.divider()
race_row = races_df[races_df["rno"] == selected_rno].iloc[0]
st.header(f"{selected_venue_label} {selected_rno}R の詳細")
st.write(f"**{race_row['title'] or ''}** / {race_row['race_type'] or ''} {race_row['distance'] or ''}")

cols = st.columns(5)
cols[0].metric("天候", race_row["weather"] or "-")
cols[1].metric("気温", f"{race_row['temperature']}℃" if pd.notna(race_row["temperature"]) else "-")
cols[2].metric("風速", f"{race_row['wind_speed']}m" if pd.notna(race_row["wind_speed"]) else "-")
cols[3].metric("水温", f"{race_row['water_temp']}℃" if pd.notna(race_row["water_temp"]) else "-")
cols[4].metric("波高", f"{race_row['wave_height']}cm" if pd.notna(race_row["wave_height"]) else "-")

tab_entries, tab_results, tab_payouts = st.tabs(["出走表", "結果", "払戻金"])

with tab_entries:
    entries_df = load_df(
        """SELECT waku AS 枠, racer_name AS 選手名, racer_class AS 級別, toban AS 登番,
                  branch AS 支部, hometown AS 出身地, age AS 年齢, weight AS 体重,
                  f_count AS F数, l_count AS L数, avg_st AS 平均ST,
                  national_win_rate AS 全国勝率, national_2rate AS 全国2連率, national_3rate AS 全国3連率,
                  local_win_rate AS 当地勝率, local_2rate AS 当地2連率, local_3rate AS 当地3連率,
                  motor_no AS モーター番号, motor_2rate AS モーター2連率, motor_3rate AS モーター3連率,
                  boat_no AS ボート番号, boat_2rate AS ボート2連率, boat_3rate AS ボート3連率
           FROM entries WHERE race_date = ? AND jcd = ? AND rno = ? ORDER BY waku""",
        (selected_date, selected_jcd, selected_rno),
    )
    if entries_df.empty:
        st.info("出走表データがありません。")
    else:
        st.dataframe(entries_df, hide_index=True, use_container_width=True)

with tab_results:
    results_df = load_df(
        """SELECT rank AS 着順, waku AS 枠, racer_name AS 選手名, toban AS 登番,
                  race_time AS レースタイム, start_timing AS ST
           FROM results WHERE race_date = ? AND jcd = ? AND rno = ? ORDER BY waku""",
        (selected_date, selected_jcd, selected_rno),
    )
    if results_df.empty:
        st.info("結果データがありません(レース未実施、または未取得の可能性があります)。")
    else:
        st.dataframe(results_df, hide_index=True, use_container_width=True)

with tab_payouts:
    payouts_df = load_df(
        """SELECT bet_type AS 勝式, combination AS 組番, payout AS 払戻金, popularity AS 人気
           FROM payouts WHERE race_date = ? AND jcd = ? AND rno = ?""",
        (selected_date, selected_jcd, selected_rno),
    )
    if payouts_df.empty:
        st.info("払戻金データがありません(レース未実施、または未取得の可能性があります)。")
    else:
        payouts_df["払戻金"] = payouts_df["払戻金"].apply(format_yen)
        st.dataframe(payouts_df, hide_index=True, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# イン逃げ狙い目レース分析(実験的機能・全期間)
#
# 「🎯 今日のおすすめ」と同じ条件・同じ選手統計(c1_stats/c2_stats)を使い、
# 対象を「当日のみ」ではなく「全期間の全レース」に広げて集計する。
# 条件に合致したレースのうち、実際に1号艇が逃げた場合の2連単の
# 出現頻度上位3つを表示する。
#
# データ量が少ないうちは該当レースが0件になったり、少数のレースだけで
# 統計を語ることになるのは想定通り。ここでは仕組みが正しく動くことの
# 確認を目的とする。
# ---------------------------------------------------------------------------
st.header("🎯 イン逃げ狙い目レース分析(実験的機能)")
st.caption(
    "①1号艇選手のイン逃げ率(1コース走行時の1着率)が80%以上、"
    "②2号艇選手の逃し率(2コース走行時に1号艇へ1着を譲った率)が50%以上、"
    "の両方を満たすレースを全期間のデータから抽出し、"
    "実際に1号艇が逃げた場合の2連単の上位3つを集計します。"
)

if c1_stats is None:
    st.info("分析に必要なデータがまだありません。")
else:
    candidates = find_qualifying_races(entries_all, c1_stats, c2_stats)

    st.subheader(f"条件に合致したレース: {len(candidates)}件")
    if candidates.empty:
        st.info("条件(①イン逃げ率80%以上 かつ ②2コース逃し率50%以上)に合致するレースは見つかりませんでした。")
    else:
        candidates = add_rate_labels(candidates, c1_stats, c2_stats)
        candidates["日付"] = candidates["race_date"].apply(fmt_date)
        display_df = candidates.rename(
            columns={"venue_name": "場", "rno": "R", "racer1_name": "1号艇選手", "racer2_name": "2号艇選手"}
        )[["日付", "場", "R", "1号艇選手", "1号艇イン逃げ率", "2号艇選手", "2号艇逃し率"]]
        st.dataframe(display_df, hide_index=True, use_container_width=True)

    st.subheader("③ 実際に1号艇が逃げた場合の2連単 上位3")
    if candidates.empty:
        st.info("条件に合致するレースがないため、集計対象がありません。")
    else:
        cand_with_result = candidates.merge(waku1_rank, on=["race_date", "jcd", "rno"], how="left")
        escaped = cand_with_result[cand_with_result["waku1_rank"] == "1"]
        rank2 = results_all[results_all["rank"] == "2"][["race_date", "jcd", "rno", "waku"]].rename(
            columns={"waku": "waku_2nd"}
        )
        escaped = escaped.merge(rank2, on=["race_date", "jcd", "rno"], how="inner")

        total_escaped = len(escaped)
        st.write(
            f"条件に合致したレース{len(candidates)}件のうち、"
            f"実際に1号艇が1着だった件数(2着艇の結果も判明済み): {total_escaped}件"
        )
        if total_escaped < SAMPLE_SIZE_WARNING_THRESHOLD:
            st.warning("⚠️ サンプル数が少なく、参考データ不足です。")

        if total_escaped == 0:
            st.info("該当レースで実際に1号艇が逃げた事例がまだありません。")
        else:
            escaped["combo"] = "1-" + escaped["waku_2nd"].astype(int).astype(str)
            top3 = escaped["combo"].value_counts().head(3)
            for combo, cnt in top3.items():
                st.write(f"- **{combo}**: {cnt}回 (n={total_escaped}中)")

st.divider()
st.caption(f"DB: {DB_PATH}")

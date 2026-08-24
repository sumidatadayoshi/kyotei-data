"""
BOATRACEデータの動作確認用ダッシュボード。

取得したデータ(出走表・結果・払戻金)が正しく保存されているかを
目視確認することが目的。統計分析はまだ実装しない。

使い方:
    streamlit run dashboard.py
"""

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

DB_PATH = Path(__file__).parent / "data" / "boatrace.db"

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


st.title("🚤 BOATRACE データ確認ダッシュボード")
st.caption("取得済みデータの中身をレース単位で目視確認するための簡易ツールです。")

conn = get_connection()

# ---------------------------------------------------------------------------
# 日付選択
# ---------------------------------------------------------------------------
dates = load_df("SELECT DISTINCT race_date FROM races ORDER BY race_date DESC")["race_date"].tolist()

if not dates:
    st.warning("データがまだありません。boatrace_scraper.py を実行してデータを取得してください。")
    st.stop()

def fmt_date(d):
    return f"{d[0:4]}-{d[4:6]}-{d[6:8]}"

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
st.caption(f"DB: {DB_PATH}")

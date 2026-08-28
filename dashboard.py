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
    または全期間)を渡す。

    starts(出走回数)がSAMPLE_SIZE_WARNING_THRESHOLD未満の選手は対象から除く。
    サンプル数が少ないと、算術的に閾値(80%/50%)を満たせるのは「全勝/全敗」の
    選手だけになり、③の「実際に逃げた割合」検証が予測に使った実績とほぼ同じ
    レースを数え直すだけになって100%に張り付くデータリーケージが起きるため。"""
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


def add_rate_labels(candidates, c1_stats, c2_stats):
    candidates = candidates.copy()
    candidates["1号艇イン逃げ率"] = candidates["toban1"].apply(lambda t: rate_label(c1_stats, t))
    candidates["2号艇逃し率"] = candidates["toban2"].apply(lambda t: rate_label(c2_stats, t))
    return candidates


def classify_composition(genders):
    """レース1走分の6艇のgender集合から、性別構成(混合戦/女子戦/男子戦)を判定する。
    genderが未取得の選手が1人でもいる場合は判定不能としてNoneを返す。"""
    genders = set(g for g in genders if g)
    if not genders:
        return None
    if genders == {"女"}:
        return "女子戦"
    if genders == {"男"}:
        return "男子戦"
    return "混合戦"


TODAY_GRADE_CHOICES = ["すべてのグレード", "SG・G1", "その他(G2・G3・一般戦)"]


def grade_bucket(grade):
    """「今日のおすすめ」のグレード絞り込み用に、gradeを3分類のどれかに丸める。"""
    if grade in ("SG", "G1"):
        return "SG・G1"
    return "その他(G2・G3・一般戦)"


st.title("🚤 BOATRACE データ確認ダッシュボード")
st.caption("取得済みデータの中身をレース単位で目視確認するための簡易ツールです。")

conn = get_connection()

entries_all = load_df("SELECT race_date, jcd, rno, waku, toban, racer_name, gender, venue_name FROM entries")
results_all = load_df("SELECT race_date, jcd, rno, waku, rank FROM results")
races_grades = load_df("SELECT race_date, jcd, rno, grade FROM races")

if not entries_all.empty and not results_all.empty:
    c1_stats, c2_stats, waku1_rank = compute_racer_rate_stats(entries_all, results_all)
else:
    c1_stats = c2_stats = waku1_rank = None

# レースごとの性別構成(混合戦/女子戦/男子戦)。グレードと合わせて
# 絞り込みフィルターや各表の付加情報として使う。
if not entries_all.empty:
    race_composition = (
        entries_all.groupby(["race_date", "jcd", "rno"])["gender"]
        .apply(classify_composition)
        .reset_index(name="composition")
    )
else:
    race_composition = pd.DataFrame(columns=["race_date", "jcd", "rno", "composition"])

race_meta = races_grades.merge(race_composition, on=["race_date", "jcd", "rno"], how="left")

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
    today_grade_choice = st.radio(
        "グレードで絞り込み", TODAY_GRADE_CHOICES, horizontal=True, key="today_grade_filter",
    )
    today_race_keys = race_meta[race_meta["race_date"] == today_str][["race_date", "jcd", "rno", "grade"]].copy()
    today_race_keys["bucket"] = today_race_keys["grade"].apply(grade_bucket)
    if today_grade_choice != "すべてのグレード":
        today_race_keys = today_race_keys[today_race_keys["bucket"] == today_grade_choice]
    today_entries_filtered = today_entries.merge(
        today_race_keys[["race_date", "jcd", "rno"]], on=["race_date", "jcd", "rno"], how="inner"
    )

    today_candidates = find_qualifying_races(today_entries_filtered, c1_stats, c2_stats)
    if today_candidates.empty:
        st.info("本日は条件に合うレースがありません。")
    else:
        today_candidates = add_rate_labels(today_candidates, c1_stats, c2_stats)
        today_candidates = today_candidates.merge(
            race_meta[["race_date", "jcd", "rno", "grade", "composition"]],
            on=["race_date", "jcd", "rno"], how="left",
        )
        today_candidates["グレード"] = today_candidates["grade"].fillna("不明")
        today_candidates["性別構成"] = today_candidates["composition"].fillna("不明")
        display_df = today_candidates.rename(
            columns={"venue_name": "場", "rno": "R", "racer1_name": "1号艇選手", "racer2_name": "2号艇選手"}
        )[["場", "R", "グレード", "性別構成", "1号艇選手", "1号艇イン逃げ率", "2号艇選手", "2号艇逃し率"]]
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
    """SELECT race_date, rno, title, race_type, distance, grade, weather, temperature,
              wind_speed, water_temp, wave_height, kimarite
       FROM races WHERE race_date = ? AND jcd = ? ORDER BY rno""",
    (selected_date, selected_jcd),
)
races_df = races_df.merge(
    race_composition[race_composition["jcd"] == selected_jcd][["race_date", "rno", "composition"]],
    on=["race_date", "rno"], how="left",
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
race_list_display = races_df.drop(columns=["race_date"]).rename(columns={
    "rno": "R", "title": "タイトル", "race_type": "種別", "distance": "距離",
    "grade": "グレード", "composition": "性別構成",
    "weather": "天候", "temperature": "気温", "wind_speed": "風速",
    "water_temp": "水温", "wave_height": "波高", "kimarite": "決まり手",
})
race_list_display["グレード"] = race_list_display["グレード"].fillna("不明")
race_list_display["性別構成"] = race_list_display["性別構成"].fillna("不明")
st.dataframe(race_list_display, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# 選択レースの詳細
# ---------------------------------------------------------------------------
st.divider()
race_row = races_df[races_df["rno"] == selected_rno].iloc[0]
st.header(f"{selected_venue_label} {selected_rno}R の詳細")
st.write(
    f"**{race_row['title'] or ''}** / {race_row['race_type'] or ''} {race_row['distance'] or ''} / "
    f"グレード: {race_row['grade'] or '不明'} / 性別構成: {race_row['composition'] or '不明'}"
)

cols = st.columns(5)
cols[0].metric("天候", race_row["weather"] or "-")
cols[1].metric("気温", f"{race_row['temperature']}℃" if pd.notna(race_row["temperature"]) else "-")
cols[2].metric("風速", f"{race_row['wind_speed']}m" if pd.notna(race_row["wind_speed"]) else "-")
cols[3].metric("水温", f"{race_row['water_temp']}℃" if pd.notna(race_row["water_temp"]) else "-")
cols[4].metric("波高", f"{race_row['wave_height']}cm" if pd.notna(race_row["wave_height"]) else "-")

tab_entries, tab_results, tab_payouts = st.tabs(["出走表", "結果", "払戻金"])

with tab_entries:
    entries_df = load_df(
        """SELECT waku AS 枠, racer_name AS 選手名, gender AS 性別, racer_class AS 級別, toban AS 登番,
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
# 🔍 絞り込み(グレード・性別構成)
#
# 以下の「イン逃げ狙い目レース分析」「回収率シミュレーター」共通の
# 絞り込み条件。デフォルト(全選択/すべて)では従来と同じ全レースが
# 対象になり、既存の集計結果は変わらない。
# ---------------------------------------------------------------------------
st.subheader("🔍 絞り込み(グレード・性別構成)")
st.caption("この下の「イン逃げ狙い目レース分析」と「回収率シミュレーター」の両方に適用されます。")

GRADE_ORDER = ["SG", "G1", "G2", "G3", "一般"]
present_grades = set(races_grades["grade"].dropna())
grade_options = [g for g in GRADE_ORDER if g in present_grades]
if races_grades["grade"].isna().any():
    grade_options = grade_options + ["不明"]

f1, f2 = st.columns([2, 1])
selected_grades = f1.multiselect("グレード", grade_options, default=grade_options)
composition_choice = f2.selectbox("性別構成", ["すべて", "混合戦", "女子戦", "男子戦"])

race_meta_filtered = race_meta.copy()
race_meta_filtered["grade_label"] = race_meta_filtered["grade"].fillna("不明")
race_meta_filtered = race_meta_filtered[race_meta_filtered["grade_label"].isin(selected_grades)]
if composition_choice != "すべて":
    race_meta_filtered = race_meta_filtered[race_meta_filtered["composition"] == composition_choice]

entries_filtered = entries_all.merge(
    race_meta_filtered[["race_date", "jcd", "rno"]], on=["race_date", "jcd", "rno"], how="inner"
)

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
    candidates = find_qualifying_races(entries_filtered, c1_stats, c2_stats)

    st.subheader(f"条件に合致したレース: {len(candidates)}件")
    if candidates.empty:
        st.info("条件(①イン逃げ率80%以上 かつ ②2コース逃し率50%以上)に合致するレースは見つかりませんでした。")
    else:
        candidates = add_rate_labels(candidates, c1_stats, c2_stats)
        candidates["日付"] = candidates["race_date"].apply(fmt_date)
        candidates = candidates.merge(race_meta[["race_date", "jcd", "rno", "grade", "composition"]],
                                       on=["race_date", "jcd", "rno"], how="left")
        candidates["グレード"] = candidates["grade"].fillna("不明")
        candidates["性別構成"] = candidates["composition"].fillna("不明")
        display_df = candidates.rename(
            columns={"venue_name": "場", "rno": "R", "racer1_name": "1号艇選手", "racer2_name": "2号艇選手"}
        )[["日付", "場", "R", "グレード", "性別構成", "1号艇選手", "1号艇イン逃げ率", "2号艇選手", "2号艇逃し率"]]
        st.dataframe(display_df, hide_index=True, use_container_width=True)

    if candidates.empty:
        cand_with_result = candidates
        known_result = candidates
        escaped = candidates
    else:
        cand_with_result = candidates.merge(waku1_rank, on=["race_date", "jcd", "rno"], how="left")
        known_result = cand_with_result[cand_with_result["waku1_rank"].notna()]
        escaped = known_result[known_result["waku1_rank"] == "1"]

    st.subheader("① 実際に1号艇が逃げた割合(結果検証)")
    st.caption(
        "①②の条件(過去実績に基づく事前の予測)に合致したレースについて、"
        "予測ではなく実際の結果として1号艇が1着(逃げ)だった割合です。"
        "「予測がどれだけ当たっていたか」を検証するための表示です。"
    )
    if candidates.empty:
        st.info("条件に合致するレースがないため、集計対象がありません。")
    elif known_result.empty:
        st.info("条件に合致したレースの中に、結果が判明しているものがまだありません。")
    else:
        total_known = len(known_result)
        escape_count = len(escaped)
        escape_rate = (escape_count / total_known * 100) if total_known > 0 else 0.0

        m1, m2, m3 = st.columns(3)
        m1.metric("対象レース数", f"{total_known}件")
        m2.metric("実際に逃げた回数", f"{escape_count}回")
        m3.metric("実際の逃げ率", f"{escape_rate:.1f}%")

        if total_known < SAMPLE_SIZE_WARNING_THRESHOLD:
            st.warning("⚠️ 対象レース数が少なく、参考データ不足です。")

    st.subheader("③ 実際に1号艇が逃げた場合の2連単 上位3")
    if candidates.empty:
        st.info("条件に合致するレースがないため、集計対象がありません。")
    elif escaped.empty:
        st.info("該当レースで実際に1号艇が逃げた事例がまだありません。")
    else:
        rank2 = results_all[results_all["rank"] == "2"][["race_date", "jcd", "rno", "waku"]].rename(
            columns={"waku": "waku_2nd"}
        )
        escaped_with_2nd = escaped.merge(rank2, on=["race_date", "jcd", "rno"], how="inner")

        total_escaped = len(escaped_with_2nd)
        st.write(
            f"条件に合致したレース{len(candidates)}件のうち、"
            f"実際に1号艇が1着だった件数(2着艇の結果も判明済み): {total_escaped}件"
        )
        if total_escaped < SAMPLE_SIZE_WARNING_THRESHOLD:
            st.warning("⚠️ サンプル数が少なく、参考データ不足です。")

        if total_escaped == 0:
            st.info("該当レースで実際に1号艇が逃げた事例がまだありません。")
        else:
            escaped_with_2nd["combo"] = "1-" + escaped_with_2nd["waku_2nd"].astype(int).astype(str)
            top3 = escaped_with_2nd["combo"].value_counts().head(3)
            for combo, cnt in top3.items():
                st.write(f"- **{combo}**: {cnt}回 (n={total_escaped}中)")

st.divider()

# ---------------------------------------------------------------------------
# 💰 回収率シミュレーター(実験的機能)
#
# 「イン逃げ狙い目レース分析」で条件(①②)に合致した過去レースのうち、
# 実際に1号艇が逃げた場合に最も出現頻度が高かった2連単の目を「毎回同じ
# 買い方」として固定し、条件に合致し結果が判明している全レースに
# 100円ずつ賭けていたと仮定した場合の回収率を計算する。
#
# 回収率 = 的中時の払戻金合計 ÷ 賭け金合計(100円 × 対象レース数)
#
# データ量が少ないうちは母数が少なく参考にならないのは想定通り。
# ここでは仕組みが正しく動くことの確認を目的とする。
# ---------------------------------------------------------------------------
st.header("💰 回収率シミュレーター(実験的機能)")
st.caption(
    "①②の条件に合致した過去レースのうち、実際に1号艇が逃げた場合の2連単で"
    "最も出現頻度が高かった目を「毎回同じ買い方」として固定し、"
    "条件に合致し結果が判明している対象レース全てに100円ずつ賭けていたと"
    "仮定した場合の回収率を計算します。"
)

BET_AMOUNT = 100

if c1_stats is None:
    st.info("分析に必要なデータがまだありません。")
else:
    sim_candidates = find_qualifying_races(entries_filtered, c1_stats, c2_stats)

    if sim_candidates.empty:
        st.info("条件に合致するレースがないため、シミュレーションできません。")
    else:
        payouts_2tan = load_df(
            "SELECT race_date, jcd, rno, combination, payout FROM payouts WHERE bet_type = '2連単'"
        )
        concluded = sim_candidates.merge(payouts_2tan, on=["race_date", "jcd", "rno"], how="inner")

        if concluded.empty:
            st.info("条件に合致したレースの中に、結果が判明しているものがまだありません。")
        else:
            escaped_combos = concluded[concluded["combination"].str.startswith("1-")]
            if escaped_combos.empty:
                st.info(
                    "条件に合致し結果が判明したレースで、実際に1号艇が逃げた事例が"
                    "まだないため、賭け目を決定できません。"
                )
            else:
                best_combo = escaped_combos["combination"].value_counts().index[0]

                total_races = len(concluded)
                hits = concluded[concluded["combination"] == best_combo]
                hit_count = len(hits)
                total_return = int(hits["payout"].sum())
                total_stake = total_races * BET_AMOUNT
                recovery_rate = (total_return / total_stake * 100) if total_stake > 0 else 0.0
                hit_rate = (hit_count / total_races * 100) if total_races > 0 else 0.0

                st.write(f"買い目(固定): **2連単 {best_combo}**(過去に1号艇が逃げた際の最頻出目)")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("対象レース数(母数)", f"{total_races}件")
                m2.metric("的中回数", f"{hit_count}回")
                m3.metric("的中率", f"{hit_rate:.1f}%")
                m4.metric("回収率", f"{recovery_rate:.1f}%")

                st.caption(
                    f"賭け金合計: {BET_AMOUNT}円 × {total_races}件 = {total_stake:,}円 / "
                    f"払戻金合計(的中分): {total_return:,}円"
                )

                if total_races < SAMPLE_SIZE_WARNING_THRESHOLD:
                    st.warning("⚠️ 対象レース数(母数)が少なく、参考データ不足です。")

st.divider()

# ---------------------------------------------------------------------------
# 💰 「1-3・1-4」固定買い回収率シミュレーター(実験的機能)
#
# 既存の回収率シミュレーター(出現頻度が最も高い目を買う方式)とは別に、
# ①②の条件に合致した対象レースすべてで、2連単「1-3」「1-4」をそれぞれ
# 毎回固定で購入した場合(1レースあたり100円)の回収率を、1-3・1-4それぞれ
# 個別に計算する。
# ---------------------------------------------------------------------------
st.header("💰「1-3・1-4」固定買い回収率シミュレーター(実験的機能)")
st.caption(
    "既存の回収率シミュレーターとは別に、①②の条件に合致した対象レースすべてで"
    "「2連単 1-3」または「2連単 1-4」を毎回固定で購入し続けた場合"
    "(1レースあたり100円)の回収率を、それぞれ個別に計算します。"
    "あわせて「1-2・1-3・1-4」の3点買いを続けた場合の的中率・回収率も計算します。"
)

FIXED_BET_AMOUNT = 100
FIXED_COMBOS = ["1-3", "1-4"]

if c1_stats is None:
    st.info("分析に必要なデータがまだありません。")
else:
    fixed_candidates = find_qualifying_races(entries_filtered, c1_stats, c2_stats)

    if fixed_candidates.empty:
        st.info("条件に合致するレースがないため、シミュレーションできません。")
    else:
        payouts_2tan_fixed = load_df(
            "SELECT race_date, jcd, rno, combination, payout FROM payouts WHERE bet_type = '2連単'"
        )
        fixed_concluded = fixed_candidates.merge(
            payouts_2tan_fixed, on=["race_date", "jcd", "rno"], how="inner"
        )

        if fixed_concluded.empty:
            st.info("条件に合致したレースの中に、結果が判明しているものがまだありません。")
        else:
            total_races = len(fixed_concluded)
            hits_13 = fixed_concluded[fixed_concluded["combination"] == "1-3"]
            hits_14 = fixed_concluded[fixed_concluded["combination"] == "1-4"]
            hit_count_13 = len(hits_13)
            hit_count_14 = len(hits_14)
            return_13 = int(hits_13["payout"].sum())
            return_14 = int(hits_14["payout"].sum())
            stake_13 = total_races * FIXED_BET_AMOUNT
            stake_14 = total_races * FIXED_BET_AMOUNT
            recovery_rate_13 = (return_13 / stake_13 * 100) if stake_13 > 0 else 0.0
            recovery_rate_14 = (return_14 / stake_14 * 100) if stake_14 > 0 else 0.0
            hit_rate_13 = (hit_count_13 / total_races * 100) if total_races > 0 else 0.0
            hit_rate_14 = (hit_count_14 / total_races * 100) if total_races > 0 else 0.0

            triple_combos = ["1-2", "1-3", "1-4"]
            hits_triple = fixed_concluded[fixed_concluded["combination"].isin(triple_combos)]
            hit_count_triple = len(hits_triple)
            return_triple = int(hits_triple["payout"].sum())
            stake_triple = total_races * FIXED_BET_AMOUNT * len(triple_combos)
            recovery_rate_triple = (return_triple / stake_triple * 100) if stake_triple > 0 else 0.0
            hit_rate_triple = (hit_count_triple / total_races * 100) if total_races > 0 else 0.0

            st.subheader("2連単 1-3 のみ買い続けた場合")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("対象レース数", f"{total_races}件")
            m2.metric("的中回数", f"{hit_count_13}回")
            m3.metric("的中率", f"{hit_rate_13:.1f}%")
            m4.metric("回収率", f"{recovery_rate_13:.1f}%")
            st.caption(f"賭け金合計: {stake_13:,}円 / 払戻金合計(的中分): {return_13:,}円")

            st.subheader("2連単 1-4 のみ買い続けた場合")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("対象レース数", f"{total_races}件")
            m2.metric("的中回数", f"{hit_count_14}回")
            m3.metric("的中率", f"{hit_rate_14:.1f}%")
            m4.metric("回収率", f"{recovery_rate_14:.1f}%")
            st.caption(f"賭け金合計: {stake_14:,}円 / 払戻金合計(的中分): {return_14:,}円")

            st.subheader("2連単 1-2・1-3・1-4 の3点買いを続けた場合")
            st.caption("3点(1-2・1-3・1-4)のうち、いずれか1点でも当たった割合・回収率です。")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("対象レース数", f"{total_races}件")
            m2.metric("的中回数", f"{hit_count_triple}回")
            m3.metric("的中率", f"{hit_rate_triple:.1f}%")
            m4.metric("回収率", f"{recovery_rate_triple:.1f}%")
            st.caption(f"賭け金合計: {stake_triple:,}円 / 払戻金合計(的中分): {return_triple:,}円")

            total_return = return_13 + return_14
            total_stake = stake_13 + stake_14
            recovery_rate = (total_return / total_stake * 100) if total_stake > 0 else 0.0
            st.caption(
                f"(参考)1-3・1-4 合計: 賭け金合計 {total_stake:,}円 / "
                f"払戻金合計 {total_return:,}円 / 回収率 {recovery_rate:.1f}%"
            )

            if total_races < SAMPLE_SIZE_WARNING_THRESHOLD:
                st.warning("⚠️ 対象レース数(母数)が少なく、参考データ不足です。")

st.divider()
st.caption(f"DB: {DB_PATH}")

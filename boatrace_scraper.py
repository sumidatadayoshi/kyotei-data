"""
BOATRACE公式サイト (https://www.boatrace.jp) から
指定日の全レース場の出走表・結果・払戻金を取得し、SQLiteに保存する。

個人の私的利用を目的としたデータ収集スクリプト。
サイトへの負荷を避けるため、リクエスト間隔を空けて順番にアクセスする。

使い方:
    python boatrace_scraper.py                 # 前日分を取得
    python boatrace_scraper.py --date 20260821  # 日付を指定して取得
    python boatrace_scraper.py --db data\\boatrace.db --interval 1.5
"""

import argparse
import html as html_module
import logging
import re
import sqlite3
import sys
import time
import unicodedata
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# 開催日の判定は常に日本時間(JST)基準で行う。
# GitHub Actions等のCIランナーはOS時刻がUTCのため、datetime.now()をそのまま使うと
# 日付がずれる可能性がある。
JST = ZoneInfo("Asia/Tokyo")

BASE_URL = "https://www.boatrace.jp"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

# 全24競艇場 (場コード: 場名)。ページ側にも場名は載っているが、
# 取得失敗時のフォールバック用に静的な参照データとして保持する。
JCD_NAMES = {
    "01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島",
    "05": "多摩川", "06": "浜名湖", "07": "蒲郡", "08": "常滑",
    "09": "津", "10": "三国", "11": "びわこ", "12": "住之江",
    "13": "尼崎", "14": "鳴門", "15": "丸亀", "16": "児島",
    "17": "宮島", "18": "徳山", "19": "下関", "20": "若松",
    "21": "芦屋", "22": "福岡", "23": "唐津", "24": "大村",
}

MAX_RACES_PER_VENUE = 12

# レース格付け(heading2_titleのCSSクラスから判定)。
# is-ippan / is-G3b は実データで確認済み。is-sg / is-g1 / is-g2 は
# サイトのナビゲーション表記(「SG・PG1」「G1・G2」「G3」)からの類推であり、
# 実際にSG/G1/G2開催があった際に検証が必要。
GRADE_CLASS_PREFIXES = [
    ("is-sg", "SG"),
    ("is-pg1", "SG"),
    ("is-g1", "G1"),
    ("is-g2", "G2"),
    ("is-g3", "G3"),
    ("is-ippan", "一般"),
]

# 対象日が「終了しているはずの日」の場合に、結果取得率がこれを下回ると
# scrape_logのstatusを'partial'にし、mainの終了コードを1にする。
RESULTS_COMPLETENESS_THRESHOLD = 0.9

logger = logging.getLogger("boatrace_scraper")


def nfkc(s):
    """全角英数字・全角スペースなどを半角に正規化する。"""
    if s is None:
        return None
    return unicodedata.normalize("NFKC", s).strip()


def to_int(s):
    if s is None:
        return None
    s = s.strip()
    if not s or s in ("-", "--", "---"):
        return None
    try:
        return int(s)
    except ValueError:
        return None


def to_float(s):
    if s is None:
        return None
    s = s.strip()
    if not s or s in ("-", "--", "---"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def extract_number(s):
    """文字列先頭の数値部分だけを取り出してfloatにする (例: '30.0℃' -> 30.0, '2m' -> 2.0)。
    ℃はNFKC正規化で'°C'に分解されるため、単位文字列を直接切り出すのではなくこちらを使う。"""
    if s is None:
        return None
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group()) if m else None


def extract_grade(soup):
    """racelist/raceresultページ共通の .heading2_title のCSSクラスから
    レース格付け(SG/G1/G2/G3/一般)を判定する。"""
    div = soup.select_one(".heading2_title")
    if not div:
        return None
    classes = [c.lower() for c in (div.get("class") or [])]
    for cls in classes:
        for prefix, label in GRADE_CLASS_PREFIXES:
            if cls.startswith(prefix):
                return label
    return None


class AccessBlockedError(Exception):
    """403/429などアクセス制限の疑いがあるレスポンスを検知した際に送出される。
    raise_on_block=Trueのセッションでのみ発生し、呼び出し元の処理を即座に
    停止させることを意図している(現状はバックフィル処理でのみ使用)。"""


class PoliteSession:
    """アクセス間隔を必ず空けるrequests.Sessionのラッパー。

    raise_on_block=True の場合、403/429を検知すると通常のリトライで
    畳みかけることを避けるため一度だけ長めに待って再確認し、それでも
    ブロックされていればAccessBlockedErrorを送出して即座に諦める。
    既存の日次スクレイピング(daily-scrape.yml等)には影響を与えないよう、
    デフォルトはFalse(従来通りログを出して他ページの取得を続ける)。"""

    def __init__(self, interval_sec=1.5, timeout=15, max_retries=3, raise_on_block=False):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.interval_sec = interval_sec
        self.timeout = timeout
        self.max_retries = max_retries
        self.raise_on_block = raise_on_block
        self._last_request_at = 0.0

    def get(self, url):
        wait = self.interval_sec - (time.time() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)

        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, timeout=self.timeout)
                self._last_request_at = time.time()
                if resp.status_code == 200:
                    resp.encoding = "utf-8"
                    return resp.text
                if resp.status_code in (403, 429) and self.raise_on_block:
                    logger.error(
                        "HTTP %s (アクセス制限の疑い) for %s (attempt %d)", resp.status_code, url, attempt
                    )
                    if attempt >= 2:
                        raise AccessBlockedError(f"HTTP {resp.status_code} received for {url}")
                    time.sleep(10)
                    continue
                logger.warning("HTTP %s for %s (attempt %d)", resp.status_code, url, attempt)
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning("Request failed for %s (attempt %d): %s", url, attempt, exc)
            time.sleep(2 * attempt)
        if last_exc:
            logger.error("Giving up on %s: %s", url, last_exc)
        else:
            logger.error("Giving up on %s: non-200 responses only", url)
        return None


# ---------------------------------------------------------------------------
# 開催場一覧の取得
# ---------------------------------------------------------------------------

def get_active_venues(session, date_str):
    """指定日 (YYYYMMDD) に開催しているレース場一覧 [(jcd, event_title), ...] を返す。"""
    url = f"{BASE_URL}/owpc/pc/race/index?hd={date_str}"
    text = session.get(url)
    if text is None:
        return []

    pattern = re.compile(
        r'raceindex\?jcd=(\d{2})&amp;hd=' + re.escape(date_str) + r'">([^<]+)</a>'
    )
    venues = []
    seen = set()
    for jcd, title in pattern.findall(text):
        if jcd in seen:
            continue
        seen.add(jcd)
        venues.append((jcd, html_module.unescape(title).strip()))
    return venues


# ---------------------------------------------------------------------------
# 出走表 (racelist) の解析
# ---------------------------------------------------------------------------

def parse_entries(html_text):
    soup = BeautifulSoup(html_text, "lxml")
    tbodies = [
        tb for tb in soup.find_all("tbody")
        if tb.get("class") and "is-fs12" in tb.get("class")
    ]

    entries = []
    for tb in tbodies:
        tr = tb.find("tr")
        if tr is None:
            continue
        tds = tr.find_all("td", recursive=False)
        if len(tds) < 8:
            continue

        waku = to_int(nfkc(tds[0].get_text(strip=True)))

        info_divs = tds[2].find_all("div")
        toban = racer_class = name = branch = hometown = None
        age = weight = None

        if len(info_divs) > 0:
            reg_text = nfkc(info_divs[0].get_text(" ", strip=True))
            if reg_text and "/" in reg_text:
                left, right = reg_text.split("/", 1)
                toban = left.strip()
                racer_class = right.strip()
        if len(info_divs) > 1:
            name = nfkc(info_divs[1].get_text(strip=True))
            if name:
                name = re.sub(r"\s+", " ", name).strip()
        if len(info_divs) > 2:
            lines = [nfkc(l) for l in info_divs[2].get_text("\n", strip=True).split("\n") if l.strip()]
            if lines:
                if "/" in lines[0]:
                    branch, hometown = [x.strip() for x in lines[0].split("/", 1)]
                if len(lines) > 1:
                    m = re.match(r"(\d+)\s*歳/([\d.]+)\s*kg", lines[1])
                    if m:
                        age = to_int(m.group(1))
                        weight = to_float(m.group(2))

        def cell_lines(td):
            return [nfkc(l) for l in td.get_text("\n", strip=True).split("\n") if l.strip()]

        fl = cell_lines(tds[3])
        f_count = to_int(fl[0][1:]) if len(fl) > 0 and fl[0].startswith("F") else None
        l_count = to_int(fl[1][1:]) if len(fl) > 1 and fl[1].startswith("L") else None
        avg_st = to_float(fl[2]) if len(fl) > 2 else None

        nat = cell_lines(tds[4])
        loc = cell_lines(tds[5])
        mot = cell_lines(tds[6])
        boat = cell_lines(tds[7])

        entries.append({
            "waku": waku,
            "toban": toban,
            "racer_class": racer_class,
            "racer_name": name,
            "branch": branch,
            "hometown": hometown,
            "age": age,
            "weight": weight,
            "f_count": f_count,
            "l_count": l_count,
            "avg_st": avg_st,
            "national_win_rate": to_float(nat[0]) if len(nat) > 0 else None,
            "national_2rate": to_float(nat[1]) if len(nat) > 1 else None,
            "national_3rate": to_float(nat[2]) if len(nat) > 2 else None,
            "local_win_rate": to_float(loc[0]) if len(loc) > 0 else None,
            "local_2rate": to_float(loc[1]) if len(loc) > 1 else None,
            "local_3rate": to_float(loc[2]) if len(loc) > 2 else None,
            "motor_no": to_int(mot[0]) if len(mot) > 0 else None,
            "motor_2rate": to_float(mot[1]) if len(mot) > 1 else None,
            "motor_3rate": to_float(mot[2]) if len(mot) > 2 else None,
            "boat_no": to_int(boat[0]) if len(boat) > 0 else None,
            "boat_2rate": to_float(boat[1]) if len(boat) > 1 else None,
            "boat_3rate": to_float(boat[2]) if len(boat) > 2 else None,
        })

    return entries


def parse_racelist_header(html_text):
    """出走表(racelist)ページのヘッダー部から title/venue_name/race_type/distance を取る。
    結果(raceresult)ページと共通のテンプレートなのでセレクタは同じ。
    レース前のため天候・決まり手などは含まれない。"""
    soup = BeautifulSoup(html_text, "lxml")

    race_info = {
        "title": None, "race_type": None, "distance": None, "grade": None,
        "venue_name": None, "weather": None, "temperature": None,
        "wind_speed": None, "water_temp": None, "wave_height": None,
        "kimarite": None,
    }

    h2 = soup.select_one(".heading2_titleName")
    if h2:
        race_info["title"] = nfkc(h2.get_text(strip=True))

    race_info["grade"] = extract_grade(soup)

    img = soup.select_one(".heading2_area img")
    if img and img.get("alt"):
        race_info["venue_name"] = img.get("alt").strip()

    detail = soup.select_one(".title16_titleDetail__add2020")
    if detail:
        parts = [nfkc(l) for l in detail.get_text("\n", strip=True).split("\n") if l.strip()]
        if parts:
            race_info["race_type"] = parts[0]
        for p in parts[1:]:
            if p.endswith("m"):
                race_info["distance"] = p

    return race_info


# ---------------------------------------------------------------------------
# 結果 (raceresult) の解析: 着順・払戻金・気象・決まり手
# ---------------------------------------------------------------------------

def parse_race_result(html_text):
    soup = BeautifulSoup(html_text, "lxml")

    race_info = {
        "title": None, "race_type": None, "distance": None, "grade": None,
        "venue_name": None, "weather": None, "temperature": None,
        "wind_speed": None, "water_temp": None, "wave_height": None,
        "kimarite": None,
    }

    h2 = soup.select_one(".heading2_titleName")
    if h2:
        race_info["title"] = nfkc(h2.get_text(strip=True))

    race_info["grade"] = extract_grade(soup)

    img = soup.select_one(".heading2_area img")
    if img and img.get("alt"):
        race_info["venue_name"] = img.get("alt").strip()

    detail = soup.select_one(".title16_titleDetail__add2020")
    if detail:
        parts = [nfkc(l) for l in detail.get_text("\n", strip=True).split("\n") if l.strip()]
        if parts:
            race_info["race_type"] = parts[0]
        for p in parts[1:]:
            if p.endswith("m"):
                race_info["distance"] = p

    results = []
    payouts = []

    for table in soup.find_all("table"):
        thead = table.find("thead")
        if not thead:
            continue
        headers = thead.get_text("|", strip=True)

        if "着" in headers and "ボートレーサー" in headers and "レースタイム" in headers:
            for tb in table.find_all("tbody"):
                tr = tb.find("tr")
                if tr is None:
                    continue
                tds = tr.find_all("td")
                if len(tds) < 4:
                    continue
                rank = nfkc(tds[0].get_text(strip=True))
                waku = to_int(nfkc(tds[1].get_text(strip=True)))
                spans = tds[2].find_all("span")
                toban = nfkc(spans[0].get_text(strip=True)) if len(spans) > 0 else None
                name = nfkc(spans[1].get_text(strip=True)) if len(spans) > 1 else None
                if name:
                    name = re.sub(r"\s+", " ", name).strip()
                race_time = nfkc(tds[3].get_text(strip=True)) or None
                results.append({
                    "rank": rank or None,
                    "waku": waku,
                    "toban": toban,
                    "racer_name": name,
                    "race_time": race_time,
                    "start_timing": None,
                })

        elif "勝式" in headers and "組番" in headers:
            for tb in table.find_all("tbody"):
                bet_type = None
                for tr in tb.find_all("tr"):
                    tds = tr.find_all("td")
                    if len(tds) == 4:
                        bet_type = nfkc(tds[0].get_text(strip=True))
                        combo_td, payout_td, pop_td = tds[1], tds[2], tds[3]
                    elif len(tds) == 3:
                        combo_td, payout_td, pop_td = tds[0], tds[1], tds[2]
                    else:
                        continue
                    combo = nfkc(combo_td.get_text(strip=True))
                    payout_text = nfkc(payout_td.get_text(strip=True))
                    pop = nfkc(pop_td.get_text(strip=True))
                    if not combo or not payout_text:
                        continue
                    payout_val = to_int(payout_text.replace("¥", "").replace(",", ""))
                    payouts.append({
                        "bet_type": bet_type,
                        "combination": combo,
                        "payout": payout_val,
                        "popularity": pop or None,
                    })

    # スタート情報 (実際の進入コース順とST) を results にマージする
    start_by_waku = {}
    for div in soup.find_all("div", class_="table1_boatImage1"):
        num_span = div.find("span", class_=lambda c: c and "table1_boatImage1Number" in c)
        time_span = div.find("span", class_=lambda c: c and "table1_boatImage1TimeInner" in c)
        if not num_span:
            continue
        waku = to_int(nfkc(num_span.get_text(strip=True)))
        if waku is None:
            continue
        time_text = nfkc(time_span.get_text(" ", strip=True)) if time_span else ""
        st = time_text.split()[0] if time_text else None
        start_by_waku[waku] = st or None

    for r in results:
        if r["waku"] in start_by_waku:
            r["start_timing"] = start_by_waku[r["waku"]]

    # 決まり手
    for th in soup.find_all("th"):
        if nfkc(th.get_text(strip=True)) == "決まり手":
            parent_table = th.find_parent("table")
            if parent_table:
                td = parent_table.find("tbody").find("td")
                if td:
                    race_info["kimarite"] = nfkc(td.get_text(strip=True)) or None
            break

    # 気象情報
    wtitle = soup.select_one(".weather1_title")
    if wtitle:
        wdiv = wtitle.find_parent("div", class_="weather1")
        if wdiv:
            for unit in wdiv.select(".weather1_bodyUnit"):
                label = unit.select_one(".weather1_bodyUnitLabelTitle")
                if not label:
                    continue
                ltext = nfkc(label.get_text(strip=True))
                data = unit.select_one(".weather1_bodyUnitLabelData")
                dtext = nfkc(data.get_text(strip=True)) if data else None
                dnum = extract_number(dtext)
                if ltext == "気温":
                    race_info["temperature"] = dnum
                elif ltext == "風速":
                    race_info["wind_speed"] = dnum
                elif ltext == "水温":
                    race_info["water_temp"] = dnum
                elif ltext == "波高":
                    race_info["wave_height"] = dnum
                elif dtext is None:
                    race_info["weather"] = ltext

    return race_info, results, payouts


# ---------------------------------------------------------------------------
# 締切時オッズ (oddstf/odds2tf/odds3f/odds3t/oddsk) の解析
#
# 単勝・複勝ページ(oddstf)は艇番ごとに1行のシンプルな表だが、
# 2連単・2連複・拡連複・3連複・3連単の各ページは「6艇の組み合わせ表」を
# HTMLのrowspanで表現したグリッドになっている。rowspanをそのまま
# BeautifulSoupで読むと後続セルの位置がずれるため、まずグリッド全体を
# 展開してから列位置(艇番ブロック)に基づいて値を取り出す。
# ---------------------------------------------------------------------------

def parse_odds_value(text):
    """'12.3' -> (12.3, None)。'1.2-1.5' (複勝・拡連複などのレンジ表記)
    -> (1.2, 1.5)。値なし('---'など)は (None, None)。"""
    text = (text or "").strip()
    if not text or text in ("-", "--", "---"):
        return None, None
    if "-" in text:
        lo, hi = text.split("-", 1)
        return to_float(lo), to_float(hi)
    return to_float(text), None


def expand_grid_rows(tbody):
    """rowspanで畳まれたtbodyを、セルが省略されない完全なグリッド
    (行ごとの {列インデックス: (テキスト, cssクラス一覧)} の辞書)に展開する。"""
    carry = {}
    grid = []
    for tr in tbody.find_all("tr", recursive=False):
        cells = tr.find_all("td", recursive=False)
        row = {}
        col = 0
        ci = 0
        while ci < len(cells) or col in carry:
            if col in carry:
                remaining, text, classes = carry[col]
                row[col] = (text, classes)
                if remaining - 1 <= 0:
                    del carry[col]
                else:
                    carry[col] = (remaining - 1, text, classes)
                col += 1
                continue
            cell = cells[ci]
            ci += 1
            text = nfkc(cell.get_text(strip=True))
            classes = cell.get("class") or []
            rowspan = int(cell.get("rowspan", 1) or 1)
            row[col] = (text, classes)
            if rowspan > 1:
                carry[col] = (rowspan - 1, text, classes)
            col += 1
        grid.append(row)
    return grid


def block_boat_numbers(thead):
    """グリッド表のヘッダーから、左から順に並ぶ艇番ブロックの艇番一覧を返す。"""
    boats = []
    for th in thead.find_all("th"):
        classes = th.get("class") or []
        if th.get("colspan"):
            continue
        if not any(c.startswith("is-boatColor") for c in classes):
            continue
        n = to_int(nfkc(th.get_text(strip=True)))
        if n is not None:
            boats.append(n)
    return boats


def parse_odds_point_table(table):
    """oddstfページ(単勝・複勝)の1艇1行×tbody形式を解析する。"""
    out = []
    for tb in table.find_all("tbody"):
        tr = tb.find("tr")
        if tr is None:
            continue
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue
        boat = to_int(nfkc(tds[0].get_text(strip=True)))
        if boat is None:
            continue
        out.append((boat, nfkc(tds[2].get_text(strip=True))))
    return out


def parse_odds_grid_2level(table):
    """2連単・2連複・拡連複ページの「6ブロック×2列(相手艇番, オッズ)」
    グリッドを解析し、(艇1, 艇2, オッズ文字列) のリストを返す。
    2連複・拡連複は艇1<艇2の組み合わせのみ埋まっており、それ以外は
    is-disabledセルなのでスキップする。"""
    thead = table.find("thead")
    tbody = table.find("tbody")
    boats = block_boat_numbers(thead)
    out = []
    for row in expand_grid_rows(tbody):
        for bi, boat1 in enumerate(boats):
            base = bi * 2
            c0, c1 = row.get(base), row.get(base + 1)
            if c0 is None or c1 is None or "is-disabled" in c0[1]:
                continue
            boat2 = to_int(c0[0])
            if boat2 is None:
                continue
            out.append((boat1, boat2, c1[0]))
    return out


def parse_odds_grid_3level(table):
    """3連単・3連複ページの「6ブロック×3列(2着候補, 3着候補, オッズ)」
    グリッドを解析し、(艇1, 艇2, 艇3, オッズ文字列) のリストを返す。"""
    thead = table.find("thead")
    tbody = table.find("tbody")
    boats = block_boat_numbers(thead)
    out = []
    for row in expand_grid_rows(tbody):
        for bi, boat1 in enumerate(boats):
            base = bi * 3
            c0, c1, c2 = row.get(base), row.get(base + 1), row.get(base + 2)
            if c0 is None or c1 is None or c2 is None or "is-disabled" in c0[1]:
                continue
            boat2, boat3 = to_int(c0[0]), to_int(c1[0])
            if boat2 is None or boat3 is None:
                continue
            out.append((boat1, boat2, boat3, c2[0]))
    return out


# (ページのURLスラッグ, [(bet_type, ページ内での表の順序(先頭のレース選択
#  ナビ表を除く), 表の形状, 艇番の並び順が意味を持つか), ...])
ODDS_PAGE_SPECS = [
    ("oddstf", [("単勝", 0, "point", None), ("複勝", 1, "point", None)]),
    ("odds2tf", [("2連単", 0, "pair", True), ("2連複", 1, "pair", False)]),
    ("odds3f", [("3連複", 0, "triple", False)]),
    ("odds3t", [("3連単", 0, "triple", True)]),
    ("oddsk", [("拡連複", 0, "pair", False)]),
]


def _odds_rows_from_table(table, shape, ordered):
    if shape == "point":
        for boat, text in parse_odds_point_table(table):
            lo, hi = parse_odds_value(text)
            if lo is not None:
                yield str(boat), lo, hi
    elif shape == "pair":
        sep = "-" if ordered else "="
        for b1, b2, text in parse_odds_grid_2level(table):
            lo, hi = parse_odds_value(text)
            if lo is not None:
                yield f"{b1}{sep}{b2}", lo, hi
    else:  # triple
        sep = "-" if ordered else "="
        for b1, b2, b3, text in parse_odds_grid_3level(table):
            lo, hi = parse_odds_value(text)
            if lo is not None:
                yield f"{b1}{sep}{b2}{sep}{b3}", lo, hi


def fetch_race_odds(session, jcd, rno, date_str):
    """締切時オッズ(単勝・複勝・2連単・2連複・3連複・3連単・拡連複の全7種)を
    5ページ取得する。1ページの取得に失敗しても、そのページ分の舟券種のみ
    欠落させて他のページの取得を続ける。"""
    rows = []
    for page, table_specs in ODDS_PAGE_SPECS:
        url = f"{BASE_URL}/owpc/pc/race/{page}?rno={rno}&jcd={jcd}&hd={date_str}"
        html_text = session.get(url)
        if html_text is None:
            logger.warning("オッズ取得失敗(%s): jcd=%s rno=%s", page, jcd, rno)
            continue
        # 先頭はどのオッズページにも共通の「レース選択」ナビ表なので除く
        tables = BeautifulSoup(html_text, "lxml").find_all("table")[1:]
        for bet_type, idx, shape, ordered in table_specs:
            if idx >= len(tables):
                logger.warning(
                    "オッズページの構造が想定と異なります(%s/%s): jcd=%s rno=%s", page, bet_type, jcd, rno
                )
                continue
            for combination, lo, hi in _odds_rows_from_table(tables[idx], shape, ordered):
                rows.append({"bet_type": bet_type, "combination": combination, "odds_low": lo, "odds_high": hi})
    return rows


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS races (
    race_date TEXT NOT NULL,
    jcd TEXT NOT NULL,
    venue_name TEXT,
    rno INTEGER NOT NULL,
    title TEXT,
    race_type TEXT,
    distance TEXT,
    grade TEXT,
    weather TEXT,
    temperature REAL,
    wind_speed REAL,
    water_temp REAL,
    wave_height REAL,
    kimarite TEXT,
    fetched_at TEXT,
    PRIMARY KEY (race_date, jcd, rno)
);

CREATE TABLE IF NOT EXISTS entries (
    race_date TEXT NOT NULL,
    jcd TEXT NOT NULL,
    venue_name TEXT,
    rno INTEGER NOT NULL,
    waku INTEGER NOT NULL,
    toban TEXT,
    racer_class TEXT,
    racer_name TEXT,
    gender TEXT,
    branch TEXT,
    hometown TEXT,
    age INTEGER,
    weight REAL,
    f_count INTEGER,
    l_count INTEGER,
    avg_st REAL,
    national_win_rate REAL,
    national_2rate REAL,
    national_3rate REAL,
    local_win_rate REAL,
    local_2rate REAL,
    local_3rate REAL,
    motor_no INTEGER,
    motor_2rate REAL,
    motor_3rate REAL,
    boat_no INTEGER,
    boat_2rate REAL,
    boat_3rate REAL,
    fetched_at TEXT,
    PRIMARY KEY (race_date, jcd, rno, waku)
);

CREATE TABLE IF NOT EXISTS results (
    race_date TEXT NOT NULL,
    jcd TEXT NOT NULL,
    venue_name TEXT,
    rno INTEGER NOT NULL,
    waku INTEGER NOT NULL,
    rank TEXT,
    is_start_trouble INTEGER,
    toban TEXT,
    racer_name TEXT,
    race_time TEXT,
    start_timing TEXT,
    fetched_at TEXT,
    PRIMARY KEY (race_date, jcd, rno, waku)
);

CREATE TABLE IF NOT EXISTS payouts (
    race_date TEXT NOT NULL,
    jcd TEXT NOT NULL,
    venue_name TEXT,
    rno INTEGER NOT NULL,
    bet_type TEXT NOT NULL,
    combination TEXT NOT NULL,
    payout INTEGER,
    popularity TEXT,
    fetched_at TEXT,
    PRIMARY KEY (race_date, jcd, rno, bet_type, combination)
);

CREATE TABLE IF NOT EXISTS odds (
    race_date TEXT NOT NULL,
    jcd TEXT NOT NULL,
    venue_name TEXT,
    rno INTEGER NOT NULL,
    bet_type TEXT NOT NULL,
    combination TEXT NOT NULL,
    odds_low REAL,
    odds_high REAL,
    fetched_at TEXT,
    PRIMARY KEY (race_date, jcd, rno, bet_type, combination)
);

CREATE TABLE IF NOT EXISTS scrape_log (
    race_date TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    venues_count INTEGER,
    races_count INTEGER,
    status TEXT,
    PRIMARY KEY (race_date, started_at)
);

CREATE TABLE IF NOT EXISTS female_racers (
    toban TEXT PRIMARY KEY,
    racer_name TEXT,
    fetched_at TEXT
);
"""

# 既存DBに後から追加した列。(table, column,型) のリストで、
# 存在しなければ ALTER TABLE ADD COLUMN する。
SCHEMA_MIGRATIONS = [
    ("races", "grade", "TEXT"),
    ("entries", "gender", "TEXT"),
    ("results", "is_start_trouble", "INTEGER"),
]


def get_connection(db_path):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    for table, column, coltype in SCHEMA_MIGRATIONS:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
    # is_start_trouble はrankから機械的に導出できるため、既存行(NULL)は
    # 再取得せずその場で補完する。F(フライング)とL(出遅れ)を同一の
    # 「スタート事故」として1にまとめる。
    conn.execute(
        "UPDATE results SET is_start_trouble = CASE WHEN rank IN ('F', 'L') THEN 1 ELSE 0 END "
        "WHERE is_start_trouble IS NULL"
    )
    conn.commit()
    return conn


def save_race(conn, race_date, jcd, venue_name, rno, race_info, fetched_at):
    conn.execute(
        """INSERT OR REPLACE INTO races
        (race_date, jcd, venue_name, rno, title, race_type, distance, grade,
         weather, temperature, wind_speed, water_temp, wave_height, kimarite, fetched_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (race_date, jcd, venue_name, rno, race_info.get("title"), race_info.get("race_type"),
         race_info.get("distance"), race_info.get("grade"), race_info.get("weather"),
         race_info.get("temperature"), race_info.get("wind_speed"), race_info.get("water_temp"),
         race_info.get("wave_height"), race_info.get("kimarite"), fetched_at),
    )


def get_female_toban_set(conn):
    """female_racersテーブルから女性選手の登番集合を取得する。"""
    return {row[0] for row in conn.execute("SELECT toban FROM female_racers").fetchall()}


def save_entries(conn, race_date, jcd, venue_name, rno, entries, fetched_at, female_tobans=None):
    if female_tobans is None:
        female_tobans = get_female_toban_set(conn)
    for e in entries:
        gender = None
        if e["toban"]:
            gender = "女" if e["toban"] in female_tobans else "男"
        conn.execute(
            """INSERT OR REPLACE INTO entries
            (race_date, jcd, venue_name, rno, waku, toban, racer_class, racer_name, gender, branch, hometown,
             age, weight, f_count, l_count, avg_st,
             national_win_rate, national_2rate, national_3rate,
             local_win_rate, local_2rate, local_3rate,
             motor_no, motor_2rate, motor_3rate,
             boat_no, boat_2rate, boat_3rate, fetched_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (race_date, jcd, venue_name, rno, e["waku"], e["toban"], e["racer_class"], e["racer_name"],
             gender, e["branch"], e["hometown"], e["age"], e["weight"], e["f_count"], e["l_count"],
             e["avg_st"], e["national_win_rate"], e["national_2rate"], e["national_3rate"],
             e["local_win_rate"], e["local_2rate"], e["local_3rate"], e["motor_no"],
             e["motor_2rate"], e["motor_3rate"], e["boat_no"], e["boat_2rate"], e["boat_3rate"],
             fetched_at),
        )


def save_results(conn, race_date, jcd, venue_name, rno, results, fetched_at):
    for r in results:
        if r["waku"] is None:
            continue
        is_start_trouble = 1 if r["rank"] in ("F", "L") else 0
        conn.execute(
            """INSERT OR REPLACE INTO results
            (race_date, jcd, venue_name, rno, waku, rank, is_start_trouble, toban, racer_name,
             race_time, start_timing, fetched_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (race_date, jcd, venue_name, rno, r["waku"], r["rank"], is_start_trouble, r["toban"],
             r["racer_name"], r["race_time"], r["start_timing"], fetched_at),
        )


def save_payouts(conn, race_date, jcd, venue_name, rno, payouts, fetched_at):
    for p in payouts:
        conn.execute(
            """INSERT OR REPLACE INTO payouts
            (race_date, jcd, venue_name, rno, bet_type, combination, payout, popularity, fetched_at)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (race_date, jcd, venue_name, rno, p["bet_type"], p["combination"], p["payout"],
             p["popularity"], fetched_at),
        )


def save_odds(conn, race_date, jcd, venue_name, rno, odds_rows, fetched_at):
    for o in odds_rows:
        conn.execute(
            """INSERT OR REPLACE INTO odds
            (race_date, jcd, venue_name, rno, bet_type, combination, odds_low, odds_high, fetched_at)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (race_date, jcd, venue_name, rno, o["bet_type"], o["combination"], o["odds_low"],
             o["odds_high"], fetched_at),
        )


# ---------------------------------------------------------------------------
# 選手の性別: 公式サイトの「ボートレーサー検索」で性別=女性を検索し、
# 登録されている女性選手の登番一覧を取得する。男性の一覧は取得せず、
# 「この一覧に無ければ男性」として扱う(男性/女性の二択のため)。
# ---------------------------------------------------------------------------

RACER_SEARCH_URL = f"{BASE_URL}/owpc/pc/data/racersearch/index"
RACER_SEARCH_RESULT_URL = f"{BASE_URL}/owpc/pc/data/racersearch/result"


def fetch_female_racers(session):
    """性別=女性で選手検索し、(toban, racer_name)のリストをページネーション込みで
    全件取得する。"""
    resp = session.session.post(
        RACER_SEARCH_URL,
        data={
            "TDATP310A_2": "TDATP310A_2",
            "TDATP310A_2:inBoatracerName": "",
            "TDATP310A_2:inTobanLeft": "",
            "TDATP310A_2:inTobanRight": "",
            "TDATP310A_2:inCultivationTerm": "",
            "TDATP310A_2:TDATP310A_3": "SELECTALL",
            "TDATP310A_2:TDATP310A_4": "SELECTALL",
            "TDATP310A_2:TDATP310A_5": "SELECTALL",
            "TDATP310A_2:TDATP310A_6": "2",  # 性別: 2=女性
            "TDATP310A_2:TDATP310A_7": "検索",
            "javax.faces.ViewState": "stateless",
        },
        headers=HEADERS,
        timeout=session.timeout,
    )
    resp.encoding = "utf-8"

    racers = {}

    def collect(html_text):
        soup = BeautifulSoup(html_text, "lxml")
        for a in soup.select('a[href*="racersearch/profile?toban="]'):
            m = re.search(r"toban=(\d+)", a["href"])
            if not m:
                continue
            toban = m.group(1)
            body = a.find_parent("li")
            name = None
            if body:
                name_tag = body.select_one(".photoGallery3_bodyName")
                if name_tag:
                    name = nfkc(name_tag.get_text(strip=True))
            racers[toban] = name
        # ページネーションリンク(orteusPageSelectBeginRow=N)を全て集める
        pages = set()
        for a in soup.select('a[href*="orteusPageSelectBeginRow="]'):
            pages.add(a["href"])
        return pages

    pages_to_visit = collect(resp.text)
    visited = set()
    while pages_to_visit:
        href = pages_to_visit.pop()
        if href in visited:
            continue
        visited.add(href)
        time.sleep(session.interval_sec)
        page_resp = session.session.get(f"{BASE_URL}{href}", headers=HEADERS, timeout=session.timeout)
        page_resp.encoding = "utf-8"
        more_pages = collect(page_resp.text)
        pages_to_visit |= (more_pages - visited)

    return [(toban, name) for toban, name in racers.items()]


def update_female_racers(db_path, interval_sec=1.5):
    """female_racersテーブルを公式サイトの検索結果で更新する。
    数が少なく変化も緩やかなため、日次ジョブの一部として毎回実行しても軽い。"""
    session = PoliteSession(interval_sec=interval_sec)
    conn = get_connection(db_path)

    racers = fetch_female_racers(session)
    fetched_at = datetime.now().isoformat(timespec="seconds")
    for toban, name in racers:
        conn.execute(
            "INSERT OR REPLACE INTO female_racers (toban, racer_name, fetched_at) VALUES (?,?,?)",
            (toban, name, fetched_at),
        )
    conn.commit()

    # 新たに判明した女性選手を、既存entriesの性別にも反映する
    conn.execute("UPDATE entries SET gender = '女' WHERE toban IN (SELECT toban FROM female_racers)")
    conn.execute("UPDATE entries SET gender = '男' WHERE gender IS NULL AND toban IS NOT NULL")
    conn.commit()
    conn.close()

    logger.info("女性選手一覧を更新しました: %d名", len(racers))
    return len(racers)


# ---------------------------------------------------------------------------
# 過去レースのグレード補完: races.gradeが未取得(NULL)の行について、
# 対応するraceresult(結果があれば)/racelist(無ければ)ページを再取得して
# gradeだけを補完する。entries/results/payoutsは触らない。
# ---------------------------------------------------------------------------

def backfill_grades(db_path, interval_sec=1.5):
    session = PoliteSession(interval_sec=interval_sec)
    conn = get_connection(db_path)

    targets = conn.execute(
        "SELECT race_date, jcd, rno FROM races WHERE grade IS NULL ORDER BY race_date, jcd, rno"
    ).fetchall()
    if not targets:
        logger.info("グレード補完対象のレースはありません。")
        conn.close()
        return 0

    updated = 0
    for race_date, jcd, rno in targets:
        has_result = conn.execute(
            "SELECT 1 FROM results WHERE race_date=? AND jcd=? AND rno=? LIMIT 1",
            (race_date, jcd, rno),
        ).fetchone()

        if has_result:
            url = f"{BASE_URL}/owpc/pc/race/raceresult?rno={rno}&jcd={jcd}&hd={race_date}"
        else:
            url = f"{BASE_URL}/owpc/pc/race/racelist?rno={rno}&jcd={jcd}&hd={race_date}"

        html_text = session.get(url)
        if html_text is None:
            logger.warning("グレード補完のための取得に失敗: date=%s jcd=%s rno=%s", race_date, jcd, rno)
            continue

        soup = BeautifulSoup(html_text, "lxml")
        grade = extract_grade(soup)
        conn.execute(
            "UPDATE races SET grade=? WHERE race_date=? AND jcd=? AND rno=?",
            (grade, race_date, jcd, rno),
        )
        conn.commit()
        updated += 1
        if updated % 20 == 0:
            logger.info("グレード補完: %d/%d件完了", updated, len(targets))

    conn.close()
    logger.info("グレード補完完了: %d/%d件", updated, len(targets))
    return updated


# ---------------------------------------------------------------------------
# 過去レースのオッズ補完: resultsが存在する(=終了している)のにoddsが
# 1件も無いレースについて、締切時オッズを後から取得する。
# 1レースあたり5ページの追加リクエストが必要なため、対象件数によっては
# 非常に時間がかかる。--backfill-odds-limit で1回の実行での処理件数の
# 上限を指定でき、複数回に分けて(例: 日次ジョブに少しずつ追記する形で)
# 完走させることを想定している。
# ---------------------------------------------------------------------------

def backfill_odds(db_path, interval_sec=1.5, limit=None):
    session = PoliteSession(interval_sec=interval_sec)
    conn = get_connection(db_path)

    targets = conn.execute(
        """SELECT DISTINCT r.race_date, r.jcd, r.venue_name, r.rno
        FROM races r
        JOIN results res ON res.race_date = r.race_date AND res.jcd = r.jcd AND res.rno = r.rno
        LEFT JOIN odds o ON o.race_date = r.race_date AND o.jcd = r.jcd AND o.rno = r.rno
        WHERE o.race_date IS NULL
        ORDER BY r.race_date, r.jcd, r.rno"""
    ).fetchall()

    total_remaining = len(targets)
    if limit:
        targets = targets[:limit]

    if not targets:
        logger.info("オッズ補完対象のレースはありません。")
        conn.close()
        return 0

    logger.info("オッズ補完対象: 今回%d件 (未補完全体: %d件)", len(targets), total_remaining)
    updated = 0
    for race_date, jcd, venue_name, rno in targets:
        odds_rows = fetch_race_odds(session, jcd, rno, race_date)
        if not odds_rows:
            logger.warning("オッズ補完: データ取得できず date=%s jcd=%s rno=%s", race_date, jcd, rno)
            continue
        fetched_at = datetime.now().isoformat(timespec="seconds")
        save_odds(conn, race_date, jcd, venue_name, rno, odds_rows, fetched_at)
        conn.commit()
        updated += 1
        if updated % 20 == 0:
            logger.info("オッズ補完: %d/%d件完了", updated, len(targets))

    conn.close()
    logger.info("オッズ補完完了: 今回%d/%d件 (残り: %d件)", updated, len(targets), total_remaining - updated)
    return updated


# ---------------------------------------------------------------------------
# 過去日付の一括バックフィル: races/entries/results/payoutsのみを対象に、
# 現在DBに入っている最古のrace_dateより前の日を、開始境界(既定2016-09-01。
# 公式サイトの現行フォーマットで閲覧できると確認できた最も古い時期)まで
# 1日ずつ古い方向へ遡って取得する。オッズは対象外(--backfill-oddsで別途扱う)。
#
# 進捗は別テーブルで管理せず、「DB内のraces.race_dateの最小値の前日」から
# 毎回自動的に再開する。1回の実行では時間予算(既定3.5時間)を使い切ったら、
# 処理中の日を最後まで終えた上で安全に終了する(日の途中で打ち切らない)。
#
# raise_on_block=Trueのセッションを使うため、403/429などアクセス制限の
# 疑いがあるレスポンスを検知するとAccessBlockedErrorがそのまま呼び出し元
# (main)まで伝播し、即座に処理を停止する。レースごとにconn.commit()して
# いるため、停止時点までに取得できたデータは失われない。
# ---------------------------------------------------------------------------

BACKFILL_HISTORY_START = "20160901"


def backfill_history(db_path, interval_sec=1.5, max_hours=3.5, start_boundary=BACKFILL_HISTORY_START):
    conn = get_connection(db_path)
    current_min = conn.execute("SELECT MIN(race_date) FROM races").fetchone()[0]
    conn.close()

    if current_min is None:
        logger.error(
            "racesテーブルが空のため、バックフィルの開始日(最古日の前日)を決定できません。"
            "先に通常のスクレイプ(直近日)でデータを入れてください。"
        )
        return 0

    if current_min <= start_boundary:
        logger.info(
            "過去データのバックフィルは既に開始境界(%s)に到達しています(現在の最古日: %s)。"
            "これ以上遡る対象はありません。",
            start_boundary, current_min,
        )
        return 0

    date_obj = datetime.strptime(current_min, "%Y%m%d") - timedelta(days=1)
    boundary_obj = datetime.strptime(start_boundary, "%Y%m%d")

    deadline = time.monotonic() + max_hours * 3600
    processed_days = 0

    logger.info(
        "過去データのバックフィル開始: %s から %s へ向けて遡って取得します(時間予算=%.1f時間、オッズは対象外)",
        date_obj.strftime("%Y%m%d"), start_boundary, max_hours,
    )

    reached_boundary = False
    while date_obj >= boundary_obj:
        date_str = date_obj.strftime("%Y%m%d")
        logger.info("=== バックフィル対象日: %s (今回%d日目) ===", date_str, processed_days + 1)

        venues_count, races_count, status = scrape_day(
            date_str, db_path, interval_sec=interval_sec, fetch_odds=False, raise_on_block=True,
        )
        processed_days += 1
        logger.info(
            "バックフィル: %s 完了 (会場数=%d, レース数=%d, status=%s)",
            date_str, venues_count, races_count, status,
        )

        date_obj -= timedelta(days=1)

        if date_obj < boundary_obj:
            reached_boundary = True
            break

        if time.monotonic() >= deadline:
            logger.info(
                "今回の実行の時間予算(%.1f時間)に達したため終了します。処理済み: %d日 "
                "(次回は%sから再開します)",
                max_hours, processed_days, date_obj.strftime("%Y%m%d"),
            )
            break

    if reached_boundary:
        logger.info(
            "開始境界(%s)まで到達しました。過去データのバックフィルは完了です(今回処理: %d日)。",
            start_boundary, processed_days,
        )

    return processed_days


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def scrape_day(date_str, db_path, interval_sec=1.5, fetch_odds=True, raise_on_block=False):
    session = PoliteSession(interval_sec=interval_sec, raise_on_block=raise_on_block)
    conn = get_connection(db_path)
    started_at = datetime.now().isoformat(timespec="seconds")
    female_tobans = get_female_toban_set(conn)

    venues = get_active_venues(session, date_str)
    if not venues:
        logger.warning("開催中のレース場が見つかりませんでした (date=%s)", date_str)

    races_count = 0
    for jcd, event_title in venues:
        venue_name_fallback = JCD_NAMES.get(jcd, jcd)
        logger.info("=== %s会場 (jcd=%s) %s の取得を開始 ===", venue_name_fallback, jcd, event_title)

        for rno in range(1, MAX_RACES_PER_VENUE + 1):
            racelist_url = f"{BASE_URL}/owpc/pc/race/racelist?rno={rno}&jcd={jcd}&hd={date_str}"
            racelist_html = session.get(racelist_url)
            if racelist_html is None:
                logger.warning("出走表取得失敗: jcd=%s rno=%s", jcd, rno)
                continue

            entries = parse_entries(racelist_html)
            if len(entries) < 6:
                logger.info("jcd=%s %dRは出走データなし。以降のレースをスキップします。", jcd, rno)
                break

            fetched_at = datetime.now().isoformat(timespec="seconds")
            save_entries(conn, date_str, jcd, venue_name_fallback, rno, entries, fetched_at, female_tobans)

            raceresult_url = f"{BASE_URL}/owpc/pc/race/raceresult?rno={rno}&jcd={jcd}&hd={date_str}"
            raceresult_html = session.get(raceresult_url)
            if raceresult_html is None:
                logger.warning("結果取得失敗: jcd=%s rno=%s", jcd, rno)
                conn.commit()
                continue

            race_info, results, payouts = parse_race_result(raceresult_html)
            venue_name = race_info.get("venue_name") or venue_name_fallback
            fetched_at = datetime.now().isoformat(timespec="seconds")

            save_race(conn, date_str, jcd, venue_name, rno, race_info, fetched_at)
            if results:
                save_results(conn, date_str, jcd, venue_name, rno, results, fetched_at)
            if payouts:
                save_payouts(conn, date_str, jcd, venue_name, rno, payouts, fetched_at)

            # 締切時オッズ(全7舟券種)。レースが終了しているため既に確定済みの値であり、
            # リアルタイム性を気にする必要はない。1レースあたり5ページ追加で
            # リクエストするため、日次取得の所要時間が大きく伸びる点に注意。
            # fetch_odds=False(過去日付の一括バックフィル向け)の場合はスキップし、
            # リクエスト数を1レースあたり2件(出走表+結果)に抑える。
            odds_rows = []
            if fetch_odds:
                odds_rows = fetch_race_odds(session, jcd, rno, date_str)
                if odds_rows:
                    odds_fetched_at = datetime.now().isoformat(timespec="seconds")
                    save_odds(conn, date_str, jcd, venue_name, rno, odds_rows, odds_fetched_at)

            conn.commit()
            races_count += 1
            logger.info(
                "jcd=%s %dR 保存完了 (entries=%d, results=%d, payouts=%d, odds=%d)",
                jcd, rno, len(entries), len(results), len(payouts), len(odds_rows),
            )

    # 結果データの完全性チェック。
    # 対象日がJST基準で「昨日以前」(=レースが終わっているはずの日)なのに
    # resultsが著しく欠けている場合は、日付境界の取り違えやスクレイピング失敗を疑う。
    # (例: cronの実行がJST 0時をまたぎ、まだ1レースも終わっていない翌日分を
    #  取得してしまうケース。この場合HTTPエラーは出ないため、exit code 0のまま
    #  見過ごされてしまう。)
    races_total = conn.execute(
        "SELECT COUNT(*) FROM races WHERE race_date = ?", (date_str,)
    ).fetchone()[0]
    results_races = conn.execute(
        "SELECT COUNT(*) FROM (SELECT DISTINCT jcd, rno FROM results WHERE race_date = ?)",
        (date_str,),
    ).fetchone()[0]
    completeness = (results_races / races_total) if races_total else 1.0
    today_jst = datetime.now(JST).strftime("%Y%m%d")
    expect_complete = date_str < today_jst

    status = "ok"
    if expect_complete and races_total > 0 and completeness < RESULTS_COMPLETENESS_THRESHOLD:
        status = "partial"
        logger.error(
            "結果データが不足しています: %d/%d レース (%.1f%%) が結果取得済み。"
            "対象日(%s)は既に終了しているはずの日ですが、閾値(%.0f%%)を下回っています。"
            "日付境界の取り違えやサイト側の変更、スクレイピング失敗の可能性があります。",
            results_races, races_total, completeness * 100, date_str,
            RESULTS_COMPLETENESS_THRESHOLD * 100,
        )
    else:
        logger.info(
            "結果取得率: %d/%d レース (%.1f%%)", results_races, races_total, completeness * 100
        )

    finished_at = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "INSERT OR REPLACE INTO scrape_log (race_date, started_at, finished_at, venues_count, races_count, status) "
        "VALUES (?,?,?,?,?,?)",
        (date_str, started_at, finished_at, len(venues), races_count, status),
    )
    conn.commit()
    conn.close()
    logger.info("完了: %s (会場数=%d, レース数=%d, status=%s)", date_str, len(venues), races_count, status)
    return len(venues), races_count, status


def scrape_today_entries(date_str, db_path, interval_sec=1.5):
    """レース開始前の出走表のみを取得する。結果(raceresult)ページはまだ
    存在しないため取得しない。「今日のおすすめ」機能向けに、当日の朝など
    レース前のタイミングで実行することを想定している。"""
    session = PoliteSession(interval_sec=interval_sec)
    conn = get_connection(db_path)
    started_at = datetime.now().isoformat(timespec="seconds")
    female_tobans = get_female_toban_set(conn)

    venues = get_active_venues(session, date_str)
    if not venues:
        logger.warning("開催中のレース場が見つかりませんでした (date=%s)", date_str)

    races_count = 0
    for jcd, event_title in venues:
        venue_name_fallback = JCD_NAMES.get(jcd, jcd)
        logger.info("=== [出走表のみ] %s会場 (jcd=%s) %s ===", venue_name_fallback, jcd, event_title)

        for rno in range(1, MAX_RACES_PER_VENUE + 1):
            racelist_url = f"{BASE_URL}/owpc/pc/race/racelist?rno={rno}&jcd={jcd}&hd={date_str}"
            racelist_html = session.get(racelist_url)
            if racelist_html is None:
                logger.warning("出走表取得失敗: jcd=%s rno=%s", jcd, rno)
                continue

            entries = parse_entries(racelist_html)
            if len(entries) < 6:
                logger.info("jcd=%s %dRは出走データなし。以降のレースをスキップします。", jcd, rno)
                break

            race_info = parse_racelist_header(racelist_html)
            venue_name = race_info.get("venue_name") or venue_name_fallback
            fetched_at = datetime.now().isoformat(timespec="seconds")

            save_entries(conn, date_str, jcd, venue_name, rno, entries, fetched_at, female_tobans)
            save_race(conn, date_str, jcd, venue_name, rno, race_info, fetched_at)
            conn.commit()

            races_count += 1
            logger.info("jcd=%s %dR 出走表保存完了 (entries=%d)", jcd, rno, len(entries))

    finished_at = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "INSERT OR REPLACE INTO scrape_log (race_date, started_at, finished_at, venues_count, races_count, status) "
        "VALUES (?,?,?,?,?,?)",
        (date_str, started_at, finished_at, len(venues), races_count, "entries_only"),
    )
    conn.commit()
    conn.close()
    logger.info("完了(出走表のみ): %s (会場数=%d, レース数=%d)", date_str, len(venues), races_count)
    return len(venues), races_count


def main():
    parser = argparse.ArgumentParser(description="BOATRACE公式サイトから日次データを取得しSQLiteに保存する")
    parser.add_argument("--date", help="取得対象日 (YYYYMMDD)。--whenより優先。")
    parser.add_argument("--when", choices=["today", "yesterday"], default="yesterday",
                         help="--date省略時に基準にする日。深夜(23時台など)に当日分を取るときは"
                              "'today'を指定する。デフォルトは'yesterday'(翌朝実行を想定)。")
    parser.add_argument("--db", default=str(Path(__file__).parent / "data" / "boatrace.db"), help="SQLiteファイルのパス")
    parser.add_argument("--interval", type=float, default=1.5, help="リクエスト間隔(秒)。デフォルト1.5秒。")
    parser.add_argument("--log", default=str(Path(__file__).parent / "data" / "scraper.log"), help="ログファイルのパス")
    parser.add_argument(
        "--entries-only", action="store_true",
        help="結果が出る前の出走表のみを取得する(「今日のおすすめ」機能向け)。"
             "resultsやpayoutsは取得せず、完全性チェックも行わない。",
    )
    parser.add_argument(
        "--update-female-racers", action="store_true",
        help="女性選手一覧(female_racersテーブル)を公式サイトの検索結果で更新し、"
             "既存entriesのgenderにも反映して終了する。他の取得は行わない。",
    )
    parser.add_argument(
        "--backfill-grades", action="store_true",
        help="racesテーブルでgradeが未取得(NULL)の行について、該当ページを"
             "再取得してgradeだけを補完し終了する。他の取得は行わない。",
    )
    parser.add_argument(
        "--backfill-odds", action="store_true",
        help="resultsが存在するのにoddsが未取得のレースについて、締切時オッズを"
             "補完し終了する。他の取得は行わない。",
    )
    parser.add_argument(
        "--backfill-odds-limit", type=int, default=None,
        help="--backfill-odds使用時、1回の実行で処理するレース数の上限。"
             "省略時は未補完分をすべて処理する。",
    )
    parser.add_argument(
        "--backfill-history", action="store_true",
        help="DB内の最古のrace_dateより前の日を、開始境界(既定%s)まで1日ずつ"
             "遡ってraces/entries/results/payoutsのみ取得する(オッズは対象外)。"
             "他の取得は行わない。" % BACKFILL_HISTORY_START,
    )
    parser.add_argument(
        "--backfill-history-hours", type=float, default=3.5,
        help="--backfill-history使用時、1回の実行に使う時間予算(時間)。既定3.5時間。",
    )
    parser.add_argument(
        "--backfill-history-start", default=BACKFILL_HISTORY_START,
        help="--backfill-history使用時に遡る開始境界日(YYYYMMDD)。既定は%(default)s。",
    )
    args = parser.parse_args()

    if args.date:
        date_str = args.date
    elif args.when == "today":
        date_str = datetime.now(JST).strftime("%Y%m%d")
    else:
        date_str = (datetime.now(JST) - timedelta(days=1)).strftime("%Y%m%d")

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args.log, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    if args.update_female_racers:
        logger.info("女性選手一覧の更新開始: db=%s", args.db)
        update_female_racers(args.db, interval_sec=args.interval)
        return

    if args.backfill_grades:
        logger.info("グレード補完開始: db=%s", args.db)
        backfill_grades(args.db, interval_sec=args.interval)
        return

    if args.backfill_odds:
        logger.info("オッズ補完開始: db=%s limit=%s", args.db, args.backfill_odds_limit)
        backfill_odds(args.db, interval_sec=args.interval, limit=args.backfill_odds_limit)
        return

    if args.backfill_history:
        logger.info(
            "過去データのバックフィル開始: db=%s hours=%.1f start=%s",
            args.db, args.backfill_history_hours, args.backfill_history_start,
        )
        try:
            backfill_history(
                args.db, interval_sec=args.interval,
                max_hours=args.backfill_history_hours, start_boundary=args.backfill_history_start,
            )
        except AccessBlockedError as exc:
            logger.error(
                "アクセス制限(403/429)の疑いを検知したため、バックフィルを即座に停止しました: %s", exc
            )
            sys.exit(3)
        return

    if args.entries_only:
        logger.info("BOATRACE出走表取得開始(結果は取得しない): date=%s db=%s interval=%.1fs", date_str, args.db, args.interval)
        scrape_today_entries(date_str, args.db, interval_sec=args.interval)
        return

    logger.info("BOATRACEデータ取得開始: date=%s db=%s interval=%.1fs", date_str, args.db, args.interval)
    _, _, status = scrape_day(date_str, args.db, interval_sec=args.interval)
    if status != "ok":
        logger.error("データが不完全なまま終了しました (status=%s)。終了コード1で終了します。", status)
        sys.exit(1)


if __name__ == "__main__":
    main()

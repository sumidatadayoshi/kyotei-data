"""指定日(省略時は前日、JST基準)のscrape_logステータスを調べ、
GITHUB_OUTPUT形式で date/status/needs_retry を標準出力に書き出す。

scrape-watchdog.ymlから複数回(初回チェック、リトライ前後の再チェック)
呼び出されるため、共通スクリプトとして切り出している。
"""

import argparse
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

JST = timezone(timedelta(hours=9))
DEFAULT_DB = Path(__file__).parent / "data" / "boatrace.db"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("date", nargs="?", help="YYYYMMDD形式。省略時は前日(JST)")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args()

    date_str = args.date or (datetime.now(JST) - timedelta(days=1)).strftime("%Y%m%d")

    con = sqlite3.connect(args.db)
    row = con.execute(
        "SELECT status FROM scrape_log WHERE race_date = ? ORDER BY finished_at DESC LIMIT 1",
        (date_str,),
    ).fetchone()
    status = row[0] if row else "missing"

    print(f"date={date_str}")
    print(f"status={status}")
    print(f"needs_retry={'true' if status != 'ok' else 'false'}")


if __name__ == "__main__":
    main()

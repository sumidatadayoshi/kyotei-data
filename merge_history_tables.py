"""backfill-history.ymlのpush直前に使う補助スクリプト。

src_db(バックフィル作業でraces/entries/results/payoutsを書き足した後の
DBコピー)の内容を、dest_db(push直前にorigin/mainへ最新化したDB)へ
テーブルごとにINSERT OR REPLACEでマージする。merge_odds_table.pyと同じ理由
(data/boatrace.dbはバイナリファイルのため、複数のワークフローが同時期に
更新するとgit rebase/mergeで解消不能なコンフリクトになる)で、「最新版のDBに、
今回追加した行だけをSQLレベルで再適用する」方式を取る。races/entries/
results/payouts/scrape_logはいずれも主キー単位の追記のみで、他ワークフロー
とキーが競合しても内容は同一になるはずなので、INSERT OR REPLACEで安全に
マージできる。
"""

import argparse
import sqlite3

TABLES = ["races", "entries", "results", "payouts", "scrape_log"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src_db", help="バックフィル後のDBコピーのパス")
    parser.add_argument("dest_db", help="マージ先(push直前に最新化した)DBのパス")
    args = parser.parse_args()

    conn = sqlite3.connect(args.dest_db)
    conn.execute("ATTACH DATABASE ? AS src", (args.src_db,))
    for table in TABLES:
        (src_count,) = conn.execute(f"SELECT COUNT(*) FROM src.{table}").fetchone()
        conn.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM src.{table}")
        conn.commit()
        (dest_count,) = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        print(f"src.{table}={src_count}行 -> マージ後のdest.{table}={dest_count}行")
    conn.close()


if __name__ == "__main__":
    main()

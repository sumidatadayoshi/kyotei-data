"""backfill-odds.ymlのpush直前に使う補助スクリプト。

src_db(バックフィル作業でoddsを書き足した後のDBコピー)のoddsテーブルの
内容を、dest_db(push直前にorigin/mainへ最新化したDB)へ
INSERT OR REPLACEでマージする。

data/boatrace.dbはバイナリファイルのため、複数のワークフローが
同時期に更新するとgit rebase/mergeで解消不能なコンフリクトになる
(scrape-watchdog.ymlで実際に発生した問題と同種)。バックフィルは
1回の実行が長時間(数十分〜数時間)かかり、実行中に他のワークフローが
先にpushしている可能性が高いため、「最新版のDBに、今回追加した行だけを
SQLレベルで再適用する」ことでバイナリのコンフリクトを回避する。
oddsテーブルは主キー(race_date, jcd, rno, bet_type, combination)単位の
追記のみで、他ワークフローとキーが競合しても内容は同一になるはずなので
INSERT OR REPLACEで安全にマージできる。
"""

import argparse
import sqlite3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src_db", help="バックフィル後のDBコピーのパス")
    parser.add_argument("dest_db", help="マージ先(push直前に最新化した)DBのパス")
    args = parser.parse_args()

    conn = sqlite3.connect(args.dest_db)
    conn.execute("ATTACH DATABASE ? AS src", (args.src_db,))
    (src_count,) = conn.execute("SELECT COUNT(*) FROM src.odds").fetchone()
    conn.execute("INSERT OR REPLACE INTO odds SELECT * FROM src.odds")
    conn.commit()
    (dest_count,) = conn.execute("SELECT COUNT(*) FROM odds").fetchone()
    conn.close()

    print(f"src.odds={src_count}行 -> マージ後のdest.odds={dest_count}行")


if __name__ == "__main__":
    main()

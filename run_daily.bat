@echo off
REM 前日分のBOATRACEデータを取得するバッチ (タスクスケジューラから起動する用)
cd /d "%~dp0"
python boatrace_scraper.py --when today --interval 1.5 >> data\run_daily.log 2>&1

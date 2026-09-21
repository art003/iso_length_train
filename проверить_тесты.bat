@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Сначала поставь окружение:
  echo python -m venv .venv
  echo .venv\Scripts\pip install -r requirements.txt
  pause
  exit /b 1
)
echo Считаю тест_1, тест_2, тест_3 — 30 листов. Минут 10–20, это нормально.
".venv\Scripts\python.exe" run_three_tests.py
pause

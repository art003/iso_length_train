@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Сначала: python -m venv .venv
  echo Потом: .venv\Scripts\pip install -r requirements.txt
  pause
  exit /b 1
)
echo Запуск http://127.0.0.1:8765
".venv\Scripts\python.exe" app.py
pause

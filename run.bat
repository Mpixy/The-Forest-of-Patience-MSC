@echo off
:: Forest of Patience helper - installs what it needs the first time, then runs.
:: Run as administrator (right click -> Run as administrator) so the keys reach the game.
cd /d "%~dp0"
if not exist venv (
    python -m venv venv || (echo Python not found - install it from python.org & pause & exit /b)
    venv\Scripts\pip install -r requirements.txt
)
venv\Scripts\python forest_helper.py
pause

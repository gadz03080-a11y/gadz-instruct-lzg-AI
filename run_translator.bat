@echo off
echo Starting translator and chat...
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python translate.py
pause

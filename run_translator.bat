@echo off
echo Запуск переводчика и Qwen (лезгинский ↔ русский)...
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python translate.py
pause

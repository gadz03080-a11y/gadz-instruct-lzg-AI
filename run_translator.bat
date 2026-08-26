<<<<<<< HEAD
@echo off
echo Запуск  (лезгинский ↔ русский)...
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python translate.py
pause
=======
@echo off
echo Запуск переводчика и Qwen (лезгинский ↔ русский)...
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python translate.py
pause
>>>>>>> 077917b28d963efa3cca55c050773a03013cebe3

@echo off
chcp 65001 >nul
cd /d "%~dp0"
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set PYTHONIOENCODING=utf-8
python run_all.py
if errorlevel 1 echo 运行失败，请检查日志和 Python 依赖。
pause

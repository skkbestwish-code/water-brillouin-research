@echo off
chcp 65001 >nul
rem 切到脚本所在目录；固定单线程 BLAS/OMP；强制 UTF-8 输出
cd /d "%~dp0"
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set PYTHONIOENCODING=utf-8
rem 先跑全部单元测试，失败则跳过模拟；通过后运行完整模拟
python -m unittest discover -s tests -v
if errorlevel 1 goto end
python run_water_simulation.py
:end
pause

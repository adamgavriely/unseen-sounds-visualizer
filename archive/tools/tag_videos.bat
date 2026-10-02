@echo off
cd /d "%~dp0"
title Benchmark Tagger
echo ============================================================
echo   BENCHMARK TAGGER
echo.
echo   Your browser will open at http://localhost:8000
echo   Tag each clip with the 1 / 2 / 3 or BAD buttons.
echo.
echo   Keep THIS window open while tagging.
echo   Close this window (or press Ctrl+C) to stop.
echo ============================================================
echo.
set "PY=C:\Users\adamg\anaconda3\envs\msproj\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" "benchmark\tagger.py"
echo.
echo Tagger stopped.
pause

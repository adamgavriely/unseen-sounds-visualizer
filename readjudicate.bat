@echo off
cd /d "%~dp0"
title Blind Re-adjudication
echo ============================================================
echo   BLIND RE-ADJUDICATION  (60 clips, ~20 minutes)
echo.
echo   Clips you already tagged, shuffled, old label hidden.
echo   No Drop button - every clip gets a real category.
echo   A short reason is REQUIRED - that text is the whole point.
echo.
echo   Keep THIS window open. Close it to stop.
echo ============================================================
echo.
set "PY=C:\Users\adamg\anaconda3\envs\msproj\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" "benchmark\readjudicate.py"
echo.
pause

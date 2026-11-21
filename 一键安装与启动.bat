@echo off
chcp 65001 >nul
title 麦弦翻唱 API - 一键安装与启动
cd /d "%~dp0"

set "PY_CMD="
where py >nul 2>&1 && (
  py -3.11 --version >nul 2>&1 && set "PY_CMD=py -3.11"
  if not defined PY_CMD (
    py -3.10 --version >nul 2>&1 && set "PY_CMD=py -3.10"
  )
  if not defined PY_CMD (
    py -3.12 --version >nul 2>&1 && set "PY_CMD=py -3.12"
  )
  if not defined PY_CMD (
    py -3 --version >nul 2>&1 && set "PY_CMD=py -3"
  )
)
if not defined PY_CMD (
  where python >nul 2>&1 && set "PY_CMD=python"
)

if not defined PY_CMD (
  echo.
  echo ==============================================================
  echo   未检测到 Python 3.10 ~ 3.12！
  echo   请先安装 Python（安装时务必勾选 Add Python to PATH）：
  echo   1) 命令行一键装：winget install Python.Python.3.11
  echo   2) 或前往官网下载：https://www.python.org/downloads/
  echo ==============================================================
  echo.
  pause
  exit /b 1
)

%PY_CMD% "%~dp0scripts\oneclick_setup.py" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo.
  echo 安装或启动中断（错误码 %RC%），排查完可直接再次双击本文件继续（支持断点续装）。
  pause
)
exit /b %RC%

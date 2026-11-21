#!/usr/bin/env bash
# 麦弦翻唱 API · Linux / WSL2 一键安装与启动脚本
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "✗ 未检测到 python3，请先运行：sudo apt update && sudo apt install -y python3 python3-venv python3-pip git ffmpeg"
  exit 1
fi

exec python3 scripts/oneclick_setup.py "$@"

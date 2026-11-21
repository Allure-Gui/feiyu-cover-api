@echo off
chcp 65001 >nul
title 停止麦弦翻唱 API
echo 正在查找并关闭监听 19655 端口的麦弦翻唱 API...
powershell -NoProfile -Command "$c = Get-NetTCPConnection -LocalPort 19655 -State Listen -ErrorAction SilentlyContinue; if ($c) { $c | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue; Write-Host ('已关闭进程 PID=' + $_.OwningProcess) } } else { Write-Host '当前没有运行在 19655 端口的服务。' }"
pause

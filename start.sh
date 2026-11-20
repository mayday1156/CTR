#!/usr/bin/env bash
# Ali-CTR Lab 一键启动脚本（Git Bash / MSYS2 环境）
# 用法：在项目根目录执行  bash start.sh
# 作用：用项目 venv 启动后端，避免系统 Python 环境混乱
set -e

cd "$(dirname "$0")"

PY=./venv/Scripts/python.exe
if [ ! -f "$PY" ]; then
    echo "[错误] 未找到 venv，请先运行： bash setup.sh"
    exit 1
fi

PORT=8000
HOST=0.0.0.0   # 0.0.0.0 = 局域网/公网可访问；仅本机调试可改回 127.0.0.1

# 端口占用检测（提示用户，而不是让 uvicorn 报错后退出）
if netstat -ano 2>/dev/null | grep -q ":$PORT .*LISTENING"; then
    echo "[提示] 端口 $PORT 已被占用。"
    echo "  若之前的后端还在运行，可直接访问 http://127.0.0.1:$PORT/ui/"
    echo "  若要重启，请先关闭占用进程，或改用其他端口："
    echo "    bash start.sh --port 8001"
fi

# 支持自定义端口：bash start.sh --port 8001
if [ "$1" = "--port" ] && [ -n "$2" ]; then
    PORT="$2"
fi

echo "======================================"
echo "  Ali-CTR Lab 启动（端口 $PORT）"
echo "  本机访问: http://127.0.0.1:$PORT/ui/"
echo "  局域网访问: http://$(hostname -I 2>/dev/null | awk '{print $1}' || echo '<本机IP>'):$PORT/ui/"
echo "  API 文档: http://127.0.0.1:$PORT/docs"
echo "======================================"

exec "$PY" -m uvicorn backend.main:app --host "$HOST" --port "$PORT"

#!/usr/bin/env bash
# Ali-CTR Lab 环境初始化脚本（Git Bash / MSYS2）
# 用法：bash setup.sh
# 作用：创建 venv 并安装全部依赖（含 CPU 版 torch）
set -e
cd "$(dirname "$0")"

# 找一个可用的 Python（优先 py launcher 3.10~3.13；torch 需匹配版本）
PY=""
if command -v py >/dev/null 2>&1; then
    # 尝试 3.13 / 3.12 / 3.11 / 3.10
    for v in 3.13 3.12 3.11 3.10; do
        if py -$v --version >/dev/null 2>&1; then PY="py -$v"; break; fi
    done
fi
if [ -z "$PY" ] && command -v python >/dev/null 2>&1; then
    PY="python"
fi
if [ -z "$PY" ]; then
    echo "[错误] 未找到 Python，请先安装 Python 3.10+"
    exit 1
fi

echo "使用解释器：$PY ($($PY --version 2>&1))"

# 创建 venv
if [ ! -f "venv/Scripts/python.exe" ]; then
    echo "[1/3] 创建虚拟环境 ..."
    $PY -m venv venv
fi

VPY=./venv/Scripts/python.exe
echo "[2/3] 安装基础依赖 ..."
"$VPY" -m pip install --upgrade pip >/dev/null 2>&1
"$VPY" -m pip install numpy pandas pyarrow fastapi uvicorn python-multipart openpyxl

echo "[3/3] 安装 CPU 版 torch（较大，请耐心等待）..."
"$VPY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "======================================"
echo "  环境初始化完成！"
echo "  启动后端：bash start.sh"
echo "  页面：http://127.0.0.1:8000/ui/"
echo "======================================"

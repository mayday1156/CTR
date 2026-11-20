#!/usr/bin/env bash
# ============================================================
# Ali-CTR Lab · 真实数据一键训练 + 同步到网页
# ============================================================
# 用法：
#   bash train_real.sh            # 用默认参数训练并同步
#   bash train_real.sh --epochs 5 # 自定义轮数
#
# 前置条件：
#   1. 已把真实 Ali-CCP 的 4 张 parquet 放到 data/real/
#   2. 后端已启动（bash start.sh），用于训练后热重载同步
# ============================================================
set -e

cd "$(dirname "$0")"

PY=./venv/Scripts/python.exe
if [ ! -f "$PY" ]; then
    # 回退：尝试系统 python / vendor
    PY=python
fi

echo "使用解释器：$PY"

# 训练 + 切换数据源 + 热重载同步
"$PY" backend/train_real.py "$@"

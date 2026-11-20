#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
真实数据一键训练 + 同步到网页
==============================
流程：
  1. 检查 data/real/ 是否放好了真实 Ali-CCP 的 4 张 parquet
  2. 用真实数据训练 CTR + CVR 两个 DeepFM（产出词表 + 模型 + 指标）
  3. 把激活数据源切到 real（写 data/active_source.json）
  4. 调用后端 /api/reload 热重载，让网页立即显示真实数据 + 真实模型

用法：
  python backend/train_real.py [--epochs 3] [--min-count 5] [--no-reload]
  或直接运行一键脚本 train_real.sh / train_real.bat

说明：
  - 训练复用 backend/train_models.py 的完整逻辑，只是数据目录指向 data/real。
  - 真实数据缺失时给出清晰指引，不做任何破坏性操作。
"""
import os
import sys
import json
import time
import shutil

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
_vendor = os.path.join(_ROOT, "vendor")
if os.path.isdir(_vendor):
    sys.path.insert(0, _vendor)

import config

REAL_DIR = os.path.join(_ROOT, "data", "real")
REQUIRED_TABLES = [
    "common_features_train.parquet",
    "common_features_test.parquet",
    "sample_skeleton_train.parquet",
    "sample_skeleton_test.parquet",
]


def check_real_data():
    """检查真实数据是否就绪，返回 (ok, 缺失列表)。"""
    missing = [t for t in REQUIRED_TABLES if not os.path.exists(os.path.join(REAL_DIR, t))]
    return len(missing) == 0, missing


def switch_source_and_reload(no_reload=False):
    """把激活数据源切到 real，并通知后端热重载。"""
    config.set_active_source("real")
    print(f"  [OK] 激活数据源已切换 -> real ({config.DATA_DIR})")

    if no_reload:
        print("  [跳过] 未调用后端重载（--no-reload）。重启后端后生效。")
        return

    # 调用后端 /api/reload 热重载
    import urllib.request
    url = f"http://{config.HOST}:{config.PORT}/api/reload"
    try:
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        print(f"  [OK] 后端已热重载：has_ctr={data.get('has_ctr')}, has_cvr={data.get('has_cvr')}")
        print(f"       数据源={data.get('data_source')}, 耗时 {data.get('reload_seconds')}s")
    except Exception as e:
        print(f"  [警告] 后端热重载失败（{e}）。")
        print(f"         若后端未运行，请先 `bash start.sh` 再运行本脚本；或运行后重启后端。")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="真实数据一键训练 + 同步到网页")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--no-reload", action="store_true",
                    help="训练完成后不调用后端热重载")
    args = ap.parse_args()

    print("=" * 60)
    print("  真实数据一键训练 · Ali-CTR Lab")
    print("=" * 60)

    # 1. 检查真实数据
    ok, missing = check_real_data()
    if not ok:
        print("\n[错误] data/real/ 缺少以下真实数据文件：")
        for m in missing:
            print(f"   - {m}")
        print("\n请把真实 Ali-CCP 的 4 张 parquet 放到 data/real/ 后重试：")
        print("   common_features_train.parquet")
        print("   common_features_test.parquet")
        print("   sample_skeleton_train.parquet")
        print("   sample_skeleton_test.parquet")
        print("\n（若只有部分文件，可先只放 train 两张，脚本会自动回退到 test。）")
        sys.exit(1)

    print(f"\n[1/3] 真实数据已就绪：{REAL_DIR}")
    for t in REQUIRED_TABLES:
        p = os.path.join(REAL_DIR, t)
        size_mb = os.path.getsize(p) / 1024 / 1024
        print(f"       {t}  ({size_mb:.1f} MB)")

    # 2. 训练（复用 train_models.py，数据目录指向 real）
    print("\n[2/3] 用真实数据训练 CTR + CVR 模型 ...")
    sys.path.insert(0, os.path.join(_ROOT, "backend"))
    from train_models import main as train_main

    # 用 train_models 的 argparse 逻辑，直接构造参数
    sys.argv = [
        "train_models.py",
        "--data-dir", REAL_DIR,
        "--source", "real",
        "--epochs", str(args.epochs),
        "--min-count", str(args.min_count),
    ]
    train_main()

    # 3. 切换数据源 + 热重载
    print("\n[3/3] 同步到网页 ...")
    switch_source_and_reload(no_reload=args.no_reload)

    print("\n" + "=" * 60)
    print("  完成！")
    print(f"  数据源已切换为 real，模型已同步到网页。")
    print(f"  打开 http://{config.HOST}:{config.PORT}/ui/home.html 查看。")
    print("=" * 60)


if __name__ == "__main__":
    main()

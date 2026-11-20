#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从真实 Ali-CCP 数据中各切取 10000 条，作为「真实数据」子集
================================================================
数据源：/data/ERA5/ali/processed_data/
  1. sample_skeleton_train.parquet   —— 样本骨架（含 click/conversion 标签、common_feature_index）
  2. sample_skeleton_test.parquet    —— 同上（测试集）
  3. common_features_train.parquet   —— 公共特征（common_feature_index 关联）
  4. common_features_test.parquet    —— 同上（测试集）

切取逻辑（保持关联完整）：
  - 以 sample_skeleton 为基准，随机抽取 N=10000 条（可 --seed 复现）
  - 从抽取样本中收集用到的 common_feature_index 集合
  - 从 common_features 中按这些 index 过滤，保证关联关系完整
  - 结果保存到「同目录」下，文件名加 _subset 后缀（不覆盖原始数据）

用法（在服务器上运行）：
  python split_subset.py                          # 默认 10000 条
  python split_subset.py --n 10000 --seed 42      # 自定义数量/随机种子
  python split_subset.py --out /path/to/outdir    # 自定义输出目录（默认同目录）

依赖：pandas + pyarrow（一般服务器上已有）
"""
import os
import sys
import argparse

import pandas as pd

SRC_DIR = "/data/ERA5/ali/processed_data"

TABLES = ["common_features_train", "common_features_test",
          "sample_skeleton_train", "sample_skeleton_test"]


def subset_sample_skeleton(df, n, seed):
    """从 sample_skeleton 抽 n 条，返回子集 + 用到的 common_feature_index。"""
    if len(df) <= n:
        return df.copy(), set(df["common_feature_index"].tolist())
    return (df.sample(n=n, random_state=seed).reset_index(drop=True),
            set(df["common_feature_index"].sample(n=n, random_state=seed).tolist()))


def main():
    ap = argparse.ArgumentParser(description="从 Ali-CCP 数据各切取子集")
    ap.add_argument("--src", type=str, default=SRC_DIR, help="源数据目录")
    ap.add_argument("--out", type=str, default=None, help="输出目录（默认同源目录）")
    ap.add_argument("--n", type=int, default=10000, help="每张表切取条数")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    args = ap.parse_args()

    out_dir = args.out or args.src
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"源目录 : {args.src}")
    print(f"输出目录: {out_dir}")
    print(f"切取条数: {args.n}（随机种子 {args.seed}）")
    print("=" * 60)

    # ---- 1. 处理 train / test 两套 ----
    for split in ("train", "test"):
        sk_path = os.path.join(args.src, f"sample_skeleton_{split}.parquet")
        cf_path = os.path.join(args.src, f"common_features_{split}.parquet")

        if not os.path.exists(sk_path) or not os.path.exists(cf_path):
            print(f"[跳过] {split}: 文件不存在，跳过")
            continue

        print(f"\n[处理 {split}]")
        sk = pd.read_parquet(sk_path)
        cf = pd.read_parquet(cf_path)
        print(f"  sample_skeleton_{split}: {len(sk):,} 行")
        print(f"  common_features_{split}: {len(cf):,} 行")

        # 抽样本骨架
        sk_sub = sk.sample(n=min(args.n, len(sk)), random_state=args.seed).reset_index(drop=True)

        # 收集用到的 common_feature_index
        used_idx = set(sk_sub["common_feature_index"].tolist())
        print(f"  抽取样本 {len(sk_sub):,} 条，涉及 {len(used_idx):,} 个 common_feature_index")

        # 过滤公共特征表
        cf_sub = cf[cf["common_feature_index"].isin(used_idx)].reset_index(drop=True)
        print(f"  关联的公共特征 {len(cf_sub):,} 行")

        # 保存（同目录，加 _subset 后缀）
        sk_out = os.path.join(out_dir, f"sample_skeleton_{split}_subset.parquet")
        cf_out = os.path.join(out_dir, f"common_features_{split}_subset.parquet")
        sk_sub.to_parquet(sk_out, index=False)
        cf_sub.to_parquet(cf_out, index=False)
        print(f"  [OK] {os.path.basename(sk_out)}  ({len(sk_sub):,} 行)")
        print(f"  [OK] {os.path.basename(cf_out)}  ({len(cf_sub):,} 行)")

    print("\n" + "=" * 60)
    print("完成！切好的文件保存在同目录，文件名带 _subset 后缀。")
    print("下载到本地项目后，将文件名去掉 _subset 后缀，放到 data/real/ 即可：")
    print("  common_features_train.parquet / sample_skeleton_train.parquet")
    print("  common_features_test.parquet  / sample_skeleton_test.parquet")
    print("=" * 60)


if __name__ == "__main__":
    main()

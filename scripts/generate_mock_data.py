#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ali-CCP 模拟数据生成器
======================

严格对齐真实 Ali-CCP 数据契约，产出 4 张 parquet，使下游（解析 / 词表 / 建模 / 网站）
可以「无缝替换」真实数据：只改数据目录，业务代码零改动。

真实数据契约（来自 ali-ccp-ctr 仓库 docs/01-data-pipeline.md 与 hive/ddl.sql）：

  1. common_features_train.parquet  (common_feature_index, feature_num, feature_list)
  2. common_features_test.parquet   (同上)
  3. sample_skeleton_train.parquet  (sample_id, click, conversion, common_feature_index,
                                     feature_num, feature_list)
  4. sample_skeleton_test.parquet   (同上)

  feature_list 编码：field_id \x02 feature_id \x03 value \x01 ...
      - \x01 分隔「字段」，\x02 分隔 field_id 与 feature_id，\x03 分隔 feature_id 与 value
  feature_num：该行 feature_list 里 token 总数

字段定义（20 个字段，14 单值 + 6 多值）：
  单值 SINGLE: 101,121,122,124,125,126,127,128,129,205,206,207,216,301
  多值 MULTI : 109_14,110_14,127_14,150_14,210,508
  多值权重 value = ln(count)

标签分布（对齐真实）：
  click 点击率 ≈ 3.89%，conversion 转化率 ≈ 0.02%

用法：
  python generate_mock_data.py [--n-users N] [--n-items N] [--n-samples N] [--out DIR] [--seed S]

默认本地小规模（几十万级），足以跑通网站全流程；调到全量规模即可压力测试。
"""
import os
import sys
import math
import json
import random
import argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ===================== 字段定义（与真实契约一致）=====================
SINGLE = ["101", "121", "122", "124", "125", "126", "127", "128", "129",
          "205", "206", "207", "216", "301"]
MULTI = ["109_14", "110_14", "127_14", "150_14", "210", "508"]

# 字段语义（用于生成有业务含义、可分析的数据，而非纯随机）
# 类目类字段：跨样本共享、可做「类目 × 时段」交叉分析
CATEGORY_FIELDS = {
    "206": 120,   # 商品类目一级（广告主行业）
    "207": 80,    # 商品类目二级
    "216": 40,    # 广告位/场景
    "205": 30,    # 投放策略
    "121": 1000,  # 广告主 id 量级
    "122": 5000,  # 商品 id 量级
    "124": 2000,  # 店铺 id 量级
}
# 用户侧字段：高基数（模拟真实里 user/item 高基数特性，其中 101 为近唯一字段）
USER_FIELDS = {
    "101": 200000,  # 近唯一 user id（真实里 field 101 高基数被频次过滤剔除）
    "125": 30000,   # 用户属性 1
    "126": 8000,    # 用户属性 2
    "127": 15000,   # 用户属性 3
    "128": 2000,    # 用户属性 4（地域）
    "129": 5000,    # 用户属性 5
}
# 多值字段：行为序列 / 兴趣标签序列，权重 ln(count)
MULTI_FIELDS = {
    "109_14": (5000, 8),    # 历史行为类目序列 (基数, 最大长度)
    "110_14": (3000, 10),
    "127_14": (2000, 6),
    "150_14": (1000, 5),
    "210":    (300, 4),     # 兴趣标签
    "508":    (200, 3),
}


def _encode_token(field_id, feature_id, value):
    """单个 token：field_id \x02 feature_id \x03 value"""
    return f"{field_id}\x02{feature_id}\x03{value}"


def build_feature_list(rng, user_id, is_clicked):
    """
    为一行样本生成 feature_list 字符串。
    让「类目类字段 + 多值序列」与 click 标签有相关性，保证数据可分析、
    可训练出有区分度的 CTR 模型（模拟真实里类目与点击率的相关性）。
    """
    tokens = []

    # ---- 单值字段 ----
    for f in SINGLE:
        fid = None
        if f == "101":
            fid = user_id                      # 近唯一
        elif f in CATEGORY_FIELDS:
            # 类目字段：点击样本略微偏向某些热门类目
            card = CATEGORY_FIELDS[f]
            # 构造：点击样本集中在少数类目，未点击均匀分布 → 产生可学信号
            if is_clicked and rng.random() < 0.6:
                fid = rng.integers(0, max(1, card // 10))      # 热门头
            else:
                fid = rng.integers(0, card)
        elif f in USER_FIELDS:
            fid = rng.integers(0, USER_FIELDS[f])
        else:
            fid = rng.integers(0, 100)
        tokens.append(_encode_token(f, fid, "1"))

    # ---- 多值字段：序列 + 权重 ln(count) ----
    for f in MULTI:
        base, max_len = MULTI_FIELDS[f]
        # 序列长度服从长尾（多数 1~3 个，少数到 max_len）
        length = min(max_len, max(1, int(rng.exponential(1.0))))
        ids = [rng.integers(0, base) for _ in range(length)]
        # 权重 ln(count)：count 用随机出现次数
        for fid in ids:
            count = rng.integers(1, 5)
            w = round(math.log(max(count, 1)), 4)
            tokens.append(_encode_token(f, fid, w))

    return "\x01".join(tokens)


def build_common_features(rng, n_users, n_items, is_train):
    """生成 common_features 表（维度表：用户/上下文特征，键唯一）。"""
    n = n_users
    idxs = [f"u{i}" for i in range(n)]
    feature_nums = []
    feature_lists = []
    for i in range(n):
        is_clicked = rng.random() < 0.039        # 与样本 CTR 一致，模拟用户偏好
        fl = build_feature_list(rng, i, is_clicked)
        feature_lists.append(fl)
        # feature_num = token 总数（\x01 分隔）
        feature_nums.append(len(fl.split("\x01")))
    return pd.DataFrame({
        "common_feature_index": idxs,
        "feature_num": feature_nums,
        "feature_list": feature_lists,
    })


def build_sample_skeleton(rng, common_idxs, n_samples, ctr=0.0389, cvr=0.0002):
    """
    生成 sample_skeleton 表（事实表：每行一次曝光，带 click/conversion 标签）。
    common_feature_index 从 common_idxs 中抽取（关联命中率 1.0）。
    """
    n = n_samples
    # 曝光从用户池抽样（有放回 → 长尾活跃度）
    pick = rng.integers(0, len(common_idxs), size=n)
    common_idx = np.array(common_idxs)[pick]

    click = (rng.random(n) < ctr).astype(np.int8)
    conversion = ((rng.random(n) < cvr) & (click == 1)).astype(np.int8)  # 转化发生在点击之后

    feature_nums = []
    feature_lists = []
    for i in range(n):
        fl = build_feature_list(rng, int(pick[i]), bool(click[i]))
        feature_lists.append(fl)
        feature_nums.append(len(fl.split("\x01")))

    return pd.DataFrame({
        "sample_id": np.arange(n, dtype=np.int64),
        "click": click,
        "conversion": conversion,
        "common_feature_index": common_idx,
        "feature_num": feature_nums,
        "feature_list": feature_lists,
    })


def write_parquet(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    print(f"  [OK] {path}  ({len(df):,} 行)")


def main():
    ap = argparse.ArgumentParser(description="生成同结构的 Ali-CCP 模拟数据")
    ap.add_argument("--n-users", type=int, default=20000,
                    help="common_features 行数（用户/上下文数），默认 20000")
    ap.add_argument("--n-samples", type=int, default=300000,
                    help="sample_skeleton 行数（曝光数），默认 300000")
    ap.add_argument("--ctr", type=float, default=0.0389, help="点击率，默认 0.0389")
    ap.add_argument("--cvr", type=float, default=0.0002, help="转化率，默认 0.0002")
    ap.add_argument("--out", type=str, default=None,
                    help="输出目录，默认 <脚本目录>/../data/mock")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "data", "mock")
    out_dir = os.path.abspath(out_dir)

    print("=" * 60)
    print("生成 Ali-CCP 模拟数据（对齐真实数据契约）")
    print(f"  用户数   : {args.n_users:,}")
    print(f"  曝光数   : {args.n_samples:,}")
    print(f"  CTR/CVR  : {args.ctr}/{args.cvr}")
    print(f"  输出目录 : {out_dir}")
    print("=" * 60)

    # common_features（train/test 各一半，模拟真实里两个不同时间段）
    half = args.n_users // 2
    cf_train = build_common_features(rng, half, args.n_users, is_train=True)
    cf_test = build_common_features(rng, half, args.n_users, is_train=False)

    # sample_skeleton（train/test 各一半）
    half_s = args.n_samples // 2
    sk_train = build_sample_skeleton(rng, cf_train["common_feature_index"].tolist(),
                                     half_s, args.ctr, args.cvr)
    sk_test = build_sample_skeleton(rng, cf_test["common_feature_index"].tolist(),
                                    args.n_samples - half_s, args.ctr, args.cvr)

    # 写出 4 张 parquet（文件名与真实契约一致）
    print("\n写出 parquet：")
    write_parquet(cf_train, os.path.join(out_dir, "common_features_train.parquet"))
    write_parquet(cf_test, os.path.join(out_dir, "common_features_test.parquet"))
    write_parquet(sk_train, os.path.join(out_dir, "sample_skeleton_train.parquet"))
    write_parquet(sk_test, os.path.join(out_dir, "sample_skeleton_test.parquet"))

    # 生成一份契约元数据，供网站/下游校验
    ctr = float(sk_train["click"].mean())
    cvr = float(sk_train["conversion"].mean())
    meta = {
        "n_common_train": int(len(cf_train)),
        "n_common_test": int(len(cf_test)),
        "n_skeleton_train": int(len(sk_train)),
        "n_skeleton_test": int(len(sk_test)),
        "ctr": ctr,
        "cvr": cvr,
        "seed": args.seed,
        "note": "mock data, same schema as real Ali-CCP",
    }
    meta_path = os.path.join(out_dir, "mock_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n  [OK] 元数据 → {meta_path}")
    print(f"\n实际生成：CTR={ctr:.4f}  CVR={cvr:.4f}")
    print("完成。下游只需把数据目录指向这里即可无缝运行。")


if __name__ == "__main__":
    main()

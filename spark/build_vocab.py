#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PySpark 分布式词表构建 —— 对应 feature_utils.build_vocab 的分布式版
====================================================================

单机版用 collections.Counter 逐行统计 (field, fid) 频次；数据量上去后
内存放不下，改用 Spark 的 flatMap + reduceByKey 在集群上统计。

产出与单机版完全一致：
    feature_index.parquet   —— field / fid / idx / cnt 四列
    vocab.json              —— 每字段规模元数据（n / raw_distinct / raw_occ）

运行：
    python spark/build_vocab.py --input data/real --output model/artifacts --min-count 3 --local 2
"""
import os
import sys
import json
import argparse
import time

from pyspark.sql import SparkSession, functions as F

SINGLE = ["101", "121", "122", "124", "125", "126", "127", "128", "129",
          "205", "206", "207", "216", "301"]
MULTI = ["109_14", "110_14", "127_14", "150_14", "210", "508"]
SEP_FIELD = "\x01"
SEP_FID = "\x02"
SEP_VAL = "\x03"


def parse_tokens(feature_list):
    """把一个 feature_list 字符串解析成 [(field_id, feature_id), ...] 列表。

    作为 Spark 的 flatMap 函数，逐行产出 token，交给 reduceByKey 聚合。
    逻辑与 feature_utils.parse_feature_list 完全一致。
    """
    out = []
    if not feature_list:
        return out
    for tok in feature_list.split(SEP_FIELD):
        if not tok:
            continue
        parts = tok.split(SEP_FID)
        if len(parts) != 2:
            continue
        fid, rest = parts
        sub = rest.split(SEP_VAL)
        if len(sub) != 2:
            continue
        feat_id = sub[0]
        if fid in SINGLE or fid in MULTI:
            try:
                out.append((fid, int(feat_id)))
            except ValueError:
                pass
    return out


def main():
    ap = argparse.ArgumentParser(description="Ali-CCP 分布式词表构建")
    ap.add_argument("--input", required=True, help="parquet 目录（含 feature_list 列）")
    ap.add_argument("--output", required=True, help="词表输出目录")
    ap.add_argument("--min-count", type=int, default=3, help="频次阈值，默认 3")
    ap.add_argument("--local", type=int, default=None, help="本地核数")
    args = ap.parse_args()

    t0 = time.time()
    builder = SparkSession.builder.appName("ali-ccp-vocab")
    if args.local:
        builder = builder.master(f"local[{args.local}]")
    spark = builder.getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    # 读所有 parquet 的 feature_list 列（显式列出文件，避免 Windows 下
    # 读目录触发 NativeIO 权限检查）
    import glob
    paths = sorted(glob.glob(os.path.join(args.input, "*.parquet")))
    if not paths:
        sys.exit(f"[错误] {args.input} 下没有 parquet 文件")
    print(f"[读入] 共 {len(paths)} 个 parquet 文件")
    df = spark.read.parquet(*paths).select("feature_list")
    print(f"[读入] 行数 = {df.count():,}")

    # flatMap 解析 → reduceByKey 统计 (field, fid) 频次
    counts = df.rdd \
        .flatMap(lambda r: parse_tokens(r["feature_list"])) \
        .map(lambda kv: (kv, 1)) \
        .reduceByKey(lambda a, b: a + b)

    # 过滤 min_count，并按 field 分组
    filtered = counts.filter(lambda kv: kv[1] >= args.min_count)
    # 收集到 driver（词表本身规模小，可安全 collect）
    kept = filtered.collect()

    # 为每个 field 生成连续 idx
    from collections import defaultdict
    by_field = defaultdict(list)  # field -> [(fid, cnt)]
    raw_occ = defaultdict(int)    # field -> 总出现次数
    raw_distinct = defaultdict(int)  # field -> 去重 fid 数（需单独统计，这里用近似）
    for (fid_field, feat_id), cnt in kept:
        by_field[fid_field].append((feat_id, cnt))
    # 统计 raw 信息
    all_counts = counts.collect()
    for (fid_field, feat_id), cnt in all_counts:
        raw_occ[fid_field] += cnt
        raw_distinct[fid_field] += 1

    rows = []
    vocab_meta = {}
    for f in SINGLE + MULTI:
        items = sorted(by_field.get(f, []), key=lambda x: x[0])
        fids = [it[0] for it in items]
        for idx, (fid, cnt) in enumerate(items):
            rows.append((f, int(fid), idx, int(cnt)))
        vocab_meta[f] = {
            "n": len(items),
            "raw_distinct": raw_distinct.get(f, 0),
            "raw_occ": raw_occ.get(f, 0),
        }

    os.makedirs(args.output, exist_ok=True)
    # 写 feature_index.parquet（结构与单机版一致）
    idx_df = spark.createDataFrame(rows, ["field", "fid", "idx", "cnt"])
    if sys.platform.startswith("win"):
        # Windows 本地无 winutils：toPandas + pyarrow 兜底写（结果等价）
        import pandas as pd
        idx_df.toPandas().to_parquet(
            os.path.join(args.output, "feature_index.parquet"),
            index=False, engine="pyarrow")
    else:
        idx_df.repartition(1).write.mode("overwrite").parquet(
            os.path.join(args.output, "feature_index.parquet"))
    # 写 vocab.json
    with open(os.path.join(args.output, "vocab.json"), "w", encoding="utf-8") as fh:
        json.dump(vocab_meta, fh, indent=2, ensure_ascii=False)

    print(f"[完成] 保留词表字段 {len(vocab_meta)} 个，min_count={args.min_count}")
    print(f"        特征映射写入 {os.path.join(args.output, 'feature_index.parquet')}")
    print(f"        规模元数据写入 {os.path.join(args.output, 'vocab.json')}")
    print(f"        耗时 {time.time() - t0:.1f}s")

    spark.stop()


if __name__ == "__main__":
    main()

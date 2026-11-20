#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PySpark 分布式特征工程 —— Ali-CCP 训练样本 ETL
=================================================

这是「单机版 scripts/split_subset.py + generate_mock_data.py」的
**分布式生产版**。数据量从几十万涨到百亿级时，pandas 单机扛不住，
改由 Spark 在集群上做同样的特征加工，代码逻辑一一对应。

对应关系（详见 docs/spark_hive.md）：
    单机 (pandas)                    分布式 (本脚本)
    ─────────────────────────────────────────────────
    pd.read_parquet(...)      →     spark.read.parquet(...)
    df.sample(n=...)          →     .sample(withReplacement, fraction)
    df[df.k.isin(set)]        →     .join(..., 'inner')
    feature_list.split(...)   →     split() + explode() 分布式展开
    df.groupby().agg()        →     groupBy().agg() 聚合派生特征
    df.to_parquet(...)        →     .write.parquet(...)

运行方式（本地单机即可验证，生产直接换 master + 改路径到 HDFS）：
    # 本地（2 核）跑通
    JAVA_HOME="..." python spark/etl_features.py \
        --input data/real --output data/spark_output --local 2

    # 集群提交（示意）
    spark-submit --master yarn --deploy-mode cluster spark/etl_features.py \
        --input hdfs:///warehouse/ali/processed_data --output hdfs:///warehouse/ali/features
"""
import os
import sys
import argparse
import time

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import ArrayType, StructType, StructField, StringType

# 与 config.py 对齐的字段契约
SINGLE = ["101", "121", "122", "124", "125", "126", "127", "128", "129",
          "205", "206", "207", "216", "301"]
MULTI = ["109_14", "110_14", "127_14", "150_14", "210", "508"]

SEP_FIELD = "\x01"   # 字段分隔
SEP_FID = "\x02"     # field_id 与 feature_id 分隔
SEP_VAL = "\x03"     # feature_id 与 value 分隔


def build_spark(master, app_name="ali-ccp-feature-etl"):
    """构建 SparkSession。master 为 None 时走集群默认，否则 local[k]。"""
    builder = SparkSession.builder.appName(app_name)
    if master:
        builder = builder.master(f"local[{master}]")
    return builder \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()


def step1_read(spark, input_dir):
    """Step 1：分布式读入 4 张 parquet（生产为 HDFS 路径）。"""
    sk_train = spark.read.parquet(os.path.join(input_dir, "sample_skeleton_train.parquet"))
    sk_test = spark.read.parquet(os.path.join(input_dir, "sample_skeleton_test.parquet"))
    cf_train = spark.read.parquet(os.path.join(input_dir, "common_features_train.parquet"))
    cf_test = spark.read.parquet(os.path.join(input_dir, "common_features_test.parquet"))
    return sk_train, sk_test, cf_train, cf_test


def step2_explode_features(skeleton):
    """Step 2：feature_list 三层编码 → 分布式展开成「字段级」明细表。

    对齐 feature_utils.parse_feature_list 的逻辑：
        feature_list:  fid\x02fid\x03val \x01 fid\x02fid\x03val ...
    展开后每行一个 token，再拆出 field_id / feature_id / value 三列。
    """
    # 先按 \x01 切成 token 数组
    df = skeleton.withColumn("token_arr", F.split("feature_list", SEP_FIELD))
    # 数组逐元素炸开（多值字段每个 feature_id 单独成行）
    df = df.withColumn("token", F.explode("token_arr"))
    df = df.filter(F.col("token") != "")
    # token 内部：fid \x02 (fid \x03 val)
    df = df.withColumn("field_id", F.split("token", SEP_FID)[0])
    inner = F.split(F.split("token", SEP_FID)[1], SEP_VAL)
    df = df.withColumn("feature_id", inner[0])
    df = df.withColumn("value", inner[1])
    return df.select("sample_id", "common_feature_index", "click", "conversion",
                     "field_id", "feature_id", "value")


def step3_user_hist(exploded):
    """Step 3：类目 / 投放策略维度的历史 CTR/CVR 聚合。

    真实 Ali-CCP 数据的 feature_list 里字段为广告/类目侧：
        field 206 = 商品类目一级（广告主行业）
        field 207 = 商品类目二级
        field 205 = 投放策略
    这些字段在事实表里 100% 存在且天然适合做「维度 × 标签」的交叉聚合，
    等价于 Hive 数仓里的 DWS 层（dws_item_hist_1d）。这是 Spark 相对
    pandas 单机的核心优势场景——百亿曝光量的分布式 groupBy。
    """
    dim_rows = exploded.filter(F.col("field_id").isin(["205", "206", "207"]))
    return dim_rows.groupBy("field_id", "feature_id").agg(
        F.count("*").alias("imp_cnt"),
        F.sum("click").alias("click_cnt"),
        F.sum("conversion").alias("conv_cnt"),
        (F.sum("click") / F.count("*")).alias("hist_ctr"),
        (F.sum("conversion") / F.when(F.sum("click") == 0, None)
         .otherwise(F.sum("click"))).alias("hist_cvr"),
    ).withColumnRenamed("feature_id", "dim_id")


def step4_join_wide(skeleton, user_hist):
    """Step 4：样本骨架 join 类目历史 CTR 特征，产出训练宽表。

    用 broadcast join 优化：类目聚合表相对样本是「小表」，广播后避免
    shuffle，是面试高频考点（spark.sql.autoBroadcastJoinThreshold）。
    这里把 field 206（类目一级）的历史 CTR 作为派生特征拼回样本，
    同时保留原始 feature_list 供下游 DeepFM 解析。
    """
    # 从 feature_list 提取 field 206 的 feature_id（类目一级）
    dim_expr = F.regexp_extract(
        F.col("feature_list"),
        r"206" + SEP_FID + r"([^" + SEP_FIELD + r"]+)" + SEP_VAL,
        1,
    )
    skeleton = skeleton.withColumn("cate1_id", dim_expr)

    # 只取 field 206 的聚合结果，与样本按类目一级 join
    cate_hist = user_hist.filter(F.col("field_id") == "206") \
        .select(F.col("dim_id").alias("cate1_id"), "hist_ctr", "hist_cvr", "imp_cnt")

    wide = skeleton.join(
        F.broadcast(cate_hist),
        on="cate1_id",
        how="left",
    )
    return wide.select(
        "sample_id", "common_feature_index", "click", "conversion",
        "feature_list", "cate1_id",
        F.col("hist_ctr").alias("cate1_hist_ctr"),
        F.col("hist_cvr").alias("cate1_hist_cvr"),
        F.col("imp_cnt").alias("cate1_imp_cnt"),
    )


def step5_write(wide, output_dir):
    """Step 5：写训练宽表。

    生产（Linux 集群）用 Spark 分区写 parquet（按 click 分区，利于下游
    按正负样本读取）。Windows 本地无 winutils.exe 时，退化为 toPandas +
    pyarrow 落盘——分布式计算（读/展开/聚合/join）已全部在 Spark 引擎内
    完成，仅「落盘」这一最终 I/O 交给 pyarrow，结果完全等价。
    """
    is_windows = sys.platform.startswith("win")
    os.makedirs(output_dir, exist_ok=True)

    if not is_windows:
        wide.repartition(4).write \
            .mode("overwrite") \
            .partitionBy("click") \
            .parquet(output_dir)
        print(f"         [Spark 分区写] {output_dir} (click=0/1 两分区)")
        return wide

    # Windows 本地：pandas 兜底写（单文件，结构一致）
    import pandas as pd
    pdf = wide.orderBy("sample_id").toPandas()
    pdf.to_parquet(os.path.join(output_dir, "train_wide.parquet"),
                   index=False, engine="pyarrow")
    print(f"         [Windows 本地 pandas 兜底写] {output_dir}/train_wide.parquet"
          f"（{len(pdf):,} 行）")
    return wide


def main():
    ap = argparse.ArgumentParser(description="Ali-CCP 分布式特征工程 ETL")
    ap.add_argument("--input", required=True, help="输入 parquet 目录（本地或 hdfs://）")
    ap.add_argument("--output", required=True, help="输出训练宽表目录")
    ap.add_argument("--local", type=int, default=None,
                    help="本地模式核数（如 2）；不传则走集群默认 master")
    ap.add_argument("--show", type=int, default=10, help="打印结果行数，0 关闭")
    args = ap.parse_args()

    t0 = time.time()
    spark = build_spark(args.local)
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    print("=" * 62)
    print(f"[Step 0] Spark {spark.version}  master={sc.master}  appId={sc.applicationId}")
    print(f"         输入: {args.input}")
    print(f"         输出: {args.output}")
    print("=" * 62)

    # Step 1 读数据
    sk_train, sk_test, cf_train, cf_test = step1_read(spark, args.input)
    print(f"[Step 1] 读入样本骨架 train={sk_train.count():,} test={sk_test.count():,} 行")
    print(f"         公共特征 train={cf_train.count():,} test={cf_test.count():,} 行")

    # 只用 train 集演示完整链路（test 逻辑一致，可同理处理）
    print("[Step 2] 展开 feature_list 三层编码 ...")
    exploded = step2_explode_features(sk_train)
    exploded.cache()
    n_tokens = exploded.count()
    print(f"         展开后 {n_tokens:,} 条 field token")

    print("[Step 3] 类目/策略维度历史 CTR/CVR 聚合 ...")
    user_hist = step3_user_hist(exploded)
    n_dims = user_hist.count()
    print(f"         聚合出 {n_dims:,} 个 (维度, 值) 的历史 CTR/CVR 特征")

    print("[Step 4] 广播 join 类目历史特征生成训练宽表 ...")
    wide = step4_join_wide(sk_train, user_hist)
    print(f"         宽表 {wide.count():,} 行")

    print("[Step 5] 分区写 parquet ...")
    step5_write(wide, args.output)

    if args.show > 0:
        print(f"\n—— 训练宽表预览（前 {args.show} 行）——")
        wide.orderBy("sample_id").limit(args.show).show(args.show, truncate=60)

    print(f"\n完成，耗时 {time.time() - t0:.1f}s。宽表已写入 {args.output}")

    spark.stop()


if __name__ == "__main__":
    main()

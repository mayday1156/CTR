# Hive / Spark 分布式数据工程

> 本文说明 CTR Insight 项目中「单机开发版」与「生产分布式版」的对应关系，以及
> Hive 数仓分层、PySpark 特征工程如何落地、如何本地验证、如何上集群。

## 一、为什么需要 Hive / Spark

项目默认用 **pandas + pyarrow** 在单机做特征加工（`scripts/`），这在小规模
（几十万行）下足够快、足够直观。但 Ali-CCP 是**广告曝光日志**，生产环境动辄
**百亿行、TB 级**，单机内存根本放不下，必须切换到分布式引擎。

因此项目在同一套数据契约下，提供了一条**一一对应的分布式实现**：

| 环节 | 单机开发版 | 生产分布式版 |
|------|-----------|-------------|
| 数据切子集 | `scripts/split_subset.py` (pandas) | Spark `sample()` + `join()` |
| 模拟数据生成 | `scripts/generate_mock_data.py` (pandas) | Hive ODS 建表 + Spark 写 |
| 特征解析 | `backend/feature_utils.py` (纯 Python) | Spark `split()` + `explode()` |
| 词表构建 | `feature_utils.build_vocab` (Counter) | `spark/build_vocab.py` (flatMap + reduceByKey) |
| 派生特征聚合 | pandas `groupby().agg()` | `spark/etl_features.py` (groupBy + broadcast join) |

**关键**：两条链路产出的数据 schema 完全一致，下游 DeepFM 训练、后端预测
**零改动**即可切换。

## 二、Hive 数仓分层（`sql/warehouse.sql`）

`sql/warehouse.sql` 定义了标准的四层数仓，对齐 Ali-CCP 数据契约：

```
ODS 原始日志   ods_ali_ccp_sample_skeleton / ods_ali_ccp_common_features
      │  (清洗、去重、类型规整)
DWD 明细       dwd_ali_ccp_feature_exploded   ← feature_list 三层编码展开
      │  (聚合派生特征)
DWS 汇总       dws_user_hist_1d / dws_item_hist_1d  ← 历史 CTR/CVR
      │  (宽表 join)
ADS 应用       ads_ali_ccp_train_wide         ← 最终训练宽表
```

- **ODS→DWD**：用 `LATERAL VIEW explode(split(feature_list, '\u0001'))` 把
  三层编码串 `field_id\x02feature_id\x03value\x01...` 拆成字段级明细。
- **DWD→DWS**：`GROUP BY field_id, feature_id` 聚合出「类目/策略维度历史
  CTR/CVR」，这是典型的广告数仓特征。
- **DWS→ADS**：`JOIN` 回样本，产出含派生特征的训练宽表，供 DeepFM 消费。

> 在真实 Hive 环境直接 `hive -f sql/warehouse.sql` 建表跑数；本地无需 Hive。

## 三、PySpark 特征工程（`spark/`）

### `spark/etl_features.py` —— 训练样本 ETL（核心）

完整实现「读 → 展开 → 聚合 → 广播 join → 落盘」五步，对应 Hive 的 DWD/DWS/ADS：

1. **Step 1 读**：`spark.read.parquet()` 分布式读 4 张 parquet
2. **Step 2 展开**：`split()` + `explode()` 把 feature_list 三层编码展开成
   字段级明细（对齐 `feature_utils.parse_feature_list`）
3. **Step 3 聚合**：`groupBy(field_id, feature_id).agg()` 算出类目/策略维度的
   历史 CTR/CVR 派生特征（对应 `dws_item_hist_1d`）
4. **Step 4 join**：`broadcast()` 小表广播，避免 shuffle，拼回样本宽表
5. **Step 5 写**：分区写 parquet（生产按 click 分区）

### `spark/build_vocab.py` —— 词表构建

用 `flatMap` + `reduceByKey` 分布式统计 `(field, fid)` 频次，按 `min_count`
过滤并生成连续 idx，产出与单机版 `build_vocab` 完全一致的
`feature_index.parquet` + `vocab.json`。

## 四、本地运行（Windows / Linux 单机）

### 前置

- JDK 17（Spark 4.x 要求，不再支持 Java 8）
- `pip install pyspark`

```bash
# Windows 需设置 JAVA_HOME（例）
export JAVA_HOME="/c/Program Files/Eclipse Adoptium/jdk-17.0.21.9-hotspot"
```

### 跑特征 ETL

```bash
python spark/etl_features.py \
    --input data/real \
    --output data/spark_output \
    --local 2 \
    --show 10
```

### 跑词表构建

```bash
python spark/build_vocab.py \
    --input data/real \
    --output data/spark_vocab \
    --min-count 3 \
    --local 2
```

### Windows 说明（winutils 问题）

Windows 本地无 HDFS，Spark 写文件时 Hadoop 层会找 `winutils.exe`（报
`HADOOP_HOME and hadoop.home.dir are unset`）。本项目脚本已内置兼容处理：

- **计算全部在 Spark 引擎内完成**（读、explode、groupBy、broadcast join 都是
  真实分布式算子）；
- **仅最终「落盘」这一 I/O** 在 Windows 下退化为 `toPandas()` + pyarrow 写，
  结果 schema 与集群版完全一致。

若你本机有 `winutils.exe`（放入 `C:\hadoop\bin` 并设 `HADOOP_HOME=C:\hadoop`），
脚本会自动走纯 Spark 写路径，无需任何改动。

## 五、上集群（生产）

Linux 集群上无需 Windows 兼容逻辑，直接 `spark-submit`：

```bash
spark-submit --master yarn --deploy-mode cluster \
    --executor-memory 8g --executor-cores 4 --num-executors 20 \
    spark/etl_features.py \
    --input hdfs:///warehouse/ali/processed_data \
    --output hdfs:///warehouse/ali/features
```

关键调优点（面试高频）：

- `spark.sql.adaptive.enabled=true` + `coalescePartitions` —— AQE 自适应优化
- `broadcast join` —— 小表广播避免 shuffle（脚本已用 `F.broadcast()`）
- `spark.sql.shuffle.partitions` —— 控制 shuffle 分区数，避免小文件/数据倾斜

## 六、技术栈总览（简历可写）

- **数仓**：Hive（ODS/DWD/DWS/ADS 四层分层建模）
- **计算引擎**：Apache Spark / PySpark（分布式 ETL + 特征工程）
- **特征**：feature_list 三层编码解析、维度交叉聚合、词表构建
- **建模**：DeepFM（CTR + CVR 双任务）
- **服务**：FastAPI + Vue3（SaaS 工作台）

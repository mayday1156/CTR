-- ============================================================================
-- CTR Insight · Ali-CCP 数仓分层脚本（Hive / Spark SQL）
-- ============================================================================
-- 说明：
--   本文件描述「生产级分布式数仓」如何对 Ali-CCP 广告点击/转化日志做
--   分层建模与特征加工。本地单机开发用 pandas/PySpark 等价实现（见
--   scripts/ 与 spark/ 目录），本脚本是与之一一对应的 Hive 版本，
--   用于在有 Hive 数仓的环境（HDFS + Spark/Hive on YARN）直接建表跑数。
--
--   数据契约（feature_list 三层编码）：
--       field_id \x02 feature_id \x03 value \x01 ... （\x01=字段分隔，\x02=field/fid，\x03=fid/value）
--   20 个字段：14 单值(SINGLE) + 6 多值(MULTI)，详见 config.py。
--
--   分层约定：
--       ODS  —— 原始日志，不加工，仅落地
--       DWD  —— 清洗、去重、类型规整、字段解析
--       DWS  —— 聚合派生特征（用户/商品/时段维度的历史 CTR 等）
--       ADS  —— 最终训练宽表（供 DeepFM 训练 / 在线预估消费）
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 0. 建库
-- ----------------------------------------------------------------------------
CREATE DATABASE IF NOT EXISTS ctr_insight
    COMMENT 'CTR Insight 广告点击率分析数仓';
USE ctr_insight;

-- ============================================================================
-- 一、ODS 层：贴源原始数据
-- ============================================================================
-- 样本骨架（事实表）：一次曝光一行，带 click/conversion 标签
CREATE EXTERNAL TABLE IF NOT EXISTS ods_ali_ccp_sample_skeleton (
    sample_id            BIGINT      COMMENT '样本 ID',
    click                TINYINT     COMMENT '是否点击 0/1',
    conversion           TINYINT     COMMENT '是否转化 0/1',
    common_feature_index STRING      COMMENT '关联维度表外键',
    feature_num          INT         COMMENT 'feature_list token 总数',
    feature_list         STRING      COMMENT '原始特征编码串',
    dt                   STRING      COMMENT '分区：曝光日期 yyyymmdd'
)
COMMENT 'Ali-CCP 样本骨架原始日志（ODS）'
PARTITIONED BY (dt STRING)
STORED AS PARQUET
LOCATION '/warehouse/ctr_insight/ods/ods_ali_ccp_sample_skeleton';

-- 公共特征（维度表）：用户/上下文特征，键唯一
CREATE EXTERNAL TABLE IF NOT EXISTS ods_ali_ccp_common_features (
    common_feature_index STRING COMMENT '维度主键',
    feature_num          INT    COMMENT 'feature_list token 总数',
    feature_list         STRING COMMENT '原始特征编码串'
)
COMMENT 'Ali-CCP 公共特征原始表（ODS）'
STORED AS PARQUET
LOCATION '/warehouse/ctr_insight/ods/ods_ali_ccp_common_features';

-- ============================================================================
-- 二、DWD 层：清洗 + feature_list 结构化解析
-- ============================================================================
-- 把 feature_list 按 \x01 拆成多行（每行一个 field token），便于后续聚合
CREATE TABLE IF NOT EXISTS dwd_ali_ccp_feature_exploded (
    sample_id            BIGINT  COMMENT '样本 ID',
    common_feature_index STRING  COMMENT '维度外键',
    click                TINYINT COMMENT '标签：点击',
    conversion           TINYINT COMMENT '标签：转化',
    field_id             STRING  COMMENT '字段 ID（101/121/.../508）',
    feature_id           STRING  COMMENT '特征 ID',
    value                STRING  COMMENT '特征值/权重'
)
COMMENT 'feature_list 按字段拆解后的明细（DWD）'
STORED AS PARQUET;

INSERT OVERWRITE TABLE dwd_ali_ccp_feature_exploded
SELECT
    s.sample_id,
    s.common_feature_index,
    s.click,
    s.conversion,
    t.field_id,
    t.feature_id,
    t.value
FROM ods_ali_ccp_sample_skeleton s
LATERAL VIEW explode(split(s.feature_list, '\u0001')) t AS token
LATERAL VIEW explode(array(split(t.token, '\u0002')[0])) tf AS field_id
LATERAL VIEW explode(array(
        split(split(t.token, '\u0002')[1], '\u0003')[0]
    )) fv AS feature_id
LATERAL VIEW explode(array(
        split(split(t.token, '\u0002')[1], '\u0003')[1]
    )) vv AS value
WHERE t.token <> '';

-- ============================================================================
-- 三、DWS 层：聚合派生特征（历史 CTR / 曝光量等）
-- ============================================================================
-- 3.1 用户历史行为特征（近 7 天滑动窗口，field 101 = user id）
CREATE TABLE IF NOT EXISTS dws_user_hist_1d (
    user_id          STRING COMMENT '用户 ID（field 101）',
    imp_cnt          BIGINT COMMENT '曝光数',
    click_cnt        BIGINT COMMENT '点击数',
    conv_cnt         BIGINT COMMENT '转化数',
    hist_ctr         DOUBLE COMMENT '历史 CTR = click/imp',
    hist_cvr         DOUBLE COMMENT '历史 CVR = conv/click'
)
COMMENT '用户历史点击/转化聚合（DWS）'
STORED AS PARQUET;

INSERT OVERWRITE TABLE dws_user_hist_1d
SELECT
    feature_id AS user_id,
    COUNT(1)   AS imp_cnt,
    SUM(click) AS click_cnt,
    SUM(conversion) AS conv_cnt,
    SUM(click) / NULLIF(COUNT(1), 0) AS hist_ctr,
    SUM(conversion) / NULLIF(SUM(click), 0) AS hist_cvr
FROM dwd_ali_ccp_feature_exploded
WHERE field_id = '101'
GROUP BY feature_id;

-- 3.2 商品/类目维度历史 CTR（field 206 = 类目一级，field 122 = 商品 id）
CREATE TABLE IF NOT EXISTS dws_item_hist_1d (
    item_id    STRING COMMENT '商品/类目 ID',
    field_id   STRING COMMENT '字段类型（122 商品 / 206 类目）',
    imp_cnt    BIGINT,
    click_cnt  BIGINT,
    hist_ctr   DOUBLE
)
COMMENT '商品/类目历史 CTR 聚合（DWS）'
STORED AS PARQUET;

INSERT OVERWRITE TABLE dws_item_hist_1d
SELECT
    feature_id AS item_id,
    field_id,
    COUNT(1)   AS imp_cnt,
    SUM(click) AS click_cnt,
    SUM(click) / NULLIF(COUNT(1), 0) AS hist_ctr
FROM dwd_ali_ccp_feature_exploded
WHERE field_id IN ('122', '206')
GROUP BY feature_id, field_id;

-- ============================================================================
-- 四、ADS 层：最终训练宽表（样本 + 原始特征 + 派生特征）
-- ============================================================================
CREATE TABLE IF NOT EXISTS ads_ali_ccp_train_wide (
    sample_id            BIGINT  COMMENT '样本 ID',
    common_feature_index STRING  COMMENT '维度外键',
    click                TINYINT COMMENT '标签：点击',
    conversion           TINYINT COMMENT '标签：转化',
    feature_list         STRING  COMMENT '原始特征编码串（下游 DeepFM 解析）',
    user_id              STRING  COMMENT '用户 ID（field 101）',
    user_hist_ctr        DOUBLE  COMMENT '用户历史 CTR',
    user_hist_cvr        DOUBLE  COMMENT '用户历史 CVR',
    user_imp_cnt         BIGINT  COMMENT '用户历史曝光数'
)
COMMENT '训练宽表：样本 + 派生特征（ADS，供 DeepFM 训练）'
STORED AS PARQUET;

INSERT OVERWRITE TABLE ads_ali_ccp_train_wide
SELECT
    s.sample_id,
    s.common_feature_index,
    s.click,
    s.conversion,
    s.feature_list,
    u.user_id,
    u.hist_ctr AS user_hist_ctr,
    u.hist_cvr AS user_hist_cvr,
    u.imp_cnt  AS user_imp_cnt
FROM ods_ali_ccp_sample_skeleton s
LEFT JOIN (
    SELECT feature_id AS user_id, imp_cnt, click_cnt, conv_cnt, hist_ctr, hist_cvr
    FROM dws_user_hist_1d
) u
ON u.user_id = (
    -- 从 feature_list 中提取 field 101 的 feature_id（单值字段，取第一个匹配）
    split(
        split(
            regexp_extract(s.feature_list, '101\\x02([^\\x01]+)', 1),
            '\\x03'
        )[0], '\\x03'
    )[0]
);

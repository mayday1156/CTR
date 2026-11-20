# CTR Insight · 点击率智能分析平台

一个面向广告点击率预估（CTR / CVR）场景的**数据分析 SaaS 平台**。平台以阿里 Ali-CCP 数据集为基准，围绕「数据工程 → 特征加工 → 模型预估 → 可视化分析」构建了一条完整的链路，并具备独立的 SaaS 产品形态：落地页、产品功能、客户案例、博客、定价、登录注册、工作台、文档与关于。

## 技术栈

- **前端**：多页面网站 + 工作台单页应用（Vue3 + ECharts + 自建设计系统，CDN 引入、无构建步骤）
- **后端**：FastAPI + Uvicorn（RESTful API，共 8 组接口），支持 gunicorn 多进程部署
- **数据**：Parquet + 实时聚合（对齐阿里 Ali-CCP 数据契约）；生产级分布式链路见 `sql/warehouse.sql`（Hive 数仓分层）+ `spark/`（PySpark 特征工程）
- **模型**：DeepFM（CTR + CVR 双任务）+ LR / GBDT 基线
- **存储与认证**：SQLite（用户 / 会话 / 数据集 / 任务），密码 PBKDF2 加盐哈希，token 会话持久化

## 核心能力

**数据工程**

- 严格对齐 Ali-CCP 数据契约：`feature_list` 三层编码（`field_id \x02 feature_id \x03 value \x01 ...`），20 个字段（14 单值 + 6 多值）
- 支持 parquet / csv / xlsx / json 多格式导入，网站引导字段映射，`feature_list` 列可自动解析
- 多维度交叉分析：类目 × 时段、投放策略 × 点击率等派生统计

**分布式链路（Hive / Spark）**

- `sql/warehouse.sql`：Hive 数仓 ODS → DWD → DWS → ADS 四层分层建模，`feature_list` 三层编码展开 + 维度历史 CTR/CVR 聚合
- `spark/etl_features.py`：PySpark 训练样本 ETL（读 → explode 展开 → groupBy 聚合 → broadcast join → 落盘）
- `spark/build_vocab.py`：PySpark 分布式词表构建（flatMap + reduceByKey 统计频次）
- 与单机 pandas 版一一对应、产出 schema 一致，下游 DeepFM 训练零改动切换

**模型预估**

- DeepFM 双任务建模（CTR 点击率 + CVR 转化率），FM 捕捉二阶交叉 + DNN 捕捉高阶非线性
- 一键执行 CTR + CVR 双预测，输出结果分布与明细，支持导出 CSV

**可视化与分析**

- 概览仪表盘、模型对比、数据分布等多维度可视化看板（ECharts）
- 模型指标（AUC / 精确率 / 召回率等）实时展示

## 网站结构

```
CTR Insight
├── 首页 /ui/home.html          落地页（Hero + 亮点 + 数据能力 + CTA）
├── 产品功能 /ui/features.html   四大核心功能
├── 客户案例 /ui/cases.html      案例展示
├── 博客 /ui/blog.html          技术文章
├── 定价 /ui/pricing.html       免费 / 专业 / 企业三档
├── 登录注册 /ui/login.html      登录 / 注册（token 认证）
├── 工作台 /ui/app.html         核心产品（概览 / 分析 / 导入预测 / 模型对比 / 数据集 / 任务历史 / 用户中心）
├── 文档 /ui/docs.html          使用指南
└── 关于 /ui/about.html         项目与技术栈
```

## 用户与数据管理

- 登录注册：SQLite 存储用户，密码 PBKDF2 加盐哈希，token 会话（7 天）
- 数据隔离：上传的数据集、预测任务按用户隔离，互不可见
- 工作台模块：数据集管理（列表 / 删除）、任务历史（列表 / 详情 / 删除）、用户中心
- 未登录可使用「数据导入与预测」（不落库），登录后自动记录到个人名下

相关接口：`/api/auth/*`（认证）、`/api/data/*`（数据集 / 任务，需 `Authorization: Bearer <token>`）。

## 本地运行

### 0. 环境

项目使用虚拟环境 `venv/` 隔离依赖（含 CPU 版 torch）。首次使用执行：

```bash
# Git Bash / MSYS2 终端
bash setup.sh

# 或 Windows CMD（setup.bat）
```

### 1. 生成模拟数据

模拟数据与真实 Ali-CCP 数据同结构，可无缝替换：

```bash
./venv/Scripts/python.exe scripts/generate_mock_data.py --n-users 20000 --n-samples 200000 --seed 42
```

产物位于 `data/mock/`，共 4 张 parquet + 一份 `mock_meta.json`。

### 2. 训练模型（CTR + CVR）

```bash
./venv/Scripts/python.exe backend/train_models.py --min-count 3 --epochs 2
```

产出词表 + CTR 模型 + CVR 模型 + 指标到 `model/artifacts/`。

### 3. 启动后端（含完整网站）

```bash
# Git Bash / MSYS2 终端
bash start.sh

# 或 Windows CMD（start.bat）
```

启动后访问 **http://127.0.0.1:8000/ui/**（根路径自动跳转）。网站首页为 `/ui/home.html`，工作台为 `/ui/app.html`。

### 4. 导入真实数据并训练（一键脚本）

平台**默认展示模拟数据**。拿到真实 Ali-CCP 数据后，两步切换：

1. 将 4 个 parquet 放入 `data/real/`（`common_features_train/test`、`sample_skeleton_train/test`）
2. 保持后端运行，执行一键脚本：

```bash
# Git Bash / MSYS2 终端
bash train_real.sh

# 或 Windows CMD（train_real.bat）
# 自定义参数：bash train_real.sh --epochs 5
```

脚本自动完成：**检测数据 → 训练 CTR+CVR → 切换数据源到 real → 热重载后端同步到网页**。

> 手动等价操作：`./venv/Scripts/python.exe backend/train_real.py`（参数 `--epochs` / `--min-count` / `--no-reload`）。

### 5. 数据导入 + 预测流程

1. 上传同格式数据（parquet / csv / xlsx / json）
2. 网站引导字段映射（含 `feature_list` 列自动解析）
3. 一键执行 CTR + CVR 双预测
4. 查看结果分布 + 明细，导出 CSV

详见 `docs/design.md` 第 9 节。

## 数据源切换机制

- 数据源由 `config.DATA_SUBDIR` 决定（`mock` / `real`），默认 `mock`
- 优先级：环境变量 `ALI_CCP_DATA` > 状态文件 `data/active_source.json` > 默认 `mock`
- `POST /api/reload` 在不重启的情况下切换数据源并重载模型；`GET /api/meta` 返回当前数据源与模型来源 / 训练时间

## Hive / Spark 分布式数据工程

平台在 pandas 单机开发之外，提供了一条**一一对应的生产分布式链路**（同一数据契约，下游 DeepFM 训练零改动切换）：

| 环节 | 单机开发版 | 生产分布式版 |
|------|-----------|-------------|
| 数据切子集 | `scripts/split_subset.py`（pandas） | Spark `sample()` + `join()` |
| 特征解析 | `backend/feature_utils.py` | Spark `split()` + `explode()` |
| 词表构建 | `feature_utils.build_vocab` | `spark/build_vocab.py`（flatMap + reduceByKey） |
| 派生特征聚合 | pandas `groupby().agg()` | `spark/etl_features.py`（groupBy + broadcast join） |

本地验证（需 JDK 17 + `pip install pyspark`）：

```bash
export JAVA_HOME="/c/Program Files/Eclipse Adoptium/jdk-17.0.21.9-hotspot"  # Windows
python spark/etl_features.py --input data/real --output data/spark_output --local 2
python spark/build_vocab.py    --input data/real --output data/spark_vocab --min-count 3 --local 2
```

集群提交使用 `spark-submit --master yarn`。详见 `docs/spark_hive.md`。

## 部署上线

本地 `bash start.sh` 为本地运行；部署为公网可访问的正式网站，使用 Docker 一键构建：

```bash
# 服务器上（已装 Docker）
docker compose up -d --build
```

部署配置包括：

- `Dockerfile` + `docker-compose.yml` —— 一键构建镜像并启动（含 CPU 版 torch）
- `deploy/nginx.conf` —— 绑定域名 + HTTPS 反代配置
- `.env.example` —— 环境变量模板
- `docs/deploy.md` —— 完整部署指南（Docker / 裸机 / 域名 / HTTPS / 生产注意事项）

服务默认绑定 `0.0.0.0`，前端 API 使用相对路径，部署到任意域名自动适配。

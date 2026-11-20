# 项目设计说明

> CTR Insight 的设计背景、架构与关键技术决策。
> 技术栈：**Vue3 + ECharts（前端） · FastAPI（后端） · Parquet + SQLite（数据） · DeepFM / LR / GBDT（模型） · Hive / PySpark（分布式链路）**

---

## 1. 项目背景与定位

项目源自对阿里 Ali-CCP 数据集（4300 万条真实广告曝光日志）的离线分析建模，目标是把「数据契约探查 → feature_list 解析 → 频次过滤建词表 → 关联 / 负采样 / 切分 → DeepFM 训练 → 评估」这一整套能力，产品化为一个**可交互的 Web 平台**，让「数据动态分析 + CTR 预估 + 模型对比」在浏览器中交互式完成。

项目定位是一条完整链路：**数据 → 特征 → 模型 → 产品化**。

---

## 2. 功能模块

| 模块 | 页面 | 核心能力 |
|------|------|---------|
| **数据概览** | 仪表盘 | 样本量、用户 / 类目 / 字段数、CTR、转化率、正负样本比例、字段覆盖分布 |
| **动态数据分析** | 交互分析 | 类目 × 时段 CTR 交叉、行为序列长度分布、曝光 → 点击 → 转化漏斗、字段 unknown 率、任意维度下钻 |
| **特征探索** | 特征 | 字段重要性、相关性热力图、类别分布对比 |
| **CTR 预估** | 在线预估 | 输入 / 选取样本 → 实时返回 pCTR 与 pCVR |
| **模型对比** | 模型 | LR / GBDT / DeepFM 的 AUC、LogLoss、GAUC 对比 |
| **数据导入与预测** | 导入 | 上传同格式数据 → 字段映射 → 批量 CTR+CVR 预测 → 结果导出 |

---

## 3. 技术架构

```
┌──────────────────────┐      ┌──────────────────────┐      ┌────────────────────────┐
│  前端（多页 + SPA）    │ ←→  │   FastAPI 后端         │ ←→  │  数据层 / 模型层         │
│  Vue3 + ECharts        │      │  8 组 RESTful 接口    │      │  Parquet + 聚合 JSON    │
│  CDN 引入、无构建步骤    │      │  认证 / 数据 / 分析   │      │  DeepFM 权重 + vocab    │
│  11 个页面              │      │  导入 / 预测 / 模型   │      │  LR / GBDT 基线         │
└──────────────────────┘      └──────────────────────┘      └────────────────────────┘
```

| 层 | 选型 | 理由 |
|----|------|------|
| 前端 | Vue3 + ECharts（CDN 引入） | 无需构建步骤，静态托管即可部署；ECharts 支持动态交互、下钻、联动 |
| 后端 | FastAPI | 与数据 / 模型同语言（Python），接口即文档（Swagger） |
| 数据 | Parquet + 预聚合 JSON 缓存 | 千万行原始数据无法实时全读，网站只查聚合结果，秒开 |
| 模型 | DeepFM + LR / GBDT 基线 | 多档对比，体现建模深度 |

**核心设计决策：网站不直接读原始大表**，而是通过聚合层把 parquet 预聚合成前端所需的 JSON 缓存（概览统计、交叉表、漏斗、unknown 率等），后端只做查询 + 轻量实时聚合 + 模型推理。数据量再大，网站也保持流畅。

---

## 4. 模拟数据与真实数据的无缝切换（重点）

模拟数据与真实数据**同一套 schema**，通过**数据目录 + 环境变量**切换，业务代码零改动。

```
数据源抽象：
  data/mock/  ──┐
               ├──> 同一套「加载 + 聚合 + 建模」代码 ──> 同一套 FastAPI 接口 ──> 同一套前端
  data/real/  ──┘
（切换：环境变量 ALI_CCP_DATA 或 config.py 的 DATA_DIR）
```

**对齐保证**（见 `scripts/generate_mock_data.py`）：

1. **4 张 parquet 同名同列同类型**，与真实 `hive/ddl.sql` 一致：
   - `common_features_train/test`：`common_feature_index(string), feature_num(int64), feature_list(string)`
   - `sample_skeleton_train/test`：`sample_id(int64), click(int8), conversion(int8), common_feature_index(string), feature_num(int64), feature_list(string)`
2. **feature_list 三层分隔编码一致**：`field_id\x02feature_id\x03value\x01...`
3. **20 个字段一致**：14 单值（101/121/122/124/125/126/127/128/129/205/206/207/216/301）+ 6 多值（109_14/110_14/127_14/150_14/210/508），多值权重 `ln(count)`
4. **标签分布一致**：CTR ≈ 3.89%，CVR ≈ 0.02%
5. **业务语义可分析**：类目字段与点击标签相关（非纯随机），保证能分析出结论、能训练出有区分度的模型

**模型侧同样抽象**：模拟数据与真实数据训练的权重均存为标准格式（`.pt` + `vocab.json`），预测接口按 config 加载对应权重。

---

## 5. 数据导入 + 批量预测

平台支持导入同格式数据（或按引导映射字段），用已有模型做 **CTR + CVR 双预测**，查看并导出结果。

### 5.1 交互流程

```
上传文件(parquet/csv/xlsx/json) → 字段映射引导 → 数据质量校验 → 批量预测(CTR+CVR) → 结果展示 / 导出
```

### 5.2 支持的格式与字段映射

| 格式 | 说明 | 字段映射 |
|------|------|---------|
| parquet | 严格对齐 Ali-CCP 4 表结构 | 自动识别，无需映射 |
| csv / xlsx / json | 非标准列名 | 引导把列映射到标准字段 |

**字段映射引导**：解析上传文件后，列出实际列名，引导映射到标准字段：

- 必备：`common_feature_index`（或 `sample_id`）、`feature_list`（或结构化特征列）
- 可选：`click`、`conversion`（有则用于评估，无则纯预测）
- 若上传结构化特征列（如 `f206_id`、`f121_id`），引导映射到 20 个标准字段

### 5.3 预测流程

1. **数据加载**：读取上传文件，解析 `feature_list`（三层分隔编码）或直接读结构化列
2. **fid → idx 映射**：用词表 `vocab.json` / `feature_index.parquet` 做 remap，未命中 → 0 号 unknown 槽
3. **CTR 预测**：加载 DeepFM 权重，`sigmoid(logit)` 得到 pCTR
4. **CVR 预测**：加载 CVR 模型，得到 pCVR
5. **结果输出**：每行 pCTR、pCVR，以及可选评估（若含 `click` / `conversion` 标签，计算 AUC / LogLoss）

### 5.4 CVR 模型的实现

CTR 与 CVR 为双任务，采用 ESMM 思路：

- **方案 A（已实现）**：复用 DeepFM 特征表示，训练独立的 CVR 分类头（`conversion` 为标签，仅在 click=1 的样本上训练），推理时 pCTCVR = pCTR × pCVR
- **方案 B（可选扩展）**：共享 embedding + 双塔输出 CTR/CVR，端到端建模

当前采用方案 A 打通链路，方案 B 作为后续扩展。

---

## 6. 后端接口设计

8 组 RESTful 接口：

| 组 | 路径前缀 | 功能 |
|----|---------|------|
| 认证 | `/api/auth/*` | 注册、登录、会话管理（token 认证） |
| 数据 | `/api/data/*` | 数据集与任务历史管理（用户隔离） |
| 概览 | `/api/overview` | 概览统计 |
| 分析 | `/api/analysis/*` | 交叉分析、漏斗、序列分布 |
| 导入 | `/api/import/*` | 文件上传、字段映射、批量预测、结果导出 |
| 预测 | `/api/predict` | CTR 在线预估 |
| 模型 | `/api/models` | 模型对比指标 |
| 元信息 | `/api/meta` | 数据源、模型来源、训练时间 |

---

## 7. CTR 模型方案（多档对比）

| 模型 | 实现 | 特点 |
|------|------|------|
| Logistic Regression | scikit-learn | 线性基线，可解释 |
| LightGBM / GBDT | lightgbm | 树模型，处理稀疏类别特征强 |
| DeepFM | PyTorch | field-aware embedding + FM 二阶 + DNN，效果最优 |

对比维度：AUC、LogLoss、GAUC、训练耗时、参数量。

# Ali-CTR Lab —— 广告点击率数据分析与预估平台

> 把已有的 Ali-CCP 离线分析/建模能力，产品化为一个可交互的 Web 展示平台。
> 技术栈：**Vue3 + ECharts（前端） · FastAPI（后端） · Parquet + JSON 缓存（数据） · DeepFM/LR/GBDT（模型）**

---

## 1. 项目背景与定位

你已有一个完整的离线仓库 `ali-ccp-ctr`，在 4300 万条阿里真实曝光日志上完成了：
数据契约探查（Hive）→ feature_list 解析（Spark）→ 频次过滤建词表 → 关联/负采样/切分 → DeepFM 训练 → 官方 test 评估（AUC 0.5969 / GAUC 0.5775 / 校准 / 上界分析）。

本项目的目标：**把这套能力做成一个网站**，让"数据动态分析 + CTR 预估 + 模型对比"可以在浏览器里交互式完成，作为简历上的完整项目成果（数据 → 特征 → 模型 → 产品化全链路）。

---

## 2. 功能模块（网站包含什么）

| 模块 | 页面 | 核心功能 | 数据来源 |
|------|------|---------|---------|
| **数据概览** | 仪表盘 Dashboard | 样本量、用户/类目/字段数、CTR、转化率、正负样本比例、字段覆盖分布 | 预聚合缓存 |
| **动态数据分析** | 交互分析 Analysis | 类目×时段 CTR 交叉、行为序列长度分布、曝光→点击→转化漏斗、字段 unknown 率、任意维度下钻 | 后端实时聚合 |
| **特征探索** | 特征 Feature | 字段重要性、相关性热力图、类别分布对比 | 离线预计算 |
| **CTR 预估** | 在线预估 Predict | 输入/选取一个样本 → 实时返回 pCTR（点击概率） | 加载模型权重推理 |
| **模型对比** | 模型 Models | LR / GBDT / DeepFM 的 AUC、LogLoss、GAUC 对比 | 离线结果 |

---

## 3. 技术架构

```
┌──────────────────────┐      ┌──────────────────────┐      ┌────────────────────────┐
│  Vue3 + Vite + ECharts│ ←→  │   FastAPI 后端         │ ←→  │  数据层 / 模型层         │
│  4~5 个页面           │      │  /overview /analysis  │      │  Parquet + 聚合JSON缓存  │
│  动态交互图表          │      │  /features /predict   │      │  DeepFM权重 + vocab.json │
│  （可按维度下钻/筛选） │      │  /models             │      │  LR/GBDT 基线模型         │
└──────────────────────┘      └──────────────────────┘      └────────────────────────┘
```

| 层 | 选型 | 理由 |
|----|------|------|
| 前端 | Vue3 + Vite + ECharts | 你已选定；ECharts 动态交互、下钻、联动强 |
| 后端 | FastAPI | 与数据/模型同语言（Python），接口即文档（Swagger） |
| 数据 | Parquet + 预聚合 JSON 缓存 | 4300 万行原始数据不能实时全读，网站只查聚合结果，秒开 |
| 模型 | DeepFM（复用）+ LR/GBDT 基线 | 多档对比，体现建模深度 |

**核心设计决策：网站不直接读原始大表**，而是通过一个 `aggregate.py` 把 parquet 预聚合成前端要的 JSON 缓存（概览统计、交叉表、漏斗、unknown 率等），后端只做查询 + 轻量实时聚合 + 模型推理。真实数据再大，网站也流畅。

---

## 4. 无缝替换模拟/真实数据（重点）

模拟数据与真实数据**同一套 schema**，通过**数据目录 + 环境变量**切换，业务代码零改动。

```
数据源抽象：
  data/mock/  ──┐
               ├──> 同一套「加载 + 聚合 + 建模」代码 ──> 同一套 FastAPI 接口 ──> 同一套前端
  data/real/  ──┘
（切换：改 ALI_CCP_ROOT 或 config.py 里的 DATA_DIR）
```

**对齐保证**（已实现，见 `scripts/generate_mock_data.py`）：

1. **4 张 parquet 同名同列同类型**，与真实 `hive/ddl.sql` 一致：
   - `common_features_train/test`：`common_feature_index(string), feature_num(int64), feature_list(string)`
   - `sample_skeleton_train/test`：`sample_id(int64), click(int8), conversion(int8), common_feature_index(string), feature_num(int64), feature_list(string)`
2. **feature_list 三层分隔编码一致**：`field_id\x02feature_id\x03value\x01...`
3. **20 个字段一致**：14 单值（101/121/122/124/125/126/127/128/129/205/206/207/216/301）+ 6 多值（109_14/110_14/127_14/150_14/210/508），多值权重 `ln(count)`
4. **标签分布一致**：CTR ≈ 3.89%，CVR ≈ 0.02%
5. **业务语义可分析**：类目字段与点击标签相关（非纯随机），保证能分析出东西、能训练出有区分度的模型

**模型侧同样抽象**：模拟数据训练的权重与真实数据训练的权重，都存成标准格式（`.pt` + `vocab.json`），`predict` 接口按 config 加载对应权重。

---

## 5. 目录结构（独立新项目 `ali-ctr-lab`）

```
ali-ctr-lab/
├── README.md                    # 项目说明
├── docs/
│   └── design.md                # 本方案文档
├── config.py                    # 统一配置（数据目录、模型路径、端口）
├── scripts/
│   └── generate_mock_data.py    # 模拟数据生成（已完成）
├── backend/                     # FastAPI 后端
│   ├── main.py                  # 应用入口 + 路由
│   ├── aggregate.py             # parquet → 聚合 JSON 缓存
│   ├── models.py                # 模型加载 + 推理（LR/GBDT/DeepFM）
│   └── routers/                 # overview/analysis/features/predict/models
├── frontend/                    # Vue3 + Vite + ECharts
│   ├── src/
│   │   ├── views/               # Dashboard/Analysis/Feature/Predict/Models
│   │   ├── components/          # 图表组件
│   │   └── api/                 # axios 封装
│   └── package.json
├── model/                       # 模型训练 + 权重
│   ├── train_baseline.py        # LR/GBDT 基线
│   ├── train_deepfm.py          # DeepFM（复用现有 torch/ 代码）
│   └── artifacts/               # 训练产物（.pt / vocab.json / 指标 json）
└── data/
    ├── mock/                    # 模拟数据（已生成）
    └── real/                    # 真实数据（用户自行放入）
```

---

## 6. 后端接口设计

| 方法 | 路径 | 功能 | 入参 |
|------|------|------|------|
| GET | `/api/overview` | 概览统计 | — |
| GET | `/api/analysis/cross` | 类目×时段 CTR 交叉 | `dim1, dim2, metric` |
| GET | `/api/analysis/funnel` | 曝光→点击→转化漏斗 | `filters` |
| GET | `/api/analysis/seq` | 行为序列长度分布 | `field` |
| GET | `/api/features/importance` | 特征重要性 | — |
| GET | `/api/features/corr` | 相关性热力图 | — |
| POST | `/api/predict` | CTR 在线预估 | `sample 特征` |
| GET | `/api/models` | 模型对比指标 | — |

---

## 7. CTR 模型方案（多档对比）

| 模型 | 实现 | 特点 | 预期指标量级 |
|------|------|------|------------|
| Logistic Regression | scikit-learn | 线性基线，可解释 | AUC 较低 |
| LightGBM / GBDT | lightgbm | 树模型，处理稀疏类别特征强 | AUC 中等 |
| DeepFM | 复用现有 `torch/step6_deepfm.py` | field-aware embedding + FM 二阶 + DNN | AUC 最高（真实 0.5969） |

对比维度：AUC、LogLoss、GAUC、训练耗时、参数量。

---

## 8. 落地里程碑

- [x] **M1 · 模拟数据生成**：`scripts/generate_mock_data.py`（本轮完成，已跑通）
- [ ] **M2 · 聚合分析层**：`backend/aggregate.py` 生成概览/交叉/漏斗/unknown 率 JSON 缓存
- [ ] **M3 · 模型层**：LR/GBDT 基线 + DeepFM 复用，产出模型对比指标
- [ ] **M4 · 后端 FastAPI**：5 组接口
- [ ] **M5 · 前端 Vue3**：4~5 个页面，ECharts 动态图表
- [ ] **M6 · 无缝切换验证**：真实数据一键切换

---

## 9. 数据导入 + 批量预测模块（新增）

支持用户在网站里**导入同格式数据**（或按网站引导映射字段），然后用已有模型做 **CTR + CVR 双预测**，查看并导出结果。

### 9.1 交互流程

```
上传文件(parquet/csv) → 字段映射引导 → 数据质量校验 → 批量预测(CTR+CVR) → 结果展示/导出
```

### 9.2 支持的导入格式与字段映射

| 格式 | 说明 | 字段映射 |
|------|------|---------|
| parquet | 严格对齐 Ali-CCP 4 表结构 | 自动识别，无需映射 |
| csv / 普通表格 | 非标准列名 | 网站引导用户把列映射到标准字段 |

**字段映射引导**：网站解析上传文件后，列出文件的实际列名，引导用户把它们映射到标准字段：
- 必备字段：`common_feature_index`（或 `sample_id`）、`feature_list`（或结构化特征列）
- 可选字段：`click`、`conversion`（有则用于评估，无则纯预测）
- 若上传的是结构化特征列（如 `f206_id`、`f121_id`...），引导映射到 20 个标准字段

### 9.3 预测流程

1. **数据加载**：读取上传文件，解析 `feature_list`（三层分隔编码）或直接读结构化列
2. **fid → idx 映射**：用词表 `vocab.json`/`feature_index.parquet` 做 remap，未命中 → 0 号 unknown 槽
3. **CTR 预测**：加载 DeepFM 权重，`sigmoid(logit)` 得到 pCTR
4. **CVR 预测**：加载 CVR 模型（见 9.4 说明），得到 pCVR
5. **结果输出**：每行 `pCTR`、`pCVR`，以及可选的评估（若导入含 `click`/`conversion` 标签，计算 AUC/LogLoss）

### 9.4 CVR 模型的实现说明（重要）

你现有 `ali-ccp-ctr` 仓库**只训练了 CTR 模型**（DeepFM 预测 click），`conversion` 列当时只是"保留给 ESMM 双任务留接口"（ROADMAP Step 9），**没有现成的 CVR 模型**。

因此"CTR + CVR 双预测"需要补一个 CVR 模型。采用**最小可用的 ESMM 思路**：

- **方案 A（推荐，简单）**：复用 DeepFM 的特征表示，训练一个独立的 CVR 分类头（`conversion` 作为标签，仅在 click=1 的样本上训练，推理时 pCTCVR = pCTR × pCVR）
- **方案 B（完整 ESMM）**：共享 embedding + 双塔输出 CTR/CVR，端到端建模（工作量更大）

先实现方案 A，验证链路跑通后再考虑方案 B。

### 9.5 后端接口（新增）

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/api/import/upload` | 上传文件，返回文件 id + 列名/预览 |
| POST | `/api/import/mapping` | 提交字段映射，返回数据质量校验报告 |
| POST | `/api/import/predict` | 执行批量预测（CTR+CVR），返回任务 id |
| GET | `/api/import/result/{task_id}` | 查询预测结果 + 汇总统计 |
| GET | `/api/import/export/{task_id}` | 导出预测结果（csv/parquet） |

### 9.6 前端页面（新增）

`Import` 页：上传 → 映射引导（表格展示列对应关系）→ 校验报告 → 预测进度 → 结果表格 + pCTR/pCVR 分布图 → 导出按钮。

---

## 10. 下一步（本轮之后）

本轮交付了**方案文档 + 模拟数据脚本**。下一步建议按顺序：

1. **M2 聚合分析层**（`backend/aggregate.py`）——把模拟数据聚合成前端要的 JSON
2. **M3 模型层**（LR/GBDT 基线 + DeepFM）——产出可对比的模型指标
3. **M4/M5 前后端**——搭出可运行的网站骨架
4. **M6 真实验证**——用你的真实数据替换模拟数据，一键切换

需要我继续做哪一步，告诉我即可。

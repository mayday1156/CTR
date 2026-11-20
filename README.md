# CTR Insight · 点击率智能分析平台

一个面向广告点击率预估场景的**数据分析 SaaS 平台**。数据分析与 CTR/CVR 预估是核心业务模块，网站本身具备完整的 SaaS 产品形态：首页、产品功能、工作台、使用文档、关于。

- **前端**：多页面网站 + 工作台单页应用（Vue3 + ECharts + 自建设计系统）
- **后端**：FastAPI + Uvicorn
- **数据**：Parquet + 实时聚合（对齐阿里 Ali-CCP 数据契约）
- **模型**：DeepFM（CTR + CVR 双任务）+ LR / GBDT 基线

## 网站结构

```
CTR Insight
├── 首页 /ui/home.html          落地页（Hero + 亮点 + 数据能力 + CTA）
├── 产品功能 /ui/features.html   四大核心功能
├── 客户案例 /ui/cases.html      案例展示
├── 博客 /ui/blog.html          技术文章
├── 定价 /ui/pricing.html       免费/专业/企业三档
├── 登录注册 /ui/login.html      登录 / 注册（token 认证）
├── 工作台 /ui/app.html         核心产品（概览 / 分析 / 导入预测 / 模型对比 / 数据集 / 任务历史 / 用户中心）
├── 文档 /ui/docs.html          使用指南
└── 关于 /ui/about.html         项目与技术栈
```

## 用户与数据管理

- **登录注册**：SQLite 存储用户，密码 PBKDF2 加盐哈希，token 会话（7 天）。数据库 `data/app.db`（首次启动自动建表）。
- **数据隔离**：登录后上传的数据集、预测任务按用户隔离，互不可见。
- **工作台新增模块**：数据集管理（列表/删除）、任务历史（列表/详情/删除）、用户中心。
- 未登录也能用「数据导入与预测」（不落库），登录后自动记录到个人名下。

相关接口：`/api/auth/*`（认证）、`/api/data/*`（数据集/任务，需 `Authorization: Bearer <token>`）。

## 快速开始

### 0. 环境初始化（首次必做）

项目使用**虚拟环境**（`venv/`），避免和系统 Python 冲突。两种方式任选其一：

```bash
# Git Bash / MSYS2 终端
bash setup.sh

# 或 Windows CMD（双击 setup.bat）
```

> 已为你创建好 `venv/` 并装好全部依赖（含 CPU 版 torch），一般无需重跑。

### 1. 生成模拟数据（与真实 Ali-CCP 数据同结构，可无缝替换）

```bash
./venv/Scripts/python.exe scripts/generate_mock_data.py --n-users 20000 --n-samples 200000 --seed 42
```

产物在 `data/mock/` 下，共 4 张 parquet + 一份 `mock_meta.json`。

### 2. 训练模型（CTR + CVR）

```bash
./venv/Scripts/python.exe backend/train_models.py --min-count 3 --epochs 2
```

产出词表 + CTR 模型 + CVR 模型 + 指标到 `model/artifacts/`。

### 3. 启动后端（含完整网站）

```bash
# Git Bash / MSYS2 终端
bash start.sh

# 或 Windows CMD（双击 start.bat）
```

然后浏览器打开 **http://127.0.0.1:8000/ui/**（根路径 http://127.0.0.1:8000/ 自动跳转）。

- 网站首页：`/ui/home.html`
- 工作台：`/ui/app.html`

### 4. 导入真实数据并训练（一键脚本）

网站**默认展示模拟数据**。拿到真实 Ali-CCP 数据后，两步切换：

1. 把 4 个 parquet 放进 `data/real/`（`common_features_train/test`、`sample_skeleton_train/test`）
2. 保持后端运行，执行一键脚本：

```bash
# Git Bash / MSYS2 终端
bash train_real.sh

# 或 Windows CMD（双击 train_real.bat）
# 自定义参数：bash train_real.sh --epochs 5
```

脚本自动完成：**检测数据 → 训练 CTR+CVR → 切换数据源到 real → 热重载后端同步到网页**。
刷新浏览器即可看到真实数据与真实模型（工作台有「真实数据」徽章）。

> 手动等价操作：`./venv/Scripts/python.exe backend/train_real.py`（参数 `--epochs` / `--min-count` / `--no-reload`）。

### 5. 数据导入 + 预测流程

1. 上传同格式数据（parquet / csv / xlsx / json）
2. 网站引导字段映射（含 `feature_list` 列可自动解析）
3. 一键执行 CTR + CVR 双预测
4. 查看结果分布 + 明细，导出 CSV

详见 `docs/design.md` 第 9 节。

## 数据源切换机制

- 数据源由 **`config.DATA_SUBDIR`** 决定（`mock` / `real`），默认 `mock`。
- 优先级：环境变量 `ALI_CCP_DATA` > 状态文件 `data/active_source.json` > 默认 `mock`。
- 后端 `POST /api/reload` 可在不重启的情况下切换数据源并重载模型；`GET /api/meta` 返回当前数据源与模型来源/训练时间，供前端展示。

## 部署上线（变成真正的网站）

本地 `bash start.sh` 只是「本地运行」。要部署成公网可访问的正式网站，项目已备好一键部署配置：

```bash
# 服务器上（已装 Docker）
docker compose up -d --build
# 访问 http://<服务器IP>:8000 即可，任何人打开链接都能用
```

已提供：
- `Dockerfile` + `docker-compose.yml` —— 一键构建镜像并启动（含 CPU 版 torch）
- `deploy/nginx.conf` —— 绑定域名 + HTTPS 反代配置
- `.env.example` —— 环境变量模板
- `docs/deploy.md` —— 完整部署指南（Docker / 裸机 / 域名 / HTTPS / 生产注意事项）

**本轮为「可部署」做的关键改造**：
- 服务默认绑定 `0.0.0.0`（公网可访问，`config.py` 可环境变量覆盖）
- token 会话从内存改为持久化到 SQLite（`sessions` 表），重启不失效
- 新增 `backend/wsgi.py` 支持 gunicorn 多进程
- 前端 API 用相对路径，部署到任何域名自动适配

> 详见 `docs/deploy.md`。

## 重要：为什么用 venv

你的终端里 `python` 可能指向多个不同解释器（如 `C:\Python315`），导致依赖互相找不到。
`venv/` 隔离了本项目所有依赖，**所有命令都用 `./venv/Scripts/python.exe`**（或 `bash start.sh`），就不会再报 `ModuleNotFoundError`。

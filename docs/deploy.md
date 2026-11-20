# Ali-CTR Lab 部署指南

本文档说明如何把 CTR Insight 网站从「本地运行」变成「真正的网站」——即部署到服务器后，任何人通过公网地址/域名即可访问，无需 clone 项目。

---

## 目录

1. [从本地到线上：你需要理解什么](#1-从本地到线上你需要理解什么)
2. [方式一：Docker 一键部署（推荐）](#2-方式一docker-一键部署推荐)
3. [方式二：裸机部署（不用 Docker）](#3-方式二裸机部署不用-docker)
4. [绑定域名 + HTTPS](#4-绑定域名--https)
5. [导入真实数据并训练（线上）](#5-导入真实数据并训练线上)
6. [生产环境注意事项](#6-生产环境注意事项)
7. [常见问题](#7-常见问题)

---

## 1. 从本地到线上：你需要理解什么

现在的项目是一个**单体 FastAPI 应用**：后端 API + 前端静态页面 + DeepFM 模型推理 + SQLite 数据库，全在同一个服务里。所谓「部署上线」，本质是：

| 本地运行 | 线上运行 |
|---------|---------|
| `bash start.sh` 在你电脑上起服务 | 在**服务器**上起同样的服务 |
| 服务绑定 `127.0.0.1`，只有本机能访问 | 服务绑定 `0.0.0.0`，公网都能访问 |
| 数据/模型在你本机磁盘 | 数据/模型在服务器磁盘（volume 挂载） |
| token 会话存内存，重启失效 | token 会话存 SQLite，重启**不**失效（已改好） |

**关键改造已经完成**（本轮已改好，你无需再动代码）：
- ✅ 服务默认绑定 `0.0.0.0`（`config.py`，可由环境变量覆盖）
- ✅ token 会话持久化到 SQLite（`backend/db.py` 新增 `sessions` 表）
- ✅ 前端 API 用相对路径（`window.location.origin`），部署到任何域名都自动适配
- ✅ 提供 `backend/wsgi.py` 支持 gunicorn 多进程

---

## 2. 方式一：Docker 一键部署（推荐）

Docker 把「装 Python、装 torch、装依赖、起服务」全部打包成一个镜像，服务器上只需 3 条命令。

### 2.1 服务器准备

```bash
# 1. 安装 Docker + Docker Compose（以 Ubuntu 为例）
curl -fsSL https://get.docker.com | sh
sudo systemctl enable --now docker
sudo apt-get install -y docker-compose-plugin   # 新版 docker compose
```

### 2.2 把项目上传到服务器

```bash
# 方式 A：git（推荐）
git clone <你的仓库地址> ali-ctr-lab && cd ali-ctr-lab

# 方式 B：scp 上传整个目录
scp -r ./ali-ctr-lab user@服务器IP:/home/user/
```

### 2.3 构建并启动

```bash
cd ali-ctr-lab
docker compose up -d --build
```

首次构建会下载 torch（约 2GB），耐心等几分钟。完成后：

```bash
# 查看状态
docker compose ps
# 查看日志
docker compose logs -f app
```

### 2.4 访问

```
http://<服务器IP>:8000
```

打开即可看到网站首页。**任何能访问这个 IP 的人都能用这个网站**（这就是「真正的网站」了）。

> 服务器安全组/防火墙需放行 8000 端口（云服务器在控制台安全组里配置）。

---

## 3. 方式二：裸机部署（不用 Docker）

如果服务器不想装 Docker，直接跑 Python 也行（需服务器已装 Python 3.10+）。

```bash
cd ali-ctr-lab

# 1. 建虚拟环境并装依赖
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. 启动（绑定 0.0.0.0 公网可访问）
ALI_CCP_HOST=0.0.0.0 ./venv/bin/gunicorn backend.wsgi:app \
    -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

访问 `http://<服务器IP>:8000` 即可。

---

## 4. 绑定域名 + HTTPS

有了域名后（阿里云/腾讯云购买，DNS 解析 A 记录到服务器 IP），配置 Nginx 反代 + 免费 HTTPS 证书。

### 4.1 安装 Nginx

```bash
sudo apt-get install -y nginx
```

### 4.2 配置反代

把 `deploy/nginx.conf` 里的 `example.com` 换成你的域名，然后：

```bash
sudo cp deploy/nginx.conf /etc/nginx/conf.d/ali-ctr-lab.conf
sudo nginx -t && sudo nginx -s reload
```

### 4.3 免费 HTTPS 证书（certbot）

```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d 你的域名.com -d www.你的域名.com
```

certbot 会自动修改 Nginx 配置、申请并续期证书。完成后打开 `https://你的域名.com` 即是带锁的正式网站。

> 说明：`deploy/nginx.conf` 里已内置了 HTTPS 配置模板（注释状态），certbot 会自动启用。

---

## 5. 导入真实数据并训练（线上）

部署上线后，网站默认展示**模拟数据**（初始状态）。要切换到真实数据：

```bash
# 1. 把真实数据放到 data/real/（4 个 parquet）
#    Docker：scp 到服务器的 <项目>/data/real/
#    裸机：cp 到 <项目>/data/real/

# 2. 训练并切换（Docker）
docker compose exec app python backend/train_real.py --epochs 5

#    裸机
./venv/bin/python backend/train_real.py --epochs 5
```

脚本会自动：训练 CTR+CVR 模型 → 切换数据源到 real → 热重载。跑完刷新浏览器即看到真实数据与真实模型。

---

## 6. 生产环境注意事项

| 事项 | 说明 |
|------|------|
| **CPU / GPU** | 当前镜像用 CPU 版 torch。真实数据全量训练/预测量大时，建议换 GPU 服务器或换 CUDA 版 torch 镜像 |
| **内存** | DeepFM 模型推理 + 多 worker 会占内存，4 worker 建议 ≥ 4GB 内存；不够就在 `docker-compose.yml` 调低 `GUNICORN_WORKERS` |
| **数据持久化** | `data/` 和 `model/` 已用 volume 挂载到宿主机，容器删了数据不丢 |
| **数据库** | 目前 SQLite 单文件（`data/app.db`）。并发用户很多时（>几十人同时写）建议迁移到 PostgreSQL |
| **备份** | 定期备份 `data/`（含 app.db、uploads、predictions）和 `model/artifacts/` |
| **安全** | 生产务必启用 HTTPS；`data/app.db` 含用户密码哈希，勿暴露 |
| **训练在线上跑** | 大训练任务会占用服务器 CPU，建议离线训练好再把模型 `scp` 到服务器 `model/artifacts/` |

---

## 7. 常见问题

**Q1：部署后打不开网站？**
- 检查服务器安全组/防火墙是否放行 8000 端口
- `docker compose logs -f app` 看服务是否正常启动

**Q2：上传的数据、训练的模型会丢吗？**
- 不会。`data/` 和 `model/` 都挂载在宿主机，删容器、重建镜像都不影响。

**Q3：登录状态为什么重启后还在？**
- 本轮已把 token 会话从内存改为存 SQLite（`sessions` 表），重启不失效。这正是「真正网站」的必要改造。

**Q4：想加 GPU？**
- 把 `Dockerfile` 里的 `torch` 换成 CUDA 版，基础镜像换成 `nvidia/cuda`，`docker-compose.yml` 加 `gpus: all`。详见 PyTorch 官方文档。

**Q5：我现在还没买服务器，怎么验证部署代码是对的？**
- 本地装 Docker 后，直接 `docker compose up -d --build`，访问 `http://127.0.0.1:8000` 即等同于线上效果（除了别人访问不到你的局域网之外，一切一致）。

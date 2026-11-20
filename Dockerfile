# ============ Ali-CTR Lab 生产镜像 ============
# 基于 CPU 版 torch，构建后约 2~3 GB（torch 占大头）。
# 构建：docker build -t ali-ctr-lab .
# 若服务器有 NVIDIA GPU，可把 torch 换成 CUDA 版镜像加速推理。

FROM python:3.11-slim

# 系统依赖（pyarrow / torch 运行时需要的库）
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖（利用 Docker 层缓存，代码改动不必重装依赖）
COPY requirements.txt .
# CPU 版 torch 用清华镜像源加速（国内服务器更快）
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
        torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 拷贝项目代码
COPY . .

# 数据 / 模型 / 数据库目录（运行时用 volume 挂载，见 docker-compose.yml）
# 这里声明为卷，避免数据写进镜像层
VOLUME ["/app/data", "/app/model"]

# 暴露服务端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://127.0.0.1:8000/api/health || exit 1

# 生产启动：gunicorn + uvicorn worker 多进程
# （本地开发仍用 `uvicorn backend.main:app`，见 start.sh）
CMD ["gunicorn", "backend.wsgi:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--access-logfile", "-"]

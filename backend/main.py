#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ali-CTR Lab 后端入口
=====================
FastAPI 应用，提供：数据概览 / 动态分析 / 特征 / 在线预测 / 模型对比 / 数据导入预测。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routers import import_router, predict_router, overview_router, models_router, analysis_router, meta_router, auth_router, data_router

app = FastAPI(title="Ali-CTR Lab API", version="0.1.0")

# CORS：允许前端开发服务器访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(overview_router.router, prefix="/api")
app.include_router(models_router.router, prefix="/api")
app.include_router(predict_router.router, prefix="/api")
app.include_router(import_router.router, prefix="/api")
app.include_router(analysis_router.router, prefix="/api")
app.include_router(meta_router.router, prefix="/api")
app.include_router(auth_router.router, prefix="/api")
app.include_router(data_router.router, prefix="/api")


@app.get("/")
def root():
    """根路径重定向到网站首页。"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ui/home.html")


# 前端静态托管：访问 http://127.0.0.1:8000/ui 打开网站
_frontend = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(_frontend):
    app.mount("/ui", StaticFiles(directory=_frontend, html=True), name="ui")


@app.get("/api/health")
def health():
    return {"status": "ok", "data_dir": config.DATA_DIR}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=False)

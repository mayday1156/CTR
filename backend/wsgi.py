#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生产 WSGI 入口（供 gunicorn 使用）
==================================
用法：
    gunicorn backend.wsgi:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000

说明：
    - 生产环境用 gunicorn + uvicorn worker 承载多进程并发
    - 本地开发仍可直接 `uvicorn backend.main:app`（见 start.sh）
"""
from backend.main import app  # noqa: F401

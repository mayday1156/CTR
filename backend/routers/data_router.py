#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据管理路由（数据集 + 预测任务历史，按用户隔离）
==================================================
GET    /api/data/datasets        列出当前用户的数据集
DELETE /api/data/datasets/{file_id}  删除数据集
GET    /api/data/tasks           列出当前用户的预测任务
GET    /api/data/tasks/{task_id} 任务详情（结果摘要 + 前若干条）
DELETE /api/data/tasks/{task_id} 删除任务
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from backend import db
from backend.routers.auth_router import _current_user

from fastapi import APIRouter, Depends, HTTPException
import pandas as pd

router = APIRouter()


@router.get("/data/datasets")
def list_datasets(user: dict = Depends(_current_user)):
    return {"datasets": db.list_datasets(user["id"])}


@router.delete("/data/datasets/{file_id}")
def remove_dataset(file_id: str, user: dict = Depends(_current_user)):
    ok = db.delete_dataset(user["id"], file_id)
    if not ok:
        raise HTTPException(404, "数据集不存在")
    return {"status": "ok"}


@router.get("/data/tasks")
def list_tasks(user: dict = Depends(_current_user)):
    return {"tasks": db.list_tasks(user["id"])}


@router.get("/data/tasks/{task_id}")
def task_detail(task_id: str, user: dict = Depends(_current_user)):
    tasks = [t for t in db.list_tasks(user["id"]) if t["task_id"] == task_id]
    if not tasks:
        raise HTTPException(404, "任务不存在")
    task = tasks[0]
    # 尝试读取结果 csv 的前若干条
    head = []
    result_path = os.path.join(config.PREDICT_DIR, f"{task_id}.csv")
    if os.path.exists(result_path):
        try:
            df = pd.read_csv(result_path)
            head = df.head(100).to_dict(orient="records")
        except Exception:
            head = []
    return {"task": task, "head": head, "result_path_exists": os.path.exists(result_path)}


@router.delete("/data/tasks/{task_id}")
def remove_task(task_id: str, user: dict = Depends(_current_user)):
    ok = db.delete_task(user["id"], task_id)
    if not ok:
        raise HTTPException(404, "任务不存在")
    return {"status": "ok"}

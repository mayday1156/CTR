#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据导入 + 批量预测路由
========================
用户上传文件（parquet/csv），网站引导字段映射，用已有模型做 CTR+CVR 预测，导出结果。

接口：
  POST /api/import/upload           上传文件 → 返回 file_id + 列名 + 预览
  POST /api/import/mapping          提交字段映射 → 返回数据质量校验报告
  POST /api/import/predict          执行批量预测 → 返回 task_id
  GET  /api/import/result/{task_id} 查询预测结果汇总
  GET  /api/import/export/{task_id} 导出预测结果（csv）
  GET  /api/import/schema           返回标准字段定义（供前端做映射引导）
"""
import os
import sys
import json
import uuid
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from backend import inference
from backend import db
from backend.feature_utils import parse_feature_list, SINGLE, MULTI

from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List

import pandas as pd
import numpy as np
import io

router = APIRouter()

# 内存中的任务状态（单机演示用；生产可换 Redis/DB）
_uploads = {}      # file_id -> {path, columns, preview, filename}
_mappings = {}     # file_id -> {mapping, report}
_tasks = {}        # task_id -> {file_id, result_path, summary, status}


def _optional_user(authorization: Optional[str] = Header(None)):
    """可选登录：返回当前用户或 None（未登录也能用，但不落库）。"""
    if authorization and authorization.startswith("Bearer "):
        try:
            user = db.get_session_user(authorization[len("Bearer "):].strip())
            return user
        except Exception:
            return None
    return None


# ===================== 辅助 =====================

def _read_upload(path, filename):
    """按扩展名读取上传文件为 DataFrame。"""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    elif ext in (".csv", ".txt"):
        return pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    elif ext == ".json":
        return pd.read_json(path)
    else:
        raise HTTPException(400, f"不支持的文件格式：{ext}")


def _parse_rows(df, mapping):
    """根据字段映射把 DataFrame 转成结构化 rows（复用 feature_utils 的解析）。"""
    # mapping: {标准字段 -> 实际列名}
    # 两种情况：feature_list 列（原始编码）或结构化列（f{f}_id / f{f}_ids / f{f}_w）
    rows = []
    fl_col = mapping.get("feature_list", "feature_list")
    has_feature_list = fl_col in df.columns

    for _, r in df.iterrows():
        if has_feature_list and pd.notna(r[fl_col]):
            single, m_ids, m_w = parse_feature_list(str(r[fl_col]))
            row = {}
            for f in SINGLE:
                row[f"f{f}_id"] = single[f]
            for f in MULTI:
                row[f"f{f}_ids"] = m_ids[f]
                row[f"f{f}_w"] = m_w[f]
        else:
            row = {}
            for f in SINGLE:
                col = mapping.get(f"f{f}_id", f"f{f}_id")
                if col in df.columns and pd.notna(r[col]):
                    row[f"f{f}_id"] = int(r[col])
                else:
                    row[f"f{f}_id"] = -1
            for f in MULTI:
                col_ids = mapping.get(f"f{f}_ids", f"f{f}_ids")
                col_w = mapping.get(f"f{f}_w", f"f{f}_w")
                ids = []
                w = []
                if col_ids in df.columns and pd.notna(r[col_ids]):
                    v = r[col_ids]
                    if isinstance(v, str):
                        ids = [int(x) for x in v.strip("[]").split(",") if x.strip()]
                    elif hasattr(v, "__iter__"):
                        ids = [int(x) for x in v]
                if col_w in df.columns and pd.notna(r[col_w]):
                    v = r[col_w]
                    if isinstance(v, str):
                        w = [float(x) for x in v.strip("[]").split(",") if x.strip()]
                    elif hasattr(v, "__iter__"):
                        w = [float(x) for x in v]
                row[f"f{f}_ids"] = ids
                row[f"f{f}_w"] = w if w else [1.0] * len(ids)
        rows.append(row)
    return rows


# ===================== 接口 =====================

@router.get("/import/schema")
def import_schema():
    """标准字段定义（供前端映射引导）。"""
    return {
        "single": SINGLE,
        "multi": MULTI,
        "required": ["feature_list"],  # 或结构化列
        "optional": ["click", "conversion"],
        "note": "上传含 feature_list 列的文件可自动解析；否则需映射到 f{field}_id / f{field}_ids / f{field}_w 结构化列。",
    }


@router.post("/import/upload")
async def import_upload(file: UploadFile = File(...),
                        authorization: Optional[str] = Header(None)):
    """上传文件，返回 file_id + 列名 + 预览。"""
    file_id = uuid.uuid4().hex[:12]
    ext = os.path.splitext(file.filename or "upload.csv")[1].lower()
    safe_path = os.path.join(config.UPLOAD_DIR, f"{file_id}{ext}")
    with open(safe_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        df = _read_upload(safe_path, file.filename)
    except Exception as e:
        raise HTTPException(400, f"文件解析失败：{e}")

    columns = [str(c) for c in df.columns]
    preview = df.head(5).where(pd.notna(df.head(5)), None).to_dict(orient="records")
    # 预览里 list 列转 str 以便 JSON 序列化
    for p in preview:
        for k, v in list(p.items()):
            if isinstance(v, (list, np.ndarray)):
                p[k] = str(v)

    _uploads[file_id] = {
        "path": safe_path, "columns": columns, "preview": preview,
        "filename": file.filename, "n_rows": int(len(df)), "ext": ext,
    }

    # 登录用户：记录数据集到 DB
    user = _optional_user(authorization)
    if user:
        db.add_dataset(user["id"], file_id, file.filename, ext,
                       int(len(df)), columns)

    return {
        "file_id": file_id, "columns": columns, "preview": preview,
        "n_rows": int(len(df)), "filename": file.filename,
    }


class MappingRequest(BaseModel):
    file_id: str
    mapping: Optional[dict] = None   # {标准字段 -> 实际列名}


@router.post("/import/mapping")
def import_mapping(req: MappingRequest):
    """提交字段映射，返回数据质量校验报告。"""
    if req.file_id not in _uploads:
        raise HTTPException(404, "file_id 不存在，请先上传")
    up = _uploads[req.file_id]
    df = _read_upload(up["path"], up["filename"])
    mapping = req.mapping or {}

    # 数据质量校验
    report = {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": up["columns"],
        "mapping": mapping,
        "checks": {},
    }

    # 检查 feature_list 或结构化字段是否可用
    fl_col = mapping.get("feature_list", "feature_list")
    if fl_col in df.columns:
        report["parse_mode"] = "feature_list"
        report["checks"]["feature_list"] = "已识别 feature_list 列，将按三层分隔编码解析"
    else:
        # 结构化字段覆盖检查
        mapped_single = [f for f in SINGLE if mapping.get(f"f{f}_id", f"f{f}_id") in df.columns]
        mapped_multi = [f for f in MULTI if mapping.get(f"f{f}_ids", f"f{f}_ids") in df.columns]
        report["parse_mode"] = "structured"
        report["checks"]["single_fields_mapped"] = len(mapped_single)
        report["checks"]["multi_fields_mapped"] = len(mapped_multi)
        if len(mapped_single) == 0 and len(mapped_multi) == 0:
            report["checks"]["warning"] = "未识别到任何特征字段，请完成字段映射"

    # 标签列（可选，用于预测后评估）
    click_col = mapping.get("click", "click")
    conv_col = mapping.get("conversion", "conversion")
    report["has_click"] = click_col in df.columns
    report["has_conversion"] = conv_col in df.columns
    if report["has_click"]:
        report["click_rate"] = float(df[click_col].mean()) if df[click_col].dtype != object else None

    _mappings[req.file_id] = {"mapping": mapping, "report": report,
                              "click_col": click_col, "conv_col": conv_col}
    return report


@router.post("/import/predict")
def import_predict(req: MappingRequest, authorization: Optional[str] = Header(None)):
    """执行批量预测，返回 task_id。"""
    if req.file_id not in _uploads:
        raise HTTPException(404, "file_id 不存在，请先上传")
    up = _uploads[req.file_id]
    df = _read_upload(up["path"], up["filename"])
    mapping = req.mapping or {}

    # 模型状态
    status = inference.model_status()
    if not status["has_ctr"]:
        raise HTTPException(400, "CTR 模型未就绪，请先运行 backend/train_models.py 训练模型")

    # 解析 rows
    rows = _parse_rows(df, mapping)

    # 预测
    predictor = inference.get_predictor()
    result = predictor.predict(rows)
    pctr = result.get("pctr")
    pcvr = result.get("pcvr")
    pctcvr = result.get("pctcvr")

    # 组装结果 DataFrame
    out = pd.DataFrame({
        "index": np.arange(len(rows)),
        "pctr": pctr,
    })
    if pcvr is not None:
        out["pcvr"] = pcvr
    if pctcvr is not None:
        out["pctcvr"] = pctcvr

    # 若有标签，计算评估指标
    summary = {"n_rows": int(len(rows))}
    click_col = mapping.get("click", "click")
    conv_col = mapping.get("conversion", "conversion")
    if click_col in df.columns:
        y = df[click_col].to_numpy().astype(float)
        summary["click_rate"] = float(y.mean())
        from backend.feature_utils import remap  # 复用 AUC 需 numpy，这里用简单实现
        summary["ctr_auc"] = _auc(y, pctr) if pctr is not None else None
    if conv_col in df.columns and pcvr is not None:
        c = df[conv_col].to_numpy().astype(float)
        summary["cvr_auc"] = _auc(c, pcvr) if c.sum() > 0 else None

    # 保存结果
    task_id = uuid.uuid4().hex[:12]
    result_path = os.path.join(config.PREDICT_DIR, f"{task_id}.csv")
    out.to_csv(result_path, index=False)

    # 分布统计
    summary["pctr_mean"] = float(pctr.mean()) if pctr is not None else None
    if pcvr is not None:
        summary["pcvr_mean"] = float(pcvr.mean())
    summary["distribution"] = {
        "pctr_bins": _histogram(pctr) if pctr is not None else [],
        "pcvr_bins": _histogram(pcvr) if pcvr is not None else [],
    }

    _tasks[task_id] = {
        "file_id": req.file_id, "result_path": result_path,
        "summary": summary, "status": "done",
    }

    # 登录用户：记录预测任务到 DB
    user = _optional_user(authorization)
    if user:
        db.add_task(user["id"], task_id, req.file_id, up["filename"],
                    int(len(rows)), summary)

    summary["task_id"] = task_id
    return summary


def _auc(y_true, y_score):
    """numpy AUC。"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(y_score, kind="stable")
    y = y_true[order]
    n = len(y)
    ranks = np.arange(1, n + 1)
    pos = y > 0.5
    n_pos = pos.sum()
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _histogram(values, bins=10):
    """直方图分布（供前端画分布图）。"""
    if values is None or len(values) == 0:
        return []
    hist, edges = np.histogram(values, bins=bins)
    return [{"bin": f"{edges[i]:.4f}-{edges[i+1]:.4f}", "count": int(hist[i])}
            for i in range(len(hist))]


@router.get("/import/result/{task_id}")
def import_result(task_id: str):
    """查询预测结果汇总 + 前若干条。"""
    if task_id not in _tasks:
        raise HTTPException(404, "task_id 不存在")
    task = _tasks[task_id]
    result_path = task["result_path"]
    df = pd.read_csv(result_path)
    head = df.head(100).to_dict(orient="records")
    return {
        "task_id": task_id,
        "summary": task["summary"],
        "head": head,
        "total_rows": int(len(df)),
    }


@router.get("/import/export/{task_id}")
def import_export(task_id: str):
    """导出预测结果为 csv。"""
    if task_id not in _tasks:
        raise HTTPException(404, "task_id 不存在")
    result_path = _tasks[task_id]["result_path"]
    return FileResponse(result_path, media_type="text/csv",
                        filename=f"prediction_{task_id}.csv")

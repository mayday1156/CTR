#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据概览 + 模型对比路由
=========================
GET /api/overview   —— 数据概览统计
GET /api/models     —— 模型对比指标
"""
import os
import sys
import json
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

from fastapi import APIRouter
import pandas as pd

router = APIRouter()


def _sanitize_json(obj):
    """递归把 NaN/Infinity 转成 None（避免 JSON 序列化崩溃）。"""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    return obj


@router.get("/overview")
def overview():
    """数据概览统计（读模拟/真实数据的元信息 + 标签分布）。"""
    out = {"data_dir": config.DATA_DIR, "tables": {}}

    for name in ["common_features_train", "common_features_test",
                 "sample_skeleton_train", "sample_skeleton_test"]:
        p = os.path.join(config.DATA_DIR, f"{name}.parquet")
        if os.path.exists(p):
            df = pd.read_parquet(p)
            info = {"n_rows": int(len(df)), "columns": list(df.columns)}
            if "click" in df.columns:
                info["ctr"] = float(df["click"].mean())
            if "conversion" in df.columns:
                info["cvr"] = float(df["conversion"].mean())
            out["tables"][name] = info

    # 汇总
    sk = out["tables"].get("sample_skeleton_train", {})
    out["summary"] = {
        "n_samples": sk.get("n_rows"),
        "ctr": sk.get("ctr"),
        "cvr": sk.get("cvr"),
    }
    return out


@router.get("/models")
def models():
    """模型对比指标（LR/GBDT/DeepFM）。"""
    metrics_path = config.MODEL_METRICS
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            # parse_constant 兜底：把文件里的 NaN/Infinity 字面量转成 None
            metrics = json.load(f, parse_constant=lambda c: None)
    else:
        metrics = {"note": "尚未训练，请运行 backend/train_models.py"}

    # 二次兜底：递归清理任何残留的 NaN/Inf
    metrics = _sanitize_json(metrics)

    # 补充模型文件状态
    return {
        "metrics": metrics,
        "artifacts": {
            "ctr_ckpt": os.path.exists(config.CTR_CKPT),
            "cvr_ckpt": os.path.exists(config.CVR_CKPT),
            "vocab": os.path.exists(config.VOCAB_PARQUET),
        },
    }

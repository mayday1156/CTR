#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据源 / 模型元信息 + 热重载路由
=================================
GET  /api/meta      —— 返回当前数据源(mock/real)、模型来源、训练时间等
POST /api/reload    —— 热重载：重新读激活数据源 + 重新加载模型（无需重启后端）
"""
import os
import sys
import json
import time
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
import backend.inference as inference

from fastapi import APIRouter

router = APIRouter()


def _clean(v):
    """把 NaN/Infinity 转成 None。"""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _model_meta():
    """读取模型元信息（来源、训练时间、指标摘要）。"""
    out = {
        "has_ctr": os.path.exists(config.CTR_CKPT),
        "has_cvr": os.path.exists(config.CVR_CKPT),
        "source": None,
        "trained_at": None,
        "ctr_auc": None,
        "cvr_auc": None,
        "n_samples": None,
    }
    if os.path.exists(config.MODEL_METRICS):
        try:
            with open(config.MODEL_METRICS, "r", encoding="utf-8") as f:
                m = json.load(f, parse_constant=lambda c: None)
            out["source"] = m.get("source")
            out["trained_at"] = m.get("trained_at")
            out["ctr_auc"] = _clean(m.get("ctr", {}).get("auc"))
            out["cvr_auc"] = _clean(m.get("cvr", {}).get("auc"))
            out["n_samples"] = m.get("n_samples")
        except Exception:
            pass
    return out


@router.get("/meta")
def meta():
    """当前数据源 + 模型状态。"""
    data_source = config.DATA_SUBDIR  # "mock" / "real"
    data_ready = os.path.exists(os.path.join(config.DATA_DIR, "sample_skeleton_train.parquet"))
    return {
        "data_source": data_source,
        "data_ready": data_ready,
        "data_dir": config.DATA_DIR,
        "model": _model_meta(),
        "status": "ok",
    }


@router.post("/reload")
def reload():
    """热重载：重新读取激活数据源，并重新加载模型（不重启进程）。

    一键脚本训练完真实模型后调用本接口，网页即可即时切换到真实数据 + 真实模型。
    """
    # 1. 重新确定数据源（config 模块级变量在进程内不会自动刷新，
    #    这里手动重读状态文件 / 环境变量）
    new_source = os.environ.get("ALI_CCP_DATA") or config._read_active_source() or "mock"
    config.set_active_source(new_source)

    # 2. 重载模型单例
    inference._predictor = None  # 重置，下次 get_predictor() 会重新加载新权重
    started = time.time()
    p = inference.get_predictor()  # 触发重新加载
    elapsed = round(time.time() - started, 2)

    return {
        "status": "ok",
        "data_source": config.DATA_SUBDIR,
        "data_dir": config.DATA_DIR,
        "has_ctr": p.has_ctr,
        "has_cvr": p.has_cvr,
        "reload_seconds": elapsed,
        "model": _model_meta(),
    }

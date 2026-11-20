#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在线单条/批量 CTR 预估路由
===========================
POST /api/predict   —— 输入样本特征，返回 pCTR / pCVR / pCTCVR
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from backend import inference
from backend.feature_utils import SINGLE, MULTI

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

router = APIRouter()


class SampleRequest(BaseModel):
    feature_list: Optional[str] = None          # 原始编码字符串
    structured: Optional[Dict[str, Any]] = None  # 结构化字段 dict


@router.post("/predict")
def predict(req: SampleRequest):
    """单条样本 CTR 预估。"""
    predictor = inference.get_predictor()
    if not predictor.has_ctr:
        raise HTTPException(400, "CTR 模型未就绪，请先训练")

    rows = []
    if req.feature_list is not None:
        from backend.feature_utils import parse_feature_list
        single, m_ids, m_w = parse_feature_list(req.feature_list)
        row = {}
        for f in SINGLE:
            row[f"f{f}_id"] = single[f]
        for f in MULTI:
            row[f"f{f}_ids"] = m_ids[f]
            row[f"f{f}_w"] = m_w[f]
        rows.append(row)
    elif req.structured is not None:
        row = {}
        for f in SINGLE:
            row[f"f{f}_id"] = int(req.structured.get(f"f{f}_id", -1))
        for f in MULTI:
            row[f"f{f}_ids"] = list(req.structured.get(f"f{f}_ids", []))
            row[f"f{f}_w"] = list(req.structured.get(f"f{f}_w", []))
        rows.append(row)
    else:
        raise HTTPException(400, "请提供 feature_list 或 structured 字段")

    result = predictor.predict(rows)
    out = {"status": "ok", "n": 1}
    for k, v in result.items():
        out[k] = float(v[0])
    return out


class BatchSampleRequest(BaseModel):
    feature_lists: Optional[List[str]] = None
    structured: Optional[List[Dict[str, Any]]] = None


@router.post("/predict/batch")
def predict_batch(req: BatchSampleRequest):
    """批量 CTR 预估。"""
    predictor = inference.get_predictor()
    if not predictor.has_ctr:
        raise HTTPException(400, "CTR 模型未就绪，请先训练")

    from backend.feature_utils import parse_feature_list
    rows = []
    if req.feature_lists is not None:
        for fl in req.feature_lists:
            single, m_ids, m_w = parse_feature_list(fl)
            row = {}
            for f in SINGLE:
                row[f"f{f}_id"] = single[f]
            for f in MULTI:
                row[f"f{f}_ids"] = m_ids[f]
                row[f"f{f}_w"] = m_w[f]
            rows.append(row)
    elif req.structured is not None:
        for st in req.structured:
            row = {}
            for f in SINGLE:
                row[f"f{f}_id"] = int(st.get(f"f{f}_id", -1))
            for f in MULTI:
                row[f"f{f}_ids"] = list(st.get(f"f{f}_ids", []))
                row[f"f{f}_w"] = list(st.get(f"f{f}_w", []))
            rows.append(row)
    else:
        raise HTTPException(400, "请提供 feature_lists 或 structured")

    result = predictor.predict(rows)
    out = {"status": "ok", "n": len(rows)}
    for k, v in result.items():
        out[k] = [float(x) for x in v]
    return out

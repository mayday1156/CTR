#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
动态数据分析路由
==================
提供多维交叉分析、漏斗、序列分布、字段覆盖等聚合接口。
数据从 parquet 实时聚合（feature_list 解析为结构化字段）。

维度说明：数据无时间戳字段，可用维度为 feature_list 中的 20 个特征字段，
其中低基数类目字段（205/206/207/216/301）适合交叉与下钻分析。

接口：
  GET /api/analysis/dims          可用分析维度列表
  GET /api/analysis/ctr_by_field  单字段各取值的 CTR 分布
  GET /api/analysis/cross         两个字段的交叉 CTR 热力图
  GET /api/analysis/funnel        曝光→点击→转化漏斗
  GET /api/analysis/seq_dist      行为序列长度分布
  GET /api/analysis/coverage      字段覆盖 / unknown 率
"""
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from backend.feature_utils import parse_feature_list, SINGLE, MULTI

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

import pandas as pd
import numpy as np

router = APIRouter()

# 低基数「类目类」字段（适合交叉/下钻），高基数字段只做分布
CATEGORICAL = ["205", "206", "207", "216", "301"]
FIELD_LABELS = {
    "101": "用户ID(近唯一)", "121": "广告主ID", "122": "商品ID", "124": "店铺ID",
    "125": "用户属性1", "126": "用户属性2", "127": "用户属性3", "128": "地域",
    "129": "用户属性5", "205": "投放策略", "206": "商品类目一级", "207": "商品类目二级",
    "216": "广告位/场景", "301": "类目标签",
    "109_14": "历史类目序列", "110_14": "历史行为序列", "127_14": "历史属性序列",
    "150_14": "历史店铺序列", "210": "兴趣标签", "508": "其他标签",
}


def _load_df():
    """加载 sample_skeleton_train（事实表），若不存在回退 test。"""
    p = os.path.join(config.DATA_DIR, "sample_skeleton_train.parquet")
    if not os.path.exists(p):
        p = os.path.join(config.DATA_DIR, "sample_skeleton_test.parquet")
    if not os.path.exists(p):
        raise HTTPException(500, "数据表不存在，请先运行 generate_mock_data.py")
    return pd.read_parquet(p)


@router.get("/analysis/dims")
def dims():
    """可用分析维度列表。"""
    return {
        "categorical": [
            {"field": f, "label": FIELD_LABELS.get(f, f)} for f in CATEGORICAL
        ],
        "single": [
            {"field": f, "label": FIELD_LABELS.get(f, f), "cardinality": "high"}
            for f in SINGLE if f not in CATEGORICAL
        ],
        "multi": [
            {"field": f, "label": FIELD_LABELS.get(f, f)} for f in MULTI
        ],
        "note": "数据无时间戳字段，维度均来自 feature_list 解析出的特征字段",
    }


@router.get("/analysis/ctr_by_field")
def ctr_by_field(field: str, top: int = 20):
    """单字段各取值的 CTR 分布（默认取样本数最多的 top 个取值）。"""
    df = _load_df()
    if field not in SINGLE:
        raise HTTPException(400, f"字段 {field} 不在单值字段中")

    # 解析该字段
    vals = []
    clicks = []
    for fl, c in zip(df["feature_list"], df["click"]):
        s, _, _ = parse_feature_list(fl)
        v = s[field]
        if v != -1:
            vals.append(v)
            clicks.append(c)

    data = pd.DataFrame({"v": vals, "click": clicks})
    # 取样本数最多的 top 个取值
    top_vals = data["v"].value_counts().head(top).index
    sub = data[data["v"].isin(top_vals)]
    grp = sub.groupby("v").agg(n=("click", "size"), ctr=("click", "mean"))
    grp = grp.sort_values("n", ascending=False)

    return {
        "field": field, "label": FIELD_LABELS.get(field, field),
        "items": [
            {"value": int(v), "n": int(row["n"]), "ctr": float(row["ctr"])}
            for v, row in grp.iterrows()
        ],
    }


@router.get("/analysis/cross")
def cross(field_x: str, field_y: str, top_x: int = 10, top_y: int = 10):
    """两个字段的交叉 CTR 热力图。"""
    df = _load_df()
    if field_x not in SINGLE or field_y not in SINGLE:
        raise HTTPException(400, "字段必须是单值字段")

    rows = []
    for fl, c in zip(df["feature_list"], df["click"]):
        s, _, _ = parse_feature_list(fl)
        vx, vy = s[field_x], s[field_y]
        if vx != -1 and vy != -1:
            rows.append((vx, vy, c))

    data = pd.DataFrame(rows, columns=["x", "y", "click"])
    top_x_vals = data["x"].value_counts().head(top_x).index
    top_y_vals = data["y"].value_counts().head(top_y).index
    sub = data[data["x"].isin(top_x_vals) & data["y"].isin(top_y_vals)]

    # 透视表：行=x取值，列=y取值，值=CTR
    pivot = sub.pivot_table(index="x", columns="y", values="click", aggfunc="mean")
    x_labels = [str(int(v)) for v in pivot.index]
    y_labels = [str(int(v)) for v in pivot.columns]
    # 转成矩阵（NaN 填 null）
    matrix = pivot.values.tolist()

    return {
        "field_x": field_x, "field_y": field_y,
        "x_labels": x_labels, "y_labels": y_labels,
        "matrix": matrix,  # matrix[i][j] = CTR(x_i, y_j)，可能为 NaN
    }


@router.get("/analysis/funnel")
def funnel():
    """曝光→点击→转化漏斗。"""
    df = _load_df()
    exposure = int(len(df))
    click = int(df["click"].sum())
    conversion = int(df["conversion"].sum()) if "conversion" in df.columns else 0
    return {
        "steps": [
            {"name": "曝光", "value": exposure},
            {"name": "点击", "value": click},
            {"name": "转化", "value": conversion},
        ],
        "rates": {
            "ctr": round(click / exposure, 5) if exposure else 0,
            "cvr": round(conversion / click, 5) if click else 0,
        },
    }


@router.get("/analysis/seq_dist")
def seq_dist(field: str = "109_14"):
    """行为序列长度分布（某多值字段）。"""
    df = _load_df()
    if field not in MULTI:
        raise HTTPException(400, f"字段 {field} 不在多值字段中")

    counter = Counter()
    for fl in df["feature_list"]:
        _, m_ids, _ = parse_feature_list(fl)
        counter[len(m_ids[field])] += 1

    return {
        "field": field, "label": FIELD_LABELS.get(field, field),
        "items": [
            {"length": int(k), "count": int(v)}
            for k, v in sorted(counter.items())
        ],
    }


@router.get("/analysis/coverage")
def coverage():
    """字段覆盖 / unknown 率。"""
    df = _load_df()
    n = len(df)

    single_cov = {f: 0 for f in SINGLE}
    multi_cov = {f: 0 for f in MULTI}
    total = 0

    for fl in df["feature_list"]:
        s, m_ids, _ = parse_feature_list(fl)
        for f in SINGLE:
            if s[f] != -1:
                single_cov[f] += 1
        for f in MULTI:
            if len(m_ids[f]) > 0:
                multi_cov[f] += 1
        total += 1

    single_items = [
        {"field": f, "label": FIELD_LABELS.get(f, f), "coverage": round(single_cov[f] / n, 4),
         "unknown_rate": round(1 - single_cov[f] / n, 4)}
        for f in SINGLE
    ]
    multi_items = [
        {"field": f, "label": FIELD_LABELS.get(f, f), "coverage": round(multi_cov[f] / n, 4),
         "unknown_rate": round(1 - multi_cov[f] / n, 4)}
        for f in MULTI
    ]
    return {"single": single_items, "multi": multi_items, "n": n}

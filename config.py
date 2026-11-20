#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ali-CTR Lab 统一配置
=====================
数据目录、模型产物路径、服务端口等，全部由本文件 + 环境变量驱动。
切换「模拟数据 / 真实数据」只改 DATA_DIR（或设置环境变量 ALI_CCP_ROOT）。
"""
import os
import json

# ---- 项目根目录 ----
ROOT = os.path.dirname(os.path.abspath(__file__))

# ---- 数据目录（模拟 / 真实切换点）----
# 优先级：环境变量 ALI_CCP_DATA > 持久化状态文件 data/active_source.json > 默认 mock
# 一键脚本 train_real.* 会把真实数据切到 real 并写状态文件；网页据此展示当前数据源。
_ACTIVE_SOURCE_FILE = os.path.join(ROOT, "data", "active_source.json")


def _read_active_source():
    """读取持久化的激活数据源（若有）。"""
    try:
        if os.path.exists(_ACTIVE_SOURCE_FILE):
            with open(_ACTIVE_SOURCE_FILE, "r", encoding="utf-8") as fh:
                d = json.load(fh)
                return d.get("source", "mock")
    except Exception:
        pass
    return None


def set_active_source(source):
    """持久化当前激活数据源（"mock" / "real"）。返回是否写入成功。"""
    try:
        with open(_ACTIVE_SOURCE_FILE, "w", encoding="utf-8") as fh:
            json.dump({"source": source}, fh, ensure_ascii=False, indent=2)
        global DATA_SUBDIR, DATA_DIR
        DATA_SUBDIR = source
        DATA_DIR = os.path.join(ROOT, "data", source)
        return True
    except Exception:
        return False


DATA_SUBDIR = os.environ.get("ALI_CCP_DATA") or _read_active_source() or "mock"
DATA_DIR = os.path.join(ROOT, "data", DATA_SUBDIR)

# ---- 模型产物目录 ----
MODEL_DIR = os.path.join(ROOT, "model", "artifacts")
os.makedirs(MODEL_DIR, exist_ok=True)

# 词表（fid -> idx 映射，模拟数据由训练脚本生成；真实数据复用现有 vocab）
VOCAB_PARQUET = os.path.join(MODEL_DIR, "feature_index.parquet")
VOCAB_JSON = os.path.join(MODEL_DIR, "vocab.json")

# CTR / CVR 模型权重
CTR_CKPT = os.path.join(MODEL_DIR, "ctr_deepfm.pt")
CVR_CKPT = os.path.join(MODEL_DIR, "cvr_deepfm.pt")

# 模型对比指标
MODEL_METRICS = os.path.join(MODEL_DIR, "model_metrics.json")

# ---- 上传文件暂存目录 ----
UPLOAD_DIR = os.path.join(ROOT, "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---- 预测任务结果目录 ----
PREDICT_DIR = os.path.join(ROOT, "data", "predictions")
os.makedirs(PREDICT_DIR, exist_ok=True)

# ---- 服务 ----
# 生产环境默认绑定 0.0.0.0（公网可访问）；本地开发可设 HOST=127.0.0.1
HOST = os.environ.get("ALI_CCP_HOST", "0.0.0.0")
PORT = int(os.environ.get("ALI_CCP_PORT", "8000"))

# ---- 字段定义（与真实 Ali-CCP 契约一致）----
SINGLE = ["101", "121", "122", "124", "125", "126", "127", "128", "129",
          "205", "206", "207", "216", "301"]
MULTI = ["109_14", "110_14", "127_14", "150_14", "210", "508"]

# feature_list 分隔符
SEP_FIELD = "\x01"   # 字段间
SEP_FID = "\x02"     # field_id 与 feature_id 间
SEP_VAL = "\x03"     # feature_id 与 value 间

# 多值序列最大长度（padding 截断）
MAX_LEN = 20

# DeepFM 超参（与现有 torch/step6_deepfm.py 一致）
EMBED_DIM = 16
DNN_HIDDEN = (256, 128, 64)
DROPOUT = 0.1
FIELD_DROPOUT = 0.15

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型加载 + 推理封装
====================
加载 CTR / CVR 的 DeepFM checkpoint，提供批量预测接口。
后端导入预测模块与 /predict 接口共用此模块。
"""
import os
import sys
import json

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
# 兼容：若存在 vendor/ 目录（本地安装的 torch），优先注入；否则用 venv/系统环境
_vendor = os.path.join(_ROOT, "vendor")
if os.path.isdir(_vendor):
    sys.path.insert(0, _vendor)
import config
from backend.feature_utils import load_luts, active_fields, rows_to_tensors

import numpy as np
import torch
import torch.nn as nn


class DeepFM(nn.Module):
    """与 train_models.py 中一致的结构（加载 checkpoint 用）。"""
    def __init__(self, luts, fields_single, fields_multi, prior_logit,
                 embed_dim=config.EMBED_DIM, dnn_hidden=config.DNN_HIDDEN,
                 dropout=config.DROPOUT, field_dropout=0.0):
        super().__init__()
        self.fields_single = list(fields_single)
        self.fields_multi = list(fields_multi)
        self.embed_dim = embed_dim
        self.field_dropout = field_dropout
        self.single_emb = nn.ModuleDict({f: nn.Embedding(len(luts[f][1]) + 1, embed_dim) for f in fields_single})
        self.single_emb1 = nn.ModuleDict({f: nn.Embedding(len(luts[f][1]) + 1, 1) for f in fields_single})
        self.multi_emb = nn.ModuleDict({f: nn.Embedding(len(luts[f][1]) + 1, embed_dim) for f in fields_multi})
        self.multi_emb1 = nn.ModuleDict({f: nn.Embedding(len(luts[f][1]) + 1, 1) for f in fields_multi})
        n_fields = len(fields_single) + len(fields_multi)
        dims = [n_fields * embed_dim] + list(dnn_hidden)
        seq = []
        for i in range(len(dims) - 1):
            seq += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        self.dnn = nn.Sequential(*seq)
        self.dnn_out = nn.Linear(dims[-1], 1)
        self.prior_bias = nn.Parameter(torch.tensor(float(prior_logit)))

    def _pool(self, emb_table, emb1_table, ids, w):
        e = emb_table(ids)
        e1 = emb1_table(ids)
        w = w.unsqueeze(-1)
        s = w.sum(dim=1) + 1e-8
        pooled = (e * w).sum(dim=1) / s
        first = (e1 * w).sum(dim=1) / s
        return pooled, first

    def forward(self, single, multi_ids, multi_w):
        parts = []
        first_parts = []
        for i, f in enumerate(self.fields_single):
            idx = single[:, i]
            parts.append(self.single_emb[f](idx))
            first_parts.append(self.single_emb1[f](idx))
        for j, f in enumerate(self.fields_multi):
            p, fir = self._pool(self.multi_emb[f], self.multi_emb1[f],
                                multi_ids[:, j, :], multi_w[:, j, :])
            parts.append(p)
            first_parts.append(fir)
        emb = torch.stack(parts, dim=1)
        first = torch.stack(first_parts, dim=1)
        first = first.sum(dim=1)
        sum_sq = emb.sum(dim=1) ** 2
        sq_sum = (emb ** 2).sum(dim=1)
        fm_second = 0.5 * (sum_sq - sq_sum).sum(dim=1, keepdim=True)
        flat = emb.reshape(emb.size(0), -1)
        dnn_out = self.dnn_out(self.dnn(flat))
        logit = first + fm_second + dnn_out + self.prior_bias
        return logit.squeeze(-1)


class Predictor:
    """加载 CTR + CVR 模型，提供批量预测。"""
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.luts = load_luts()
        self.fs, self.fm = active_fields(self.luts)
        self.ctr_model = None
        self.cvr_model = None
        self._load(load_ctr=True, load_cvr=True)

    def _build_from_ckpt(self, ckpt_path):
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        fs = ck["fields_single"]
        fm = ck["fields_multi"]
        prior_logit = ck.get("prior_logit", 0.0)
        model = DeepFM(self.luts, fs, fm, prior_logit,
                       dnn_hidden=ck.get("dnn_hidden", config.DNN_HIDDEN)).to(self.device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        return model

    def _load(self, load_ctr=True, load_cvr=True):
        if load_ctr and os.path.exists(config.CTR_CKPT):
            self.ctr_model = self._build_from_ckpt(config.CTR_CKPT)
        if load_cvr and os.path.exists(config.CVR_CKPT):
            self.cvr_model = self._build_from_ckpt(config.CVR_CKPT)

    @property
    def has_ctr(self):
        return self.ctr_model is not None

    @property
    def has_cvr(self):
        return self.cvr_model is not None

    def predict(self, rows, max_len=config.MAX_LEN):
        """对若干结构化行做 CTR/CVR 预测，返回 dict。

        返回：
          pctr  : np.ndarray (n,)
          pcvr  : np.ndarray (n,) 或 None（无 CVR 模型）
          pctcvr: np.ndarray (n,) 或 None
        """
        single, multi_ids, multi_w = rows_to_tensors(rows, self.luts, max_len)
        result = {}
        if self.ctr_model is not None:
            result["pctr"] = self._run(self.ctr_model, single, multi_ids, multi_w)
        if self.cvr_model is not None:
            pcvr = self._run(self.cvr_model, single, multi_ids, multi_w)
            result["pcvr"] = pcvr
            if "pctr" in result:
                result["pctcvr"] = result["pctr"] * pcvr
        return result

    def _run(self, model, single, multi_ids, multi_w, batch_size=8192):
        n = single.shape[0]
        outs = []
        with torch.no_grad():
            for i in range(0, n, batch_size):
                s = torch.as_tensor(single[i:i + batch_size], dtype=torch.long, device=self.device)
                mi = torch.as_tensor(multi_ids[i:i + batch_size], dtype=torch.long, device=self.device)
                mw = torch.as_tensor(multi_w[i:i + batch_size], dtype=torch.float32, device=self.device)
                outs.append(torch.sigmoid(model(s, mi, mw)).cpu().numpy())
        return np.concatenate(outs)


# 全局单例（懒加载）
_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


def model_status():
    """返回模型可用状态，供接口返回。"""
    p = get_predictor()
    return {
        "has_ctr": p.has_ctr,
        "has_cvr": p.has_cvr,
        "fields_single": p.fs,
        "fields_multi": p.fm,
        "device": p.device,
    }

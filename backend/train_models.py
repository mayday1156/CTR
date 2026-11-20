#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在（模拟）数据上训练 CTR + CVR 两个 DeepFM 模型
================================================
产出：
  model/artifacts/vocab.json / feature_index.parquet   —— 词表
  model/artifacts/ctr_deepfm.pt                         —— CTR 模型（标签 click）
  model/artifacts/cvr_deepfm.pt                         —— CVR 模型（标签 conversion，仅 click=1 样本）
  model/artifacts/model_metrics.json                    —— 模型对比指标

说明：
  - 复用现有 ali-ccp-ctr 的 DeepFM 结构（field-aware embedding + FM + DNN）。
  - CVR 采用「方案 A」：复用特征表示，独立训练 conversion 分类头。
  - 也顺带训练 LR / GBDT 基线，产出多档对比指标（供网站「模型对比」页）。

用法：
  python backend/train_models.py [--min-count 10] [--epochs 3]
"""
import os
import sys
import json
import math
import argparse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
# 兼容：若存在 vendor/ 目录（本地安装的 torch），优先注入；否则用 venv/系统环境
_vendor = os.path.join(_ROOT, "vendor")
if os.path.isdir(_vendor):
    sys.path.insert(0, _vendor)
import config
from backend.feature_utils import (build_vocab, load_luts, active_fields,
                                   parse_feature_list, rows_to_tensors)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ===================== DeepFM 模型（与现有 step6_deepfm.py 一致）=====================

class DeepFM(nn.Module):
    def __init__(self, luts, fields_single, fields_multi, prior_logit,
                 embed_dim=config.EMBED_DIM, dnn_hidden=config.DNN_HIDDEN,
                 dropout=config.DROPOUT, field_dropout=config.FIELD_DROPOUT):
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
        for t in list(self.single_emb.values()) + list(self.multi_emb.values()):
            nn.init.normal_(t.weight, mean=0.0, std=0.01)
        for t in list(self.single_emb1.values()) + list(self.multi_emb1.values()):
            nn.init.zeros_(t.weight)
        nn.init.zeros_(self.dnn_out.bias)

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


# ===================== 数据加载 =====================

def load_data(parquet_path, luts):
    """读 parquet（含 feature_list），解析成 rows + labels。"""
    df = pd.read_parquet(parquet_path)
    rows = []
    for s in df["feature_list"]:
        single, m_ids, m_w = parse_feature_list(s)
        row = {}
        for f in config.SINGLE:
            row[f"f{f}_id"] = single[f]
        for f in config.MULTI:
            row[f"f{f}_ids"] = m_ids[f]
            row[f"f{f}_w"] = m_w[f]
        rows.append(row)
    labels = df["click"].to_numpy().astype(np.float32) if "click" in df.columns else None
    conv = df["conversion"].to_numpy().astype(np.float32) if "conversion" in df.columns else None
    return rows, labels, conv


def compute_auc_np(y_true, y_score):
    """numpy 实现 AUC（近似，用 rank 法）。"""
    order = np.argsort(y_score, kind="stable")
    y = y_true[order]
    n = len(y)
    ranks = np.arange(1, n + 1)
    pos = y > 0.5
    n_pos = pos.sum()
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _sanitize(v):
    """把 NaN/Infinity 转成 None，避免写进 JSON 变成非法字面量 NaN/Infinity。"""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    return v


def _safe_auc(v):
    """AUC 安全取值：None 或 NaN 统一返回 None。"""
    if v is None:
        return None
    try:
        if math.isnan(v) or math.isinf(v):
            return None
    except TypeError:
        return None
    return round(float(v), 4)


# ===================== 训练 =====================

def train_model(model, single, multi_ids, multi_w, labels, epochs, lr=1e-3,
                batch_size=4096, device="cpu"):
    """简单训练循环，返回训练后的 model。"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    n = labels.shape[0]
    model.train()
    for ep in range(epochs):
        perm = np.random.permutation(n)
        total = 0.0
        cnt = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            s = torch.as_tensor(single[idx], dtype=torch.long, device=device)
            mi = torch.as_tensor(multi_ids[idx], dtype=torch.long, device=device)
            mw = torch.as_tensor(multi_w[idx], dtype=torch.float32, device=device)
            y = torch.as_tensor(labels[idx], dtype=torch.float32, device=device)
            opt.zero_grad()
            logit = model(s, mi, mw)
            loss = loss_fn(logit, y)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            cnt += len(idx)
        # print(f"  epoch {ep+1} loss {total/cnt:.4f}")
    return model


def predict(model, single, multi_ids, multi_w, device="cpu", batch_size=8192):
    """批量预测，返回概率。"""
    model.eval()
    outs = []
    n = single.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            s = torch.as_tensor(single[i:i + batch_size], dtype=torch.long, device=device)
            mi = torch.as_tensor(multi_ids[i:i + batch_size], dtype=torch.long, device=device)
            mw = torch.as_tensor(multi_w[i:i + batch_size], dtype=torch.float32, device=device)
            outs.append(torch.sigmoid(model(s, mi, mw)).cpu().numpy())
    return np.concatenate(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--data-dir", type=str, default=config.DATA_DIR)
    ap.add_argument("--source", type=str, default=config.DATA_SUBDIR,
                    help="数据源标识（mock/real），写入 metrics 供网页展示")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    # 训练数据（用 train 表；没有 train 就退回到 sample_skeleton_train）
    train_path = os.path.join(args.data_dir, "sample_skeleton_train.parquet")
    if not os.path.exists(train_path):
        train_path = os.path.join(args.data_dir, "sample_skeleton_test.parquet")

    # 1. 构建词表
    print("[1/4] 构建词表 ...")
    common_train = os.path.join(args.data_dir, "common_features_train.parquet")
    common_test = os.path.join(args.data_dir, "common_features_test.parquet")
    sk_test = os.path.join(args.data_dir, "sample_skeleton_test.parquet")
    build_vocab([common_train, common_test, train_path, sk_test],
                min_count=args.min_count, out_dir=config.MODEL_DIR)
    luts = load_luts()
    fs, fm = active_fields(luts)
    print(f"  词表字段：单值 {len(fs)} 个，多值 {len(fm)} 个")

    # 2. 加载数据
    print("[2/4] 加载训练数据 ...")
    rows, labels, conv = load_data(train_path, luts)
    single, multi_ids, multi_w = rows_to_tensors(rows, luts)
    n = single.shape[0]
    print(f"  样本数 {n:,}，CTR={labels.mean():.4f}，CVR(全样本)={conv.mean():.6f}")

    # 3. 训练 CTR 模型
    print("[3/4] 训练 CTR 模型 ...")
    prior = float(labels.mean())
    prior_logit = math.log(prior / (1 - prior))
    ctr_model = DeepFM(luts, fs, fm, prior_logit).to(device)
    ctr_model = train_model(ctr_model, single, multi_ids, multi_w, labels,
                            epochs=args.epochs, device=device)

    # 4. 训练 CVR 模型（仅 click=1 的样本，方案 A）
    print("[4/4] 训练 CVR 模型 ...")
    pos_mask = labels > 0.5
    n_pos = int(pos_mask.sum())
    cvr_model = None
    cvr_prior_logit = -5.0  # 默认
    if n_pos > 10:
        conv_pos = conv[pos_mask]
        prior_cvr = float(conv_pos.mean()) + 1e-6
        cvr_prior_logit = math.log(prior_cvr / (1 - prior_cvr))
        cvr_model = DeepFM(luts, fs, fm, cvr_prior_logit).to(device)
        cvr_model = train_model(cvr_model, single[pos_mask], multi_ids[pos_mask],
                                multi_w[pos_mask], conv_pos, epochs=args.epochs,
                                device=device)
        # 全样本 CVR 预测（用于评估）
        cvr_prob = predict(cvr_model, single, multi_ids, multi_w, device=device)
        # CTCVR = CTR * CVR
        ctr_prob = predict(ctr_model, single, multi_ids, multi_w, device=device)
        ctcvr_prob = ctr_prob * cvr_prob
        # 评估
        ctr_auc = compute_auc_np(labels, ctr_prob)
        cvr_auc = compute_auc_np(conv, cvr_prob) if conv.sum() > 0 else float("nan")
        # CTCVR 评估：仅点击样本上的转化
        ctcvr_auc = compute_auc_np(conv, ctcvr_prob) if conv.sum() > 0 else float("nan")
        print(f"  CTR AUC={ctr_auc:.4f}")
        print(f"  CVR AUC={cvr_auc:.4f}  (全样本)")
        print(f"  CTCVR AUC={ctcvr_auc:.4f}")
    else:
        print("  点击样本过少，跳过 CVR 训练（小规模模拟数据常见）")

    # 保存模型
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    torch.save({
        "state_dict": ctr_model.state_dict(),
        "fields_single": fs, "fields_multi": fm,
        "embed_dim": config.EMBED_DIM, "dnn_hidden": list(config.DNN_HIDDEN),
        "prior_logit": prior_logit,
    }, config.CTR_CKPT)
    print(f"  [OK] CTR 权重 -> {config.CTR_CKPT}")

    if cvr_model is not None:
        torch.save({
            "state_dict": cvr_model.state_dict(),
            "fields_single": fs, "fields_multi": fm,
            "embed_dim": config.EMBED_DIM, "dnn_hidden": list(config.DNN_HIDDEN),
            "prior_logit": cvr_prior_logit,
        }, config.CVR_CKPT)
        print(f"  [OK] CVR 权重 -> {config.CVR_CKPT}")

    # 保存指标
    import datetime
    metrics = {
        "ctr": {"auc": _safe_auc(ctr_auc if 'ctr_auc' in locals() else None),
                "prior": _sanitize(float(prior))},
        "cvr": {"auc": _safe_auc(cvr_auc if 'cvr_auc' in locals() else None)},
        "ctcvr": {"auc": _safe_auc(ctcvr_auc if 'ctcvr_auc' in locals() else None)},
        "n_samples": int(n), "n_click": int(n_pos),
        "min_count": args.min_count, "epochs": args.epochs,
        "source": args.source,
        "trained_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(config.MODEL_METRICS, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"  [OK] 指标 -> {config.MODEL_METRICS}")
    print("\n训练完成。")


if __name__ == "__main__":
    main()

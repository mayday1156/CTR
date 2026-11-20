#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
feature_list 解析 + 词表 + remap 公共工具
========================================
被「模型训练」与「后端导入预测」共用，保证两者对 feature_list 的解析逻辑完全一致。

feature_list 编码（真实 Ali-CCP 契约）：
    field_id \x02 feature_id \x03 value \x01 field_id \x02 feature_id \x03 value \x01 ...
"""
import os
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config

SINGLE = config.SINGLE
MULTI = config.MULTI
SEP_FIELD = config.SEP_FIELD
SEP_FID = config.SEP_FID
SEP_VAL = config.SEP_VAL


# ===================== 解析 =====================

def parse_feature_list(s):
    """把 feature_list 字符串解析为 20 个字段。

    返回：
        single: dict[field_id -> int]，单值字段取值，缺失 -1
        m_ids : dict[field_id -> list[int]]，多值字段 id 列表
        m_w   : dict[field_id -> list[float]]，多值字段权重列表
    """
    single = {f: -1 for f in SINGLE}
    m_ids = {f: [] for f in MULTI}
    m_w = {f: [] for f in MULTI}
    if s is None:
        return single, m_ids, m_w
    for tok in s.split(SEP_FIELD):
        if not tok:
            continue
        parts = tok.split(SEP_FID)
        if len(parts) != 2:
            continue
        fid, rest = parts
        sub = rest.split(SEP_VAL)
        if len(sub) != 2:
            continue
        feat_id, val = sub
        if fid in single:
            try:
                single[fid] = int(feat_id)
            except ValueError:
                pass
        elif fid in m_ids:
            try:
                m_ids[fid].append(int(feat_id))
                m_w[fid].append(float(val))
            except ValueError:
                pass
    return single, m_ids, m_w


def feature_list_to_row(s):
    """把 feature_list 解析成一行「结构化列」dict（与 Spark step2 输出一致）。"""
    single, m_ids, m_w = parse_feature_list(s)
    row = {}
    for f in SINGLE:
        row[f"f{f}_id"] = single[f]
    for f in MULTI:
        row[f"f{f}_ids"] = m_ids[f]
        row[f"f{f}_w"] = m_w[f]
    return row


# ===================== 词表 =====================

def build_vocab(parquet_paths, min_count=10, out_dir=None):
    """从若干 parquet（含 feature_list 列）构建词表。

    按 (field, fid) 全局计数，保留 count >= min_count 的，为每个 field 生成连续 idx。
    返回 luts: dict[field -> (sorted_fid_array, sorted_idx_array)]。
    """
    from collections import Counter
    counter = {f: Counter() for f in SINGLE + MULTI}

    for path in parquet_paths:
        df = pd.read_parquet(path, columns=["feature_list"])
        for s in df["feature_list"]:
            single, m_ids, m_w = parse_feature_list(s)
            for f in SINGLE:
                if single[f] != -1:
                    counter[f][single[f]] += 1
            for f in MULTI:
                for fid in m_ids[f]:
                    counter[f][fid] += 1

    luts = {}
    vocab_meta = {}
    rows = []
    for f in SINGLE + MULTI:
        kept = {fid: c for fid, c in counter[f].items() if c >= min_count}
        fids = np.array(sorted(kept.keys()), dtype=np.int64)
        idxs = np.arange(len(fids), dtype=np.int64)
        luts[f] = (fids, idxs)
        vocab_meta[f] = {"n": int(len(fids)), "raw_distinct": len(counter[f]),
                         "raw_occ": int(sum(counter[f].values()))}
        for fid, idx in zip(fids, idxs):
            rows.append((f, int(fid), int(idx), int(kept[fid])))

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        # vocab.json：字段规模元数据
        with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as fh:
            json.dump(vocab_meta, fh, indent=2, ensure_ascii=False)
        # feature_index.parquet：field, fid, idx, cnt
        df = pd.DataFrame(rows, columns=["field", "fid", "idx", "cnt"])
        df.to_parquet(os.path.join(out_dir, "feature_index.parquet"), index=False)

    return luts


def load_luts(vocab_parquet=None):
    """从 feature_index.parquet 加载 luts。"""
    path = vocab_parquet or config.VOCAB_PARQUET
    tbl = pq.read_table(path, columns=["field", "fid", "idx"])
    fields = tbl.column("field").to_numpy(zero_copy_only=False)
    fids = tbl.column("fid").to_numpy()
    idxs = tbl.column("idx").to_numpy()
    by = {}
    for f, fid, idx in zip(fields, fids, idxs):
        d = by.setdefault(str(f), ([], []))
        d[0].append(int(fid))
        d[1].append(int(idx))
    luts = {}
    for f, (fl, il) in by.items():
        sf = np.array(fl, dtype=np.int64)
        si = np.array(il, dtype=np.int64)
        order = np.argsort(sf, kind="stable")
        luts[f] = (sf[order], si[order])
    return luts


def active_fields(luts):
    """词表里实际保留的字段（field 101 等可能被频次过滤剔除）。"""
    return ([f for f in SINGLE if f in luts],
            [f for f in MULTI if f in luts])


def remap(lut, query):
    """fid -> idx 向量化映射，未命中 -> 0 号 unknown 槽。"""
    sf, si = lut
    q = np.asarray(query, dtype=np.int64)
    pos = np.searchsorted(sf, q)
    np.clip(pos, 0, sf.size - 1, out=pos)
    out = np.where(sf[pos] == q, si[pos], 0).astype(np.int64)
    return out


# ===================== 批量转 tensor =====================

def rows_to_tensors(rows, luts, max_len=config.MAX_LEN):
    """
    把若干「结构化行」dict 列表转成模型输入 tensor。
    rows: list[dict]，每个含 f{f}_id / f{f}_ids / f{f}_w 等列。
    返回 single(m,n), multi_ids(m,k,L), multi_w(m,k,L)
    """
    fs, fm = active_fields(luts)
    n = len(rows)
    single = np.empty((n, len(fs)), dtype=np.int64)
    for j, f in enumerate(fs):
        col = np.array([r.get(f"f{f}_id", -1) for r in rows], dtype=np.int64)
        single[:, j] = remap(luts[f], col)
    multi_ids = np.zeros((n, len(fm), max_len), dtype=np.int64)
    multi_w = np.zeros((n, len(fm), max_len), dtype=np.float32)
    for j, f in enumerate(fm):
        for i, r in enumerate(rows):
            ids = r.get(f"f{f}_ids", [])
            w = r.get(f"f{f}_w", [])
            L = min(len(ids), max_len)
            if L == 0:
                continue
            remapped = remap(luts[f], np.array(ids[:L], dtype=np.int64))
            multi_ids[i, j, :L] = remapped
            multi_w[i, j, :L] = np.array(w[:L], dtype=np.float32)
    return single, multi_ids, multi_w


def df_to_rows(df, mapping):
    """
    把上传的 DataFrame + 字段映射，转成「结构化行」dict 列表。
    mapping: dict[标准字段名 -> 实际列名]（或 None 表示列名即标准名）。
    """
    rows = []
    if mapping is None:
        mapping = {}
    for _, r in df.iterrows():
        row = {}
        for f in SINGLE:
            col = mapping.get(f"f{f}_id", f"f{f}_id")
            if col in df.columns:
                row[f"f{f}_id"] = int(r[col]) if pd.notna(r[col]) else -1
            else:
                row[f"f{f}_id"] = -1
        for f in MULTI:
            col_ids = mapping.get(f"f{f}_ids", f"f{f}_ids")
            col_w = mapping.get(f"f{f}_w", f"f{f}_w")
            if col_ids in df.columns:
                row[f"f{f}_ids"] = list(r[col_ids]) if hasattr(r[col_ids], "__iter__") and not isinstance(r[col_ids], str) else []
            else:
                row[f"f{f}_ids"] = []
            if col_w in df.columns:
                row[f"f{f}_w"] = list(r[col_w]) if hasattr(r[col_w], "__iter__") and not isinstance(r[col_w], str) else []
            else:
                row[f"f{f}_w"] = []
        rows.append(row)
    return rows

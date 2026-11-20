#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SQLite 数据库 + 用户认证模块
=============================
- 用户表 users：注册/登录，密码哈希存储
- 数据集表 datasets：用户上传的数据集元信息（按 user 隔离）
- 预测任务表 tasks：用户的预测任务记录（按 user 隔离）
- token 会话：内存 token -> user_id 映射（单机演示；生产可换 JWT/Redis）

注意：不引入额外依赖（bcrypt/passlib），密码用 hashlib.pbkdf2_hmac 加盐哈希。
"""
import os
import sys
import json
import time
import uuid
import sqlite3
import hashlib
import secrets

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

DB_PATH = os.path.join(config.ROOT, "data", "app.db")

TOKEN_TTL = 60 * 60 * 24 * 7  # 7 天


# ===================== 数据库初始化 =====================

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """建表（幂等）。"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT,
            created_at TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            file_id TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            ext TEXT,
            n_rows INTEGER,
            columns TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            task_id TEXT UNIQUE NOT NULL,
            file_id TEXT,
            filename TEXT,
            n_rows INTEGER,
            summary TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    # 持久化会话表：token 存数据库，服务重启后登录态不丢失（生产环境必需）
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            expires_at REAL NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()


init_db()


# ===================== 密码哈希 =====================

def _hash_password(password: str, salt: str = None) -> str:
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                             bytes.fromhex(salt), 100_000)
    return f"{salt}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt, expected = stored.split("$", 1)
        return secrets.compare_digest(_hash_password(password, salt).split("$", 1)[1], expected)
    except Exception:
        return False


# ===================== 用户 =====================

def create_user(username: str, password: str, email: str = "") -> dict:
    conn = get_conn()
    cur = conn.cursor()
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        cur.execute("INSERT INTO users (username, password_hash, email, created_at) VALUES (?,?,?,?)",
                    (username, _hash_password(password), email or None, now))
        conn.commit()
        user_id = cur.lastrowid
    except sqlite3.IntegrityError:
        conn.close()
        raise ValueError("用户名已存在")
    conn.close()
    return {"id": user_id, "username": username, "email": email, "created_at": now}


def authenticate(username: str, password: str) -> dict:
    conn = get_conn()
    cur = conn.cursor()
    row = cur.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    if not row or not _verify_password(password, row["password_hash"]):
        raise ValueError("用户名或密码错误")
    return {"id": row["id"], "username": row["username"], "email": row["email"],
            "created_at": row["created_at"]}


def get_user_by_id(user_id: int) -> dict:
    conn = get_conn()
    cur = conn.cursor()
    row = cur.execute("SELECT id, username, email, created_at FROM users WHERE id = ?",
                      (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


# ===================== 会话 token（持久化到 SQLite，重启不失效） =====================

def create_session(user_id: int) -> str:
    token = secrets.token_hex(32)
    expires_at = time.time() + TOKEN_TTL
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO sessions (token, user_id, expires_at) VALUES (?,?,?)",
                (token, user_id, expires_at))
    conn.commit()
    conn.close()
    return token


def get_session_user(token: str) -> dict:
    if not token:
        return None
    conn = get_conn()
    cur = conn.cursor()
    row = cur.execute("SELECT user_id, expires_at FROM sessions WHERE token = ?", (token,)).fetchone()
    if not row:
        conn.close()
        return None
    if time.time() > row["expires_at"]:
        cur.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
        conn.close()
        return None
    user_id = row["user_id"]
    conn.close()
    return get_user_by_id(user_id)


def destroy_session(token: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()
    conn.close()


# ===================== 数据集 / 任务记录 =====================

def add_dataset(user_id: int, file_id: str, filename: str, ext: str,
                n_rows: int, columns: list) -> None:
    conn = get_conn()
    cur = conn.cursor()
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("""INSERT OR REPLACE INTO datasets
                   (user_id, file_id, filename, ext, n_rows, columns, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (user_id, file_id, filename, ext, n_rows, json.dumps(columns), now))
    conn.commit()
    conn.close()


def list_datasets(user_id: int) -> list:
    conn = get_conn()
    cur = conn.cursor()
    rows = cur.execute("""SELECT id, file_id, filename, ext, n_rows, columns, created_at
                          FROM datasets WHERE user_id = ? ORDER BY id DESC""",
                       (user_id,)).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["columns"] = json.loads(d["columns"] or "[]")
        except Exception:
            d["columns"] = []
        out.append(d)
    return out


def delete_dataset(user_id: int, file_id: str) -> bool:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM datasets WHERE user_id = ? AND file_id = ?", (user_id, file_id))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    # 同步删除磁盘上的上传文件
    for ext in (".parquet", ".csv", ".xlsx", ".xls", ".json", ".txt"):
        p = os.path.join(config.UPLOAD_DIR, f"{file_id}{ext}")
        if os.path.exists(p):
            os.remove(p)
    return deleted


def add_task(user_id: int, task_id: str, file_id: str, filename: str,
             n_rows: int, summary: dict) -> None:
    conn = get_conn()
    cur = conn.cursor()
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("""INSERT OR REPLACE INTO tasks
                   (user_id, task_id, file_id, filename, n_rows, summary, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (user_id, task_id, file_id, filename, n_rows, json.dumps(summary), now))
    conn.commit()
    conn.close()


def list_tasks(user_id: int) -> list:
    conn = get_conn()
    cur = conn.cursor()
    rows = cur.execute("""SELECT id, task_id, file_id, filename, n_rows, summary, created_at
                          FROM tasks WHERE user_id = ? ORDER BY id DESC""",
                       (user_id,)).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["summary"] = json.loads(d["summary"] or "{}")
        except Exception:
            d["summary"] = {}
        out.append(d)
    return out


def delete_task(user_id: int, task_id: str) -> bool:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM tasks WHERE user_id = ? AND task_id = ?", (user_id, task_id))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    p = os.path.join(config.PREDICT_DIR, f"{task_id}.csv")
    if os.path.exists(p):
        os.remove(p)
    return deleted

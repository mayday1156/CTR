#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用户认证路由
=============
POST /api/auth/register   注册
POST /api/auth/login      登录 → 返回 token
POST /api/auth/logout     登出
GET  /api/auth/me         当前用户（需 Authorization: Bearer <token>）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from backend import db

from fastapi import APIRouter, Header, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class RegisterRequest(BaseModel):
    username: str
    password: str
    email: Optional[str] = ""


class LoginRequest(BaseModel):
    username: str
    password: str


def _current_user(authorization: Optional[str] = Header(None)) -> dict:
    """从 Authorization: Bearer <token> 解析当前用户。"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "未登录")
    token = authorization[len("Bearer "):].strip()
    user = db.get_session_user(token)
    if not user:
        raise HTTPException(401, "登录已过期，请重新登录")
    return user


@router.post("/auth/register")
def register(req: RegisterRequest):
    username = req.username.strip()
    password = req.password
    if not username or len(username) < 2:
        raise HTTPException(400, "用户名至少 2 个字符")
    if not password or len(password) < 4:
        raise HTTPException(400, "密码至少 4 位")
    try:
        user = db.create_user(username, password, req.email or "")
    except ValueError as e:
        raise HTTPException(409, str(e))
    token = db.create_session(user["id"])
    return {"token": token, "user": user}


@router.post("/auth/login")
def login(req: LoginRequest):
    try:
        user = db.authenticate(req.username.strip(), req.password)
    except ValueError as e:
        raise HTTPException(401, str(e))
    token = db.create_session(user["id"])
    return {"token": token, "user": user}


@router.post("/auth/logout")
def logout(authorization: Optional[str] = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        db.destroy_session(authorization[len("Bearer "):].strip())
    return {"status": "ok"}


@router.get("/auth/me")
def me(user: dict = Depends(_current_user)):
    return {"user": user}

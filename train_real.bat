@echo off
REM ============================================================
REM Ali-CTR Lab · 真实数据一键训练 + 同步到网页（Windows CMD）
REM ============================================================
REM 用法：双击运行，或在 cmd 里执行  train_real.bat [--epochs 5]
REM
REM 前置条件：
REM   1. 已把真实 Ali-CCP 的 4 张 parquet 放到 data\real\
REM   2. 后端已启动（双击 start.bat），用于训练后热重载同步
REM ============================================================
chcp 65001 >nul
cd /d "%~dp0"

set PY=venv\Scripts\python.exe
if not exist "%PY%" set PY=python

echo 使用解释器：%PY%
"%PY%" backend\train_real.py %*
if errorlevel 1 (
    echo.
    echo [失败] 训练或同步出错，请检查上面的错误信息。
    pause
    exit /b 1
)

echo.
echo 完成！请刷新浏览器 http://127.0.0.1:8000/ui/home.html 查看。
pause

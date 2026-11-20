@echo off
REM Ali-CTR Lab 一键启动脚本（Windows CMD）
REM 用法：双击运行，或在 cmd 里执行  start.bat [端口]
chcp 65001 >nul
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [错误] 未找到 venv，请先运行 setup.bat
    pause
    exit /b 1
)

set PORT=8000
if not "%~1"=="" set PORT=%~1

echo ======================================
echo   Ali-CTR Lab 启动（端口 %PORT%）
echo   本机访问: http://127.0.0.1:%PORT%/ui/
echo   局域网访问: http://本机IP:%PORT%/ui/
echo   API 文档: http://127.0.0.1:%PORT%/docs
echo   若端口被占用，可换端口启动：start.bat 8001
echo ======================================

venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port %PORT%
pause

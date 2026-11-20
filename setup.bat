@echo off
REM Ali-CTR Lab 环境初始化脚本（Windows CMD）
REM 用法：双击运行，或在 cmd 里执行  setup.bat
chcp 65001 >nul
cd /d "%~dp0"

set PY=python
where py >nul 2>&1 && set PY=py -3.13
where py >nul 2>&1 && (py -3.13 --version >nul 2>&1 || set PY=py)

echo 使用解释器：%PY%

if not exist "venv\Scripts\python.exe" (
    echo [1/3] 创建虚拟环境 ...
    %PY% -m venv venv
)

echo [2/3] 安装基础依赖 ...
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install numpy pandas pyarrow fastapi uvicorn python-multipart openpyxl

echo [3/3] 安装 CPU 版 torch（较大，请耐心等待）...
venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu

echo.
echo ======================================
echo   环境初始化完成！
echo   启动后端：start.bat
echo   页面：http://127.0.0.1:8000/ui/
echo ======================================
pause

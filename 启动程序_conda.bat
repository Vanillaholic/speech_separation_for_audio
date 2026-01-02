@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo 说话人分离程序启动器 (Conda环境版)
echo ========================================
echo.

REM 设置conda环境路径
set CONDA_ENV=%~dp0venv_portable
set CONDA_PYTHON=%CONDA_ENV%\python.exe

REM 检查conda环境是否存在
if not exist "%CONDA_PYTHON%" (
    echo [错误] 未找到Conda环境
    echo [信息] 预期路径: %CONDA_PYTHON%
    echo.
    echo [提示] 请先运行"复制conda环境.bat"复制环境
    echo 或确保venv_portable目录存在且包含python.exe
    echo.
    pause
    exit /b 1
)

echo [信息] 使用Conda环境启动
echo [信息] Python路径: %CONDA_PYTHON%
echo.

REM 设置CONDA_PREFIX环境变量（某些conda包可能需要）
set CONDA_PREFIX=%CONDA_ENV%

REM 设置SSL证书路径（如果需要）
if exist "%CONDA_ENV%\Library\ssl\cacert.pem" (
    set SSL_CERT_FILE=%CONDA_ENV%\Library\ssl\cacert.pem
)

REM 验证Python是否可用
"%CONDA_PYTHON%" --version >nul 2>&1
if !errorlevel! neq 0 (
    echo [错误] Python无法运行
    echo [提示] 请检查环境是否完整复制
    pause
    exit /b 1
)

REM 显示Python版本
echo [信息] Python版本:
"%CONDA_PYTHON%" --version
echo.

REM 检查依赖
echo [信息] 检查依赖...
"%CONDA_PYTHON%" -c "import torch; import gradio; import pyannote.audio; print('[成功] 所有依赖已就绪')" 2>nul
if !errorlevel! neq 0 (
    echo [警告] 某些依赖可能缺失，但继续启动...
    echo.
)

echo [信息] 正在启动Gradio应用...
echo.

REM 启动应用
"%CONDA_PYTHON%" gradio_app.py

if !errorlevel! neq 0 (
    echo.
    echo [错误] 应用启动失败
    echo [提示] 请检查错误信息
    pause
    exit /b 1
)

pause


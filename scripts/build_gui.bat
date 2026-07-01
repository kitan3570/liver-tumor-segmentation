@echo off
REM =============================================================================
REM scripts/build_gui.bat
REM -----------------------------------------------------------------------------
REM 用途：在 Windows 上用 PyInstaller 把 PySide6 GUI 打包为单个可执行文件。
REM
REM 应用名称：LiverTumorSegToolkit
REM 入口    ：gui/app.py
REM 输出    ：dist/LiverTumorSegToolkit/LiverTumorSegToolkit.exe
REM           （onedir 模式，启动更快、调试更友好；如需单文件改 --onefile）
REM
REM 重要说明：
REM   1) 不建议把 PyTorch / MONAI / CUDA 全部打包进 exe，体积会达数 GB 且易
REM      因 CUDA 运行时缺失而启动失败。本脚本默认排除这些大包，仅打包 GUI
REM      外壳；运行时仍依赖 conda 环境中的 torch/monai（见 README §6B.5）。
REM   2) 更推荐在 conda 环境中直接 `python gui/app.py` 运行 GUI。
REM   3) 打包版本主要用于界面启动，不负责内置大型模型和数据；模型权重、
REM      数据集、配置文件仍需在运行时通过 GUI 选择。
REM
REM 用法（在项目根目录执行）：
REM   scripts\build_gui.bat
REM =============================================================================

setlocal enabledelayedexpansion

REM ---------------- 用户配置 ----------------
set APP_NAME=LiverTumorSegToolkit
set ENTRY=gui\app.py
set ICON=
REM 如有图标可设：set ICON=--icon=assets\app.ico

REM ---------------- 切到项目根 ----------------
cd /d "%~dp0\.."

echo ============================================================
echo  PyInstaller 打包：%APP_NAME%
echo  入口: %ENTRY%
echo ============================================================

REM ---------------- 检查 PyInstaller ----------------
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo [INFO] 未检测到 PyInstaller，正在安装...
    python -m pip install pyinstaller
    if errorlevel 1 (
        echo [ERROR] PyInstaller 安装失败，请手动执行：python -m pip install pyinstaller
        exit /b 1
    )
)

REM ---------------- 清理旧产物 ----------------
if exist build rmdir /s /q build
if exist dist\%APP_NAME% rmdir /s /q dist\%APP_NAME%
if exist %APP_NAME%.spec del /q %APP_NAME%.spec

REM ---------------- 执行打包 ----------------
REM --name             指定应用名称（生成 LiverTumorSegToolkit.exe）
REM --windowed         GUI 应用，不弹黑色控制台窗口
REM --noconfirm        覆盖旧 dist 不询问
REM --collect-data PySide6  收集 Qt 插件/翻译等数据文件
REM --exclude          排除大包：torch/monai/nibabel/cv2 等（运行时走 conda 环境）
REM --add-data         把 configs/ 与 src/ 一并打包，便于 GUI 调用 src 脚本
REM     Windows 分隔符用 ';'  （Linux/macOS 用 ':'）
pyinstaller ^
    --name %APP_NAME% ^
    --windowed ^
    --noconfirm ^
    --collect-data PySide6 ^
    --exclude-module torch ^
    --exclude-module torchvision ^
    --exclude-module monai ^
    --exclude-module nibabel ^
    --exclude-module cv2 ^
    --exclude-module tensorboard ^
    --exclude-module matplotlib ^
    --add-data "configs;configs" ^
    --add-data "src;src" ^
    %ICON% ^
    %ENTRY%

if errorlevel 1 (
    echo.
    echo [ERROR] 打包失败，请查看上方日志。
    exit /b 1
)

echo.
echo ============================================================
echo  打包成功！
echo  产物目录: dist\%APP_NAME%\
echo  可执行文件: dist\%APP_NAME%\%APP_NAME%.exe
echo ============================================================
echo.
echo 注意：
echo   1. 本打包版本仅含 GUI 外壳，不含 PyTorch/MONAI/CUDA。
echo   2. 运行 exe 时仍需在 conda 环境中（torch/monai 已安装），
echo      或在 exe 同级目录提供 src/ 与 configs/。
echo   3. 更推荐直接在 conda 环境运行：python gui\app.py
echo.

endlocal

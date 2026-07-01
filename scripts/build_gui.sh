#!/usr/bin/env bash
# =============================================================================
# scripts/build_gui.sh
# -----------------------------------------------------------------------------
# 用途：在 Linux / macOS 上用 PyInstaller 把 PySide6 GUI 打包为可执行文件。
#
# 应用名称：LiverTumorSegToolkit
# 入口    ：gui/app.py
# 输出    ：dist/LiverTumorSegToolkit/LiverTumorSegToolkit
#           （onedir 模式，启动更快、调试更友好；如需单文件改 --onefile）
#
# 重要说明：
#   1) 不建议把 PyTorch / MONAI / CUDA 全部打包进可执行文件，体积会达数 GB
#      且易因 CUDA 运行时缺失而启动失败。本脚本默认排除这些大包，仅打包 GUI
#      外壳；运行时仍依赖 conda 环境中的 torch/monai（见 README §6B.5）。
#   2) 更推荐在 conda 环境中直接 `python gui/app.py` 运行 GUI。
#   3) 打包版本主要用于界面启动，不负责内置大型模型和数据；模型权重、
#      数据集、配置文件仍需在运行时通过 GUI 选择。
#
# 用法（在项目根目录执行）：
#   bash scripts/build_gui.sh
# =============================================================================

set -euo pipefail

# ---------------- 用户配置 ----------------
APP_NAME="LiverTumorSegToolkit"
ENTRY="gui/app.py"
ICON=""
# 如有图标可设：ICON="--icon=assets/app.ico"

# ---------------- 切到项目根 ----------------
cd "$(dirname "$0")/.."

echo "============================================================"
echo " PyInstaller 打包：${APP_NAME}"
echo " 入口: ${ENTRY}"
echo "============================================================"

# ---------------- 检查 PyInstaller ----------------
if ! python -c "import PyInstaller" 2>/dev/null; then
    echo "[INFO] 未检测到 PyInstaller，正在安装..."
    python -m pip install pyinstaller
fi

# ---------------- 清理旧产物 ----------------
rm -rf build
rm -rf "dist/${APP_NAME}"
rm -f "${APP_NAME}.spec"

# ---------------- 执行打包 ----------------
# --name             指定应用名称（生成 LiverTumorSegToolkit 可执行文件）
# --windowed         GUI 应用，不弹终端窗口
# --noconfirm        覆盖旧 dist 不询问
# --collect-data PySide6  收集 Qt 插件/翻译等数据文件
# --exclude          排除大包：torch/monai/nibabel/cv2 等（运行时走 conda 环境）
# --add-data         把 configs/ 与 src/ 一并打包，便于 GUI 调用 src 脚本
#     Linux/macOS 分隔符用 ':'  （Windows 用 ';'）
pyinstaller \
    --name "${APP_NAME}" \
    --windowed \
    --noconfirm \
    --collect-data PySide6 \
    --exclude-module torch \
    --exclude-module torchvision \
    --exclude-module monai \
    --exclude-module nibabel \
    --exclude-module cv2 \
    --exclude-module tensorboard \
    --exclude-module matplotlib \
    --add-data "configs:configs" \
    --add-data "src:src" \
    ${ICON} \
    "${ENTRY}"

echo ""
echo "============================================================"
echo " 打包成功！"
echo " 产物目录: dist/${APP_NAME}/"
echo " 可执行文件: dist/${APP_NAME}/${APP_NAME}"
echo "============================================================"
echo ""
echo "注意："
echo "  1. 本打包版本仅含 GUI 外壳，不含 PyTorch/MONAI/CUDA。"
echo "  2. 运行可执行文件时仍需在 conda 环境中（torch/monai 已安装），"
echo "    或在可执行文件同级目录提供 src/ 与 configs/。"
echo "  3. 更推荐直接在 conda 环境运行：python gui/app.py"
echo ""

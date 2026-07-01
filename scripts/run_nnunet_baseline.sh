#!/usr/bin/env bash
# =============================================================================
# nnU-Net v2 baseline 一键流程脚本：Medical Segmentation Decathlon Task03_Liver
# -----------------------------------------------------------------------------
# 用途：作为本项目自写 MONAI 3D U-Net 的"对照组"。不覆盖、不修改任何
#       src/*.py / scripts/*.py（除本文件）/ configs/* 等 MONAI 自写代码。
#       nnU-Net 仅作为外部工具跑一遍 baseline，结果放在独立的 nnunet/ 目录。
#
# 前置条件（务必按顺序安装）：
#   1) 先按 https://pytorch.org/get-started/locally/ 安装匹配本机 CUDA 的 PyTorch
#      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
#   2) 再安装 nnU-Net v2
#      pip install nnunetv2
#   建议在独立 conda 环境：conda create -n nnunet python=3.10 -y && conda activate nnunet
#
# 用法：
#   bash scripts/run_nnunet_baseline.sh
#
# 可调参数：见下方 USER CONFIG 区块；默认 DATASET_ID=3（对应 Dataset003_Liver）。
# 若已完成某步骤，可把对应命令注释掉，只跑剩余步骤。
# =============================================================================

set -euo pipefail

# ---------------- USER CONFIG ----------------
# MSD Task03_Liver 原始数据路径（其下应含 imagesTr/labelsTr/dataset.json）
MSD_TASK_DIR="data/raw/Task03_Liver"

# nnU-Net 用 DATASET_ID；3 对应 MSD Task03，nnU-Net 内部记为 Dataset003_Liver
DATASET_ID=3

# 训练配置：3d_fullres / 2d / 3d_lowres（显存不足可降级为 2d 或 3d_lowres）
CONFIG="3d_fullres"

# fold 编号（5 折交叉验证，0~4）
FOLD=0

# 推理输入文件夹（待预测 CT .nii.gz 所在目录），留空则跳过推理
INPUT_FOLDER="data/raw/Task03_Liver/imagesTs"

# 推理输出文件夹
OUTPUT_FOLDER="nnunet/predictions"
# ---------------------------------------------

# -----------------------------------------------------------------------------
# 1) 设置 nnU-Net v2 必需的三个环境变量
#    注意：换终端需重新设置，或写进 ~/.bashrc / 系统环境变量。
# -----------------------------------------------------------------------------
export nnUNet_raw="./nnunet/nnUNet_raw"
export nnUNet_preprocessed="./nnunet/nnUNet_preprocessed"
export nnUNet_results="./nnunet/nnUNet_results"
echo "[1/4] 环境变量：nnUNet_raw=${nnUNet_raw}"
echo "      nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "      nnUNet_results=${nnUNet_results}"

if [[ ! -d "${MSD_TASK_DIR}" ]]; then
  echo "[ERROR] 未找到 MSD Task03_Liver 原始数据目录：${MSD_TASK_DIR}"
  echo "        请先把 Task03_Liver 解压到该目录，或修改脚本顶部的 MSD_TASK_DIR。"
  exit 1
fi

# -----------------------------------------------------------------------------
# 2) 将 MSD Task03_Liver 转换为 nnU-Net v2 原始数据格式
#    产物：$nnUNet_raw/Dataset003_Liver/{imagesTr,labelsTr,dataset.json}
# -----------------------------------------------------------------------------
echo "[2/4] 转换 MSD 数据 -> nnU-Net v2 格式 (Dataset003_Liver) ..."
nnUNetv2_convert_MSD_dataset -i "${MSD_TASK_DIR}" -d "${DATASET_ID}"

# -----------------------------------------------------------------------------
# 3) 预处理与规划，并校验数据完整性（图像/标签一一对应、形状一致、标签合法）
#    首次运行耗时较长；--verify_dataset_integrity 确保标签取值如 {0,1,2}
# -----------------------------------------------------------------------------
echo "[3/4] 预处理与规划 (plan_and_preprocess) + 数据完整性校验 ..."
nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" --verify_dataset_integrity

# -----------------------------------------------------------------------------
# 4) 训练（按 CONFIG 训练；显存不足请把 CONFIG 改为 2d 或 3d_lowres）
#    用法：nnUNetv2_train DATASET_ID CONFIG FOLD
# -----------------------------------------------------------------------------
echo "[4/4] 训练：nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD}"
nnUNetv2_train "${DATASET_ID}" "${CONFIG}" "${FOLD}"

# -----------------------------------------------------------------------------
# (可选) 推理：对一批新图像 (.nii.gz) 预测，输出同结构与命名的 .nii.gz mask
#    若不需要推理，可把 INPUT_FOLDER 置空或整段注释。
# -----------------------------------------------------------------------------
if [[ -n "${INPUT_FOLDER}" && -d "${INPUT_FOLDER}" ]]; then
  echo "[extra] 推理：nnUNetv2_predict -i ${INPUT_FOLDER} -o ${OUTPUT_FOLDER} -d ${DATASET_ID} -c ${CONFIG} -f ${FOLD}"
  mkdir -p "${OUTPUT_FOLDER}"
  nnUNetv2_predict \
    -i "${INPUT_FOLDER}" \
    -o "${OUTPUT_FOLDER}" \
    -d "${DATASET_ID}" \
    -c "${CONFIG}" \
    -f "${FOLD}"
else
  echo "[extra] 跳过推理（INPUT_FOLDER 为空或不存在）。"
fi

echo "================================================================"
echo "nnU-Net v2 baseline 流程结束。"
echo "结果目录：${nnUNet_results}"
echo "预测目录：${OUTPUT_FOLDER}"
echo "可用本项目 src/evaluate.py 评价："
echo "  python src/evaluate.py --pred_dir ${OUTPUT_FOLDER} \\"
echo "    --label_dir data/raw/Task03_Liver/labelsTr \\"
echo "    --output_csv outputs/evaluation_nnunet.csv"
echo "================================================================"
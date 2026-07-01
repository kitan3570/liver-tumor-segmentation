"""scripts/export_stl.py — 把预测的 NIfTI 分割导出为 STL 三维网格

作用
----
读取模型推理产出的 NIfTI 分割预测（多类标签：0=背景、1=肝脏、2=肿瘤），
从中取出指定类别（--label_id）的二值掩膜，用 marching cubes 提取等值面并
生成三角网格，最后导出为 STL 文件。STL 是 3D 打印与教学展示的通用网格格式，
可用 MeshLab / Blender / 在线 3D 查看器打开，直观展示肝脏或肿瘤的立体形态，
帮助大一同学建立从「体素分割」到「三维结构」的直观认识。

运行方式
--------
显式指定输出路径：

    python scripts/export_stl.py --input outputs/predictions/case0.nii.gz --label_id 1 --output outputs/stl/case0_liver.stl

不指定 --output 时会自动生成到 outputs/stl/<输入文件名>_<类别名>.stl，例如：

    python scripts/export_stl.py --input outputs/predictions/case0.nii.gz --label_id 2
    # → outputs/stl/case0_tumor.stl

输入
----
- 预测 NIfTI 文件（.nii / .nii.gz），体素值为类别 id（0/1/2），
  通常由 scripts/infer.py 滑动窗口推理后保存得到。

输出
----
- STL 三角网格文件（.stl），默认写入 outputs/stl/。

常见错误
--------
1. 输入文件不存在：检查 --input 路径是否正确、推理（infer.py）是否已运行。
2. label_id 在预测中无前景体素（例如该病例本就没有肿瘤）：marching cubes
   提取不到任何表面，将跳过导出并打印警告；请确认 --label_id 与预测的类别
   编码一致（1=肝脏, 2=肿瘤）。
3. 缺少 scikit-image / trimesh 依赖：STL 导出依赖这两个包，请执行
   `pip install scikit-image trimesh`（或按 requirements.txt 安装完整依赖）。
"""

import argparse
import os
import sys

# 把项目根目录加入 sys.path，使得直接 `python scripts/export_stl.py ...` 运行时可
# import src.*（否则 Python 只把 scripts/ 当搜索路径，找不到 src 包）。os/sys 已导入。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_nifti, export_label_stl, ensure_dir


# 类别 id → 人类可读名称的映射，用于自动命名与日志打印。
# 1=肝脏、2=肿瘤；其它 id 退化为 class{id}，保证脚本对未知类别也不会崩溃。
CLASS_NAMES = {1: "liver", 2: "tumor"}


def class_name(label_id: int) -> str:
    """根据 label_id 返回类别名（1→liver, 2→tumor，其它→class{id}）。"""
    return CLASS_NAMES.get(label_id, f"class{label_id}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    --help 由 argparse 自动提供，无需手动定义。
    """
    parser = argparse.ArgumentParser(
        description=(
            "把预测的 NIfTI 分割中的指定类别（liver=1, tumor=2）转为 STL 三维网格。"
            "基于 scikit-image 的 marching cubes 提取等值面，trimesh 平滑与导出。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="预测 NIfTI 文件路径（.nii / .nii.gz），体素值为类别 id。",
    )
    parser.add_argument(
        "--label_id",
        type=int,
        default=1,
        help="要导出的类别 id：1=肝脏 liver，2=肿瘤 tumor。",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出 STL 路径；若为 None 则自动生成到 "
             "outputs/stl/<输入文件名>_<类别名>.stl。",
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        default=False,
        help="是否对网格做 Taubin 平滑（去锯齿，会略微改变形状）。",
    )
    parser.add_argument(
        "--spacing",
        nargs=3,
        type=float,
        default=[1.0, 1.0, 1.0],
        metavar=("SX", "SY", "SZ"),
        help="体素间距 (mm)，影响输出网格的物理尺寸，应与 CT 的 spacing 一致；"
             "spacing 错误会导致导出的肝脏变形（被拉长或压扁）。",
    )
    return parser.parse_args()


def main() -> None:
    """脚本主流程：解析参数 → 校验输入 → 加载 NIfTI → 自动命名 → 导出 STL。"""
    args = parse_args()

    input_path = args.input
    label_id = args.label_id
    smooth = args.smooth
    spacing = tuple(args.spacing)  # marching_cubes 需要元组形式的体素间距
    cname = class_name(label_id)

    # —— 1) 校验输入文件存在，缺失时给出清晰提示并退出 ——
    if not os.path.isfile(input_path):
        print(f"[错误] 输入文件不存在：{input_path}\n"
              "请检查 --input 路径，并确认推理（infer.py）已生成预测文件。")
        sys.exit(1)

    # —— 2) 加载 NIfTI 预测，得到多类标签数组与仿射矩阵 ——
    # load_nifti 返回 (data, affine)：data 是 3D 体素标签（0/1/2），
    # affine 是「体素索引→物理坐标」的 4x4 矩阵，其中也编码了真实 spacing。
    # 本脚本为教学简洁起见使用命令行 --spacing；若要精确还原物理尺寸，
    # 应令 --spacing 与 affine 中的体素间距一致（下方会打印两者供对照）。
    label, affine = load_nifti(input_path)

    # 从仿射矩阵推算真实体素间距：取 affine 左上 3x3 每一列向量的长度。
    # 这里只用数组自带的运算，不需要在本文件额外 import numpy。
    spacing_from_affine = tuple(
        float((affine[:3, i] ** 2).sum() ** 0.5) for i in range(3)
    )

    # —— 3) 确定输出路径：未指定 --output 时自动生成到 outputs/stl/ ——
    # 命名规则：outputs/stl/<输入文件词干>_<类别名>.stl
    if args.output is None:
        stem = os.path.basename(input_path)
        # 去掉 .nii.gz（双扩展名）或 .nii 后缀，得到文件词干
        if stem.endswith(".nii.gz"):
            stem = stem[: -len(".nii.gz")]
        elif stem.endswith(".nii"):
            stem = stem[: -len(".nii")]
        output_path = os.path.join("outputs", "stl", f"{stem}_{cname}.stl")
    else:
        output_path = args.output

    # 确保输出目录存在（export_stl 内部也会创建，这里显式做一次更直观）
    ensure_dir(os.path.dirname(os.path.abspath(output_path)))

    # —— 4) 打印关键信息，让初学者清楚脚本在做什么、输入输出是什么 ——
    print("=" * 60)
    print(f"输入预测 : {input_path}")
    print(f"输出 STL : {output_path}")
    print(f"导出类别 : label_id={label_id}（{cname}）")
    print(f"网格平滑 : {smooth}")
    print(f"体素间距 : {spacing} (mm)  ← 本次导出使用")
    print(f"affine间距: {spacing_from_affine} (mm)  ← 原图真实间距（供对照）")
    print("=" * 60)

    # —— 5) 检查该类别是否有前景体素 ——
    # 若整卷没有该类，marching cubes 提取不到任何等值面；提前提示更友好。
    # （export_label_stl 内部也会对空掩膜跳过，这里先检查是为了给出更明确的原因）
    foreground_count = int((label == label_id).sum())
    if foreground_count == 0:
        print(f"[警告] 预测中 label_id={label_id}（{cname}）无前景体素，"
              "无法生成网格，跳过导出。\n"
              "请确认 --label_id 与预测的类别编码一致（1=肝脏, 2=肿瘤）。")
        sys.exit(0)

    # —— 6) 调用 src.utils 的 STL 导出函数 ——
    # export_label_stl 流程：先取 (label==label_id) 的 0/1 二值掩膜，再调用
    # export_stl：用 scikit-image 的 marching_cubes(level=0.5) 在掩膜上提取
    # 等值面（即前景/背景交界处的三角片），trimesh 组成网格、可选 Taubin
    # 平滑后导出 STL。scikit-image 与 trimesh 在函数内部「延迟导入」，缺失时
    # 会抛 ImportError，这里捕获后给出面向新手的安装提示。
    try:
        export_label_stl(
            label,
            label_id,
            output_path,
            spacing=spacing,
            smooth=smooth,
        )
    except ImportError as e:
        print(f"[错误] 缺少 STL 导出依赖：{e}\n"
              "请安装 scikit-image 与 trimesh：pip install scikit-image trimesh")
        sys.exit(1)
    except Exception as e:
        print(f"[错误] STL 导出失败：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

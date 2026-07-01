"""导出 STL 网格：把预测 NIfTI mask 转成 3D 表面网格（STL）。

作用
====
本脚本读取一个模型预测出的 3D 分割 mask（NIfTI 格式 .nii/.nii.gz），
对其中指定的每个标签（如 1=肝脏、2=肿瘤）单独提取出表面三角网格，
并导出为 STL 文件，方便用 3DViewer / MeshLab / Blender 等软件查看，
也可用于 3D 打印或教学演示。

对于大一医学生：理解"分割 mask → 3D 表面模型"的直观意义就是——
把一摞 2D 切片里勾画出来的器官/病灶轮廓，重新堆叠拼出一个立体的
三角面片外壳，这样就能在三维空间里看到它的整体形状。

运行方式
========
在项目根目录（d:\\项目1）下运行：

    python src/export_stl.py \
        --mask_path outputs/predictions/case.nii.gz \
        --output_dir outputs/stl \
        --labels 1 2 \
        --smooth \
        --min_component_size 100

参数说明
--------
--mask_path        : 必填，预测 mask 的 .nii.gz 路径
--output_dir       : 必填，STL 文件输出文件夹（不存在会自动创建）
--labels           : 要导出的标签 id 列表，默认 [1, 2]（1=liver, 2=tumor）
--smooth           : 是否对网格做 Taubin 平滑（开启后表面更光滑）
--min_component_size: 去除体素数小于该值的零碎连通域，默认 100
--help             : 查看帮助

输入输出
========
输入：一个预测 NIfTI mask（整数标签，0=背景，1=肝脏，2=肿瘤……）
输出：每个标签一个 STL 文件，命名 <文件名>_<类名>.stl，例如
      case_liver.stl、case_tumor.stl

常见错误
========
1. mask 文件不存在 → 检查路径是否正确、是否已运行 infer.py 生成预测。
2. 某个 label 清理后没有有效体素 → 脚本会打印警告并跳过该标签。
3. 缺少 trimesh / scikit-image → 本脚本顶部会优雅提示安装命令。
4. STL 体积很大、打开很卡 → 可加 --smooth，或调整 --min_component_size。

⚠⚠ 临床免责提醒 ⚠⚠
本脚本生成的 STL 文件仅供 **教学、展示与科研模型生成** 之用，
不可直接用于临床诊断、治疗规划或手术导航。请勿据此进行任何
医疗决策，一切以医生与正规医疗流程为准。
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
from scipy import ndimage

# trimesh 与 scikit-image 是较重的可选依赖：用 try/except 包起来，
# 这样即使没装也不会让 `python src/export_stl.py --help` 直接崩溃，
# 而是给出友好的中文安装提示。
try:
    import trimesh
except Exception:
    trimesh = None
    print("[警告] 未检测到 trimesh。请在 conda 环境内执行：")
    print("       conda install -c conda-forge trimesh  或  pip install trimesh")

try:
    from skimage import measure
except Exception:
    measure = None
    print("[警告] 未检测到 scikit-image。请在 conda 环境内执行：")
    print("       conda install -c conda-forge scikit-image  或  pip install scikit-image")


# 标签 id → 可读类名 的映射，用于 STL 文件命名与打印信息。
# 1=肝脏、2=肿瘤是本项目 phase2 的常用约定；遇到其它标签则用 class<label> 占位。
LABEL_NAME_MAP = {
    1: "liver",
    2: "tumor",
}


def get_class_name(label):
    """根据标签 id 返回类名；映射表里没有就返回 class<label>。"""
    return LABEL_NAME_MAP.get(label, f"class{label}")


def remove_small_components(binary, min_component_size):
    """去除二值 mask 中过小的连通域（假阳性碎片）。

    为什么去小连通域？
    ----------------
    模型在预测肝脏/肿瘤时，有时会在肝脏外面、肺里或皮下"误判"出几个
    孤立的小体素团——这些是"假阳性碎片"。如果在 STL 里保留它们，肉眼
    会看到器官旁边飘着一堆小颗粒，既不真实也影响美观。这里用连通域
    分析统计每个独立团块的体素数，把小于 min_component_size 的清零，
    就能去掉这些噪声碎片，只留下主要的器官/病灶主体。

    参数
    ----
    binary            : np.uint8 的 3D 0/1 数组
    min_component_size: 体素数阈值，体素数小于该值的连通域会被清零

    返回
    ----
    cleaned : 清理后的 0/1 数组（np.uint8）
    """
    # ndimage.label 给每个连通域分配一个整数编号，背景为 0。
    labeled_array, num_features = ndimage.label(binary)

    # 没有任何连通域就直接返回空数组，避免后续索引出错。
    if num_features == 0:
        return binary.copy()

    # bincount 统计每个编号（含 0=背景）的出现次数，即每个连通域的体素数。
    # 注意 labeled_array 的编号从 1 开始，bincount[0] 是背景体素数。
    counts = np.bincount(labeled_array.ravel())

    # 构造"保留掩码"：编号 i 保留当且仅当 counts[i] >= 阈值（i=0 背景恒不保留）。
    keep = counts >= min_component_size
    keep[0] = False  # 背景永远不需要进入 cleaned

    # 用保留掩码筛选：把 keep 按 labeled_array 的值映射回 3D 形状，再转 0/1。
    cleaned = keep[labeled_array].astype(np.uint8)
    return cleaned


def export_label_to_stl(mask, label, spacing, args, stem, output_dir):
    """对单个 label 完成二值化 → 清理 → marching cubes → 平滑 → 保存 STL。

    参数
    ----
    mask       : 整数标签 3D 数组（已从 NIfTI 读入）
    label      : 要导出的标签 id，如 1 或 2
    spacing    : 体素物理间距 (sx, sy, sz)，单位 mm
    args       : argparse 解析结果，用到 smooth
    stem       : mask 文件名（去扩展名），用于命名输出 STL
    output_dir : STL 输出文件夹

    返回
    ----
    bool：True 表示成功导出，False 表示跳过（无有效体素）。
    """
    # ---- 1) 二值化：只保留等于当前 label 的体素 ----
    # (mask == label) 是 bool 数组，*.astype(np.uint8) 转成 0/1，便于后续处理。
    binary = (mask == label).astype(np.uint8)

    # ---- 2) 去小连通域：去掉模型产生的假阳性碎片 ----
    binary = remove_small_components(binary, args.min_component_size)

    # ---- 3) 若清理后完全没有体素，打印警告并跳过 ----
    if binary.sum() == 0:
        print(f"[警告] label {label}（{get_class_name(label)}）无有效体素，跳过。")
        return False

    # ---- 4) Marching Cubes：把体素栅格等值面化成三角网格 ----
    # level=0.5 在 0/1 二值数据上正好落在二值分界面；spacing 让顶点
    # 坐标使用真实物理尺寸（mm），后续 STL 量纲才是毫米而不是体素。
    vertices, faces, normals, values = measure.marching_cubes(
        binary, level=0.5, spacing=spacing
    )

    # ---- 5) 构造 trimesh 网格 ----
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # ---- 6) 可选平滑：Taubin 滤波让表面更光滑、保留体积 ----
    # Taubin 滤波交替做"膨胀/收缩"两步，能在抹平毛刺的同时避免网格缩水，
    # 比单次高斯平滑更稳。try/except 兜底是为了在 trimesth 版本差异或
    # 网格退化时不至于整段失败。
    if args.smooth:
        try:
            mesh = trimesh.smoothing.filter_taubin(mesh)
        except Exception as e:
            print(f"[提示] 平滑失败，将保存未平滑网格：{e}")

    # ---- 7) 保存 STL ----
    class_name = get_class_name(label)
    out_path = os.path.join(output_dir, f"{stem}_{class_name}.stl")
    mesh.export(out_path)

    # ---- 8) 打印信息 ----
    print(f"[完成] label {label}（{class_name}）→ {out_path}")
    print(f"       顶点数: {len(mesh.vertices)}")
    print(f"       面数  : {len(mesh.faces)}")
    print(f"       体素 spacing: {spacing}")
    return True


def parse_args():
    """构建并解析命令行参数。"""
    parser = argparse.ArgumentParser(
        prog="export_stl",
        description="把预测 NIfTI mask 转成 STL 网格（仅供教学/展示，不可临床使用）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mask_path",
        required=True,
        help="预测 mask 的 .nii.gz 路径（整数标签，0=背景）",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="STL 输出文件夹（不存在会自动创建）",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        type=int,
        default=[1, 2],
        help="要导出的标签 id 列表，默认 [1 2]（1=liver, 2=tumor）",
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="是否对网格做 Taubin 平滑（开启后表面更光滑）",
    )
    parser.add_argument(
        "--min_component_size",
        type=int,
        default=100,
        help="体素数小于该值的连通域会被清零（去除假阳性碎片），默认 100",
    )
    return parser.parse_args()


def main():
    """主流程：读 mask → 对每个 label 导出 STL。"""
    args = parse_args()

    # 检查关键依赖是否存在；缺了就直接退出，并给出安装提示。
    if trimesh is None or measure is None:
        print("[错误] 缺少 trimesh 或 scikit-image，无法导出 STL。")
        sys.exit(1)

    # 检查 mask 文件存在性，给出比 FileNotFoundError 更友好的提示。
    if not os.path.isfile(args.mask_path):
        print(f"[错误] mask 文件不存在：{args.mask_path}")
        print("       请先运行 scripts/infer.py 生成预测 NIfTI，再执行本脚本。")
        sys.exit(1)

    # 自动创建输出文件夹（含中间层）。
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 读 NIfTI ----
    # nibabel 加载 .nii.gz；get_fdata() 返回 float64 数组，转成 int 便于 ==label 比较。
    # spacing 用 header.get_zooms()[:3] 取前三个轴（x/y/z）的体素间距，单位 mm。
    nii = nib.load(args.mask_path)
    mask = np.asarray(nii.get_fdata(), dtype=np.int32)
    spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])

    print("=" * 60)
    print("export_stl：预测 mask → STL 网格导出")
    print(f"  mask_path : {args.mask_path}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  labels    : {args.labels}")
    print(f"  smooth    : {args.smooth}")
    print(f"  min_component_size: {args.min_component_size}")
    print(f"  mask 形状 : {mask.shape}    spacing: {spacing}")
    print("=" * 60)

    # 临床免责提醒（运行时再强调一次）。
    print("[提醒] STL 仅供教学/展示模型生成，不可直接用于临床治疗或手术导航。")

    # 取不带扩展名的文件名 stem：如 case.nii.gz → case。
    stem = os.path.basename(args.mask_path)
    for ext in (".nii.gz", ".nii"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break

    # 逐个标签导出。
    success = 0
    for label in args.labels:
        ok = export_label_to_stl(
            mask=mask,
            label=label,
            spacing=spacing,
            args=args,
            stem=stem,
            output_dir=args.output_dir,
        )
        if ok:
            success += 1

    print("=" * 60)
    print(f"导出结束：成功 {success} / {len(args.labels)} 个标签。")
    if success == 0:
        print("[警告] 没有任何标签成功导出，请检查 mask 内容与 --labels 设置。")


if __name__ == "__main__":
    main()
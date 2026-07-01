"""scripts/check_data.py — 数据检查脚本（Task03_Liver 肝脏 3D 分割项目）

作用
====
扫描 Medical Segmentation Decathlon Task03_Liver 数据集的 imagesTr/labelsTr
（以及 imagesTs/labelsTs）目录，对训练集或测试集做一次「数据体检」：
  1. 统计样本数量；
  2. 统计图像与标签的体素形状分布（每轴的 min / max / mean）；
  3. 统计体素间距 spacing 分布（从 nibabel header.get_zooms() 读取，单位 mm）；
  4. 检查每个样本的图像与标签形状是否一致，不一致则记录警告与样本名；
  5. 对（可抽样）样本加载标签，统计 0/1/2 各类别像素占比并打印平均占比——
     其中 >0 即「前景」，对应阶段一(phase1) 的肝脏整体；1=肝脏、2=肿瘤
     对应阶段二(phase2) 的三分类。
最终打印汇总表与警告列表，帮助在训练前尽早发现数据问题。

运行方式
========
需在项目根目录（即 d:\\项目1）下运行，以便 ``from src.utils import load_nifti`` 可用：
    python scripts/check_data.py --data_root ./data/Task03_Liver
可选参数：
    --split tr|ts|both      检查训练集 / 测试集 / 两者，默认 tr
    --sample_ratio 0~1      标签类别统计的抽样比例，默认 1.0（全量）
更多示例：
    python scripts/check_data.py --data_root ./data/Task03_Liver --split both
    python scripts/check_data.py --data_root ./data/Task03_Liver --split tr --sample_ratio 0.3
    python scripts/check_data.py --help        # 查看参数帮助

输入
====
--data_root 指向解压后的 Task03_Liver 根目录，其下应包含：
    imagesTr/liver_*.nii.gz   训练图像
    labelsTr/liver_*.nii.gz   训练标签
    imagesTs/liver_*.nii.gz   测试图像（可选）
    labelsTs/liver_*.nii.gz   测试标签（可选）

输出
====
仅打印到终端（stdout）：各子集的汇总统计表 + 警告列表。
本脚本不写任何文件、不修改数据，可安全反复运行。

常见错误
========
1. --data_root 路径错误：根目录不存在 → 脚本提示并 exit(1)。
2. imagesTr/labelsTr（或 imagesTs/labelsTs）子目录缺失 → 训练集缺失视为硬错误
   exit(1)；测试集为可选项，缺失时跳过并提示（若单独 --split ts 则 exit(1)）。
3. image 与 label 文件名不配对（如某图像没有同名标签）→ 记入警告列表。
4. 图像与标签体素形状不匹配（如 (512,512,100) vs (512,512,99)）→ 记入警告并
   给出样本名与各自形状；此类样本若直接送入训练会因尺寸不一致报错。
5. 图像与标签 spacing 不一致 → 记入警告；spacing 决定重采样后的真实物理体积。
6. 标签出现 0/1/2 以外的值 → 记入警告（提示标签可能损坏或类别定义不符）。
"""

# ===================== 标准库导入 =====================
import os
import sys
import glob
import argparse

# ===================== 第三方库导入 =====================
# numpy：数组与统计计算；nibabel：读取 NIfTI 文件头（shape / spacing）。
import numpy as np
import nibabel as nib

# ===================== 项目内模块导入 =====================
# 先把项目根目录加入 sys.path，使得直接 `python scripts/check_data.py ...`
# 运行时可 import src.*（否则 Python 只把 scripts/ 当搜索路径，找不到 src 包）。
# os 与 sys 已在上方标准库导入块中导入。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# load_nifti 返回 (data, affine)；本脚本仅在「标签类别抽样统计」时需要真正读入
# 体素数据，故用到它；而形状/spacing 的扫描只用 nibabel 读文件头（懒加载，更快）。
from src.utils import load_nifti


# 子集名 -> (图像子目录, 标签子目录, 中文标签)，集中维护便于扩展。
SPLIT_DIRS = {
    "tr": ("imagesTr", "labelsTr", "训练集"),
    "ts": ("imagesTs", "labelsTs", "测试集"),
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 含 data_root / split / sample_ratio 三个字段。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Task03_Liver 数据检查脚本：扫描 imagesTr/labelsTr（与 imagesTs/labelsTs），"
            "统计样本数量、图像/标签形状与 spacing、检查配对与形状一致性、"
            "抽样统计标签各类别像素占比，并打印汇总与警告。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例（在项目根目录运行）：\n"
            "  python scripts/check_data.py --data_root ./data/Task03_Liver\n"
            "  python scripts/check_data.py --data_root ./data/Task03_Liver "
            "--split both --sample_ratio 0.3\n"
            "（需在项目根目录运行，以便 `from src.utils import load_nifti` 可用）"
        ),
    )
    # --data_root：必填，数据集根目录
    parser.add_argument(
        "--data_root", required=True,
        help="数据集根目录，例如 ./data/Task03_Liver（其下应含 imagesTr/labelsTr 等）",
    )
    # --split：选择检查训练集 / 测试集 / 两者，默认训练集
    parser.add_argument(
        "--split", choices=["tr", "ts", "both"], default="tr",
        help="检查哪个子集：tr=训练集, ts=测试集, both=两者均检查（默认 tr）",
    )
    # --sample_ratio：标签类别统计的抽样比例，0~1，默认全量
    parser.add_argument(
        "--sample_ratio", type=float, default=1.0,
        help="标签类别统计的抽样比例，范围 (0,1]；1.0=全量统计，<1 则均匀抽样"
             "以节省时间（默认 1.0）",
    )
    # 注：argparse 自动提供 -h/--help，无需手动添加。
    return parser.parse_args()


def scan_and_pair(images_dir: str, labels_dir: str):
    """用 glob 找出 images_dir 与 labels_dir 下的 liver_*.nii.gz，按文件名配对。

    配对规则：图像与标签文件名完全一致（如 liver_0.nii.gz）即视为一对；
    任一方缺失则记入警告。

    Args:
        images_dir: 图像目录绝对/相对路径。
        labels_dir: 标签目录绝对/相对路径。

    Returns:
        tuple(list, list):
            - samples: list[(basename, image_path, label_path)]，已配对样本；
            - warnings: list[str]，未配对（图像无标签 / 标签无图像）的警告信息。
    """
    # glob 匹配 liver_ 开头、.nii.gz 结尾的文件；sorted 保证顺序确定、可复现
    image_paths = sorted(glob.glob(os.path.join(images_dir, "liver_*.nii.gz")))
    label_paths = sorted(glob.glob(os.path.join(labels_dir, "liver_*.nii.gz")))

    # 以「文件名(basename)」为 key 建索引，便于按名配对
    image_map = {os.path.basename(p): p for p in image_paths}
    label_map = {os.path.basename(p): p for p in label_paths}

    samples = []
    warnings = []
    # 所有出现过的文件名取并集，逐个判断是否成对
    for bn in sorted(set(image_map) | set(label_map)):
        img = image_map.get(bn)
        lab = label_map.get(bn)
        if img is not None and lab is not None:
            samples.append((bn, img, lab))
        elif img is not None:
            warnings.append(
                f"图像缺少对应标签：{bn}（{images_dir} 有图像，"
                f"但 {labels_dir} 无同名标签）"
            )
        else:
            warnings.append(
                f"标签缺少对应图像：{bn}（{labels_dir} 有标签，"
                f"但 {images_dir} 无同名图像）"
            )
    return samples, warnings


def _per_axis_min_max_mean(list_of_tuples):
    """把一组 (a, b, c) 元组列表转成 (N, 3) 数组，返回每轴的 min/max/mean。

    用于汇总「形状分布」「spacing 分布」：每例是一个 3D 元组，逐轴统计可看出
    各维度在整批数据中的范围与典型大小。

    Args:
        list_of_tuples: list[tuple]，每项为长度 3 的数值元组。

    Returns:
        tuple: (mins, maxs, means)，均为长度 3 的 numpy 数组；输入为空时返回
        (None, None, None)。
    """
    if len(list_of_tuples) == 0:
        return None, None, None
    arr = np.asarray(list_of_tuples, dtype=float)  # (N, 3)
    return arr.min(axis=0), arr.max(axis=0), arr.mean(axis=0)


def collect_shape_spacing(samples):
    """遍历配对样本，用 nibabel「懒加载」读取文件头，收集形状与 spacing 并做一致性检查。

    概念说明
    --------
    - nibabel 默认懒加载：``nib.load(path).shape`` 与 ``.header.get_zooms()`` 只读
      文件头、不读全体素数据，因此扫描上百例也很快。
    - **spacing（体素间距）**：每个体素在物理空间中的尺寸，单位一般为 mm（可通过
      ``header.get_xyzt_units()`` 查询）。同样大小的数组，spacing 不同则代表的
      物理体积不同；这是后续 MONAI ``SpacingD`` 重采样的关键依据。
    - **轴向**：原始 NIfTI 前 3 维是空间维度，具体物理轴含义由 affine 方向决定；
      MONAI 加载并 ``Orientation('RAS')`` 后会统一为 channel-first 的 (D, H, W)
      顺序。本脚本统计的是「原始」前 3 维，供训练前人工核对。

    Args:
        samples: list[(basename, image_path, label_path)]，由 scan_and_pair 返回。

    Returns:
        dict: 含 image_shapes / label_shapes / image_spacings / label_spacings
        （均为 list[tuple]），以及 shape_mismatch / spacing_mismatch
        （list[(bn, img_val, lab_val)]，记录不一致样本）。
    """
    image_shapes, label_shapes = [], []
    image_spacings, label_spacings = [], []
    shape_mismatch, spacing_mismatch = [], []

    for bn, img_path, lab_path in samples:
        # 懒加载图像/标签对象（仅读头信息，不读全体素）
        img_nii = nib.load(img_path)
        lab_nii = nib.load(lab_path)

        # 取前 3 维（Task03_Liver 为 3D 单模态；若偶遇 4D 也只看空间三维）
        img_shape = tuple(int(s) for s in img_nii.shape[:3])
        lab_shape = tuple(int(s) for s in lab_nii.shape[:3])
        # get_zooms() 返回每维间距；前 3 个为空间 spacing(mm)
        img_sp = tuple(float(s) for s in img_nii.header.get_zooms()[:3])
        lab_sp = tuple(float(s) for s in lab_nii.header.get_zooms()[:3])

        image_shapes.append(img_shape)
        label_shapes.append(lab_shape)
        image_spacings.append(img_sp)
        label_spacings.append(lab_sp)

        # 形状一致性：整数精确比较即可
        if img_shape != lab_shape:
            shape_mismatch.append((bn, img_shape, lab_shape))
        # spacing 一致性：浮点比较用容差，避免极小数值误差误报
        if not np.allclose(img_sp, lab_sp, atol=1e-3):
            spacing_mismatch.append((bn, img_sp, lab_sp))

    return {
        "image_shapes": image_shapes,
        "label_shapes": label_shapes,
        "image_spacings": image_spacings,
        "label_spacings": label_spacings,
        "shape_mismatch": shape_mismatch,
        "spacing_mismatch": spacing_mismatch,
    }


def compute_label_category_stats(samples, sample_ratio: float):
    """对（抽样）样本加载标签体素数据，统计 0/1/2 各类像素占比，返回平均占比。

    概念说明
    --------
    - 标签取值约定：0=背景(background), 1=肝脏(liver), 2=肿瘤(tumor)。
    - **阶段一(phase1, 二分类)**：>0 即视为「肝脏整体」前景，因此额外统计
      (>0) 的占比；对应 src/data.py 的 LabelMapTransform 在 num_classes=2 时
      做 ``label>0 → 1`` 的映射。
    - **阶段二(phase2, 三分类)**：保留 1/2 区分肝脏与肿瘤，对应 num_classes>=3。
    - **「占比」**= 该类体素数 / 该样本体素总数，按样本逐一计算后取平均（而非
      把所有样本体素混在一起算全局占比），这样每个样本权重相同，避免少数大体积
      样本主导统计结果。

    抽样
    ----
    sample_ratio ∈ (0, 1]：1.0=全量统计；<1 时按等间隔(linspace)抽样，结果确定、
    可复现。本函数用 ``src.utils.load_nifti`` 加载标签（返回 (data, affine)），
    只取 data；这是脚本中唯一需要真正读入全体素数据的环节，故支持抽样以节省时间。

    Args:
        samples: list[(basename, image_path, label_path)]。
        sample_ratio: 抽样比例，范围 (0, 1]。

    Returns:
        dict: 含 n_sampled（抽样数）、ratio_bg/ratio_liver/ratio_tumor/ratio_fg
        （各类平均占比，0~1 浮点）、unexpected（含 0/1/2 以外值的样本名列表）。
        samples 为空时返回 None。
    """
    n = len(samples)
    if n == 0:
        return None

    # 确定抽样数量与索引（等间隔抽样，确定性、可复现）
    if sample_ratio >= 1.0:
        chosen = samples
    else:
        n_sample = max(1, int(round(n * sample_ratio)))
        idx = np.linspace(0, n - 1, n_sample).astype(int)
        # round 后可能产生重复索引，去重并保持升序
        idx = sorted(set(int(i) for i in idx))
        chosen = [samples[i] for i in idx]

    ratios_bg, ratios_liver, ratios_tumor, ratios_fg = [], [], [], []
    unexpected = []  # 出现 0/1/2 以外值的样本

    for bn, _img_path, lab_path in chosen:
        # 加载标签体素数据（保留磁盘原始 dtype，通常为 int16/uint8）
        label, _ = load_nifti(lab_path)
        flat = np.asarray(label).ravel()
        total = int(flat.size)
        if total == 0:
            continue

        # np.bincount 一次扫描得到每个取值的体素数（要求非负整数，标签满足）
        # counts[k] = 值等于 k 的体素数；长度为 max(值)+1
        counts = np.bincount(flat.astype(np.int64))
        n0 = int(counts[0]) if len(counts) > 0 else 0
        n1 = int(counts[1]) if len(counts) > 1 else 0
        n2 = int(counts[2]) if len(counts) > 2 else 0
        # 前景 = 全体素 - 背景，避免再做一次 (flat>0) 的全量扫描
        n_pos = total - n0
        # 是否存在 >2 的取值（counts[3:] 之和 > 0 表示有非法值）
        if len(counts) > 3 and int(counts[3:].sum()) > 0:
            unexpected.append(bn)

        ratios_bg.append(n0 / total)
        ratios_liver.append(n1 / total)
        ratios_tumor.append(n2 / total)
        ratios_fg.append(n_pos / total)

    def _mean(xs):
        return float(np.mean(xs)) if len(xs) > 0 else 0.0

    return {
        "n_sampled": len(chosen),
        "ratio_bg": _mean(ratios_bg),          # 类别 0 背景
        "ratio_liver": _mean(ratios_liver),    # 类别 1 肝脏
        "ratio_tumor": _mean(ratios_tumor),    # 类别 2 肿瘤
        "ratio_fg": _mean(ratios_fg),          # >0 前景（phase1 肝脏整体）
        "unexpected": unexpected,
    }


def _print_axis_block(title, list_of_tuples):
    """打印某项（形状或 spacing）的逐轴 min/max/mean 分布。"""
    print(title)
    mins, maxs, means = _per_axis_min_max_mean(list_of_tuples)
    if mins is None:
        print("    （无样本）")
        return
    axis_names = ["第0轴", "第1轴", "第2轴"]
    for i, ax in enumerate(axis_names):
        print(f"    {ax}: min={mins[i]:.3f}, max={maxs[i]:.3f}, mean={means[i]:.3f}")


def print_summary(split_label, samples, stats, label_stats, pair_warnings):
    """打印某个子集的汇总统计表，并返回该子集产生的全部警告字符串列表。

    Args:
        split_label: 子集中文标签，如 "训练集"。
        samples: 该子集配对样本列表。
        stats: collect_shape_spacing 返回的 dict。
        label_stats: compute_label_category_stats 返回的 dict（可为 None）。
        pair_warnings: scan_and_pair 返回的配对警告 list[str]。

    Returns:
        list[str]: 该子集所有警告（配对、形状不匹配、spacing 不匹配、标签非法值），
        供主流程汇总到全局警告列表。
    """
    warns = []
    bar = "=" * 70
    print("\n" + bar)
    print(f"数据检查汇总 —— {split_label}")
    print(bar)
    print(f"配对样本数量：{len(samples)}")

    # —— 图像形状分布 ——
    _print_axis_block(
        "图像形状分布（原始 NIfTI 前 3 维；MONAI 加载后会统一为 channel-first 的 (D,H,W)）：",
        stats["image_shapes"],
    )

    # —— 标签形状分布 ——
    _print_axis_block("标签形状分布（前 3 维）：", stats["label_shapes"])

    # —— 图像 spacing 分布 ——
    _print_axis_block(
        "图像 spacing 分布（体素间距，单位 mm，来自 header.get_zooms()）：",
        stats["image_spacings"],
    )

    # —— 图像-标签形状一致性 ——
    sm = stats["shape_mismatch"]
    if not sm:
        print("\n图像-标签形状一致性：全部一致 ✓")
    else:
        print(f"\n图像-标签形状一致性：存在 {len(sm)} 例不一致 ✗")
        for bn, ish, lsh in sm:
            line = f"{split_label} [形状不匹配] {bn}: 图像={ish}, 标签={lsh}"
            warns.append(line)
            print("    " + line)

    # —— 图像-标签 spacing 一致性 ——
    spm = stats["spacing_mismatch"]
    if not spm:
        print("图像-标签 spacing 一致性：全部一致 ✓")
    else:
        print(f"图像-标签 spacing 一致性：存在 {len(spm)} 例不一致 ✗")
        for bn, isp, lsp in spm:
            line = f"{split_label} [spacing 不匹配] {bn}: 图像={isp}, 标签={lsp}"
            warns.append(line)
            print("    " + line)

    # —— 标签类别像素占比 ——
    print("\n标签类别像素占比（按样本计算占比后取平均）：")
    if label_stats is not None:
        print(f"    抽样样本数：{label_stats['n_sampled']}")
        print(f"    类别 0 背景        平均占比：{label_stats['ratio_bg'] * 100:6.2f}%")
        print(f"    类别 1 肝脏(liver) 平均占比：{label_stats['ratio_liver'] * 100:6.2f}%")
        print(f"    类别 2 肿瘤(tumor) 平均占比：{label_stats['ratio_tumor'] * 100:6.2f}%")
        print(f"    >0 前景(phase1)   平均占比：{label_stats['ratio_fg'] * 100:6.2f}%")
        if label_stats["unexpected"]:
            for bn in label_stats["unexpected"]:
                line = f"{split_label} [标签含非法值] {bn} 标签出现 0/1/2 以外的值"
                warns.append(line)
                print("    " + line)

    # —— 配对警告 ——
    if pair_warnings:
        print(f"\n配对警告（{len(pair_warnings)} 条）：")
        for w in pair_warnings:
            warns.append(f"{split_label} {w}")
            print(f"    - {w}")

    print(bar)
    return warns


def main():
    """主流程：解析参数 → 逐子集扫描统计 → 打印汇总与全局警告列表。"""
    args = parse_args()

    # —— 校验 --sample_ratio 取值范围 (0, 1] ——
    if not (0.0 < args.sample_ratio <= 1.0):
        print(f"[错误] --sample_ratio 必须在 (0, 1] 范围内，当前为 {args.sample_ratio}")
        sys.exit(1)

    data_root = args.data_root
    # —— 校验数据集根目录存在 ——
    if not os.path.isdir(data_root):
        print(f"[错误] 数据集根目录不存在：{data_root}")
        print("       请检查 --data_root 路径是否正确，并确认 Task03_Liver 已解压。")
        sys.exit(1)

    # 根据 --split 决定要检查哪些子集
    splits = ["tr", "ts"] if args.split == "both" else [args.split]

    any_checked = False
    all_warnings = []

    for split in splits:
        img_sub, lab_sub, label = SPLIT_DIRS[split]
        images_dir = os.path.join(data_root, img_sub)
        labels_dir = os.path.join(data_root, lab_sub)

        # —— 目录缺失处理 ——
        if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
            if split == "tr":
                # 训练集目录缺失：硬错误
                print(f"[错误] {label}目录缺失：")
                print(f"       期望图像目录：{images_dir}")
                print(f"       期望标签目录：{labels_dir}")
                print("       请确认数据集已正确解压，且 --data_root 指向 Task03_Liver 根目录。")
                sys.exit(1)
            else:
                # 测试集缺失：单独请求 ts 时为硬错误，both 时跳过并提示
                if args.split == "ts":
                    print(f"[错误] 未找到测试集目录：")
                    print(f"       {images_dir} / {labels_dir}")
                    print("       测试集为可选项；若确需检查，请先放置 imagesTs/labelsTs。")
                    sys.exit(1)
                print(f"[提示] 未找到测试集目录，跳过测试集检查（测试集为可选项）：")
                print(f"       {images_dir} / {labels_dir}")
                continue

        # —— 扫描与配对 ——
        samples, pair_warnings = scan_and_pair(images_dir, labels_dir)

        # —— 无配对样本：硬错误 ——
        if len(samples) == 0:
            print(f"[错误] {label}未找到任何 liver_*.nii.gz 配对样本：")
            print(f"       图像目录：{images_dir}")
            print(f"       标签目录：{labels_dir}")
            print("       请确认目录内文件命名符合 liver_*.nii.gz 约定。")
            sys.exit(1)

        # —— 统计形状与 spacing（仅读文件头，快）——
        stats = collect_shape_spacing(samples)

        # —— 标签类别抽样统计（需读体素数据，支持抽样以省时）——
        label_stats = compute_label_category_stats(samples, args.sample_ratio)

        # —— 打印该子集汇总，并收集其警告 ——
        split_warns = print_summary(label, samples, stats, label_stats, pair_warnings)
        all_warnings.extend(split_warns)
        any_checked = True

    # —— 兜底：没有任何子集被检查 ——
    if not any_checked:
        print("[错误] 没有可检查的数据子集。")
        sys.exit(1)

    # —— 全局警告列表 ——
    print("\n" + "=" * 70)
    if all_warnings:
        print(f"全局警告列表（共 {len(all_warnings)} 条）")
        print("=" * 70)
        for w in all_warnings:
            print(f"  - {w}")
    else:
        print("全局警告列表：无配对/形状/spacing/标签异常警告 ✓")
        print("=" * 70)

    print("\n数据检查结束。请重点关注：图像-标签形状/spacing 不匹配、未配对文件、"
          "标签含非法值——这些若不处理，可能在训练时引发报错。")


if __name__ == "__main__":
    main()


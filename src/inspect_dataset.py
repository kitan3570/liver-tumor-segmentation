"""src/inspect_dataset.py —— Task03_Liver 数据集检查脚本（面向初学者）

================================================================
作用
================================================================
扫描 Medical Segmentation Decathlon 的 Task03_Liver（肝脏肿瘤分割）数据集，
对每一对 image/label 做一次「数据体检」，并把结果写成 CSV 与示意图 PNG：
  1. 在 imagesTr/ 与 labelsTr/ 下用 glob 找出 liver_*.nii.gz，按文件名配对；
  2. 用 nibabel 读取每例的体素形状 (shape)、体素间距 (spacing)、强度统计
     (min/max/mean)、标签取值集合 (np.unique)；
  3. 用 pandas 把逐例信息汇总成一张表，存为 <output_dir>/dataset_summary.csv；
  4. 固定随机种子 (seed=42) 随机挑 3 例，分别沿 axial / coronal / sagittal
     三方向画「灰度图 + 半透明彩色标签叠加」的示意图，各存一张 PNG 到
     <output_dir>/figures/。

整个脚本只读数据、不改数据，可安全反复运行。

================================================================
运行方式
================================================================
需已安装 numpy、nibabel、pandas、matplotlib。在项目根目录 (d:\\项目1) 下运行：
    python src/inspect_dataset.py --data_dir data/raw/Task03_Liver --output_dir outputs
也可只给 --data_dir（--output_dir 默认为 outputs）：
    python src/inspect_dataset.py --data_dir data/raw/Task03_Liver
查看参数帮助：
    python src/inspect_dataset.py --help

================================================================
输入
================================================================
--data_dir 指向解压后的 Task03_Liver 根目录，其下应包含：
    imagesTr/liver_*.nii.gz   训练图像（CT 体素）
    labelsTr/liver_*.nii.gz   训练标签（与同名图像对应；0=背景, 1=肝脏, 2=肿瘤）
    dataset.json              数据集说明（本脚本不强制依赖该文件）

================================================================
输出
================================================================
- <output_dir>/dataset_summary.csv   逐例统计表（pandas DataFrame 写出）
- <output_dir>/figures/<case>_axial.png
- <output_dir>/figures/<case>_coronal.png
- <output_dir>/figures/<case>_sagittal.png
  每方向单独一张图；标题含病例名、方向、标签取值集合。

================================================================
常见错误
================================================================
1. --data_dir 路径错误：根目录不存在 → 脚本打印提示并以非 0 退出。
2. imagesTr/ 或 labelsTr/ 子目录缺失 → 脚本提示并退出（这是硬错误）。
3. 无任何配对样本（例如文件命名不是 liver_*.nii.gz）→ 脚本提示并退出。
4. 某图像缺少同名标签（或反之）→ 打印警告、跳过该例、继续其余样本。
5. 图像与标签形状不一致 → 仍按图像方向切片，可能在某切片索引越界；脚本会
   在逐例输出中标注 shape，请人工留意。
6. 标签出现 0/1/2 以外的取值 → 本脚本「不假设」取值，一律以实际的
   np.unique 结果为准并打印/写入 CSV，方便你发现问题。
"""

# ===================== 标准库导入 =====================
import os
import sys
import glob
import random
import argparse

# ===================== 第三方库导入 =====================
import numpy as np            # 数组与统计
import nibabel as nib         # 读取 NIfTI (.nii / .nii.gz) 文件与文件头
import pandas as pd           # 把逐例统计整理成表格并写 CSV

# matplotlib：先指定非交互后端 'Agg'，再 import pyplot，
# 这样 plt.savefig 可在无显示器（服务器/命令行）环境下正常工作。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===================== 工具函数 =====================
def ensure_dir(path: str) -> None:
    """若目录不存在则创建（含中间目录）；已存在则不报错。

    os.makedirs(exist_ok=True) 表示「目录已存在也视为成功」，
    非常适合输出目录这种「可能已经被人手动建好」的场景。
    """
    os.makedirs(path, exist_ok=True)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    设计要点
    --------
    - --data_dir 必填：数据集根目录。
    - --output_dir 默认 'outputs'：CSV 与示意图的输出目录。
    - argparse 会自动给 -h/--help 生成帮助，无需手动添加。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Task03_Liver 数据集检查脚本：扫描 imagesTr/labelsTr，逐例打印 "
            "shape/spacing/强度/标签取值，汇总成 CSV，并随机挑 3 例画 "
            "axial/coronal/sagittal 三方向叠加示意图。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例（在项目根目录运行）：\n"
            "  python src/inspect_dataset.py "
            "--data_dir data/raw/Task03_Liver --output_dir outputs\n"
            "  python src/inspect_dataset.py --help"
        ),
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="数据集根目录，例如 data/raw/Task03_Liver（其下应含 imagesTr/labelsTr）",
    )
    parser.add_argument(
        "--output_dir", default="outputs",
        help="输出目录：CSV 存 <output_dir>/dataset_summary.csv，"
             "图存 <output_dir>/figures/（默认 outputs）",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="快速模式：只从文件头读取 shape/spacing，跳过图像强度统计"
             "（intensity min/max/mean）与画图，大幅加速大体积数据集检查",
    )
    parser.add_argument(
        "--max_cases", type=int, default=0,
        help="最多检查的病例数（0=全部）。用于大数据集快速抽检",
    )
    parser.add_argument(
        "--no_plots", action="store_true",
        help="跳过三方向叠加图绘制（仍写 CSV 与汇总）",
    )
    return parser.parse_args()


def scan_and_pair(images_dir: str, labels_dir: str):
    """扫描 imagesTr 与 labelsTr，按 basename 配对 image/label。

    概念说明
    --------
    - glob 用通配符 liver_*.nii.gz 匹配文件名；sorted 保证顺序确定、可复现。
    - basename 即文件名本身（如 liver_0.nii.gz）。图像与标签「同名」即视为一对。
    - 任一方缺失则记入 warnings 并跳过该例。

    Args:
        images_dir: imagesTr 的路径。
        labels_dir: labelsTr 的路径。

    Returns:
        tuple(list, list):
            - samples: list[(basename, image_path, label_path)] 已配对样本；
            - warnings: list[str] 未配对警告。
    """
    image_paths = sorted(glob.glob(os.path.join(images_dir, "liver_*.nii.gz")))
    label_paths = sorted(glob.glob(os.path.join(labels_dir, "liver_*.nii.gz")))

    # 以文件名(basename)为 key 建索引，便于按名配对
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
            warnings.append(f"图像缺少对应标签：{bn}（{images_dir} 有，但 {labels_dir} 无）")
        else:
            warnings.append(f"标签缺少对应图像：{bn}（{labels_dir} 有，但 {images_dir} 无）")
    return samples, warnings


def pick_largest_label_slice(label_vol: np.ndarray, axis: int) -> int:
    """在指定轴向上挑选「含前景(label>0)面积最大」的切片索引。

    概念说明
    --------
    - 把 3D 体素沿某轴「切片」，就得到一张 2D 图。axis=0 对应 sagittal、
      axis=1 对应 coronal、axis=2 对应 axial（这是 Task03_Liver 常见约定）。
    - 直接取中心切片可能切到全背景、看不见结构，所以这里挑「前景像素数最多」
      的切片，能保证叠加图里能看到肝脏/肿瘤轮廓。
    - np.argmax 返回最大值首次出现的索引；若该轴长为 0，返回 0 作兜底。

    Args:
        label_vol: 标签体素数组 (X, Y, Z)。
        axis: 切片轴，0/1/2。

    Returns:
        int: 该轴上前景面积最大的切片索引。
    """
    if label_vol.shape[axis] == 0:
        return 0
    # (label_vol > 0) 得到布尔体素；要取「沿 axis 每张切片」的前景像素数，
    # 需对「除 axis 外的其它两个轴」求和，得到长度等于 axis 维度的一维数组。
    # 例如 axis=2 时，应对 (0, 1) 求和 -> shape (Z,) 的每张轴向切片前景数。
    other_axes = tuple(i for i in range(label_vol.ndim) if i != axis)
    fg_per_slice = (label_vol > 0).sum(axis=other_axes)
    return int(np.argmax(fg_per_slice))


def overlay_label_on_slice(image_slice: np.ndarray, label_slice: np.ndarray,
                           alpha: float = 0.4) -> np.ndarray:
    """把灰度 CT 切片与彩色标签混合成一张 RGB 图（用于叠加可视化）。

    概念说明
    --------
    - matplotlib 的 imshow 接受 (H, W, 3) 的 RGB 数组（值范围 0~1）。
    - 我们先把灰度切片归一化到 [0,1]，复制 3 份成 RGB；
    - 再在「label==1」的像素处叠加红色、在「label==2」处叠加绿色，
      用 alpha 做比例混合：result = (1-alpha)*gray + alpha*color。
    - 这样既保留了 CT 解剖结构（灰度背景），又突出标签区域（红/绿）。

    Args:
        image_slice: 2D 灰度切片（任意数值范围）。
        label_slice: 与之同形状的 2D 标签切片。
        alpha: 标签颜色的不透明度，0~1，默认 0.4。

    Returns:
        np.ndarray: (H, W, 3) float，范围 0~1。
    """
    # —— 1. 归一化灰度切片到 [0,1] ——
    vmin = float(image_slice.min())
    vmax = float(image_slice.max())
    if vmax - vmin < 1e-8:
        # 极端情况：所有体素值相同，避免除零；归一化全设 0
        gray = np.zeros_like(image_slice, dtype=float)
    else:
        gray = (image_slice.astype(float) - vmin) / (vmax - vmin)

    # —— 2. 复制成 RGB（3 通道）——
    rgb = np.stack([gray, gray, gray], axis=-1)

    # —— 3. 用半透明红/绿覆盖标签区域 ——
    # label==1 红色 (1,0,0)；label==2 绿色 (0,1,0)。
    mask1 = (label_slice == 1)
    mask2 = (label_slice == 2)
    color_red = np.array([1.0, 0.0, 0.0])
    color_green = np.array([0.0, 1.0, 0.0])
    for mask, color in ((mask1, color_red), (mask2, color_green)):
        if not mask.any():
            continue
        # 在 mask 处做 alpha 混合：(1-alpha)*gray + alpha*color
        rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * color

    return rgb


def plot_one_direction(image_vol: np.ndarray, label_vol: np.ndarray,
                       axis: int, direction_name: str, case: str,
                       label_uniques, out_dir: str) -> str:
    """沿指定轴向切片并保存一张 image+label 叠加 PNG。

    Args:
        image_vol: 图像体素 (X, Y, Z)。
        label_vol: 标签体素 (X, Y, Z)。
        axis: 切片轴，0=sagittal / 1=coronal / 2=axial。
        direction_name: 方向名（用于标题与文件名，如 'axial'）。
        case: 病例名（basename 去掉扩展名）。
        label_uniques: 该例标签取值集合（放进标题）。
        out_dir: 图片输出目录（已存在）。

    Returns:
        str: 保存的 PNG 绝对路径。
    """
    # —— 1. 选切片：沿 axis 取前景面积最大的切片 ——
    slice_idx = pick_largest_label_slice(label_vol, axis)
    # np.take 沿指定轴取切片；再 squeeze 去掉长度为 1 的维度得到 2D
    image_slice = np.take(image_vol, slice_idx, axis=axis)
    label_slice = np.take(label_vol, slice_idx, axis=axis)

    # —— 2. 把灰度切片与标签合成 RGB 叠加图 ——
    rgb = overlay_label_on_slice(image_slice, label_slice, alpha=0.4)

    # —— 3. 画图并保存 ——
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    ax.imshow(rgb, origin="lower")  # origin='lower' 与 NIfTI 体素坐标方向更一致
    ax.set_title(
        f"{case} | {direction_name} | slice={slice_idx} | "
        f"label={label_uniques}",
        fontsize=10,
    )
    ax.axis("off")

    out_path = os.path.join(out_dir, f"{case}_{direction_name}.png")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    """主流程：解析参数 → 校验目录 → 扫描配对 → 逐例统计 → 写 CSV → 画图。"""
    args = parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir

    # —— 1. 校验数据集根目录存在 ——
    if not os.path.isdir(data_dir):
        print(f"[错误] 数据集根目录不存在：{data_dir}")
        print("       请检查 --data_dir 路径是否正确，并确认 Task03_Liver 已解压。")
        sys.exit(1)

    images_dir = os.path.join(data_dir, "imagesTr")
    labels_dir = os.path.join(data_dir, "labelsTr")

    # —— 2. 校验 imagesTr / labelsTr 子目录存在 ——
    if not os.path.isdir(images_dir):
        print(f"[错误] 缺少图像目录：{images_dir}")
        print("       请确认 Task03_Liver 已正确解压（应包含 imagesTr/ 子目录）。")
        sys.exit(1)
    if not os.path.isdir(labels_dir):
        print(f"[错误] 缺少标签目录：{labels_dir}")
        print("       请确认 Task03_Liver 已正确解压（应包含 labelsTr/ 子目录）。")
        sys.exit(1)

    # —— 3. 扫描与配对 ——
    samples, warnings = scan_and_pair(images_dir, labels_dir)

    # 打印配对警告（缺失的一方）
    for w in warnings:
        print(f"[警告] {w}")

    # —— 4. 无配对样本：硬错误 ——
    if len(samples) == 0:
        print(f"[错误] 未找到任何 liver_*.nii.gz 配对样本：")
        print(f"       图像目录：{images_dir}")
        print(f"       标签目录：{labels_dir}")
        print("       请确认目录内文件命名符合 liver_*.nii.gz 约定。")
        sys.exit(1)

    # —— 应用 --max_cases 抽检限制 ——
    if args.max_cases and args.max_cases > 0 and args.max_cases < len(samples):
        samples = samples[:args.max_cases]
        print(f"[信息] --max_cases={args.max_cases}，只检查前 {len(samples)} 例。")

    quick_mode = args.quick
    do_plots = (not args.no_plots) and (not quick_mode)
    if quick_mode:
        print("[信息] --quick 快速模式：仅读取文件头获取 shape/spacing，"
              "跳过强度统计与画图。")
    if not do_plots:
        print("[信息] 跳过三方向叠加图绘制。")

    print(f"[信息] 共配对 {len(samples)} 例 liver_*.nii.gz。开始逐例检查...\n")

    # —— 5. 准备输出目录 ——
    ensure_dir(output_dir)
    figures_dir = os.path.join(output_dir, "figures")
    ensure_dir(figures_dir)

    # —— 6. 逐例统计 ——
    # 设计要点（针对 27GB 大数据集的性能优化）：
    #   a) shape/spacing 从文件头直接读取，无需加载体素（header 信息极小）；
    #   b) quick 模式下完全不加载图像体素，只加载标签体素用于 np.unique；
    #   c) 非 quick 模式用 np.asarray(nii.dataobj) 取原生 dtype（通常 int16），
    #      避免 get_fdata() 把 int16 拷贝成 float64（内存 ×4 且慢）；
    #   d) 逐例打印 [进度] i/N，方便 GUI 解析显示进度条；
    #   e) 缓存被随机选中的 3 例的体素数据，避免画图时重复 I/O；
    #   f) 捕获 KeyboardInterrupt，保留已检查的部分结果写入 CSV。
    rows = []
    total = len(samples)
    # 预先选定要画图的 3 例（seed=42 可复现），用于在主循环中缓存其体素
    rng = random.Random(42)
    n_pick = min(3, len(samples)) if do_plots else 0
    chosen_set = set()
    chosen_list = []
    if n_pick > 0:
        chosen_list = rng.sample(samples, n_pick)
        chosen_set = {os.path.basename(c[1]) for c in chosen_list}
    # 缓存：basename -> (image_vol, label_vol, label_uniques)
    cached_vols = {}

    interrupted = False
    try:
        for i, (case_bn, img_path, lab_path) in enumerate(samples, start=1):
            case = case_bn.replace(".nii.gz", "")
            # —— 进度提示（GUI 解析 [进度] i/N 显示进度条） ——
            print(f"[进度] {i}/{total} ({case})")

            img_nii = nib.load(img_path)
            # shape/spacing 直接来自文件头，不需要加载体素
            shape = tuple(int(s) for s in img_nii.shape[:3])
            spacing = tuple(float(s) for s in img_nii.header.get_zooms()[:3])

            # —— quick 模式：跳过图像强度统计 ——
            if quick_mode:
                intensity_min = float("nan")
                intensity_max = float("nan")
                intensity_mean = float("nan")
                image_vol = None
            else:
                # 用 dataobj 取原生 dtype（int16），避免 float64 拷贝
                image_vol = np.asarray(img_nii.dataobj)
                intensity_min = float(image_vol.min())
                intensity_max = float(image_vol.max())
                intensity_mean = float(image_vol.mean())

            # 标签取值：仍需加载标签体素，但用 dataobj 取原生 dtype（uint8）
            lab_nii = nib.load(lab_path)
            label_vol = np.asarray(lab_nii.dataobj)
            label_uniques = [int(v) for v in np.unique(label_vol)]

            # —— 缓存被选中画图案例的体素，避免重复加载 ——
            if case_bn in chosen_set:
                cached_vols[case_bn] = (image_vol, label_vol, label_uniques)

            # 终端逐例输出
            print(f"病例 {case}：")
            print(f"  image shape   = {shape}")
            print(f"  voxel spacing = {spacing} (mm)")
            if quick_mode:
                print(f"  intensity     : (skipped in --quick mode)")
            else:
                print(f"  intensity     : min={intensity_min:.2f}, "
                      f"max={intensity_max:.2f}, mean={intensity_mean:.2f}")
            print(f"  label unique  = {label_uniques}")

            rows.append({
                "case": case,
                "image_shape_x": shape[0],
                "image_shape_y": shape[1],
                "image_shape_z": shape[2],
                "spacing_x": spacing[0],
                "spacing_y": spacing[1],
                "spacing_z": spacing[2],
                "intensity_min": intensity_min,
                "intensity_max": intensity_max,
                "intensity_mean": intensity_mean,
                "label_unique": str(label_uniques),
            })

            # —— 释放本例体素引用（除非已缓存），降低峰值内存 ——
            if case_bn not in chosen_set:
                image_vol = None
                label_vol = None

    except KeyboardInterrupt:
        interrupted = True
        print("\n[停止] 收到中断信号，正在保存已检查的部分结果...")

    # —— 7. 写 CSV ——
    df = pd.DataFrame(rows, columns=[
        "case", "image_shape_x", "image_shape_y", "image_shape_z",
        "spacing_x", "spacing_y", "spacing_z",
        "intensity_min", "intensity_max", "intensity_mean",
        "label_unique",
    ])
    csv_path = os.path.join(output_dir, "dataset_summary.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    if interrupted:
        print(f"\n[信息] 已写入部分 CSV（{len(rows)}/{total} 例）：{csv_path}")
    else:
        print(f"\n[信息] 已写入 CSV：{csv_path}")

    # —— 8. 汇总分布打印（min/max/mean） ——
    print("\n" + "=" * 60)
    print("形状与 spacing 分布汇总（逐轴 min/max/mean）")
    print("=" * 60)
    for col, label in (
        ("image_shape", "image shape"),
        ("spacing", "voxel spacing (mm)"),
    ):
        if col == "image_shape":
            cols = ["image_shape_x", "image_shape_y", "image_shape_z"]
        else:
            cols = ["spacing_x", "spacing_y", "spacing_z"]
        sub = df[cols].astype(float)
        print(f"- {label}:")
        for c in cols:
            print(f"    {c}: min={sub[c].min():.3f}, "
                  f"max={sub[c].max():.3f}, mean={sub[c].mean():.3f}")

    # —— 9. 画三方向叠加图（仅非 quick 模式且未被中断时） ——
    if do_plots and not interrupted and n_pick > 0:
        print(f"\n[信息] 随机挑选 {n_pick} 例（seed=42）绘制三方向叠加图：")
        for case_bn, img_path, lab_path in chosen_list:
            case = case_bn.replace(".nii.gz", "")
            # 从主循环缓存中取出体素，避免重复 I/O
            if case_bn in cached_vols:
                image_vol, label_vol, label_uniques = cached_vols[case_bn]
            else:
                # 极端情况（被 --max_cases 截断导致缓存未命中）：按需加载
                img_nii = nib.load(img_path)
                lab_nii = nib.load(lab_path)
                image_vol = np.asarray(img_nii.dataobj)
                label_vol = np.asarray(lab_nii.dataobj)
                label_uniques = [int(v) for v in np.unique(label_vol)]

            for axis, direction_name in ((2, "axial"), (1, "coronal"), (0, "sagittal")):
                out_path = plot_one_direction(
                    image_vol, label_vol, axis, direction_name,
                    case, label_uniques, figures_dir,
                )
                print(f"  保存 {out_path}")

    if interrupted:
        print("\n[信息] 检查已被中止（部分结果已保存）。")
    else:
        print("\n[信息] 全部完成。")
    print(f"       CSV：{csv_path}")
    print(f"       图片目录：{figures_dir}")


if __name__ == "__main__":
    main()
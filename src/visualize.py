"""src/visualize.py — 医学图像分割可视化（轴向切片对比 PNG）

作用
----
读取一张 CT 图像（NIfTI）与其对应的真值标签、模型预测（均为 NIfTI，可选），
在「轴向切片 (axial slice)」上生成多张对比 PNG，帮助大一医学生直观认识：
  1. 原始 CT（灰度，肝窗显示）；
  2. CT + 真值标签叠加（liver 红、tumor 绿半透明）；
  3. CT + 预测标签叠加（liver 红、tumor 绿）；
  4. 真值 vs 预测的对比（并排两 overlay 或差异图）。
脚本会自动选取含「肿瘤」面积最大的若干轴向切片（tumor 优先），逐切片输出一张
组合 PNG，便于翻阅查看病灶及其分割结果。

轴向切片说明（面向初学者）
--------------------------
医学 CT 沿 Z 轴逐层扫描，每一层叫一个「轴向切片 (axial slice)」。本项目约定
3D 体素数组形状为 (H, W, D)，第 3 个维度（axis=2）为轴向方向，故
``image[:, :, z]`` 即第 z 张轴向切片。本脚本固定取 axis=2 切片进行可视化。

颜色叠加说明
------------
- liver（类别 1）用红色 (1,0,0) 半透明叠加；
- tumor（类别 2）用绿色 (0,1,0) 半透明叠加；
- 背景（类别 0）不叠加。
肝脏与肿瘤在标签中互斥（同一体素不会既是肝脏又是肿瘤），因此各自区域分别
着色不会互相覆盖。

运行方式
--------
需在项目根目录（d:\\项目1）下运行。单例模式（一对图像/标签/预测）：

    python src/visualize.py --image_path data/Task03_Liver/imagesTr/liver_0.nii.gz --label_path data/Task03_Liver/labelsTr/liver_0.nii.gz --pred_path outputs/predictions/liver_0.nii.gz --output_dir outputs/figures

只画 CT + 预测（无真值）：

    python src/visualize.py --image_path data/Task03_Liver/imagesTr/liver_0.nii.gz --pred_path outputs/predictions/liver_0.nii.gz --output_dir outputs/figures

批量模式（按同名文件词干配对，逐例处理）：

    python src/visualize.py --image_dir data/Task03_Liver/imagesTr --label_dir data/Task03_Liver/labelsTr --pred_dir outputs/predictions --output_dir outputs/figures

输入
----
- --image_path  ：CT 图像 NIfTI（.nii / .nii.gz），体素为 HU 值（单例必填）。
- --label_path  ：（可选）真值标签 NIfTI，体素为类别 id（0/1/2）。
- --pred_path   ：（可选）预测 NIfTI，体素为类别 id（0/1/2）。
- --output_dir  ：输出目录（必填），PNG 默认写入其下的 figures/ 子目录。
- --num_slices  ：每例选取的轴向切片数，默认 4。
- --image_dir / --label_dir / --pred_dir ：批量模式目录，按同名词干配对，
  与单例参数互斥（--batch_mode 可选标志，开启后强制批量模式）。

输出
----
- 每例每切片一张组合 PNG，命名 <case>_slice<s>.png，默认写入 <output_dir>/figures/。

常见错误
--------
1. 文件不存在：检查 --image_path / --label_path / --pred_path 路径是否正确。
2. label/pred 缺失：若某例的 label 或 pred 文件缺失，会跳过对应子图（GT overlay
   或 Pred overlay 或 GT vs Pred 对比图），其余子图照常绘制。
3. 切片越界：num_slices 大于实际切片数时，脚本会按实际切片数截断，避免越界。
4. 批量配对失败：批量模式下若某词干在 label_dir / pred_dir 找不到同名文件，
   会打印警告并跳过该对应输入，而非中断整个流程。
"""

import os
import argparse
import sys
import glob

import numpy as np
import nibabel as nib

# matplotlib 必须在 import pyplot 之前切换到 Agg 后端：Agg 是「无显示」后端，
# 只把图渲染到文件、不打开 GUI 窗口，从而在远程服务器 / 纯命令行环境不会报错。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  （上一行切换后端，本行才能 import pyplot）

# 中文字体：默认 DejaVu Sans 不含中文 glyph，标题里的「红/绿/原色/冲突」等会
# 渲染成方框。优先用 Windows 自带的 Microsoft YaHei，其次 SimHei；都不在则回退
# 默认。axes.unicode_minus=False 避免 minus sign 也变方框。
import matplotlib.font_manager as _fm
_zh_fonts = [f.name for f in _fm.fontManager.ttflist]
for _zh in ("Microsoft YaHei", "SimHei", "Microsoft JhengHei", "SimSun", "KaiTi"):
    if _zh in _zh_fonts:
        plt.rcParams["font.sans-serif"] = [_zh, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

# 把项目根目录加入 sys.path，使得直接 `python src/visualize.py ...` 运行时
# 可 import src.*（否则 Python 只把 src/ 当搜索路径，找不到 src 包）。sys/os 已导入。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# CT 肝脏软组织窗：HU 范围 [-200, 250]。肝脏实质 HU 约 60，落在该窗中部；
# 超出范围的极亮（骨骼/造影剂）与极暗（空气）被截断，便于观察肝脏与病灶。
CT_WINDOW_MIN = -200
CT_WINDOW_MAX = 250

# 叠加透明度：alpha 越大颜色越浓；0.5 兼顾看到底层 CT 与颜色辨识度。
OVERLAY_ALPHA = 0.5

# 各类别 RGB 颜色（取值 0~1，便于 matplotlib 叠加）。与全项目配色保持一致：
# 0 背景、1 肝脏 liver 红色、2 肿瘤 tumor 绿色。
CLASS_COLORS = {
    0: (0.0, 0.0, 0.0),
    1: (1.0, 0.0, 0.0),
    2: (0.0, 1.0, 0.0),
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。``--help`` 由 argparse 自动提供，无需手动定义。"""
    parser = argparse.ArgumentParser(
        description=(
            "绘制 CT / 真值 / 预测的轴向切片对比 PNG。"
            "liver 红色、tumor 绿色叠加，自动选取含肿瘤（tumor 优先）的若干切片。"
            "支持单例模式与批量模式（按同名词干配对）。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # —— 单例模式参数 ——
    parser.add_argument(
        "--image_path",
        default=None,
        help="CT 图像 NIfTI 路径（.nii / .nii.gz），单例模式必填。",
    )
    parser.add_argument(
        "--label_path",
        default=None,
        help="真值标签 NIfTI 路径（可选）；不提供则不画 Ground Truth 相关子图。",
    )
    parser.add_argument(
        "--pred_path",
        default=None,
        help="预测 NIfTI 路径（可选）；不提供则不画 Prediction 相关子图。",
    )
    # —— 输出与切片数 ——
    parser.add_argument(
        "--output_dir",
        required=True,
        help="输出目录（必填）；PNG 默认写入其下的 figures/ 子目录。",
    )
    parser.add_argument(
        "--num_slices",
        type=int,
        default=4,
        help="每例选取的轴向切片数（按 tumor 面积优先排序取前 N 张）。",
    )
    # —— 批量模式参数 ——
    parser.add_argument(
        "--image_dir",
        default=None,
        help="批量模式：CT 图像所在目录，扫描 .nii / .nii.gz 文件。",
    )
    parser.add_argument(
        "--label_dir",
        default=None,
        help="批量模式：真值标签所在目录，按同名词干与图像配对。",
    )
    parser.add_argument(
        "--pred_dir",
        default=None,
        help="批量模式：预测所在目录，按同名词干与图像配对。",
    )
    parser.add_argument(
        "--batch_mode",
        action="store_true",
        help="批量模式开关；--image_dir 存在时默认即批量模式，此标志可显式开启。",
    )
    return parser.parse_args()


def _load_nifti(path: str) -> np.ndarray:
    """用 nibabel 读 NIfTI，返回体素数组（保留磁盘原始 dtype 概念但转 float32 用于图像）。

    本函数封装 nibabel 加载，便于统一错误处理。注意：本脚本对图像会再
    ``astype(np.float32)``，对标签/预测保留整数类型以供类别判断。

    Args:
        path: .nii / .nii.gz 文件路径。

    Returns:
        np.ndarray: 3D 体素数组（保留原始 dtype）。
    """
    img = nib.load(path)                  # 懒加载 NIfTI 图像对象
    data = np.asanyarray(img.dataobj)     # 按磁盘原始 dtype 读取体素
    return data


def _stem(path: str) -> str:
    """取文件词干：去掉 .nii.gz（双扩展名）或 .nii 后缀，仅保留文件主名。"""
    base = os.path.basename(path)
    if base.endswith(".nii.gz"):
        return base[: -len(".nii.gz")]
    if base.endswith(".nii"):
        return base[: -len(".nii")]
    return base


def _find_paired(base_dir: str, stem: str) -> str:
    """在目录中按词干查找配对的 NIfTI 文件，返回第一个匹配路径，找不到返回 None。

    支持两种扩展名优先级：.nii.gz 优先于 .nii。
    """
    if base_dir is None:
        return None
    for ext in (".nii.gz", ".nii"):
        cand = os.path.join(base_dir, stem + ext)
        if os.path.isfile(cand):
            return cand
    return None


def select_slices(label: np.ndarray = None, pred: np.ndarray = None,
                  num_slices: int = 4, total_slices: int = None) -> list:
    """自动选取含 liver/tumor 面积大的若干轴向切片（tumor 优先）。

    选择策略（tumor 优先）：
        1) 若 label 或 pred 中存在类别 2（肿瘤）：沿 axis=2 统计每张切片中
           tumor 像素数，按面积从大到小排序，取前 num_slices 个切片索引。
        2) 若无肿瘤但存在前景（label>0 或 pred>0）：按前景面积排序取前 N。
        3) 若 label 与 pred 都为空或无任何前景：在 [0, D-1] 内均匀取 num_slices 个切片。

    Args:
        label: 3D 真值标签（可选，整数）。
        pred:  3D 预测（可选，整数）。
        num_slices: 要取的切片数。
        total_slices: 轴向方向切片总数 D，用于均匀采样回退；若 None 则从 label/pred 推断。

    Returns:
        list[int]: 选中的轴向切片索引列表（未排序的具体顺序由面积排定）。
    """
    # 优先肿瘤面积统计：合并 label==2 与 pred==2（二者取并集，使含肿瘤的切片均被考虑）
    tumor_mask = None
    if label is not None and (label == 2).any():
        tumor_mask = (label == 2)
        if pred is not None and (pred == 2).any():
            tumor_mask = tumor_mask | (pred == 2)
    elif pred is not None and (pred == 2).any():
        tumor_mask = (pred == 2)

    if tumor_mask is not None:
        # 沿前两轴 (H, W) 求和，得到每张切片的肿瘤像素数，形状 (D,)
        counts = tumor_mask.sum(axis=(0, 1))
        order = np.argsort(-counts)  # 降序排序的索引
        chosen = [int(z) for z in order[:num_slices] if counts[z] > 0]
        if chosen:
            return chosen
        # tumor 全为 0 的极端情况，落入下面的前景回退

    # 无肿瘤：按前景（label>0 或 pred>0）面积排序
    fg = None
    if label is not None:
        fg = (label > 0)
        if pred is not None:
            fg = fg | (pred > 0)
    elif pred is not None:
        fg = (pred > 0)

    if fg is not None:
        counts = fg.sum(axis=(0, 1))
        if counts.max() > 0:
            order = np.argsort(-counts)
            return [int(z) for z in order[:num_slices] if counts[z] > 0]

    # 前景也全空：均匀采样回退
    D = total_slices
    if D is None:
        if label is not None:
            D = label.shape[2]
        elif pred is not None:
            D = pred.shape[2]
        else:
            D = 1
    n = min(num_slices, D)
    return [int(round(z)) for z in np.linspace(0, D - 1, n)]


def overlay_multiclass(image_slice: np.ndarray, mask_slice: np.ndarray,
                       classes_to_draw: list = (1, 2)) -> np.ndarray:
    """在灰度 CT 切片上叠加多类掩膜（liver 红、tumor 绿），返回 RGB 图。

    实现思路：先把灰度切片做肝窗裁剪并归一化到 [0,1]；扩展为 3 通道灰度 RGB；
    再对每个类别在其二值区域内按 alpha 混合该类颜色。由于肝脏(1)与肿瘤(2)在
    标签中互斥，逐类区域写入不会互相覆盖。

    Args:
        image_slice: 2D 灰度 CT 切片（HU 值，范围任意）。
        mask_slice:  2D 多类标签切片（0=背景, 1=肝脏, 2=肿瘤）。
        classes_to_draw: 要叠加的类别 id 列表，默认 (1, 2)。

    Returns:
        np.ndarray: 形状 (H, W, 3) 的 RGB float32，范围 0~1。
    """
    # 1) 肝窗裁剪：把 HU 限制在 [CT_WINDOW_MIN, CT_WINDOW_MAX]，超出范围截断
    img = image_slice.astype(np.float32)
    img = np.clip(img, CT_WINDOW_MIN, CT_WINDOW_MAX)
    # 2) 线性归一化到 [0,1]，便于与颜色叠加
    if CT_WINDOW_MAX - CT_WINDOW_MIN > 1e-8:
        img = (img - CT_WINDOW_MIN) / (CT_WINDOW_MAX - CT_WINDOW_MIN)
    else:
        img = np.zeros_like(img)
    # 3) 扩展为 3 通道灰度 RGB
    rgb = np.stack([img, img, img], axis=-1)  # (H, W, 3)

    # 4) 逐类叠加：互斥类别可直接区域写入，不会互相覆盖
    for cls in classes_to_draw:
        region = (mask_slice == cls)
        if not region.any():
            continue
        color_arr = np.array(CLASS_COLORS[cls], dtype=np.float32)  # (3,)
        blend = (OVERLAY_ALPHA * region.astype(np.float32))[..., None]  # (H, W, 1)
        rgb = rgb * (1.0 - blend) + color_arr * blend
    return rgb.astype(np.float32)


def _opaque_gray(image_slice: np.ndarray) -> np.ndarray:
    """把灰度 CT 切片做肝窗归一化后扩展为 3 通道 RGB，用于原始 CT 子图显示。"""
    img = image_slice.astype(np.float32)
    img = np.clip(img, CT_WINDOW_MIN, CT_WINDOW_MAX)
    if CT_WINDOW_MAX - CT_WINDOW_MIN > 1e-8:
        img = (img - CT_WINDOW_MIN) / (CT_WINDOW_MAX - CT_WINDOW_MIN)
    return np.stack([img, img, img], axis=-1).astype(np.float32)


def _difference_rgb(gt_slice: np.ndarray, pred_slice: np.ndarray,
                    image_slice: np.ndarray = None) -> np.ndarray:
    """构造「真值 vs 预测」的差异对比图（RGB），用于第 4 子图。

    着色约定（便于初学者快速辨识对错）：
        - TP（真值与预测均为前景，且类别一致）：用该类别原色（liver 红 / tumor 绿）；
        - FP（预测有、真值无）：青色 (0,1,1)，表示「多画了」；
        - FN（真值有、预测无）：橙色 (1,0.5,0)，表示「漏画了」；
        - 类别冲突（真值类别≠预测类别）：紫色 (1,0,1)，醒目提示错分；
        - 背景一致（均为 0）：显示底层灰度 CT。

    若未提供 image_slice，背景一致区域填黑。

    Args:
        gt_slice:   2D 真值标签切片。
        pred_slice: 2D 预测切片。
        image_slice: 2D 灰度 CT 切片（可选），用作背景一致的灰度底。

    Returns:
        np.ndarray: (H, W, 3) RGB float32，范围 0~1。
    """
    H, W = gt_slice.shape
    if image_slice is not None:
        rgb = _opaque_gray(image_slice)
    else:
        rgb = np.zeros((H, W, 3), dtype=np.float32)

    TP_COLOR = np.array([0.0, 1.0, 1.0])   # FP 青色
    FN_CARD = np.array([1.0, 0.5, 0.0])    # FN 橙色
    CONFLICT = np.array([1.0, 0.0, 1.0])   # 类别冲突紫色

    # 真值前景区域
    gt_fg = (gt_slice > 0)
    pred_fg = (pred_slice > 0)

    # TP：两者都为前景且类别相同 → 用对应类别颜色
    same_class = (gt_slice == pred_slice) & gt_fg & pred_fg
    for cls in (1, 2):
        seg = same_class & (gt_slice == cls)
        if seg.any():
            rgb[seg] = np.array(CLASS_COLORS[cls], dtype=np.float32)

    # 类别冲突：都为前景但类别不同
    conflict = gt_fg & pred_fg & ~same_class
    if conflict.any():
        rgb[conflict] = CONFLICT

    # FP：预测有真值无
    fp = pred_fg & ~gt_fg
    if fp.any():
        rgb[fp] = TP_COLOR  # 青色表示「多画了」

    # FN：真值有预测无
    fn = gt_fg & ~pred_fg
    if fn.any():
        rgb[fn] = FN_CARD  # 橙色表示「漏画了」

    return rgb


def visualize_case(image_path: str, label_path: str, pred_path: str,
                   output_dir: str, num_slices: int) -> list:
    """对单个病例完成「加载→选切片→构图→保存」流程，返回生成的 PNG 路径列表。

    Args:
        image_path: CT 图像 NIfTI 路径（必填，缺失则报错返回 []）。
        label_path: 真值标签路径（可选，可传 None）。
        pred_path:  预测路径（可选，可传 None）。
        output_dir: 输出根目录；PNG 实际写入其下 figures/ 子目录。
        num_slices: 每例要画的切片数。

    Returns:
        list[str]: 本次生成的 PNG 路径列表；若图像缺失则为空列表。
    """
    # —— 1) 校验图像存在 ——
    if image_path is None or not os.path.isfile(image_path):
        print(f"[错误] CT 图像文件不存在：{image_path}\n请检查路径是否正确。")
        return []

    case = _stem(image_path)  # 病例名（文件词干），用于标题与输出命名

    # —— 2) 加载 CT 图像（float）——
    image = _load_nifti(image_path).astype(np.float32)

    # —— 3) 可选加载真值 / 预测，缺失则跳过对应子图 ——
    label = None
    if label_path is not None and os.path.isfile(label_path):
        label = _load_nifti(label_path)
    elif label_path is not None:
        print(f"[警告] 病例 {case} 的真值标签缺失，跳过 GT 子图：{label_path}")

    pred = None
    if pred_path is not None and os.path.isfile(pred_path):
        pred = _load_nifti(pred_path)
    elif pred_path is not None:
        print(f"[警告] 病例 {case} 的预测缺失，跳过 Pred 子图：{pred_path}")

    # —— 4) 形状一致性简单校验（不一致会给出警告，但尽量继续）——
    D = image.shape[2]
    if label is not None and label.shape != image.shape:
        print(f"[警告] 病例 {case} 的 label 形状 {label.shape} 与 image {image.shape} 不一致，"
              "可能配对错误；将按 image 的轴向切片索引截取。")
    if pred is not None and pred.shape != image.shape:
        print(f"[警告] 病例 {case} 的 pred 形状 {pred.shape} 与 image {image.shape} 不一致，"
              "可能配对错误；将按 image 的轴向切片索引截取。")

    # —— 5) 自动选切片（tumor 优先）——
    slices = select_slices(label, pred, num_slices=num_slices, total_slices=D)
    # 防越界：把超出 [0, D-1] 的索引截掉
    slices = [z for z in slices if 0 <= z < D]
    if not slices:
        slices = [D // 2]  # 极端回退：只画中间切片

    # —— 6) 确定输出目录与待画类别 ——
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    classes_to_draw = [1, 2]  # 默认 liver + tumor；无 tumor 的切片自动只画 liver

    out_paths = []
    for z in slices:
        # 取该切片的 2D 数据
        image_slice = image[:, :, z]
        label_slice = label[:, :, z] if label is not None else None
        pred_slice = pred[:, :, z] if pred is not None else None

        # —— 构图：根据可用输入决定子图数量（最多 4 类图）——
        have_gt = label_slice is not None
        have_pred = pred_slice is not None
        # 子图：CT(必) + GT(可选) + Pred(可选) + GTvsPred(需两者都有)
        n = 1 + (1 if have_gt else 0) + (1 if have_pred else 0) + \
            (1 if (have_gt and have_pred) else 0)
        n = max(n, 1)

        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        axes = np.atleast_1d(axes).tolist()
        pos = 0

        # 子图1：原始 CT（灰度）
        axes[pos].imshow(_opaque_gray(image_slice))
        axes[pos].set_title(f"{case}\nCT  slice={z}")
        axes[pos].axis("off")
        pos += 1

        # 子图2：CT + Ground Truth overlay
        if have_gt:
            rgb_gt = overlay_multiclass(image_slice, label_slice, classes_to_draw)
            axes[pos].imshow(rgb_gt)
            axes[pos].set_title(f"{case}\nCT + GT  slice={z}\n(label: liver红/tumor绿)")
            axes[pos].axis("off")
            pos += 1

        # 子图3：CT + Prediction overlay
        if have_pred:
            rgb_pred = overlay_multiclass(image_slice, pred_slice, classes_to_draw)
            axes[pos].imshow(rgb_pred)
            axes[pos].set_title(f"{case}\nCT + Pred  slice={z}\n(pred: liver红/tumor绿)")
            axes[pos].axis("off")
            pos += 1

        # 子图4：GT vs Pred 对比（差异图）
        if have_gt and have_pred:
            diff_rgb = _difference_rgb(label_slice, pred_slice, image_slice)
            axes[pos].imshow(diff_rgb)
            axes[pos].set_title(
                f"{case}\nGT vs Pred  slice={z}\n"
                "(TP原色 / FP青 / FN橙 / 冲突紫)"
            )
            axes[pos].axis("off")
            pos += 1

        plt.tight_layout()
        out_path = os.path.join(fig_dir, f"{case}_slice{z}.png")
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)  # 关闭图对象，释放内存，避免多次调用时图叠加
        out_paths.append(out_path)

    print(f"[完成] 病例 {case}：生成 {len(out_paths)} 张 PNG -> {fig_dir}")
    return out_paths


def _collect_batch(args: argparse.Namespace) -> list:
    """批量模式：扫描 image_dir 下的 NIfTI，并按同名词干在 label_dir / pred_dir 配对。

    Returns:
        list[tuple]: 每项为 (image_path, label_path, pred_path)；未找到则对应项为 None。
    """
    image_files = []
    for ext in ("*.nii.gz", "*.nii"):
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext)))
    # 去重（.nii 与 .nii.gz 同名时各扫一次会重复）
    seen = set()
    unique_images = []
    for p in image_files:
        if p not in seen:
            seen.add(p)
            unique_images.append(p)
    unique_images.sort()

    cases = []
    for img in unique_images:
        stem = _stem(img)
        lab = _find_paired(args.label_dir, stem) if args.label_dir else None
        prd = _find_paired(args.pred_dir, stem) if args.pred_dir else None
        cases.append((img, lab, prd))
    return cases


def main() -> None:
    """脚本主流程：解析参数 → 判别单例/批量模式 → 逐例可视化 → 汇总输出。"""
    args = parse_args()

    # —— 确保输出目录存在 ——
    os.makedirs(args.output_dir, exist_ok=True)

    # —— 判别模式：批量 vs 单例（互斥）——
    batch_trigger = args.image_dir is not None or args.batch_mode
    single_trigger = args.image_path is not None

    if batch_trigger and single_trigger:
        print("[错误] 批量模式（--image_dir/--batch_mode）与单例模式（--image_path）"
              "互斥，请二选一。")
        sys.exit(1)

    if not batch_trigger and not single_trigger:
        print("[错误] 请提供 --image_path（单例）或 --image_dir（批量）之一。")
        sys.exit(1)

    if batch_trigger:
        if args.image_dir is None:
            print("[错误] 批量模式必须提供 --image_dir。")
            sys.exit(1)
        if not os.path.isdir(args.image_dir):
            print(f"[错误] --image_dir 目录不存在：{args.image_dir}")
            sys.exit(1)
        cases = _collect_batch(args)
        if not cases:
            print(f"[错误] 在 {args.image_dir} 下未找到任何 .nii / .nii.gz 文件。")
            sys.exit(1)
        print(f"[批量] 共发现 {len(cases)} 个病例，开始逐例可视化 ...")
        total_png = 0
        for img, lab, prd in cases:
            outs = visualize_case(img, lab, prd, args.output_dir, args.num_slices)
            total_png += len(outs)
        print("=" * 60)
        print(f"批量完成：共 {len(cases)} 例，累计生成 {total_png} 张 PNG")
        print(f"输出目录：{os.path.join(args.output_dir, 'figures')}")
        print("=" * 60)
    else:
        # 单例模式
        outs = visualize_case(args.image_path, args.label_path,
                              args.pred_path, args.output_dir, args.num_slices)
        print("=" * 60)
        print(f"单例完成：共生成 {len(outs)} 张 PNG")
        print(f"输出目录：{os.path.join(args.output_dir, 'figures')}")
        print("=" * 60)


if __name__ == "__main__":
    main()
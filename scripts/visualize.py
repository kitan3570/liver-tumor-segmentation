"""scripts/visualize.py — 绘制 CT / 真值标签 / 预测标签的轴向切片对比图（PNG）

作用
----
读取一张 CT 图像（NIfTI）与其对应的真值标签、模型预测（均为 NIfTI，可选），
在「轴向切片 (axial slice)」上把三者并排画成一张 PNG：
  - 子图1：CT 灰度图（肝窗 [-100, 400]，让肝脏实质落在可见范围中段）；
  - 子图2（若提供 --label）：CT 底图 + 真值标签叠加（liver 红、tumor 绿）；
  - 子图3（若提供 --prediction）：CT 底图 + 预测标签叠加（liver 红、tumor 绿）。
帮助大一同学直观对比「模型预测」与「人工标注」的差异，建立对 3D 分割输出
形态的直观认识。

轴向切片说明
------------
医学 CT 沿 Z 轴逐层扫描，每一层叫一个「轴向切片 (axial slice)」。本项目约定
3D 体素数组形状为 (H, W, D)，第 3 个维度（axis=2）为轴向方向，故
``image[:, :, z]`` 即第 z 张轴向切片。本脚本固定取 axis=2 切片进行可视化。

颜色叠加说明
------------
- liver（类别 1）用红色 (1,0,0) 叠加；
- tumor（类别 2）用绿色 (0,1,0) 叠加；
- 背景（类别 0）不叠加。
颜色由 ``src.utils.get_class_colors()`` 统一提供，保证全项目配色一致。
叠加由 ``src.utils.overlay_mask`` 完成：它把单类二值掩膜按 alpha 混合到灰度底
上。由于肝脏与肿瘤在标签中互斥（同一体素不会既是肝脏又是肿瘤），本脚本对
两类分别调用 overlay_mask，再把各自区域写回同一张灰度底图。

运行方式
--------
需在项目根目录（d:\\项目1）下运行，以便 ``from src.utils import ...`` 可用。
完整对比（CT + 真值 + 预测）：

    python scripts/visualize.py --image data/Task03_Liver/imagesTr/liver_0.nii.gz --label data/Task03_Liver/labelsTr/liver_0.nii.gz --prediction outputs/predictions/liver_0.nii.gz --output outputs/figures/liver_0.png

只画 CT + 预测（无真值）：

    python scripts/visualize.py --image data/Task03_Liver/imagesTr/liver_0.nii.gz --prediction outputs/predictions/liver_0.nii.gz

只画 CT（仅查看图像，自动取中间切片）：

    python scripts/visualize.py --image data/Task03_Liver/imagesTr/liver_0.nii.gz

输入
----
- --image      ：CT 图像 NIfTI（.nii / .nii.gz），体素为 HU 值。
- --label      ：（可选）真值标签 NIfTI，体素为类别 id（0/1/2）。
- --prediction ：（可选）预测 NIfTI，体素为类别 id（0/1/2）。

输出
----
- PNG 图片，默认写入 outputs/figures/<图像文件词干>.png。

常见错误
--------
1. 文件不存在：检查 --image / --label / --prediction 路径是否正确。
2. matplotlib 在无显示环境（远程服务器 / 纯命令行）下报 ``no display`` 错误：
   本脚本已在 import pyplot 之前切换到 Agg 后端，仅渲染到文件、不打开 GUI
   窗口，故无需图形界面即可保存 PNG。
3. --slice 越界：轴向切片索引必须在 [0, D-1]（D = image.shape[2]）；若越界
   脚本会提示并退出。不指定 --slice 时，脚本会基于 --label 自动选择含病灶
   最多的一层，若 --label 也未提供则取中间切片。
"""

import argparse
import os
import sys

import numpy as np

# matplotlib 必须在 import pyplot 之前切换到 Agg 后端：Agg 是「无显示」后端，
# 只把图渲染到文件、不尝试打开 GUI 窗口，从而在远程服务器 / 纯命令行环境不会报错。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  （上一行切换后端，本行才能 import pyplot）

# 项目内工具：
#   load_nifti          读 NIfTI -> (data, affine)
#   ensure_dir          确保目录存在
#   get_class_colors    取类别配色 {0:黑, 1:红 liver, 2:绿 tumor}
#   pick_slice_with_lesion  自动选含病灶最多的轴向切片（axis=2）
#   overlay_mask        把单类二值掩膜按 alpha 混合到灰度切片，返回 RGB
#   load_config         读 YAML 配置（用于 --config 取 num_classes）
# 把项目根目录加入 sys.path，使得直接 `python scripts/visualize.py ...` 运行时可
# import src.*（否则 Python 只把 scripts/ 当搜索路径，找不到 src 包）。sys/os 已导入。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    load_nifti,
    ensure_dir,
    get_class_colors,
    pick_slice_with_lesion,
    overlay_mask,
    load_config,
)


# CT 肝脏软组织窗：HU 范围 [-100, 400]。肝脏实质 HU 约 60，落在该窗中部；
# 超出范围的极亮（骨骼/造影剂）与极暗（空气）被截断，便于观察肝脏与病灶。
CT_WINDOW_MIN = -100
CT_WINDOW_MAX = 400

# 叠加透明度：alpha 越大颜色越浓；0.5 兼顾看到底层 CT 与颜色辨识度。
OVERLAY_ALPHA = 0.5


def parse_args() -> argparse.Namespace:
    """解析命令行参数。``--help`` 由 argparse 自动提供，无需手动定义。"""
    parser = argparse.ArgumentParser(
        description=(
            "绘制 CT / 真值 / 预测的轴向切片对比 PNG。"
            "liver 红色、tumor 绿色叠加，默认自动选取含病灶切片。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        required=True,
        help="CT 图像 NIfTI 路径（.nii / .nii.gz），必填。",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="真值标签 NIfTI 路径（可选）；不提供则不画 Ground Truth 子图。",
    )
    parser.add_argument(
        "--prediction",
        default=None,
        help="预测 NIfTI 路径（可选）；不提供则不画 Prediction 子图。",
    )
    parser.add_argument(
        "--slice",
        type=int,
        default=None,
        help="指定轴向切片索引；若为 None 则自动选择含病灶切片（基于 --label），"
             "若 --label 也未提供则取中间切片 image.shape[2]//2。",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出 PNG 路径；若为 None 则默认 outputs/figures/<image词干>.png。",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="可选 YAML 配置路径，用于读取 num_classes 决定颜色映射；"
             "若为 None 则按标签中是否出现类别 2 自动判断（出现→3 类，否则→2 类）。",
    )
    return parser.parse_args()


def overlay_multiclass(image_slice: np.ndarray, mask_slice: np.ndarray,
                       classes_to_draw: list) -> np.ndarray:
    """在灰度 CT 切片上叠加多类掩膜（liver 红、tumor 绿），返回 RGB 图。

    实现思路
    --------
    ``overlay_mask`` 一次只能叠加「单类」掩膜，且内部会把灰度图线性归一化到
    [0,1]（不做 CT 窗裁剪）。由于肝脏(1)与肿瘤(2)在标签中互斥（同一体素不会
    既是肝脏又是肿瘤），我们采用「同一灰度底 + 互斥区域写入」的复合策略：
      1) 先用「全零掩膜 + alpha=0」调用一次 overlay_mask，得到归一化后的纯
         灰度 RGB 底图——alpha=0 时混合项为 0，结果即 overlay_mask 内部归一
         化的灰度 RGB，不引入任何颜色；
      2) 对每个待画类别，用该类的二值掩膜再调用一次 overlay_mask（传入同一
         image_slice，故归一化方式一致、灰度底相同），把对应区域的颜色叠加
         结果写回底图。
    这样每个体素最终只呈现其所属类别的颜色，未标注区域保持灰度。

    Args:
        image_slice: 2D 灰度 CT 切片（HU 值，范围任意）。
        mask_slice:  2D 多类标签切片（0=背景, 1=肝脏, 2=肿瘤）。
        classes_to_draw: 要叠加的类别 id 列表，例如 [1] 或 [1, 2]。

    Returns:
        np.ndarray: 形状 (H, W, 3) 的 RGB float32，范围 0~1。
    """
    colors = get_class_colors()
    # 1) 取归一化灰度 RGB 底图：全零掩膜 + alpha=0 → 不混合任何颜色，仅返回
    #    overlay_mask 内部归一化后的纯灰度底（颜色参数此时不起作用，任取一类即可）。
    base = overlay_mask(image_slice, np.zeros_like(mask_slice),
                        colors[1], alpha=0.0)
    # 2) 逐类叠加：同一 image_slice 保证灰度底一致；互斥类别可直接区域写入，
    #    不会出现两种颜色互相覆盖的问题。
    for cls in classes_to_draw:
        region = (mask_slice == cls)  # 当前类的二值掩膜
        if not region.any():
            continue  # 该切片无此类，跳过
        overlayed = overlay_mask(image_slice, region, colors[cls],
                                 alpha=OVERLAY_ALPHA)
        base[region] = overlayed[region]  # 把该类区域替换为带颜色叠加的结果
    return base


def main() -> None:
    """脚本主流程：解析参数 → 校验文件 → 加载 NIfTI → 选切片 → 构图 → 保存 PNG。"""
    args = parse_args()

    # —— 1) 校验 --image 必须存在 ——
    if not os.path.isfile(args.image):
        print(f"[错误] CT 图像文件不存在：{args.image}\n"
              "请检查 --image 路径是否正确。")
        sys.exit(1)

    # —— 2) 加载 CT 图像 ——
    # load_nifti 返回 (data, affine)；可视化只需 data。转 float32 便于归一化。
    image, _ = load_nifti(args.image)
    image = image.astype(np.float32)

    # —— 3) 可选加载真值标签与预测 ——
    label = None
    if args.label is not None:
        if not os.path.isfile(args.label):
            print(f"[错误] 真值标签文件不存在：{args.label}")
            sys.exit(1)
        label, _ = load_nifti(args.label)  # 标签保留整数类型即可

    prediction = None
    if args.prediction is not None:
        if not os.path.isfile(args.prediction):
            print(f"[错误] 预测文件不存在：{args.prediction}")
            sys.exit(1)
        prediction, _ = load_nifti(args.prediction)

    # —— 4) 确定 num_classes（决定是否画 tumor 绿色）——
    # 优先级：--config > --label 含类别 2 > --prediction 含类别 2 > 默认 2。
    # 二分类(num_classes=2) 只画 liver；三分类(num_classes=3) 画 liver + tumor。
    if args.config is not None:
        if not os.path.isfile(args.config):
            print(f"[错误] 配置文件不存在：{args.config}")
            sys.exit(1)
        config = load_config(args.config)  # 读 YAML 顶层字段，含 num_classes
        num_classes = int(config.get("num_classes", 2))
    elif label is not None and (label == 2).any():
        num_classes = 3  # 真值里出现肿瘤 → 三分类
    elif prediction is not None and (prediction == 2).any():
        num_classes = 3  # 无真值但预测含肿瘤 → 三分类
    else:
        num_classes = 2  # 仅肝脏二分类

    # 待叠加的类别列表：num_classes=2 → [1]；num_classes=3 → [1, 2]。
    classes_to_draw = [c for c in (1, 2) if c < num_classes]

    # —— 5) 选择轴向切片索引 ——
    # 优先用 --slice；其次基于 --label 自动选含病灶层；再次取中间层。
    if args.slice is not None:
        slice_idx = args.slice
    elif label is not None:
        # pick_slice_with_lesion 沿 axis=2 统计前景像素，返回病灶最多的切片索引
        slice_idx = pick_slice_with_lesion(label, num_classes)
    else:
        slice_idx = image.shape[2] // 2  # 无标签时取中间切片

    # —— 6) 校验切片索引未越界 ——
    # axial 方向为 axis=2，合法范围 [0, D-1]，D = image.shape[2]。
    if not (0 <= slice_idx < image.shape[2]):
        print(f"[错误] 切片索引越界：slice={slice_idx}，"
              f"合法范围为 [0, {image.shape[2] - 1}]。\n"
              "请调整 --slice，或不指定以让脚本自动选择。")
        sys.exit(1)

    # —— 7) 取 2D 轴向切片：image[:, :, z]（axis=2 为轴向）——
    image_slice = image[:, :, slice_idx]
    label_slice = label[:, :, slice_idx] if label is not None else None
    pred_slice = prediction[:, :, slice_idx] if prediction is not None else None

    # —— 8) 确定输出路径 ——
    if args.output is None:
        stem = os.path.basename(args.image)
        # 去掉 .nii.gz（双扩展名）或 .nii 后缀，得到文件词干
        if stem.endswith(".nii.gz"):
            stem = stem[: -len(".nii.gz")]
        elif stem.endswith(".nii"):
            stem = stem[: -len(".nii")]
        output_path = os.path.join("outputs", "figures", f"{stem}.png")
    else:
        output_path = args.output
    ensure_dir(os.path.dirname(os.path.abspath(output_path)))

    # —— 9) 构图：1~3 个并排子图（CT / Ground Truth / Prediction）——
    # 子图数量 = 1(CT) + 有无 label + 有无 prediction。
    n_subplots = 1 + (1 if label_slice is not None else 0) + \
        (1 if pred_slice is not None else 0)
    fig, axes = plt.subplots(1, n_subplots, figsize=(5 * n_subplots, 5))
    # subplots 在 n=1 时返回单个 Axes，统一成列表便于按下标访问。
    axes = np.atleast_1d(axes).tolist()

    # 子图1：CT 灰度。用肝窗 [-100,400] 设 vmin/vmax，让肝脏实质落在可见范围中段；
    # 注意 overlay 子图由 overlay_mask 内部按全图 min/max 归一化，二者灰度底可能略有差异。
    axes[0].imshow(image_slice, cmap="gray",
                   vmin=CT_WINDOW_MIN, vmax=CT_WINDOW_MAX)
    axes[0].set_title("CT")
    axes[0].axis("off")

    # 子图2：真值标签叠加（若提供 --label）
    pos = 1
    if label_slice is not None:
        rgb_gt = overlay_multiclass(image_slice, label_slice, classes_to_draw)
        axes[pos].imshow(rgb_gt)
        axes[pos].set_title("Ground Truth")
        axes[pos].axis("off")
        pos += 1

    # 子图3：预测标签叠加（若提供 --prediction）
    if pred_slice is not None:
        rgb_pred = overlay_multiclass(image_slice, pred_slice, classes_to_draw)
        axes[pos].imshow(rgb_pred)
        axes[pos].set_title("Prediction")
        axes[pos].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)  # 关闭图对象，释放内存，避免多次调用时图叠加

    # —— 10) 打印关键信息，让同学清楚脚本做了什么、产物在哪 ——
    print("=" * 60)
    print(f"CT 图像    : {args.image}")
    print(f"真值标签   : {args.label if label is not None else '（无）'}")
    print(f"预测       : {args.prediction if prediction is not None else '（无）'}")
    print(f"轴向切片   : {slice_idx} / {image.shape[2] - 1}")
    print(f"num_classes: {num_classes}（叠加类别 {classes_to_draw}）")
    print(f"输出 PNG   : {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

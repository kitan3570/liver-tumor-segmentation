"""scripts/evaluate.py — 肝脏 3D 分割评价脚本（Task 7 SubTask 7.6）

文件作用
========
加载「预测分割 NIfTI」与「真值分割 NIfTI」，对每个样本计算分割指标
（Dice / IoU / Precision / Recall，分 liver / tumor 两类），再汇总成验证集
级别的均值与标准差，最终输出 JSON 与 CSV 两份结果文件，并在终端打印一张
格式化的汇总表。

本脚本本身不训练、不推理，只做「读文件 → 算指标 → 写结果」三件事，适合在
scripts/infer.py 生成预测之后，独立、快速地评估模型在测试/验证集上的表现。
指标计算与汇总逻辑复用 src.trainer 的 compute_metrics / aggregate_metrics，
与训练过程中验证集用的指标完全一致，保证「训练时看的分数」与「事后评估的
分数」口径相同、可对比。

面向大一医学生的指标速览
========================
- **Dice**：衡量预测与真值的重叠程度，0~1，越大越好；等同于 F1-score。
- **IoU（交并比）**：重叠体积 / 合并体积，比 Dice 更严格，0~1，越大越好。
- **Precision（精确率）**：预测为肝脏/肿瘤的体素里，有多少真是该类；
  偏高代表「不乱报」。
- **Recall（召回率）**：真值为肝脏/肿瘤的体素里，有多少被找出来；
  偏高代表「不漏诊」。临床中漏诊（低 Recall）往往比误报更危险。

运行方式
========
典型命令（阶段一、二分类）：

    python scripts/evaluate.py --predictions outputs/predictions --labels data/Task03_Liver/labelsTr --config configs/phase1_binary.yaml

阶段二（三分类）只需把 --config 换成 phase2 配置即可：

    python scripts/evaluate.py --predictions outputs/predictions --labels data/Task03_Liver/labelsTr --config configs/phase2_ternary.yaml

不指定 --output 时，结果默认写到 config['output']['metrics_dir']
（即 outputs/metrics/），文件名为 metrics.json 与 metrics.csv。

输入
====
- --predictions：预测 NIfTI 目录（如 outputs/predictions），其下是若干
  *.nii.gz，文件名需与真值目录中的同名文件一一对应（由 scripts/infer.py
  生成预测时已保证同名）。
- --labels：真值 NIfTI 目录（如 data/Task03_Liver/labelsTr），原始标签值
  为 0=背景、1=肝脏、2=肿瘤。
- --config：YAML 配置文件，用于读取 num_classes（2=二分类，3=三分类），
  并在 --output 为空时取其中的 output.metrics_dir 作为输出目录。

输出
====
- metrics.json：结构 {'summary': 汇总dict, 'per_sample': [每样本dict, ...]}，
  含每样本的 liver_*/tumor_* 指标与汇总均值±标准差。
- metrics.csv：逐样本表格，字段 [sample, liver_dice, liver_iou,
  liver_precision, liver_recall]，三分类时追加 [tumor_dice, tumor_iou,
  tumor_precision, tumor_recall]，可用 Excel/pandas 直接打开。
- 终端打印一张格式化汇总表（liver/tumor 四项指标的均值±标准差）。

常见错误
========
1) 预测/真值不配对：predictions 目录下某 .nii.gz 在 labels 目录找不到同名
   真值 —— 脚本会打印警告并跳过该样本，不会中断整体评估。
2) num_classes 与预测类别数不符：例如用 phase1 配置（num_classes=2）去评估
   三分类预测（含类别 2 的体素），会导致「标签对齐」把真值里肿瘤并入肝脏，
   而预测仍保留肿瘤类别，两者类别空间不一致，指标无意义。请确保 --config
   与「生成这批预测时所用的配置」一致。
3) 空真值样本记 NaN：若某样本真值里根本没有肿瘤（label 中类别 2 为空），
   compute_metrics 会把该样本的 tumor_* 记为 NaN；aggregate_metrics 汇总时
   会忽略 NaN 再求均值，使肿瘤指标只反映「真正含肿瘤的样本」上的表现。
"""

# ===================== 标准库导入 =====================
import argparse
import os
import sys
import glob
import csv

# ===================== 第三方库导入 =====================
# numpy：标签对齐与类型转换；本脚本不直接用 torch/MONAI，指标全在 CPU 用 numpy 算。
import numpy as np

# 把项目根目录（本文件所在目录的上一级）加入 sys.path，使得直接运行
# `python scripts/evaluate.py ...` 时能正确 import src.* 模块。
# 说明：Python 运行脚本时只会把脚本所在目录(scripts/)加入 sys.path[0]，
# 不会自动加入项目根；此处手动补上，保证 src.utils / src.trainer 可被导入。
# （与 scripts/train.py、scripts/infer.py 的做法保持一致。）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ===================== 本项目模块导入 =====================
# load_config      : 读 YAML 配置（文件不存在时抛 FileNotFoundError 并附中文提示）
# load_nifti       : 加载 NIfTI，返回 (体素数据, 仿射矩阵)，这里只用 [0] 取体素数据
# save_json        : 把结果 dict 写成 UTF-8 JSON（内部已 ensure_dir）
# ensure_dir       : 确保输出目录存在
# compute_metrics  : 单样本指标（Dice/IoU/Precision/Recall，空真值记 NaN）
# aggregate_metrics: 多样本指标汇总（均值±标准差，忽略 NaN）
from src.utils import load_config, load_nifti, save_json, ensure_dir
from src.trainer import compute_metrics, aggregate_metrics


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    --help 由 argparse 自动提供，无需手动定义；这里只定义本项目专用参数。

    Returns:
        argparse.Namespace: 含 predictions、labels、config、output。
    """
    parser = argparse.ArgumentParser(
        description=(
            "加载预测与真值 NIfTI，计算逐样本与汇总分割指标"
            "（Dice/IoU/Precision/Recall，分 liver/tumor），输出 JSON+CSV 并打印汇总表。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # --help 自动显示默认值
    )
    parser.add_argument(
        "--predictions", required=True,
        help="预测 NIfTI 目录（如 outputs/predictions），其下是与真值同名的 .nii.gz。",
    )
    parser.add_argument(
        "--labels", required=True,
        help="真值 NIfTI 目录（如 data/Task03_Liver/labelsTr）。",
    )
    parser.add_argument(
        "--config", required=True,
        help="YAML 配置文件路径（如 configs/phase1_binary.yaml），用于读取 num_classes。",
    )
    parser.add_argument(
        "--output", default=None,
        help="结果输出目录；若为 None 则用 config['output']['metrics_dir']。",
    )
    return parser.parse_args()


def build_csv_fieldnames(num_classes: int) -> list:
    """根据类别数构造 CSV 表头字段顺序。

    二分类只记录肝脏四项；三分类追加肿瘤四项。'sample' 始终在第一列，
    便于在 Excel 中按样本名筛选/排序。

    Args:
        num_classes: 类别数（2=二分类，3=三分类）。

    Returns:
        list: 表头字段名列表。
    """
    fields = ["sample", "liver_dice", "liver_iou", "liver_precision", "liver_recall"]
    if num_classes >= 3:
        fields += ["tumor_dice", "tumor_iou", "tumor_precision", "tumor_recall"]
    return fields


def _fmt_pair(summary: dict, key: str) -> str:
    """把汇总 dict 中某指标的「均值±标准差」格式化为一个单元格字符串。

    summary 中每个原始指标键 key 对应均值，key+'_std' 对应标准差；
    若均值为 NaN（如整个集合都没有该类真值），整格直接显示 'NaN'。

    Args:
        summary: aggregate_metrics 返回的汇总 dict。
        key: 指标键名，如 'liver_dice'。

    Returns:
        str: 形如 '0.9234±0.0123' 或 'NaN'。
    """
    mean = summary.get(key, float("nan"))
    std = summary.get(f"{key}_std", float("nan"))
    # 均值或标准差为 NaN 时，无法给出有意义的「均值±标准差」，统一显示 NaN
    if (isinstance(mean, float) and np.isnan(mean)) or \
       (isinstance(std, float) and np.isnan(std)):
        return "NaN"
    return f"{mean:.4f}±{std:.4f}"


def print_summary_table(summary: dict, num_classes: int, n_samples: int) -> None:
    """在终端打印一张格式化的指标汇总表。

    表格列出 liver（及三分类时的 tumor）在 Dice/IoU/Precision/Recall 上的
    均值±标准差；空真值导致的 NaN 会在对应单元格显示 'NaN'。

    Args:
        summary: aggregate_metrics 返回的汇总 dict。
        num_classes: 类别数，决定是否打印 tumor 行。
        n_samples: 本次评估的样本总数（用于表头说明）。
    """
    # 列宽设计：类别列 16 字符 + 4 个指标列各 14 字符 = 72 字符，可容纳
    # 形如 '0.9234±0.0123'（13 字符）的「均值±标准差」单元格。
    print("=" * 72)
    print(f"分割评价指标汇总（共评估 {n_samples} 个样本）")
    print("-" * 72)
    # 表头：类别 + 四项指标，每列右对齐
    header = f"{'类别 Class':<16}{'Dice':>14}{'IoU':>14}{'Precision':>14}{'Recall':>14}"
    print(header)
    print("-" * 72)

    # 肝脏行：两个阶段都打印
    liver_cells = "".join(
        f"{_fmt_pair(summary, f'liver_{m}'):>14}" for m in ["dice", "iou", "precision", "recall"]
    )
    print(f"{'肝脏 Liver':<16}{liver_cells}")

    # 肿瘤行：仅三分类打印
    if num_classes >= 3:
        tumor_cells = "".join(
            f"{_fmt_pair(summary, f'tumor_{m}'):>14}" for m in ["dice", "iou", "precision", "recall"]
        )
        print(f"{'肿瘤 Tumor':<16}{tumor_cells}")

    print("-" * 72)
    print("说明：数值为「均值±标准差」；空真值样本已记为 NaN 并在汇总时忽略。")
    print("=" * 72)


def main() -> None:
    """脚本主入口：解析参数 → 加载配置 → 配对预测与真值 → 逐样本算指标 →
    汇总 → 输出 JSON/CSV → 打印汇总表。"""
    args = parse_args()

    # —— 1) 读取配置，确定类别数与输出目录 ——
    config = load_config(args.config)
    num_classes = config["num_classes"]

    # 输出目录：命令行 --output 优先，否则用配置中的 metrics_dir
    output_dir = args.output if args.output is not None else config["output"]["metrics_dir"]
    ensure_dir(output_dir)

    print("=" * 60)
    print("分割评估配置 Evaluate config")
    print("-" * 60)
    print(f"  predictions : {args.predictions}")
    print(f"  labels      : {args.labels}")
    print(f"  config      : {args.config}")
    print(f"  num_classes : {num_classes}（{'二分类' if num_classes == 2 else '三分类'}）")
    print(f"  output_dir  : {output_dir}")
    print("=" * 60)

    # —— 2) 扫描预测目录，按文件名在 labels 目录配对同名真值 ——
    # glob 拿到所有 *.nii.gz；sorted 让顺序确定，便于复现与排查
    pred_files = sorted(glob.glob(os.path.join(args.predictions, "*.nii.gz")))
    if not pred_files:
        print(f"[错误] 预测目录下没有 .nii.gz 文件：{args.predictions}\n"
              "请先运行 scripts/infer.py 生成预测，或检查 --predictions 路径。")
        sys.exit(1)

    # 逐样本配对：真值路径 = labels 目录 + 预测文件名；配对失败则警告并跳过
    pairs = []  # 每项 (sample_name, pred_path, label_path)
    for pred_path in pred_files:
        fname = os.path.basename(pred_path)            # 如 liver_001.nii.gz
        label_path = os.path.join(args.labels, fname)  # 真值同名文件
        if not os.path.isfile(label_path):
            print(f"[警告] 找不到同名真值，跳过该样本：{fname}（期望：{label_path}）")
            continue
        pairs.append((fname, pred_path, label_path))

    if not pairs:
        print(f"[错误] 没有成功配对的样本（predictions 与 labels 文件名不匹配）。\n"
              f"  predictions: {args.predictions}\n  labels     : {args.labels}")
        sys.exit(1)

    print(f"[信息] 成功配对 {len(pairs)} 个样本，开始逐样本计算指标 ...")

    # —— 3) 逐样本：加载 → 标签对齐 → 算指标 ——
    # per_sample_metrics：含 'sample' 字段，供 JSON/CSV 输出；
    # raw_metrics：不含 'sample'，专供 aggregate_metrics 汇总（见下方说明）。
    per_sample_metrics = []
    raw_metrics = []

    for sample_name, pred_path, label_path in pairs:
        # load_nifti 返回 (data, affine)；评估只用体素数据 data，丢弃 affine
        pred_data = load_nifti(pred_path)[0]
        label_data = load_nifti(label_path)[0]

        # —— 标签对齐（重要）——
        # 原始真值标签恒为 0=背景/1=肝脏/2=肿瘤。但训练时 src.data.LabelMapTransform
        # 会按阶段做映射：阶段一(num_classes==2) 把「肝脏+肿瘤」都并入肝脏(label>0→1)，
        # 阶段二(num_classes>=3) 保留 0/1/2。模型是在「映射后的标签」上训练并预测的，
        # 因此评估时必须对真值做**同样的映射**，使真值与预测处于同一类别空间；否则
        # 真值里的肿瘤体素会被当成「类别 2」去和「二分类预测里不存在的类别 2」比较，
        # 导致肝脏指标被严重低估。这里复刻 LabelMapTransform 的映射规则。
        if num_classes == 2:
            # 阶段一：>0 的体素（肝脏1+肿瘤2）都视作肝脏前景 1
            label_data = (label_data > 0).astype(np.int64)
        else:
            # 阶段二：保留原始 0/1/2；统一成 int64 便于后续逐体素比较
            label_data = np.asarray(label_data).astype(np.int64)

        # 预测也统一成 int64（infer 保存时多为 uint8，这里与真值对齐类型）
        pred_data = np.asarray(pred_data).astype(np.int64)

        # 单样本指标：空真值类别记 NaN（见 compute_metrics docstring）
        m = compute_metrics(pred_data, label_data, num_classes)

        # 收集「不含 sample」的原始指标，供 aggregate_metrics 聚合
        raw_metrics.append(m)

        # 组装「含 sample」的逐样本记录，供 JSON/CSV 输出
        row = {"sample": sample_name}
        row.update(m)
        per_sample_metrics.append(row)

    # —— 4) 汇总：均值±标准差，忽略 NaN ——
    # 说明：aggregate_metrics 会对每个键在所有样本上取均值；若直接传 per_sample_metrics，
    # 其中的字符串键 'sample' 无法转 float64 求均值会报错，故这里传「不含 sample」的
    # raw_metrics，得到的 summary 仅含 liver_*/tumor_* 及其 *_std。
    summary = aggregate_metrics(raw_metrics)

    # —— 5) 输出 metrics.json ——
    # save_json 内部已 ensure_dir，并保证中文不转义、缩进 2 空格。
    json_path = os.path.join(output_dir, "metrics.json")
    save_json({"summary": summary, "per_sample": per_sample_metrics}, json_path)
    print(f"[输出] JSON 汇总已保存：{json_path}")

    # —— 6) 输出 metrics.csv（用 csv 模块一次写整表，比逐行 append 更简单）——
    # 项目里 src.utils.append_csv_row 适合「逐行追加」（如训练每个 epoch 记一行）；
    # 这里是一次性写完整表格，直接用 csv.DictWriter 一次性写入更直观。
    csv_path = os.path.join(output_dir, "metrics.csv")
    fieldnames = build_csv_fieldnames(num_classes)
    # newline="" 是 csv 模块在 Windows 下的推荐写法，避免行间出现多余空行
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()                       # 写表头
        for row in per_sample_metrics:
            # row 中的 NaN（float('nan')）会被 csv 序列化为字符串 'nan'，便于查看
            writer.writerow(row)
    print(f"[输出] CSV 逐样本表已保存：{csv_path}")

    # —— 7) 打印汇总表 ——
    print_summary_table(summary, num_classes, len(pairs))


if __name__ == "__main__":
    main()
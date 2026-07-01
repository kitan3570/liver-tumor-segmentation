"""分割性能评价脚本 (Task 7)。

作用
----
对模型预测的肝脏/肿瘤分割结果 (.nii.gz) 与真值标注进行对比，
逐例计算 Dice、IoU、Precision、Recall 等指标，并汇总 mean/std，
输出到 CSV 文件，便于大一医学生理解模型好坏。

运行方式
--------
在项目根目录 (d:\\项目1) 下执行：

    # 方式一：用 --label_dir 按目录配对（旧用法，保留）
    python src/evaluate.py --pred_dir outputs/predictions ^
        --label_dir data/raw/Task03_Liver/labelsTr ^
        --output_csv outputs/evaluation.csv

    # 方式二：用 --label_json 按 splits JSON 配对（正式测试集推荐）
    python src/evaluate.py --pred_dir outputs/predictions_test ^
        --label_json data/splits/test.json ^
        --output_csv outputs/evaluation_test.csv

输入
----
--pred_dir   : 预测 .nii.gz 所在目录（模型推理输出）。必填。
--label_dir  : 真值 .nii.gz 所在目录（数据集 labelsTr）。与 --label_json 二选一。
--label_json : 样本清单 JSON（如 data/splits/test.json），脚本会读取每条
               记录的 label 字段，按 basename 到 pred_dir 找同名预测文件。
               与 --label_dir 同时给出时，**优先使用 --label_json** 并打印提示。
--output_csv : 评价结果 CSV 路径（父目录会自动创建）。必填。

输出
----
1) CSV 文件，逐例指标 + 末尾 summary 行 (mean/std)。
2) 控制台打印 mean±std 汇总表，以及清单路径 / 病例数 / 成功数 / CSV 路径。

常见错误 / 注意事项
------------------
1) 预测与真值按 basename 配对，文件名不一致会跳过并警告。
2) 肿瘤空病例：真值中若无 label=2（肿瘤）体素，该类指标记为 NaN，
   不报错。医学分割中常见肿瘤缺失病例，需特殊处理。
3) num_classes：本脚本只评价 label=1 (liver) 与 label=2 (tumor) 两类，
   若数据集类别标签不同，需要修改下方 CLASSES。
4) HD95（95% 豪斯多夫距离）反映边界距离误差，更贴近临床"轮廓偏差"，
   但 MONAI 依赖较重，本脚本暂作 TODO 跳过。
5) 正式测试集评估**必须**用 --label_json data/splits/test.json，避免
   直接对整个 labelsTr 文件夹评估而混入训练集。
"""

import os
import glob
import json
import argparse
import sys

import numpy as np
import nibabel as nib
import pandas as pd

# 把 src 目录加入 sys.path（本脚本不强依赖其它 src 模块，但便于未来扩展）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# 需要评价的类别标签：1=肝脏 liver，2=肿瘤 tumor
CLASSES = {1: "liver", 2: "tumor"}


def compute_class_metrics(pred, label, c):
    """对某一类别 c 计算四项指标。

    参数
    ----
    pred  : np.ndarray，整型预测体素，值为类别编号
    label : np.ndarray，整型真值体素，值为类别编号
    c     : int，要评价的类别编号（1 或 2）

    返回
    ----
    dict，含 dice/iou/precision/recall 四项。
    若真值中不存在该类（label_c 体素数为 0），四项记 NaN。

    医学背景
    --------
    - Dice  = 2|A∩B| / (|A|+|B|)，衡量预测 A 与真值 B 的重叠程度，
              1 表示完全一致，0 表示无重叠。医学分割中 Dice>0.7
              通常被认为可接受，>0.9 为优秀。
    - IoU   = |A∩B| / |A∪B|，又称 Jaccard，对边界更敏感。
    - Precision = TP/(TP+FP)：预测为该类的体素里，真值也是该类的比例，
                   高表示"少误判"。
    - Recall    = TP/(TP+FN)：真值为该类的体素里，被预测到的比例，
                   高表示"少漏诊"。
    """
    # 二值化：把当前类 c 的体素置为 True
    pred_c = (pred == c)
    label_c = (label == c)

    # 交集 = TP（True Positive，真正例）
    tp = np.logical_and(pred_c, label_c).sum()
    # 预测为 c 但真值不是 c = FP（False Positive，假正例，误判）
    fp = np.logical_and(pred_c, np.logical_not(label_c)).sum()
    # 真值为 c 但预测不是 c = FN（False Negative，假负例，漏诊）
    fn = np.logical_and(np.logical_not(pred_c), label_c).sum()

    # 真值中不存在该类：肿瘤空病例处理，指标记 NaN
    # 医学分割中常见整个病例没有肿瘤（如健康肝或肿瘤被切除），
    # 此时无法有意义地计算重叠指标，记为 NaN 以避免误导。
    if label_c.sum() == 0:
        return {
            "dice": float("nan"),
            "iou": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
        }

    # 各指标公式（分母为 0 时返回 nan，避免除零）
    dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


def compute_hausdorff95(pred, label, c, spacing=(1, 1, 1)):
    """HD95（95百分位豪斯多夫距离）占位函数。

    HD95 衡量预测边界与真值边界之间第 95 百分位的距离（毫米），
    较 Dice 更关注轮廓的空间偏差，对临床"切缘评估"更有意义。

    本项目暂不实现：MONAI 的 compute_hausdorff_distance 依赖较重，
    且需对边界体素处理较繁琐。TODO：后续接入 MONAI 实现。
    """
    # TODO: 如需启用 HD95，使用 MONAI metrics.compute_hausdorff_distance
    return float("nan")


def evaluate_pair(pred_path, label_path):
    """对一对预测/真值文件计算 liver 与 tumor 的所有指标。

    返回 dict，键形如 liver_dice、tumor_iou 等。
    """
    # 用 nibabel 读取 .nii.gz，np.asanyarray 获取数组（保持 dtype）
    pred_data = np.asanyarray(nib.load(pred_path).dataobj)
    label_data = np.asanyarray(nib.load(label_path).dataobj)

    metrics = {}
    for c, name in CLASSES.items():
        m = compute_class_metrics(pred_data, label_data, c)
        # 展平成列名：liver_dice, liver_iou, ...
        metrics[f"{name}_dice"] = m["dice"]
        metrics[f"{name}_iou"] = m["iou"]
        metrics[f"{name}_precision"] = m["precision"]
        metrics[f"{name}_recall"] = m["recall"]

    return metrics


def _load_label_list_from_json(label_json_path: str) -> list:
    """从样本清单 JSON 读取真值 label 路径列表。

    JSON 格式（与 data/splits/test.json 一致）：
        [{"image": ".../liver_x.nii.gz", "label": ".../liver_x.nii.gz"}, ...]

    本函数只取 label 字段用于评估；image 字段不参与评估。

    Args:
        label_json_path: JSON 文件路径。

    Returns:
        list[str]: label 路径列表（保持 JSON 中的顺序）。

    Raises:
        SystemExit: 当 JSON 不存在 / 不是有效 JSON / 不是 list /
            某条记录缺 label 字段时。
    """
    # 1) 文件存在性
    if not os.path.isfile(label_json_path):
        print(f"[错误] --label_json 指定的 JSON 不存在：{label_json_path}")
        print("       正式测试集评估请用 data/splits/test.json。")
        sys.exit(1)

    # 2) 读取并解析 JSON
    try:
        with open(label_json_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[错误] --label_json 不是有效的 JSON：{label_json_path}")
        print(f"       解析失败：{e}")
        sys.exit(1)

    # 3) 必须是 list
    if not isinstance(records, list):
        print(f"[错误] --label_json 的顶层必须是数组（list），"
              f"实际类型：{type(records).__name__}")
        sys.exit(1)

    if len(records) == 0:
        print(f"[错误] --label_json 是空数组，没有待评估样本：{label_json_path}")
        sys.exit(1)

    # 4) 逐条校验：必须有 label 字段（不强制文件存在，缺失在配对时再 warning，
    #    因为有时 JSON 里写的是相对路径或环境差异导致路径暂时不可达）
    label_paths = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            print(f"[错误] --label_json 第 {i} 条记录不是字典：{rec}")
            sys.exit(1)
        if "label" not in rec or not rec["label"]:
            print(f"[错误] --label_json 第 {i} 条记录缺少 label 字段：{rec}")
            print("       每条记录形如 "
                  "{\"image\": \".../liver_x.nii.gz\", \"label\": \"...\"}。")
            sys.exit(1)
        label_paths.append(rec["label"])

    return label_paths


def main():
    """命令行入口：按 label_dir 或 label_json 配对 pred_dir，逐例评价并汇总。

    配对规则：
        - --label_json 模式（推荐，正式测试集用）：从 JSON 读 label 路径列表，
          按 basename 到 pred_dir 找同名预测文件。
        - --label_dir 模式（旧用法，保留）：扫描 pred_dir 下所有 *.nii.gz，
          按 basename 到 label_dir 找同名真值。
        - 两者同时给出：优先使用 --label_json，并打印提示。
        - 两者皆无：报错退出。
    """
    # ---------------- argparse 参数解析 ----------------
    parser = argparse.ArgumentParser(
        description="对肝脏/肿瘤分割预测进行 Dice/IoU/Precision/Recall 评价"
    )
    parser.add_argument("--pred_dir", required=True,
                        help="预测 .nii.gz 所在目录")
    parser.add_argument("--label_dir", default=None,
                        help="真值 .nii.gz 所在目录；与 --label_json 二选一，"
                             "同时给出时优先使用 --label_json")
    parser.add_argument("--label_json", default=None,
                        help="样本清单 JSON（如 data/splits/test.json）；"
                             "脚本会读取每条记录的 label 字段，按 basename 到 "
                             "pred_dir 找同名预测文件。与 --label_dir 同时给出时"
                             "优先使用 --label_json。")
    parser.add_argument("--output_csv", required=True,
                        help="输出 CSV 路径，如 outputs/evaluation_test.csv")
    args = parser.parse_args()

    pred_dir = args.pred_dir
    output_csv = args.output_csv

    # ---------------- 确定真值来源模式 ----------------
    has_label_dir = args.label_dir is not None
    has_label_json = args.label_json is not None

    if not has_label_dir and not has_label_json:
        print("[错误] 必须提供 --label_dir 或 --label_json 之一。"
              "正式测试集评估请用 --label_json data/splits/test.json。")
        sys.exit(1)

    use_label_json = has_label_json  # 同时给定时优先 JSON
    if has_label_json and has_label_dir:
        print(f"[提示] 同时指定了 --label_json 与 --label_dir，"
              f"优先使用 --label_json：{args.label_json}")

    # ---------------- 构造 (pred_path, label_path) 配对列表 ----------------
    # pairs: list of (basename, pred_path, label_path)
    pairs = []
    if use_label_json:
        # JSON 模式：以 JSON 中 label 为基准，按 basename 找 pred_dir 下同名预测
        label_paths = _load_label_list_from_json(args.label_json)
        for label_path in label_paths:
            basename = os.path.basename(label_path)
            pred_path = os.path.join(pred_dir, basename)
            pairs.append((basename, pred_path, label_path))
        total_cases = len(pairs)
    else:
        # 目录模式（旧用法）：扫描 pred_dir 下所有 *.nii.gz，按 basename 找真值
        pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz")))
        if len(pred_files) == 0:
            print(f"[警告] {pred_dir} 下没有找到 *.nii.gz 文件，请检查路径。")
            sys.exit(1)
        for pred_path in pred_files:
            basename = os.path.basename(pred_path)
            label_path = os.path.join(args.label_dir, basename)
            pairs.append((basename, pred_path, label_path))
        total_cases = len(pairs)

    # ---------------- 逐例评价 ----------------
    rows = []  # 每个元素是一例的指标 dict
    success_count = 0  # 成功评估的病例数

    for basename, pred_path, label_path in pairs:
        # 预测文件缺失（JSON 模式下常见）：记录 warning 并跳过该病例。
        # 这里选择「跳过」而非「CSV 标记 missing」，因为缺失预测无法计算
        # 任何指标，标 missing 反而会污染 mean/std 统计。
        if not os.path.exists(pred_path):
            print(f"[警告] 未找到预测 {pred_path}，跳过 {basename}。")
            continue
        # 真值文件缺失：记录 warning 并跳过
        if not os.path.exists(label_path):
            print(f"[警告] 未找到真值 {label_path}，跳过 {basename}。")
            continue

        try:
            m = evaluate_pair(pred_path, label_path)
        except Exception as e:
            print(f"[警告] 评价 {basename} 时出错：{e}")
            continue

        m["case"] = basename
        rows.append(m)
        success_count += 1
        # tumor_dice 可能是 NaN（肿瘤空病例），用 NaN 安全的格式化
        tumor_dice_str = (f"{m['tumor_dice']:.4f}"
                          if not np.isnan(m["tumor_dice"]) else "NaN")
        print(f"[完成] {basename}: "
              f"liver_dice={m['liver_dice']:.4f}, "
              f"tumor_dice={tumor_dice_str}")

    # ---------------- 汇总打印（无论是否有成功病例都先打印配对统计） ----------------
    # 正式测试集评估后需明确告知：清单路径、测试病例数、成功数、CSV 路径
    print("\n" + "=" * 60)
    print(f"[评估汇总]")
    if use_label_json:
        print(f"  真值清单     : {args.label_json}")
    else:
        print(f"  真值目录     : {args.label_dir}")
    print(f"  测试病例数量 : {total_cases}")
    print(f"  成功评估数量 : {success_count}")
    print(f"  预测目录     : {pred_dir}")

    if len(rows) == 0:
        print("[错误] 没有成功配对评价的病例，请检查 pred_dir / label_json。")
        print("=" * 60)
        sys.exit(1)

    # ---------------- 组织 DataFrame ----------------
    # 指定列顺序，方便阅读
    columns = [
        "case",
        "liver_dice", "liver_iou", "liver_precision", "liver_recall",
        "tumor_dice", "tumor_iou", "tumor_precision", "tumor_recall",
    ]
    df = pd.DataFrame(rows)[columns]

    # ---------------- 整体 mean/std 汇总 ----------------
    # numeric_only=True 避免 case 字符串参与计算；skipna 默认跳过 NaN 病例
    metric_cols = columns[1:]
    mean_series = df[metric_cols].mean(numeric_only=True)
    std_series = df[metric_cols].std(numeric_only=True)

    # 追加 summary 行：mean 与 std 各一行，case 字段标注
    summary_mean = {"case": "MEAN"}
    summary_std = {"case": "STD"}
    for col in metric_cols:
        summary_mean[col] = mean_series[col]
        summary_std[col] = std_series[col]

    df_with_summary = pd.concat(
        [df, pd.DataFrame([summary_mean, summary_std])],
        ignore_index=True
    )

    # ---------------- 保存 CSV ----------------
    # 确保父目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df_with_summary.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"  评价 CSV     : {output_csv}")
    print("=" * 60)

    # ---------------- 打印 mean±std 汇总表 ----------------
    print("\n===== 汇总 (mean ± std) =====")
    for col in metric_cols:
        mu = mean_series[col]
        sd = std_series[col]
        # 全 NaN 列（如所有病例都无肿瘤）打印 NaN
        if np.isnan(mu):
            print(f"{col:<16s}: NaN")
        else:
            print(f"{col:<16s}: {mu:.4f} ± {sd:.4f}")
    print("===============================")


if __name__ == "__main__":
    main()
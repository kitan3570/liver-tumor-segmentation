"""src/verify_split_integrity.py —— 划分独立性校验脚本

文件作用
========
    检查 data/splits/ 下的 train.json / val.json / test.json 三个划分是否
    相互独立——即同一病例不会同时出现在训练集和测试集里。这是医学影像分割
    实验的「底线」：若测试集病例曾参与训练，评估指标会被严重高估，结论
    不可信。

    本脚本面向大一医学生，因此代码注释格外详尽。

检查内容
========
    1) 三个 JSON 中 image 字段的 basename 是否有重叠（同一 CT 出现在多个划分）。
    2) 三个 JSON 中 label 字段的 basename 是否有重叠（同一标注出现在多个划分）。
    3) 输出每个划分的病例数。
    4) 有重叠 → 打印 ERROR 并以非 0 退出码退出，便于 CI / 脚本感知。
    5) 无重叠 → 打印 "OK: train/val/test are independent."。
    6) 把完整报告写入 outputs/split_integrity_report.txt。

运行方式
========
    在项目根目录 (d:\\项目1) 下执行：

    python src/verify_split_integrity.py \
        --train_json data/splits/train.json \
        --val_json data/splits/val.json \
        --test_json data/splits/test.json

输入
====
    --train_json : 训练集划分 JSON
    --val_json   : 验证集划分 JSON
    --test_json  : 测试集划分 JSON

    JSON 格式（由 src/create_splits.py 生成）：
        [{"image": ".../liver_x.nii.gz", "label": ".../liver_x.nii.gz"}, ...]

输出
====
    1) 控制台：每个划分的病例数 + 重叠检查结果。
    2) outputs/split_integrity_report.txt：完整报告。
    3) 退出码：0=独立无重叠；1=存在重叠或读取失败。

常见错误
========
    1) JSON 不存在：报错退出，请先用 src/create_splits.py 生成划分。
    2) JSON 不是 list：报错退出，格式不符合 create_splits 产物约定。
    3) 某条记录缺 image / label 字段：报错退出。
"""

import os
import sys
import json
import argparse


def _load_split(json_path: str, split_name: str):
    """读取一个划分 JSON，返回 (image_basenames, label_basenames) 列表。

    Args:
        json_path : JSON 文件路径。
        split_name: 划分名称（train/val/test），仅用于错误提示。

    Returns:
        tuple(list[str], list[str]): image 与 label 的 basename 列表，
        保持 JSON 中的顺序（不去重，便于发现 JSON 内部自身的重复）。

    Raises:
        SystemExit: 当 JSON 不存在 / 不是有效 JSON / 不是 list /
            某条记录缺 image/label 字段时。
    """
    # 1) 文件存在性
    if not os.path.isfile(json_path):
        print(f"[错误] {split_name} 划分 JSON 不存在：{json_path}")
        print("       请先用 src/create_splits.py 生成划分。")
        sys.exit(1)

    # 2) 读取并解析 JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[错误] {split_name} 划分 JSON 解析失败：{json_path}")
        print(f"       原因：{e}")
        sys.exit(1)

    # 3) 必须是 list
    if not isinstance(records, list):
        print(f"[错误] {split_name} 划分 JSON 的顶层必须是数组（list），"
              f"实际类型：{type(records).__name__}")
        sys.exit(1)

    # 4) 逐条提取 image / label 的 basename
    image_basenames = []
    label_basenames = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            print(f"[错误] {split_name} JSON 第 {i} 条记录不是字典：{rec}")
            sys.exit(1)
        if "image" not in rec or not rec["image"]:
            print(f"[错误] {split_name} JSON 第 {i} 条记录缺少 image 字段：{rec}")
            sys.exit(1)
        if "label" not in rec or not rec["label"]:
            print(f"[错误] {split_name} JSON 第 {i} 条记录缺少 label 字段：{rec}")
            sys.exit(1)
        image_basenames.append(os.path.basename(rec["image"]))
        label_basenames.append(os.path.basename(rec["label"]))

    return image_basenames, label_basenames


def _find_cross_split_overlaps(splits: dict, field: str) -> list:
    """找出三个划分中某字段（image/label）basename 的两两重叠。

    Args:
        splits: dict，键为划分名（train/val/test），值为该划分的
                (image_basenames, label_basenames) 元组。
        field : "image" 或 "label"，指定检查哪个字段。

    Returns:
        list[tuple(str, str, set[str])]：每项为 (split_a, split_b, overlap_set)，
        overlap_set 为两个划分中重复出现的 basename 集合。仅返回非空重叠。
    """
    # 取出每个划分对应字段的 basename 集合
    names = ["train", "val", "test"]
    # field == "image" → 元组下标 0；field == "label" → 下标 1
    idx = 0 if field == "image" else 1
    sets = {n: set(splits[n][idx]) for n in names}

    # 两两组合检查：train-vs-val, train-vs-test, val-vs-test
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    overlaps = []
    for a, b in pairs:
        overlap = sets[a] & sets[b]
        if overlap:
            overlaps.append((a, b, overlap))
    return overlaps


def main() -> None:
    """主入口：解析参数 → 读取三个划分 → 检查重叠 → 打印报告 + 写文件。"""
    parser = argparse.ArgumentParser(
        description="检查 train/val/test 三个划分 JSON 是否相互独立（无病例重叠）"
    )
    parser.add_argument("--train_json", required=True,
                        help="训练集划分 JSON，如 data/splits/train.json")
    parser.add_argument("--val_json", required=True,
                        help="验证集划分 JSON，如 data/splits/val.json")
    parser.add_argument("--test_json", required=True,
                        help="测试集划分 JSON，如 data/splits/test.json")
    args = parser.parse_args()

    # ---------- 读取三个划分 ----------
    # splits[name] = (image_basenames, label_basenames)
    splits = {
        "train": _load_split(args.train_json, "train"),
        "val":   _load_split(args.val_json,   "val"),
        "test":  _load_split(args.test_json,  "test"),
    }

    # ---------- 检查 image / label 的两两重叠 ----------
    image_overlaps = _find_cross_split_overlaps(splits, "image")
    label_overlaps = _find_cross_split_overlaps(splits, "label")
    has_overlap = len(image_overlaps) > 0 or len(label_overlaps) > 0

    # ---------- 组织报告文本 ----------
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("划分独立性校验报告 (split integrity report)")
    report_lines.append("=" * 60)
    report_lines.append("")
    report_lines.append("输入划分：")
    report_lines.append(f"  train_json : {args.train_json}")
    report_lines.append(f"  val_json   : {args.val_json}")
    report_lines.append(f"  test_json  : {args.test_json}")
    report_lines.append("")
    report_lines.append("各划分病例数：")
    for n in ["train", "val", "test"]:
        img_n = len(splits[n][0])
        lbl_n = len(splits[n][1])
        report_lines.append(f"  {n:<5s}: {img_n} 例 (image) / {lbl_n} 例 (label)")
    report_lines.append("")

    if has_overlap:
        report_lines.append("ERROR: 划分之间存在病例重叠，正式测试集独立性被破坏！")
        report_lines.append("")
        if image_overlaps:
            report_lines.append("[image 字段重叠]")
            for a, b, overlap in image_overlaps:
                report_lines.append(f"  {a} ∩ {b} = {len(overlap)} 个重复文件：")
                for fn in sorted(overlap):
                    report_lines.append(f"    - {fn}")
            report_lines.append("")
        if label_overlaps:
            report_lines.append("[label 字段重叠]")
            for a, b, overlap in label_overlaps:
                report_lines.append(f"  {a} ∩ {b} = {len(overlap)} 个重复文件：")
                for fn in sorted(overlap):
                    report_lines.append(f"    - {fn}")
            report_lines.append("")
        report_lines.append("请重新运行 src/create_splits.py 生成互斥的划分。")
    else:
        report_lines.append("OK: train/val/test are independent.")
    report_lines.append("=" * 60)

    report_text = "\n".join(report_lines)

    # ---------- 控制台输出 ----------
    print(report_text)

    # ---------- 写入 outputs/split_integrity_report.txt ----------
    # 报告路径固定为项目根/outputs/split_integrity_report.txt，便于统一查看
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    report_dir = os.path.join(proj_root, "outputs")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "split_integrity_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print(f"\n[已保存] 完整报告 -> {report_path}")

    # ---------- 退出码：有重叠则非 0 ----------
    if has_overlap:
        sys.exit(1)


if __name__ == "__main__":
    main()

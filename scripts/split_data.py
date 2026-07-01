"""scripts/split_data.py — 数据划分脚本

文件作用
========
把 Medical Segmentation Decathlon Task03_Liver 的训练集 (imagesTr / labelsTr)
按比例划分为 train / val / test 三份 JSON 清单，供后续训练、验证、推理脚本读取。

每个 JSON 文件是一个 list，元素形如：
    {"image": "<绝对路径>", "label": "<绝对路径>"}
这与 MONAI 的数据清单风格一致，可直接喂给 src.data.build_dataloader。

运行方式
========
    python scripts/split_data.py \
        --data_root ./data/Task03_Liver \
        --ratio 0.8 0.1 0.1 \
        --seed 42 \
        --output outputs/splits

输入
====
- --data_root：数据集根目录，其下应含 imagesTr/ 与 labelsTr/ 两个子目录，
  文件名形如 liver_001.nii.gz ... liver_131.nii.gz，image 与 label 同名配对。

输出
====
- 在 --output 目录下生成 train.json / val.json / test.json 三个清单文件。
- 终端打印每段样本数与若干示例条目。

常见错误
========
1) imagesTr 或 labelsTr 目录不存在 —— 检查 --data_root 路径是否正确。
2) --ratio 三个数之和不为 1 —— 请保证比例之和为 1（如 0.8 0.1 0.1）。
3) 找不到任何 liver_*.nii.gz —— 确认数据集已解压到 data_root 下。
4) 同名 image/label 不配对 —— 脚本会跳过并打印警告，请检查文件是否缺失。
"""

# ===================== 标准库导入 =====================
import os
import glob
import random
import argparse
import sys

# 把项目根目录（本文件所在目录的上一级）加入 sys.path，使得直接
# `python scripts/split_data.py ...` 运行时可 import src.*（否则 Python 只把
# scripts/ 当作包搜索路径，找不到 src 包）。与 train.py / evaluate.py 一致。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ===================== 本项目模块导入 =====================
# 复用 src.utils 中的目录创建与 JSON 保存工具，保持与项目其他脚本一致
from src.utils import ensure_dir, save_json


# Task03_Liver 的图像/标签子目录名（约定俗成，来自 MSD 数据集结构）
IMAGES_DIR = "imagesTr"
LABELS_DIR = "labelsTr"
# 文件名前缀：Task03 的文件形如 liver_001.nii.gz
FILE_PATTERN = "liver_*.nii.gz"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 含 data_root, ratio(list[float]), seed, output。
    """
    parser = argparse.ArgumentParser(
        description="把 Task03_Liver 训练集划分为 train/val/test JSON 清单（可复现）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # --help 自动显示默认值
    )
    parser.add_argument(
        "--data_root", required=True,
        help="数据集根目录，其下应含 imagesTr/ 与 labelsTr/",
    )
    parser.add_argument(
        "--ratio", nargs=3, type=float, default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
        help="train/val/test 比例，三个数之和应为 1",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子，相同种子下划分结果一致（可复现）",
    )
    parser.add_argument(
        "--output", default="outputs/splits",
        help="输出目录，将生成 train.json / val.json / test.json",
    )
    return parser.parse_args()


def pair_samples(data_root: str) -> list:
    """在 imagesTr/ 与 labelsTr/ 中按同名配对样本。

    Args:
        data_root: 数据集根目录。

    Returns:
        list: 每个元素是 {"image": <abs path>, "label": <abs path>}，
              按文件名升序排列；仅返回 image 与 label 都存在的样本。
    """
    images_dir = os.path.join(data_root, IMAGES_DIR)
    labels_dir = os.path.join(data_root, LABELS_DIR)

    # 用 glob 找到所有 liver_*.nii.gz，sorted 保证顺序稳定（可复现）
    image_paths = sorted(glob.glob(os.path.join(images_dir, FILE_PATTERN)))
    label_paths = sorted(glob.glob(os.path.join(labels_dir, FILE_PATTERN)))

    # 以文件 basename 为 key 建索引，便于按名配对
    label_by_name = {os.path.basename(p): p for p in label_paths}

    samples = []
    missing_label = []
    for img in image_paths:
        name = os.path.basename(img)
        if name in label_by_name:
            # 配对成功：用绝对路径，后续脚本读取不受工作目录影响
            samples.append({
                "image": os.path.abspath(img),
                "label": os.path.abspath(label_by_name[name]),
            })
        else:
            missing_label.append(name)

    if missing_label:
        print(f"[警告] 以下 image 没有同名 label，已跳过：{missing_label}")

    return samples


def split_samples(samples: list, ratio: list, seed: int) -> dict:
    """按比例把样本列表切成 train/val/test 三段（可复现）。

    切分策略：
        - 先用固定种子打乱（random.Random(seed).shuffle），保证相同 seed 结果一致；
        - 按比例计算各段样本数（整数），余数依次补到 train，保证总数不丢。

    Args:
        samples: pair_samples 返回的样本列表。
        ratio: [train, val, test] 比例，和应为 1。
        seed: 随机种子。

    Returns:
        dict: {"train": list, "val": list, "test": list}。
    """
    n = len(samples)
    # 用独立的 Random 对象，避免影响全局 random 状态
    rng = random.Random(seed)
    shuffled = list(samples)  # 浅拷贝，不修改原列表
    rng.shuffle(shuffled)

    # 计算各段样本数：先按比例向下取整
    n_train = int(round(n * ratio[0]))
    n_val = int(round(n * ratio[1]))
    n_test = n - n_train - n_val  # test 取剩余，保证总数恰好为 n
    # 极端情况下 round 可能造成 n_train+n_val>n，此时修正
    if n_test < 0:
        n_train = max(0, n_train + n_test)
        n_test = n - n_train - n_val

    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:n_train + n_val + n_test],
    }
    return splits


def main() -> None:
    """脚本主入口：解析参数 → 配对样本 → 划分 → 保存 JSON → 打印汇总。"""
    args = parse_args()

    # 1) 校验比例之和
    if abs(sum(args.ratio) - 1.0) > 1e-6:
        print(f"[错误] --ratio 三个数之和应为 1，当前为 {sum(args.ratio)}。")
        raise SystemExit(1)

    # 2) 校验数据目录存在
    if not os.path.isdir(args.data_root):
        print(f"[错误] data_root 不存在：{args.data_root}")
        raise SystemExit(1)
    images_dir = os.path.join(args.data_root, IMAGES_DIR)
    labels_dir = os.path.join(args.data_root, LABELS_DIR)
    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        print(f"[错误] imagesTr/ 或 labelsTr/ 不存在：\n  {images_dir}\n  {labels_dir}")
        raise SystemExit(1)

    # 3) 配对样本
    samples = pair_samples(args.data_root)
    if not samples:
        print(f"[错误] 未找到任何配对样本（{FILE_PATTERN}）。请确认数据已解压。")
        raise SystemExit(1)
    print(f"[信息] 共配对到 {len(samples)} 个样本。")

    # 4) 划分
    splits = split_samples(samples, args.ratio, args.seed)

    # 5) 保存 JSON 清单
    ensure_dir(args.output)
    for name, items in splits.items():
        out_path = os.path.join(args.output, f"{name}.json")
        save_json(items, out_path)
        print(f"[输出] {name}.json -> {out_path}（{len(items)} 个样本）")
        # 打印最多 2 条示例，便于人工核对路径正确
        for ex in items[:2]:
            print(f"        示例: image={os.path.basename(ex['image'])}")

    # 6) 复现性提示
    print(f"[信息] 划分已固定 seed={args.seed}，相同参数重复运行结果一致。")


if __name__ == "__main__":
    main()

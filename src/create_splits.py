#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
create_splits.py —— Task03_Liver 训练数据划分脚本（train / val / test）

【作用】
    将 Task03_Liver 数据集中的训练图像与标签划分为 train / val / test 三个子集，
    并分别保存为 JSON 文件，便于后续训练（train）与推理（infer）脚本统一读取。

【运行方式】
    在项目根目录（d:\\项目1）下执行：
        python src/create_splits.py --data_dir data/raw/Task03_Liver --output_dir data/splits --train_ratio 0.7 --val_ratio 0.15 --seed 42

    说明：
        --data_dir      数据集根目录，其下应包含 imagesTr 与 labelsTr 两个子目录
        --output_dir    划分结果 JSON 的输出目录（默认 data/splits）
        --train_ratio   训练集比例（默认 0.7）
        --val_ratio     验证集比例（默认 0.15）
        --seed          随机种子，用于可复现划分（默认 42）

【输入与输出】
    输入：
        <data_dir>/imagesTr/liver_*.nii.gz   训练图像
        <data_dir>/labelsTr/liver_*.nii.gz   训练标签（与图像同名配对）
    输出：
        <output_dir>/train.json   训练集列表
        <output_dir>/val.json     验证集列表
        <output_dir>/test.json    测试集列表
    每条 JSON 条目格式：
        {"image": "<图像绝对路径>", "label": "<标签绝对路径>"}

【常见错误】
    1) imagesTr 或 labelsTr 目录不存在    —— 提示目录不存在并 exit(1)。
    2) train_ratio + val_ratio + test_ratio != 1 —— 提示比例之和不等于 1 并 exit(1)。
    3) image 与 label 不能一一对应（缺失或多出） —— 提示配对失败并 exit(1)。
"""

import os          # 操作系统接口：路径拼接、绝对路径、目录创建
import glob        # 文件模式匹配：扫描 liver_*.nii.gz
import json        # JSON 读写：将划分结果序列化到文件
import random      # 随机数生成：使用独立 Random 实例打乱数据，保证可复现
import argparse    # 命令行参数解析
import sys         # 退出程序：sys.exit(1)


def parse_args():
    """解析命令行参数。详见下方对每个参数的中文说明。"""
    parser = argparse.ArgumentParser(
        # prog 与 description 用于 --help 输出，便于大一同学理解脚本用途
        prog="create_splits.py",
        description="将 Task03_Liver 训练数据划分为 train/val/test 三个子集，输出 JSON 列表。",
    )
    # --data_dir：必填参数，数据集根目录，其下应含 imagesTr 与 labelsTr
    parser.add_argument(
        "--data_dir",
        required=True,
        help="数据集根目录，例如 data/raw/Task03_Liver（其下需含 imagesTr 与 labelsTr）",
    )
    # --output_dir：可选，默认 data/splits
    parser.add_argument(
        "--output_dir",
        default="data/splits",
        help="划分结果输出目录，默认 data/splits",
    )
    # --train_ratio：训练集比例，浮点数，默认 0.7
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="训练集比例（浮点数），默认 0.7",
    )
    # --val_ratio：验证集比例，浮点数，默认 0.15
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="验证集比例（浮点数），默认 0.15",
    )
    # --seed：随机种子，保证同样参数可得到同样划分（可复现实验）
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，用于可复现划分，默认 42",
    )
    return parser.parse_args()


def scan_and_pair(data_dir):
    """
    扫描 imagesTr 与 labelsTr 中的 liver_*.nii.gz，并按 basename（文件名）一一配对。

    配对策略：
        1) 分别用 glob 收集 imagesTr 与 labelsTr 下全部 liver_*.nii.gz；
        2) 以 basename（不含目录的文件名）作为 key 构建字典；
        3) 两个字典的 key 集合必须完全一致，否则视为配对失败。

    返回：
        list[list[abs_image, abs_label]]：配对好的 (图像绝对路径, 标签绝对路径) 列表
    """
    images_dir = os.path.join(data_dir, "imagesTr")
    labels_dir = os.path.join(data_dir, "labelsTr")

    # 校验：imagesTr / labelsTr 必须存在，否则报错退出
    if not os.path.isdir(images_dir):
        print("[错误] imagesTr 目录不存在: {}".format(images_dir))
        sys.exit(1)
    if not os.path.isdir(labels_dir):
        print("[错误] labelsTr 目录不存在: {}".format(labels_dir))
        sys.exit(1)

    # 用 glob 扫描 liver_*.nii.gz；结果转为绝对路径，便于后续 JSON 直接读取
    image_paths = [os.path.abspath(p) for p in glob.glob(os.path.join(images_dir, "liver_*.nii.gz"))]
    label_paths = [os.path.abspath(p) for p in glob.glob(os.path.join(labels_dir, "liver_*.nii.gz"))]

    # 以 basename 为 key 建立字典，方便一一对齐
    image_map = {os.path.basename(p): p for p in image_paths}
    label_map = {os.path.basename(p): p for p in label_paths}

    # 校验：image 与 label 必须一一对应（数量一致且同名）
    image_keys = set(image_map.keys())
    label_keys = set(label_map.keys())
    if image_keys != label_keys:
        # 找出只出现在一侧的文件名，帮助定位问题
        only_in_images = image_keys - label_keys
        only_in_labels = label_keys - image_keys
        print("[错误] image 与 label 不能一一对应，配对失败。")
        if only_in_images:
            print("       仅在 imagesTr 中存在: {}".format(sorted(only_in_images)))
        if only_in_labels:
            print("       仅在 labelsTr 中存在: {}".format(sorted(only_in_labels)))
        sys.exit(1)

    # 按 basename 排序后组装成 [(image, label), ...]，排序是为了后续打乱前的确定性
    paired = []
    for name in sorted(image_keys):
        paired.append([image_map[name], label_map[name]])
    return paired


def split_list(paired, train_ratio, val_ratio, seed):
    """
    按 train/val/test 比例对配对列表进行随机划分。

    切分策略：
        1) test_ratio = 1 - train_ratio - val_ratio（主程序已校验比例和为 1）；
        2) 使用 random.Random(seed) 创建独立随机数生成器，不影响全局 random 状态；
        3) 先复制并打乱列表，保证可复现性（相同 seed -> 相同结果）；
        4) 用整数取整计算各子集大小，余数全部补到 train，保证总数不丢失。

    返回：
        train, val, test 三个子列表
    """
    # 复制一份再打乱，避免修改外部传入的列表
    shuffled = paired[:]
    rng = random.Random(seed)         # 独立 Random 实例：不影响全局 random
    rng.shuffle(shuffled)             # 原地打乱

    total = len(shuffled)
    test_ratio = 1.0 - train_ratio - val_ratio

    # 整数取整：int() 向下取整，可能产生余数
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = int(total * test_ratio)

    # 余数补偿：把丢失的样本数全部补到 train，保证 train+val+test == total
    remainder = total - (n_train + n_val + n_test)
    n_train += remainder

    # 按顺序切片：train 取前 n_train 个，val 取其后 n_val 个，test 取剩余
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:n_train + n_val + n_test]
    return train, val, test


def write_json(path, items):
    """
    将划分条目写入 JSON 文件。

    每条目格式：{"image": "<绝对路径>", "label": "<绝对路径>"}
    ensure_ascii=False：保证路径中的中文不会转义；
    indent=2：美化输出，便于人工核对。

    注意：items 内部元素可能是 [image, label] 列表（scan_pairs 产物），
    这里统一转成 {"image":..., "label":...} 字典，以便 MONAI LoadImaged 直接读取。
    """
    normalized = []
    for it in items:
        if isinstance(it, dict):
            normalized.append(it)
        elif isinstance(it, (list, tuple)) and len(it) == 2:
            normalized.append({"image": it[0], "label": it[1]})
        else:
            raise ValueError("无法识别的划分条目格式: {}".format(it))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)


def main():
    # 解析命令行参数
    args = parse_args()

    # 校验：比例之和应为 1（test_ratio 由 1 - train - val 推得，间接校验）
    if abs(args.train_ratio + args.val_ratio - 1.0) > 1e-6 and \
       (args.train_ratio + args.val_ratio) > 1.0 + 1e-9:
        # 实际只要 train+val 超过 1，test_ratio 必为负，需要报错
        print("[错误] train_ratio + val_ratio = {}，超过 1，test_ratio 将为负。".format(
            args.train_ratio + args.val_ratio))
        sys.exit(1)

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    # 二次校验：三段比例之和应等于 1（考虑浮点误差用容差判断）
    if abs(args.train_ratio + args.val_ratio + test_ratio - 1.0) > 1e-6:
        print("[错误] train_ratio + val_ratio + test_ratio != 1（当前分别为 {}/{}/{}）。".format(
            args.train_ratio, args.val_ratio, test_ratio))
        sys.exit(1)

    # 扫描并配对图像与标签
    paired = scan_and_pair(args.data_dir)
    print("[信息] 共配对到 {} 例 liver 样本。".format(len(paired)))

    # 划分 train / val / test
    train, val, test = split_list(paired, args.train_ratio, args.val_ratio, args.seed)

    # 创建输出目录（已存在则忽略）
    os.makedirs(args.output_dir, exist_ok=True)

    # 写出三个 JSON 文件
    train_path = os.path.join(args.output_dir, "train.json")
    val_path = os.path.join(args.output_dir, "val.json")
    test_path = os.path.join(args.output_dir, "test.json")
    write_json(train_path, train)
    write_json(val_path, val)
    write_json(test_path, test)

    # 打印各 split 数量
    print("[结果] train: {} 例 -> {}".format(len(train), train_path))
    print("[结果] val  : {} 例 -> {}".format(len(val), val_path))
    print("[结果] test : {} 例 -> {}".format(len(test), test_path))

    # 打印示例条目（每个 split 最多 1 条），便于人工核对路径是否正确
    if train:
        print("[示例 train] {}".format(train[0]))
    if val:
        print("[示例 val  ] {}".format(val[0]))
    if test:
        print("[示例 test ] {}".format(test[0]))


if __name__ == "__main__":
    main()
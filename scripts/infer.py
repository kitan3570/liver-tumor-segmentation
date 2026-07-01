"""scripts/infer.py — 肝脏 3D 分割推理脚本

文件作用
========
用训练好的 3D U-Net 权重，对 test/val 集（整张 CT 体积）做「滑动窗口推理」：
把大体积切成 96³ 小块逐块送网络预测，再拼接回原尺寸，得到多类分割图，并
保存为 NIfTI 文件（.nii.gz）。可选启用「保留最大连通域」后处理，去除零散
假阳性碎片。本脚本是 Task 7 SubTask 7.5 的产物，生成的预测可供可视化切片
对比、STL 三维导出与指标评估使用。

滑动窗口推理为什么必要
----------------------
整张 CT 体积（如 512×512×100）远大于训练 patch（96³），一次性送入网络会
爆显存。SlidingWindowInferer 把大体积切成若干 96³ 窗口逐块前向，再拼接回
原尺寸；相邻窗口 overlap=0.25（重叠 25%），重叠区做加权平均，减轻拼接处
的接缝/块效应。

运行方式
========
典型命令（阶段一、二分类）：

    python scripts/infer.py --config configs/phase1_binary.yaml --model_ckpt outputs/models/best.pt --data_list outputs/splits/test.json

不指定 --data_list 时默认用 config['data']['splits_dir']/test.json；
不指定 --output    时默认用 config['output']['predictions_dir']；
不指定 --device    时默认用 config['device']（标准设置为 cuda）。

输入
====
- --config      ：YAML 配置文件（如 configs/phase1_binary.yaml）。
- --model_ckpt  ：训练好的权重路径（.pt/.pth），如 outputs/models/best.pt。
- --data_list   ：JSON 数据清单（list of {image,label}）；默认 splits_dir/test.json。
- --output      ：预测 NIfTI 输出目录；默认 predictions_dir。

输出
====
- 在 --output 目录下，为每个样本生成与输入图像同名的 NIfTI 预测文件
  （如 liver_001.nii.gz），体素值为类别 id（phase1: 0/1；phase2: 0/1/2）。
- 终端用 tqdm 显示推理进度，结束时打印完成数量与输出目录。

常见错误
========
1) --model_ckpt 不存在 —— 检查路径，确认训练（train.py）已产出 best.pt。
2) --data_list 不存在 —— 先运行 scripts/split_data.py 生成 test.json。
3) CUDA 不可用 —— get_device 会自动回退到 CPU 并打印警告；3D 整卷推理在
   CPU 上很慢，请耐心等待，或换用支持 CUDA 的机器。
4) 显存不足（CUDA out of memory）—— 调小 config['infer']['sw_batch_size']
   （如设为 1），或减小 roi_size；也可关闭其他占用显存的程序后重试。
"""

# ===================== 标准库导入 =====================
import os
import sys
import argparse

import numpy as np

# ===================== 第三方库导入 =====================
# torch：模型前向与张量搬运到 GPU；tqdm：终端进度条。
# monai.inferers.SlidingWindowInferer：MONAI 提供的滑动窗口推理器。
# scipy.ndimage：用于连通域标注（保留最大连通域后处理）。
import torch
from tqdm import tqdm
from monai.inferers import SlidingWindowInferer
from scipy import ndimage

# ===================== 本项目模块导入 =====================
# 先把项目根目录加入 sys.path，使得直接 `python scripts/infer.py ...` 运行时可
# import src.*（否则 Python 只把 scripts/ 当搜索路径，找不到 src 包）。sys/os 已导入。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 复用 src.utils 的配置/IO/设备工具，src.data 的数据加载器，src.models 的模型构建。
from src.utils import (
    load_config,
    load_json,
    ensure_dir,
    load_nifti,
    save_nifti,
    get_device,
)
from src.data import build_dataloader
from src.models import build_model, load_model


def keep_largest_connected_component(mask_3d: np.ndarray) -> np.ndarray:
    """保留 3D 二值掩膜中体积最大的连通域，去除零散小碎片。

    【为什么要做这个后处理】
        滑动窗口推理后，网络常会在肝实质外「画」出一些零星的小斑点（假阳性
        碎片），它们的成因往往是：远离肝脏的背景区域被错分成前景、或拼接边
        缘的轻微伪影。这些碎片体积小、与主肝区不连通，临床上不合理（肝脏是
        一个连续器官）。保留最大连通域可有效去除这些碎片，提升预测的「形态
        合理性」与下游可视化/STL 的质量。

    【实现】
        用 scipy.ndimage.label 对二值前景做连通域标注：每个连通体素群获得
        一个独立编号；再统计各连通域体素数，仅保留最大的那一个。默认使用
        scipy 的「6-邻接（面邻接）」连通判定（3D 下即上下/左右/前后相邻）。

    Args:
        mask_3d: 3D 二值数组（bool 或 0/1），前景视为 >0 / True。

    Returns:
        np.ndarray: 与输入同形状的 bool 数组，仅在最大连通域处为 True，其余为 False。
        若输入全为空（无前景），直接返回全 False 的 bool 数组。
    """
    # 统一成 bool 语义：>0 视为前景，避免传入 int 0/1 时歧义
    binary = mask_3d.astype(bool)

    # 全空：没有前景可处理，直接返回，避免后续 label 报错或得到空结果
    if not binary.any():
        return binary

    # 标注连通域：labeled 与 binary 同形，每个连通域赋一个 >=1 的整数编号；
    # num 为连通域总数（0 始终是背景）。
    labeled, num = ndimage.label(binary)

    # 0 或 1 个连通域：无需删除任何碎片，原样返回
    if num <= 1:
        return binary

    # 统计每个连通域（编号 1..num）的体素数：以 binary 为权重在 labeled 上求和
    sizes = ndimage.sum(binary, labeled, index=range(1, num + 1))

    # argmax 给出最大域在 sizes 中的下标（0 起），+1 还原为连通域编号
    largest_label = int(np.argmax(sizes)) + 1

    # 返回仅最大连通域为 True 的 bool 掩膜
    return labeled == largest_label


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    --help 由 argparse 自动提供，无需手动定义。

    Returns:
        argparse.Namespace: 含 config、model_ckpt、data_list、output、device、no_cc。
    """
    parser = argparse.ArgumentParser(
        description=(
            "用训练好的 3D U-Net 对 test/val 集做滑动窗口推理，"
            "保存 NIfTI 预测，可选保留最大连通域后处理。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # --help 自动显示默认值
    )
    parser.add_argument(
        "--config",
        required=True,
        help="YAML 配置文件路径，如 configs/phase1_binary.yaml。",
    )
    parser.add_argument(
        "--model_ckpt",
        required=True,
        help="模型权重路径（.pt/.pth），如 outputs/models/best.pt。",
    )
    parser.add_argument(
        "--data_list",
        default=None,
        help="JSON 数据清单路径（list of {image,label}）；"
             "若为 None 则用 config['data']['splits_dir']/test.json。",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="预测 NIfTI 输出目录；若为 None 则用 config['output']['predictions_dir']。",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="覆盖配置中的设备（如 cuda / cpu）；若为 None 则用 config['device']。",
    )
    parser.add_argument(
        "--no-cc",
        action="store_true",
        default=False,
        help="禁用「保留最大连通域」后处理（默认是否启用由 config['infer']['keep_largest_cc'] 决定）。",
    )
    return parser.parse_args()


def main() -> None:
    """脚本主入口：解析参数 → 加载配置/数据/模型 → 滑动窗口推理 → 后处理 → 保存 NIfTI。"""
    args = parse_args()

    # —— 1) 读取配置 ——
    config = load_config(args.config)

    # —— 2) 确定设备：命令行 --device 优先，否则用 config['device'] ——
    # get_device 会在请求 cuda 但不可用时自动回退到 CPU 并打印警告。
    device_str = args.device if args.device is not None else config["device"]
    device = get_device(device_str)

    # —— 3) 加载数据清单（list of {image, label}）——
    if args.data_list is not None:
        data_list_path = args.data_list
    else:
        # 默认用 splits_dir 下的 test.json
        data_list_path = os.path.join(config["data"]["splits_dir"], "test.json")
    if not os.path.isfile(data_list_path):
        print(f"[错误] 数据清单不存在：{data_list_path}\n"
              "请先用 scripts/split_data.py 生成 test.json，或用 --data_list 指定正确路径。")
        sys.exit(1)
    data_list = load_json(data_list_path)
    print(f"[信息] 数据清单：{data_list_path}（共 {len(data_list)} 个样本）")

    # —— 4) 确定输出目录并创建 ——
    output_dir = args.output if args.output is not None else config["output"]["predictions_dir"]
    ensure_dir(output_dir)

    # —— 5) 构建测试 DataLoader（test 变换：不裁剪、不增强，保留原尺寸，交给滑动窗口分块）——
    # build_dataloader 在 phase='test' 时 batch_size=1、shuffle=False，便于逐样本整卷推理。
    loader = build_dataloader(data_list, config, "test")

    # —— 6) 构建模型并加载权重，切换到 eval 模式（关闭 BN/Dropout 的训练行为）——
    if not os.path.isfile(args.model_ckpt):
        print(f"[错误] 模型权重不存在：{args.model_ckpt}\n"
              "请确认训练（train.py）已产出权重文件，或用 --model_ckpt 指定正确路径。")
        sys.exit(1)
    model = build_model(config)
    model = load_model(model, args.model_ckpt, device)
    model.eval()  # 推理必须 eval：BatchNorm 用全局统计量、Dropout 关闭

    # —— 7) 构建滑动窗口推理器（参数取自 config['infer']，与 trainer 验证时一致）——
    infer_cfg = config["infer"]
    inferer = SlidingWindowInferer(
        roi_size=tuple(infer_cfg["roi_size"]),        # 窗口块大小，如 (96, 96, 96)
        sw_batch_size=infer_cfg["sw_batch_size"],     # 并发窗口数；显存不足时调小
        overlap=infer_cfg["overlap"],                 # 相邻窗口重叠比例，重叠区加权平均
    )

    # —— 8) 后处理开关：默认由 config 决定，--no-cc 可强制关闭 ——
    use_cc = infer_cfg.get("keep_largest_cc", True) and (not args.no_cc)
    num_classes = config["num_classes"]  # 2=二分类(仅肝脏), 3=三分类(肝脏+肿瘤)

    print("=" * 60)
    print(f"配置文件 : {args.config}")
    print(f"模型权重 : {args.model_ckpt}")
    print(f"数据清单 : {data_list_path}")
    print(f"输出目录 : {output_dir}")
    print(f"设备     : {device}")
    print(f"滑动窗口 : roi_size={tuple(infer_cfg['roi_size'])}, "
          f"sw_batch_size={infer_cfg['sw_batch_size']}, overlap={infer_cfg['overlap']}")
    print(f"类别数   : {num_classes}（{'二分类' if num_classes == 2 else '三分类'}）")
    print(f"最大连通域后处理 : {'启用' if use_cc else '关闭'}")
    print("=" * 60)

    # —— 9) 推理主循环：torch.no_grad() 关闭梯度，节省显存与加速 ——
    sample_idx = 0  # 已处理样本计数（同时作为 data_list 的索引：test 不打乱、batch=1）
    with torch.no_grad():
        # tqdm 显示进度条；unit="case" 表示单位为「病例」
        for batch in tqdm(loader, desc="滑动窗口推理", unit="case"):
            # 取图像并搬到设备：(B, 1, D, H, W) float32
            inputs = batch["image"].to(device)

            # 滑动窗口推理：inferer 内部把整卷切成若干 roi_size 小块逐块前向再拼接；
            # 输出 (B, C, D, H, W) 是每类的 logit 得分（C=num_classes）。
            outputs = inferer(inputs, model)

            # argmax 解码：沿类别维(dim=1)取得分最大的类别，得到 (B, 1, D, H, W) 整数分割图。
            pred = outputs.argmax(dim=1, keepdim=True)

            # 逐样本处理（test 的 batch_size=1，循环通常只跑 1 次，写循环是为稳健）
            for i in range(pred.shape[0]):
                # 取该样本的 3D 预测 (D, H, W) 并转成紧凑的 uint8（类别 id 0/1/2 足够）
                pred_np = pred[i, 0].cpu().numpy().astype(np.uint8)

                # —— 后处理：保留最大连通域（去假阳性碎片）——
                if use_cc:
                    # 对预测中出现的「每个前景类别」分别保留最大连通域：
                    #   phase1（num_classes=2）：前景只有「肝脏」(类别 1) 一类；
                    #   phase2（num_classes=3）：前景有「肝脏」(1) 与「肿瘤」(2) 两类。
                    # 分别处理每一类，可避免把两类碎片互相串连、误删——例如不会因某片
                    # 肿瘤小于肝脏主连通域，就把它当作"碎片"清掉。
                    for cls in np.unique(pred_np):
                        if cls == 0:
                            continue  # 跳过背景
                        # 该类的二值掩膜
                        binary = (pred_np == cls)
                        # 仅保留该类的最大连通域
                        largest = keep_largest_connected_component(binary)
                        # 把该类中「非最大连通域」的体素清零（去碎片）
                        pred_np[binary & (~largest)] = 0

                # —— 取仿射矩阵：按项目约定，从原始图像文件重新读取 affine ——
                # 说明：MONAI 的 test 变换含 Orientation(RAS)+Spacing(重采样到 1mm)，
                # 故 pred_np 处于「重采样后的 RAS/1mm」空间，数组尺寸与原图不同；
                # 这里沿用项目简化约定，仍挂上原始图像的 affine，便于后续按原文件名
                # 与原空间对照做可视化/STL 导出。
                image_path = data_list[sample_idx]["image"]
                _, affine = load_nifti(image_path)

                # —— 输出文件名与输入图像同名，放到 output 目录 ——
                out_path = os.path.join(output_dir, os.path.basename(image_path))
                save_nifti(pred_np, affine, out_path)

                # 用 tqdm.write 输出，避免打断进度条
                tqdm.write(f"  -> 已保存：{out_path}")

                sample_idx += 1

    # —— 10) 打印完成汇总 ——
    print("=" * 60)
    print(f"[完成] 共推理并保存 {sample_idx} 个样本")
    print(f"[输出] 预测目录：{output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

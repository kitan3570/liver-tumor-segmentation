#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py —— 基于 MONAI 3D U-Net 的肝脏 / 肝肿瘤分割训练脚本（自包含）

【作用】
    本脚本是“自包含”的训练入口：读 YAML 配置 -> 构建 MONAI 3D U-Net ->
    在 Task03_Liver 三分类（背景 0 / 肝脏 1 / 肿瘤 2）数据上训练：
    完成 CT 加载、重采样、HU 窗、前景裁剪、随机裁剪与数据增强，
    在验证集上用滑动窗口推理 + DiceMetric 计算肝脏 / 肿瘤 dice，
    保存 best_model.pth / last_model.pth 并写 TensorBoard。
    本脚本不复用既有 src/trainer.py，所有逻辑写在同一文件内，便于大一同学阅读。

【运行方式】
    在项目根目录（d:\\项目1）下执行（推荐先冒烟测试 max_epochs=2）：
        python src/train.py --config configs/liver_unet_3d.yaml \
            --train_json data/splits/train.json --val_json data/splits/val.json \
            --output_dir outputs/models --max_epochs 2

    参数说明（均支持 --help 查看）：
        --config        YAML 配置文件路径（默认 configs/liver_unet_3d.yaml）
        --train_json    覆盖 config.data.train_json
        --val_json      覆盖 config.data.val_json
        --output_dir    模型与 checkpoint_meta 的输出目录
        --max_epochs    覆盖 config.training.max_epochs
        --batch_size    覆盖 config.training.batch_size
        --lr            覆盖 config.training.lr
        --num_workers   覆盖 config.training.num_workers
        --device        指定设备（如 "cuda" / "cpu"），未指定则自动检测

【输入与输出】
    输入：
        splits JSON：list，元素为 {"image": <abs path>, "label": <abs path>}（create_splits 产物）
    输出：
        <output_dir>/best_model.pth        验证 mean_dice 最高的模型权重（state_dict）
        <output_dir>/last_model.pth        最近一个 epoch 的模型权重（用于 resume）
        <output_dir>/checkpoint_meta.json   存 start_epoch，用于断点续训
        TensorBoard 日志默认写入 <output_dir>/runs（SummaryWriter）

【常见错误】
    1) CUDA 不可用         —— 脚本会自动回退到 CPU（速度很慢，仅小规模可用）。
    2) 显存不足（OOM）     —— 捕获 OutOfMemoryError 后提示：减小 roi_size / batch_size / num_samples(=4)。
    3) splits 未生成       —— 提示 train_json / val_json 不存在并退出，请先运行 create_splits.py。
    4) Windows 多进程报错  —— 可把 --num_workers 设为 0 规避 DataLoader 多进程问题。
"""

# ====================================================================
# 一、import 分组
# ====================================================================
# 1) 标准库
import os           # 路径拼接、目录创建、文件存在性判断
import json         # 读写 checkpoint_meta.json、读取 splits JSON
import argparse     # 命令行参数解析
import sys          # sys.path 注入项目根、sys.exit 退出
import time         # 计时：benchmark 模式下记录每 epoch 耗时

# 2) 第三方：PyTorch
import torch                        # 张量与设备管理
import torch.optim as optim         # AdamW 优化器
from torch.optim.lr_scheduler import OneCycleLR  # 学习率调度器
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 日志

# 3) 第三方：MONAI
import monai
from monai.utils import set_determinism                   # 设置确定性随机种子（MONAI 1.6 正确 API）
from monai.transforms import (                             # 数据预处理 / 增强变换
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    SpatialPadd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    EnsureTyped,
    AsDiscrete,
    MapLabelValued,
    MapTransform,
)
from monai.data import CacheDataset, Dataset, PersistentDataset, DataLoader, decollate_batch  # 数据集与加载器
from monai.networks.nets import UNet                          # 3D U-Net 模型
from monai.losses import DiceCELoss                           # 组合 Dice + CE 损失
from monai.inferers import sliding_window_inference           # 滑动窗口推理
from monai.metrics import DiceMetric                          # 验证期 Dice 指标

# 4) 第三方：其它
import numpy as np   # 随机种子 / numpy 端可复现
import yaml          # 读取 YAML 配置
from tqdm import tqdm   # 训练 / 验证进度条

# 把项目根目录加入 sys.path，便于按模块化路径 import（如 from src import ...）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ====================================================================
# 二、配置加载与命令行覆盖
# ====================================================================
def parse_args():
    """
    解析命令行参数。

    策略：
        --config 先加载 YAML 作为默认值；
        --train_json / --val_json / --output_dir / --max_epochs / --batch_size
        / --lr / --num_workers / --device 这些参数默认值均为 None，
        仅在命令行显式给出（非 None）时覆盖配置文件的对应字段。
    """
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="基于 MONAI 3D U-Net 的肝脏 / 肝肿瘤分割训练脚本（自包含）。",
    )
    parser.add_argument("--config", default="configs/liver_unet_3d.yaml",
                        help="YAML 配置文件路径，默认 configs/liver_unet_3d.yaml")
    parser.add_argument("--train_json", default=None,
                        help="覆盖 config.data.train_json 的训练集 JSON 路径")
    parser.add_argument("--val_json", default=None,
                        help="覆盖 config.data.val_json 的验证集 JSON 路径")
    parser.add_argument("--output_dir", default="outputs/models",
                        help="模型与 checkpoint_meta 的输出目录，默认 outputs/models")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="覆盖 config.training.max_epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="覆盖 config.training.batch_size")
    parser.add_argument("--lr", type=float, default=None,
                        help="覆盖 config.training.lr（AdamW 学习率）")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="覆盖 config.training.num_workers（Windows 可设为 0）")
    parser.add_argument("--cache", "--cache_strategy", dest="cache", default=None,
                        choices=["none", "disk", "memory"],
                        help="训练集缓存策略：none=不缓存(最省内存,最慢)；"
                             "disk=磁盘持久化缓存到项目盘 outputs/.cache/（首次慢,后续快,省内存）；"
                             "memory=内存全缓存(最快,但需~120GB RAM,大数据集慎用)。默认 disk。"
                             "--cache_strategy 为同义别名，供 benchmark 脚本调用。")
    # AMP 混合精度：用 BooleanOptionalAction 同时支持 --amp / --no_amp。
    # 默认 False（关闭），与原项目一致；optimized 模式传 --amp，baseline 传 --no_amp。
    parser.add_argument("--amp", "--no_amp", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="启用 AMP 自动混合精度训练（fp16 前向 + fp32 反向）。"
                             "GPU 有 Tensor Core 时显存减半、速度提升 1.5-2x，精度损失可忽略。"
                             "仅 CUDA 设备生效；CPU 训练时自动忽略。默认关闭。"
                             "用 --no_amp 显式关闭（baseline 用）。")
    # —— 学习率调度器开关（benchmark baseline 用 none 关闭 OneCycleLR）——
    parser.add_argument("--scheduler", default=None, choices=["none", "onecycle"],
                        help="学习率调度器：none=不调度(恒定 lr，baseline)；"
                             "onecycle=OneCycleLR warmup+cosine（默认，与原项目一致）。")
    # —— 梯度裁剪开关与阈值 ——
    # 默认 None→开启（与原项目一致，max_norm=12.0）；baseline 用 --no_grad_clip 关闭。
    parser.add_argument("--grad_clip", "--no_grad_clip", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="梯度裁剪防爆炸。默认开启(max_norm=12.0，与原项目一致)；"
                             "用 --no_grad_clip 关闭（baseline 用）。")
    parser.add_argument("--grad_clip_max_norm", type=float, default=12.0,
                        help="梯度裁剪最大范数，默认 12.0（与原项目一致）。")
    # —— DataLoader 高级优化开关 ——
    # 默认 None→自动（cuda 时 pin_memory=True；num_workers>0 时 persistent_workers=True）。
    # baseline 用 --no_pin_memory / --no_persistent_workers / --no_non_blocking 显式关闭。
    parser.add_argument("--pin_memory", "--no_pin_memory", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="DataLoader pin_memory（锁页内存，加速 CPU→GPU 传输）。"
                             "默认自动(cuda→True)；--no_pin_memory 关闭（baseline 用）。")
    parser.add_argument("--persistent_workers", "--no_persistent_workers",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="DataLoader persistent_workers（复用 worker 进程，省启动开销）。"
                             "默认自动(num_workers>0→True)；--no_persistent_workers 关闭（baseline 用）。"
                             "Windows 下 num_workers=0 时自动关闭，避免报错。")
    parser.add_argument("--non_blocking", "--no_non_blocking", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="数据 .to(device, non_blocking=...)，让搬运与计算重叠。"
                             "默认 True（与原项目一致）；--no_non_blocking 关闭（baseline 用）。")
    # —— 验证期 AMP 开关（独立于训练 AMP）——
    # 默认 None→跟随 use_amp；baseline 用 --no_val_amp 显式关闭。
    parser.add_argument("--val_amp", "--no_val_amp", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="验证期滑动窗口推理用 AMP 加速。默认跟随 --amp；"
                             "--no_val_amp 关闭（baseline 用）。")
    # —— benchmark 模式 ——
    parser.add_argument("--benchmark_mode", action="store_true", default=False,
                        help="benchmark 模式：记录每 epoch 的 loss/dice/耗时/显存/lr，"
                             "训练结束（含失败）时把结果写到 --benchmark_output_json。"
                             "不影响训练流程本身，仅在末尾多写一个 JSON。")
    parser.add_argument("--benchmark_output_json", default=None,
                        help="benchmark 模式结果 JSON 输出路径（仅 --benchmark_mode 时生效）。")
    parser.add_argument("--device", default=None,
                        help='指定设备，如 "cuda" 或 "cpu"；未指定则自动检测')
    parser.add_argument(
        "--task_mode", default=None, choices=["binary_liver", "multiclass"],
        help=(
            "任务模式：binary_liver=二分类(背景/肝脏，label>0→1，out_channels=2)；"
            "multiclass=三分类(背景/肝脏/肿瘤，保持 0/1/2，out_channels=3)。"
            "未指定时取 config.data.task_mode，默认 multiclass。"
        ),
    )
    return parser.parse_args()


def build_config(args):
    """
    读取 YAML，并把命令行非 None 参数覆盖到配置上。

    返回：
        cfg: dict，包含 data / preprocessing / model / training / inference /
             output_dir / device 等运行所需全部字段。
    """
    # 1) 读取 YAML 配置文件
    if not os.path.isfile(args.config):
        print("[错误] 配置文件不存在: {}".format(args.config))
        sys.exit(1)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2) 命令行覆盖：仅在参数非 None 时覆盖
    if args.train_json is not None:
        cfg["data"]["train_json"] = args.train_json
    if args.val_json is not None:
        cfg["data"]["val_json"] = args.val_json
    if args.max_epochs is not None:
        cfg["training"]["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.num_workers is not None:
        cfg["training"]["num_workers"] = args.num_workers

    # 缓存策略：none/disk/memory。默认 disk（持久化到项目盘，省内存且加速二次训练）。
    if args.cache is not None:
        cfg["training"]["cache"] = args.cache
    else:
        cfg["training"].setdefault("cache", "disk")

    # 3) 输出目录与设备单独存放（不在 YAML 内）
    cfg["output_dir"] = args.output_dir
    # 设备自动检测：未显式指定时，有 CUDA 用 CUDA，否则用 CPU
    if args.device is not None:
        cfg["device"] = args.device
    else:
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # AMP 混合精度开关：仅 CUDA 生效（必须在 device 确定后判断）
    # args.amp 为 bool（BooleanOptionalAction，默认 False）；--no_amp → False
    cfg["use_amp"] = args.amp and cfg["device"].startswith("cuda")

    # —— 速度优化相关开关（benchmark 用，默认保持与原项目一致）——
    is_cuda = cfg["device"].startswith("cuda")
    num_workers = cfg["training"]["num_workers"]

    # 学习率调度器：默认 onecycle（原项目行为）；--scheduler none 关闭（恒定 lr）
    cfg["scheduler"] = args.scheduler if args.scheduler is not None else "onecycle"

    # 梯度裁剪：默认开启（原项目行为，max_norm=12.0）；--no_grad_clip 关闭
    cfg["grad_clip"] = True if args.grad_clip is None else args.grad_clip
    cfg["grad_clip_max_norm"] = args.grad_clip_max_norm

    # DataLoader pin_memory：默认自动(cuda→True)；--no_pin_memory 关闭
    cfg["pin_memory"] = is_cuda if args.pin_memory is None else args.pin_memory

    # persistent_workers：默认自动(num_workers>0→True)；
    # Windows 下 num_workers=0 时强制 False，避免 "persistent_workers needs num_workers>0" 报错
    if args.persistent_workers is None:
        cfg["persistent_workers"] = num_workers > 0
    else:
        # 即使用户传 --persistent_workers，num_workers=0 时仍强制关闭（避免报错）
        cfg["persistent_workers"] = args.persistent_workers and num_workers > 0

    # non_blocking：默认 True（原项目行为）；--no_non_blocking 关闭
    cfg["non_blocking"] = True if args.non_blocking is None else args.non_blocking

    # 验证期 AMP：默认跟随训练 use_amp；--no_val_amp / --val_amp 显式覆盖
    cfg["val_amp"] = cfg["use_amp"] if args.val_amp is None else (args.val_amp and is_cuda)

    # benchmark 模式与结果 JSON 路径
    cfg["benchmark_mode"] = args.benchmark_mode
    cfg["benchmark_output_json"] = args.benchmark_output_json

    # 4) 任务模式 task_mode：命令行优先，其次 config.data.task_mode，默认 multiclass
    #    - binary_liver : 背景vs肝脏二分类，label>0→1，out_channels=2，num_classes=2
    #    - multiclass   : 背景/肝脏/肿瘤三分类，保持 0/1/2，out_channels=3，num_classes=3
    if args.task_mode is not None:
        task_mode = args.task_mode
    else:
        task_mode = cfg.get("data", {}).get("task_mode", "multiclass")
    if task_mode not in ("binary_liver", "multiclass"):
        print("[错误] 未知 task_mode: {}（仅支持 binary_liver / multiclass）".format(task_mode))
        sys.exit(1)
    cfg["task_mode"] = task_mode
    num_classes = 2 if task_mode == "binary_liver" else 3
    # 自动覆盖模型输出通道与类别数（保证与模式一致，避免配置不匹配）
    cfg["model"]["out_channels"] = num_classes
    if "data" not in cfg:
        cfg["data"] = {}
    cfg["data"]["num_classes"] = num_classes
    cfg["num_classes"] = num_classes  # 顶层别名，便于各函数取用
    return cfg


# ====================================================================
# 三、数据预处理 / 增强链
# ====================================================================
def _maybe_binarize_label(cfg):
    """二分类模式时返回把 label>0 映射为 1 的 MapLabelValued 变换；
    三分类模式返回 None（保持 0/1/2 不变）。

    放在 Spacingd 之后、CropForegroundd 之前，确保重采样插值用的是原始标签，
    再做二值化（nearest 插值不会产生非法值，二值化后只剩 0/1）。
    """
    if cfg.get("task_mode") != "binary_liver":
        return None
    # orig_labels=[0,1,2] -> target_labels=[0,1,1]：背景=0，肝脏与肿瘤都并为肝脏=1
    return MapLabelValued(keys="label", orig_labels=[0, 1, 2], target_labels=[0, 1, 1])


def build_train_transforms(cfg):
    """
    构建训练数据变换链（含随机增强与随机裁剪）。

    步骤说明（面向大一同学）：
        LoadImaged            : 读取 .nii.gz 为 (H,W,D) numpy/tensor；
        EnsureChannelFirstd   : 把通道维提到最前 -> (C,H,W,D)，CT 单通道即 (1,H,W,D)；
        Orientationd(axcodes="RAS") : 统一到右前上(RAS)坐标系，消除采集方向差异；
        Spacingd               : 按目标体素间距(1.5,1.5,2.0)重采样，图像用 bilinear，
                                  标签用 nearest（避免插值产生非法类别值）；
        [binary_liver] MapLabelValued : 二分类模式把 1/2 都映射为 1（背景vs肝脏）；
        ScaleIntensityRanged   : 对 image 做 HU 窗 [-200,250] 线性映射到 [0,1] 并 clip；
        CropForegroundd        : 用 image 的非零前景裁掉空气区域，减小计算量；
        RandCropByPosNegLabeld : 随机裁 96x96x96 patch，正负样本各 1 个，num_samples=4
                                  -> 单条样本会被展开成 4 个 patch 列表，
                                  DataLoader 的 list_data_collate 会自动 flatten；
        RandFlipd / RandRotate90d / RandShiftIntensityd : 几何与强度增强，提升泛化；
        EnsureTyped            : 转为 torch.Tensor 并保证设备类型一致。
    """
    spacing = tuple(cfg["preprocessing"]["spacing"])
    roi = tuple(cfg["preprocessing"]["roi_size"])
    a_min, a_max = cfg["preprocessing"]["ct_window"]
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
    ]
    binarize = _maybe_binarize_label(cfg)
    if binarize is not None:
        transforms.append(binarize)
    transforms += [
        ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max,
                             b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # 关键修复：某些 CT 重采样后某轴体素数 < roi_size（如 z=62 < 96），
        # RandCropByPosNegLabeld 会抛 ValueError。SpatialPadd 用 0 填充到至少
        # roi_size（已大于则不操作），保证后续随机裁剪合法。
        # Task03_Liver 的 liver_0 等薄切片案例需此修复。
        SpatialPadd(keys=["image", "label"], spatial_size=roi, mode="constant"),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
                               spatial_size=roi, pos=1, neg=1, num_samples=4),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
        RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        EnsureTyped(keys=["image", "label"]),
    ]
    return Compose(transforms)


def build_val_transforms(cfg):
    """
    构建验证数据变换链（**不含 RandCrop 与任何随机增强**）。

    验证只做确定性预处理：加载 -> 通道优先 -> RAS -> 重采样 ->
    [binary_liver] 二值化 -> HU 窗 -> 前景裁剪 -> 转类型，
    保证每次评估结果可复现，再用 sliding_window_inference 在整图上推理。
    """
    spacing = tuple(cfg["preprocessing"]["spacing"])
    a_min, a_max = cfg["preprocessing"]["ct_window"]
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
    ]
    binarize = _maybe_binarize_label(cfg)
    if binarize is not None:
        transforms.append(binarize)
    transforms += [
        ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max,
                             b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ]
    return Compose(transforms)


# ====================================================================
# 四、模型 / 损失 / 优化器 构建与 resume
# ====================================================================
def build_model(cfg):
    """
    根据 cfg['model'] 构建 MONAI 3D U-Net，并移动到目标设备。

    关键参数：
        spatial_dims=3, in_channels=1(CT单通道), out_channels=3(背景/肝脏/肿瘤),
        channels=(16,32,64,128,256), strides=(2,2,2,2), num_res_units=2
    """
    m = cfg["model"]
    model = UNet(
        spatial_dims=m["spatial_dims"],
        in_channels=m["in_channels"],
        out_channels=m["out_channels"],
        channels=tuple(m["channels"]),
        strides=tuple(m["strides"]),
        num_res_units=m["num_res_units"],
    )
    model = model.to(cfg["device"])
    return model


def build_loss_optimizer(cfg, model):
    """
    构建损失函数、优化器与学习率调度器：
        DiceCELoss(to_onehot_y=True, softmax=True)：自动把标签做 one-hot，
        并对网络 logits 做 softmax，再算 Dice + CE 的组合损失，适合多分类分割。
        AdamW(lr, weight_decay)：自适应矩估计 + 解耦权重衰减。
        OneCycleLR：学习率 warmup + cosine 衰减，收敛更快、最终精度更高。
            max_lr = 配置的 lr；pct_start=0.3 前 30% 轮数线性升温到 max_lr，
            后 70% 轮数 cosine 降到 max_lr/1000。

    scheduler 开关：cfg["scheduler"] == "none" 时返回 None（恒定 lr，benchmark
    baseline 用）；默认 "onecycle" 与原项目一致。真正的 total_steps 在 main()
    拿到 train_loader 后重建。
    """
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    # scheduler == "none"：不建调度器（恒定 lr，benchmark baseline）；否则建占位 OneCycleLR
    if cfg.get("scheduler", "onecycle") == "none":
        scheduler = None
    else:
        # OneCycleLR 需要 total_steps = epochs * steps_per_epoch
        # 此处先建占位，真正 step 数在 main() 里拿到 train_loader 后再修正
        scheduler = OneCycleLR(
            optimizer,
            max_lr=cfg["training"]["lr"],
            total_steps=cfg["training"]["max_epochs"] * 100,  # 占位，main 中覆盖
            pct_start=0.3,
            anneal_strategy="cos",
            cycle_momentum=True,
        )
    return loss_function, optimizer, scheduler


def maybe_resume(cfg, model):
    """
    断点续训：若 <output_dir>/last_model.pth 存在，则加载权重到 model，
    并从同名 checkpoint_meta.json 读 start_epoch；若 meta 缺失则 start_epoch=0。

    返回：
        start_epoch: int，本轮训练应从第几个 epoch 开始（已训练的数量）。
    """
    last_path = os.path.join(cfg["output_dir"], "last_model.pth")
    meta_path = os.path.join(cfg["output_dir"], "checkpoint_meta.json")
    if not os.path.isfile(last_path):
        return 0
    # 加载权重
    state_dict = torch.load(last_path, map_location=cfg["device"])
    model.load_state_dict(state_dict)
    # 读取起始 epoch
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        start_epoch = int(meta.get("start_epoch", 0))
    else:
        start_epoch = 0
    print("[resume] 从 epoch {} 恢复（已加载 {}）。".format(start_epoch, last_path))
    return start_epoch


# ====================================================================
# 五、数据集与 DataLoader
# ====================================================================
def load_split(json_path):
    """读取 splits JSON 文件并返回 list[{image,label}]。"""
    if not os.path.isfile(json_path):
        print("[错误] splits 文件不存在: {}（请先运行 create_splits.py）".format(json_path))
        sys.exit(1)
    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    return items


def build_dataloaders(cfg):
    """
    构建训练 / 验证数据集与 DataLoader。

    训练集缓存策略由 cfg['training']['cache'] 控制（三档）：
      - "none"   ：Dataset，不缓存，每 epoch 实时重采样。最省内存但最慢。
      - "disk"   ：PersistentDataset，把预处理后的体素持久化到项目盘
                   outputs/.cache/。首次运行计算并写入，后续 epoch/训练直接读缓存，
                   既省内存（仅当前 batch 在内存）又加速。27GB 大数据集推荐此模式。
      - "memory" ：CacheDataset(cache_rate=1.0)，全量缓存到内存。最快但需 ~120GB RAM，
                   大数据集慎用（会撑爆 pagefile）。
    验证集始终不缓存（Dataset + num_workers=0），避免整卷重采样 OOM。
    """
    train_json = cfg["data"]["train_json"]
    val_json = cfg["data"]["val_json"]
    train_files = load_split(train_json)
    val_files = load_split(val_json)
    print("[数据] 训练样本 {} 例，验证样本 {} 例。".format(len(train_files), len(val_files)))

    train_transforms = build_train_transforms(cfg)
    val_transforms = build_val_transforms(cfg)

    cache_mode = cfg["training"].get("cache", "disk")

    # —— 训练集：按缓存策略选择 Dataset 类型 ——
    if cache_mode == "memory":
        print("[数据] 训练集 CacheDataset（内存全缓存，最快但占大量 RAM）")
        train_ds = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=1.0,
            num_workers=cfg["training"]["num_workers"],
        )
    elif cache_mode == "disk":
        # 持久化缓存目录放在项目盘，避免写 C 盘
        cache_dir = os.path.join(cfg["output_dir"], ".cache", "train")
        os.makedirs(cache_dir, exist_ok=True)
        print("[数据] 训练集 PersistentDataset（磁盘缓存到 {}，省内存且二次训练快）".format(cache_dir))
        train_ds = PersistentDataset(
            data=train_files,
            transform=train_transforms,
            cache_dir=cache_dir,
        )
    else:  # "none"
        print("[数据] 训练集 Dataset（不缓存，实时处理，最省内存）")
        train_ds = Dataset(
            data=train_files,
            transform=train_transforms,
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        # pin_memory / persistent_workers 由 cfg 控制（benchmark baseline 可关闭）；
        # build_config 已保证 num_workers=0 时 persistent_workers=False（Windows 安全）
        pin_memory=cfg.get("pin_memory", cfg["device"].startswith("cuda")),
        persistent_workers=cfg.get("persistent_workers", cfg["training"]["num_workers"] > 0),
    )

    # —— 验证集：不缓存，num_workers=0 避免 OOM ——
    val_ds = Dataset(
        data=val_files,
        transform=val_transforms,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=0,
                            pin_memory=cfg.get("pin_memory", cfg["device"].startswith("cuda")))
    return train_loader, val_loader


# ====================================================================
# 六、训练一个 epoch
# ====================================================================
def train_one_epoch(cfg, model, loader, loss_function, optimizer, scheduler,
                    scaler, epoch, writer):
    """
    训练单个 epoch（支持 AMP 混合精度 + OneCycleLR 调度 + 梯度裁剪）：
        1) model.train()；遍历 DataLoader；
        2) 数据 non_blocking 搬到 GPU（与数据加载重叠）；
        3) autocast(fp16) 前向 -> GradScaler 反向 + step；
        4) 梯度裁剪防止爆炸；optimizer.step 后 scheduler.step()；
        5) tqdm 显示 loss 与 lr。
    """
    model.train()
    epoch_loss = 0.0
    step = 0
    use_amp = cfg.get("use_amp", False) and cfg["device"].startswith("cuda")
    # non_blocking / grad_clip / grad_clip_max_norm 由 cfg 控制（benchmark baseline 可关闭）
    non_blocking = cfg.get("non_blocking", True)
    do_grad_clip = cfg.get("grad_clip", True)
    grad_clip_max_norm = cfg.get("grad_clip_max_norm", 12.0)
    pbar = tqdm(loader, desc="train epoch {}".format(epoch), leave=False)
    for batch in pbar:
        # non_blocking=True 让数据搬运与上一 step 的 GPU 计算重叠
        inputs = batch["image"].to(cfg["device"], non_blocking=non_blocking)
        labels = batch["label"].to(cfg["device"], non_blocking=non_blocking)

        optimizer.zero_grad(set_to_none=True)  # set_to_none 比 zero_grad 更快

        if use_amp:
            # AMP 混合精度：前向用 fp16 加速，反向用 GradScaler 防梯度下溢
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            if do_grad_clip:
                # 梯度裁剪（先 unscale 再 clip，否则裁剪的是缩放后的梯度）
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            if do_grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip_max_norm)
            optimizer.step()

        # OneCycleLR 每个 batch step 一次；scheduler=None（baseline 恒定 lr）时跳过
        if scheduler is not None:
            scheduler.step()

        epoch_loss += loss.item()
        step += 1
        pbar.set_postfix({
            "loss": "{:.4f}".format(loss.item()),
            "lr": "{:.2e}".format(optimizer.param_groups[0]["lr"]),
        })
    avg_loss = epoch_loss / max(step, 1)
    return avg_loss


# ====================================================================
# 七、验证
# ====================================================================
def run_validation(cfg, model, val_loader, epoch, writer):
    """
    在验证集上执行滑动窗口推理并计算 Dice。

    流程：
        1) model.eval()，不计算梯度；
        2) 对每例整图用 sliding_window_inference(roi_size=(96,96,96), sw_batch_size=4)
           得到 (1,C,D,H,W) 的 logits；
        3) 后处理 AsDiscrete(argmax=True, to_onehot=num_classes)：先 argmax 得预测类别，
           再转成 num_classes 通道 one-hot，与 one-hot 标签对齐；
        4) DiceMetric(include_background=False, reduction="mean_batch") 计算：
           aggregate() 返回形状 (num_classes-1,)，
           - multiclass: (2,) -> result[0]=liver dice(类1), result[1]=tumor dice(类2)
           - binary_liver: (1,) -> result[0]=liver dice(类1)，tumor_dice 记 0.0
        5) mean_dice = (liver_dice + tumor_dice)/2（binary 时即 liver_dice），
           作为挑选 best 的依据。
    """
    model.eval()
    roi_size = tuple(cfg["inference"]["roi_size"])
    sw_batch_size = cfg["inference"]["sw_batch_size"]
    num_classes = cfg["num_classes"]  # 2 (binary_liver) 或 3 (multiclass)
    task_mode = cfg.get("task_mode", "multiclass")
    # 验证期 AMP 独立于训练 AMP：cfg["val_amp"] 默认跟随 use_amp（benchmark baseline 可关）
    use_amp = cfg.get("val_amp", cfg.get("use_amp", False)) and cfg["device"].startswith("cuda")
    non_blocking = cfg.get("non_blocking", True)
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=num_classes)])
    post_label = Compose([AsDiscrete(to_onehot=num_classes)])  # 标签已是类别图，直接 one-hot

    # mean_batch：对每个类在 batch 内取均值，结果按类排列
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    liver_vals, tumor_vals = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="val epoch {}".format(epoch), leave=False):
            inputs = batch["image"].to(cfg["device"], non_blocking=non_blocking)
            labels = batch["label"].to(cfg["device"], non_blocking=non_blocking)
            # 验证时用 AMP autocast + half 精度输入，加速滑动窗口推理
            if use_amp:
                inputs = inputs.half()
                with torch.cuda.amp.autocast():
                    outputs = sliding_window_inference(
                        inputs, roi_size, sw_batch_size, model,
                        device=cfg["device"], sw_device=cfg["device"],
                    )
                outputs = outputs.float()  # 后处理用 fp32 保证精度
            else:
                # 滑动窗口推理：输出 (1,C,D,H,W)
                outputs = sliding_window_inference(
                    inputs, roi_size, sw_batch_size, model, device=cfg["device"]
                )
            # decollate 拆分 batch 后逐例做 argmax+onehot
            outputs_list = [post_pred(o) for o in decollate_batch(outputs)]
            labels_list = [post_label(l) for l in decollate_batch(labels)]
            dice_metric(y_pred=outputs_list, y=labels_list)
            # aggregate() 形状 (num_classes-1,)：[0]=liver；multiclass 时 [1]=tumor
            result = dice_metric.aggregate()
            result_np = result.cpu().numpy()
            liver_dice = float(result_np[0])
            if task_mode == "binary_liver":
                tumor_dice = 0.0  # 二分类无肿瘤类别
            else:
                tumor_dice = float(result_np[1])
            liver_vals.append(liver_dice)
            tumor_vals.append(tumor_dice)
            dice_metric.reset()

    # 整个验证集上的平均
    liver_dice = float(np.mean(liver_vals)) if liver_vals else 0.0
    if task_mode == "binary_liver":
        tumor_dice = 0.0
        mean_dice = liver_dice  # 二分类只看 liver dice
    else:
        tumor_dice = float(np.mean(tumor_vals)) if tumor_vals else 0.0
        mean_dice = (liver_dice + tumor_dice) / 2.0
    return liver_dice, tumor_dice, mean_dice


# ====================================================================
# 八、保存与元信息
# ====================================================================
def save_checkpoint(cfg, model, meta):
    """保存 last_model.pth 并写出 checkpoint_meta.json（记录 start_epoch 供下次 resume）。"""
    last_path = os.path.join(cfg["output_dir"], "last_model.pth")
    meta_path = os.path.join(cfg["output_dir"], "checkpoint_meta.json")
    torch.save(model.state_dict(), last_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def save_best(cfg, model):
    """当验证 mean_dice 创新高时保存最佳权重。"""
    best_path = os.path.join(cfg["output_dir"], "best_model.pth")
    torch.save(model.state_dict(), best_path)


# ====================================================================
# 九、主训练循环
# ====================================================================
def main():
    # ---------- 配置加载 ----------
    args = parse_args()
    cfg = build_config(args)

    # ---------- 重定向临时目录到项目盘 ----------
    # 防止 PyTorch / MONAI / tqdm 往 C:\Users\...\Temp 写大文件（如 DataLoader
    # 的 shared memory、TensorBoard 事件文件等），把临时目录设到项目所在盘。
    _proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _tmp_dir = os.path.join(_proj_root, "outputs", ".tmp")
    os.makedirs(_tmp_dir, exist_ok=True)
    os.environ["TEMP"] = _tmp_dir
    os.environ["TMP"] = _tmp_dir

    # ---------- 随机种子 ----------
    seed = cfg["training"]["seed"]
    set_determinism(seed)          # MONAI 提供的统一确定性种子设置（兼容 1.6+）
    # 额外再固定 torch / numpy 端，进一步保证可复现
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)

    # ---------- 输出目录 ----------
    os.makedirs(cfg["output_dir"], exist_ok=True)

    print("[配置] 设备: {}".format(cfg["device"]))
    print("[配置] task_mode={}, num_classes={}, out_channels={}".format(
        cfg["task_mode"], cfg["num_classes"], cfg["model"]["out_channels"]))
    print("[配置] max_epochs={}, batch_size={}, lr={}, num_workers={}".format(
        cfg["training"]["max_epochs"], cfg["training"]["batch_size"],
        cfg["training"]["lr"], cfg["training"]["num_workers"]))
    print("[配置] cache={}（none=不缓存/disk=磁盘缓存/memory=内存缓存）".format(
        cfg["training"].get("cache", "disk")))
    print("[配置] AMP 混合精度={}（{}）".format(
        cfg.get("use_amp", False),
        "已启用，显存减半+加速" if cfg.get("use_amp", False) else "未启用"))
    # 打印速度优化开关状态（benchmark 与日常训练都可据此核对实际配置）
    print("[配置] scheduler={}, grad_clip={} (max_norm={}), val_amp={}".format(
        cfg.get("scheduler", "onecycle"), cfg.get("grad_clip", True),
        cfg.get("grad_clip_max_norm", 12.0), cfg.get("val_amp", cfg.get("use_amp", False))))
    print("[配置] pin_memory={}, persistent_workers={}, non_blocking={}".format(
        cfg.get("pin_memory"), cfg.get("persistent_workers"),
        cfg.get("non_blocking")))
    if cfg.get("benchmark_mode", False):
        print("[配置] benchmark 模式已开启，结果将写入 {}".format(
            cfg.get("benchmark_output_json")))

    # ---------- 数据预处理 / DataLoader ----------
    try:
        train_loader, val_loader = build_dataloaders(cfg)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
            print("[错误] 显存不足，请减小 roi_size / batch_size / num_samples(=4)。")
            sys.exit(1)
        raise

    # ---------- 模型 / 损失 / 优化器 / 调度器 ----------
    model = build_model(cfg)
    loss_function, optimizer, scheduler = build_loss_optimizer(cfg, model)

    # 修正 OneCycleLR 的 total_steps 为真实值（epochs × steps_per_epoch）
    # RandCropByPosNegLabeld(num_samples=4) 把每例展开成 4 个 patch，
    # 故 len(train_loader) = ceil(样本数 × 4 / batch_size)
    # scheduler == "none"（benchmark baseline）时跳过，保持恒定 lr
    steps_per_epoch = len(train_loader)
    if cfg.get("scheduler", "onecycle") == "onecycle":
        total_steps = max(steps_per_epoch * cfg["training"]["max_epochs"], 1)
        # OneCycleLR 已构建，用新 total_steps 重建（避免 total_steps=0 报错）
        scheduler = OneCycleLR(
            optimizer,
            max_lr=cfg["training"]["lr"],
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy="cos",
            cycle_momentum=True,
        )
        print("[配置] OneCycleLR: total_steps={}, steps_per_epoch={}, max_lr={}".format(
            total_steps, steps_per_epoch, cfg["training"]["lr"]))
    else:
        # baseline：恒定 lr，不调度
        print("[配置] scheduler=none（恒定 lr={}，benchmark baseline）".format(
            cfg["training"]["lr"]))

    # AMP GradScaler（仅 CUDA 时启用；CPU 时 scaler 为 None，train_one_epoch 走 fp32 分支）
    if cfg.get("use_amp", False):
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # ---------- resume ----------
    start_epoch = maybe_resume(cfg, model)

    # ---------- TensorBoard ----------
    writer = SummaryWriter(log_dir=os.path.join(cfg["output_dir"], "runs"))

    # ---------- benchmark 状态初始化 ----------
    # benchmark_mode 下记录每 epoch 的 loss/dice/耗时/显存/lr，结束时写 JSON。
    # 非 benchmark 模式这些变量保持空，不影响原有流程。
    is_bench = cfg.get("benchmark_mode", False)
    bench_epochs = []           # 每 epoch 一条记录
    bench_best_liver = -1.0     # 历次验证 liver dice 最大值
    bench_best_tumor = -1.0     # 历次验证 tumor dice 最大值
    bench_final_liver = float("nan")  # 最后一次验证的 liver dice
    bench_final_tumor = float("nan")  # 最后一次验证的 tumor dice
    bench_total_time = 0.0      # 累计 epoch 耗时（秒）
    bench_status = "ok"
    bench_error = ""
    # CUDA 峰值显存统计：训练开始前重置，每 epoch 末读取累计峰值
    is_cuda = cfg["device"].startswith("cuda")
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    # ---------- 训练循环 ----------
    best_dice = -1.0
    max_epochs = cfg["training"]["max_epochs"]
    val_interval = cfg["training"]["val_interval"]

    try:
        for epoch in range(start_epoch, max_epochs):
            epoch_t0 = time.time()
            # 训练
            avg_loss = train_one_epoch(cfg, model, train_loader, loss_function,
                                       optimizer, scheduler, scaler, epoch, writer)
            # 写 TensorBoard：训练损失
            writer.add_scalar("train/loss", avg_loss, epoch)
            print("[epoch {}/{}] train loss = {:.4f}".format(epoch + 1, max_epochs, avg_loss))

            # 验证（每 val_interval 个 epoch 一次）
            liver_dice, tumor_dice, mean_dice = float("nan"), float("nan"), float("nan")
            do_val = (epoch + 1) % val_interval == 0 or (epoch + 1) == max_epochs
            if do_val:
                liver_dice, tumor_dice, mean_dice = run_validation(
                    cfg, model, val_loader, epoch, writer)
                # 写 TensorBoard：各类 dice
                writer.add_scalar("val/liver_dice", liver_dice, epoch)
                writer.add_scalar("val/tumor_dice", tumor_dice, epoch)
                writer.add_scalar("val/mean_dice", mean_dice, epoch)
                print("[epoch {}/{}] val liver dice = {:.4f}, tumor dice = {:.4f}, "
                      "mean dice = {:.4f}".format(epoch + 1, max_epochs,
                                                  liver_dice, tumor_dice, mean_dice))
                # 保存 best
                if mean_dice > best_dice:
                    best_dice = mean_dice
                    save_best(cfg, model)
                    print("[保存] 新最佳 mean dice = {:.4f}，已写 best_model.pth".format(best_dice))
                # benchmark：更新 best/final dice
                if is_bench:
                    if liver_dice > bench_best_liver:
                        bench_best_liver = liver_dice
                    if tumor_dice > bench_best_tumor:
                        bench_best_tumor = tumor_dice
                    bench_final_liver = liver_dice
                    bench_final_tumor = tumor_dice

            # 每 epoch 末保存 last 与元信息
            save_checkpoint(cfg, model, meta={"start_epoch": epoch + 1})

            # benchmark：记录本 epoch 指标
            epoch_time = time.time() - epoch_t0
            if is_bench:
                bench_total_time += epoch_time
                gpu_mem = float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if is_cuda else float("nan")
                bench_epochs.append({
                    "epoch": epoch + 1,
                    "train_loss": float(avg_loss),
                    "val_liver_dice": liver_dice if do_val else float("nan"),
                    "val_tumor_dice": tumor_dice if do_val else float("nan"),
                    "epoch_time_sec": float(epoch_time),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "gpu_memory_MB": gpu_mem,
                })
                print("[benchmark] epoch {} 耗时 {:.1f}s，累计 {:.1f}s，峰值显存 {:.0f}MB".format(
                    epoch + 1, epoch_time, bench_total_time,
                    gpu_mem if not (gpu_mem != gpu_mem) else 0))
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
            print("[错误] 显存不足，请减小 roi_size / batch_size / num_samples(=4)。")
            if is_bench:
                bench_status = "failed"
                bench_error = "CUDA out of memory"
            sys.exit(1)
        if is_bench:
            bench_status = "failed"
            bench_error = "{}: {}".format(type(e).__name__, str(e))
        raise
    except Exception as e:
        # 任意异常：benchmark 记录失败原因后再抛出，保证 JSON 仍能写出
        if is_bench:
            bench_status = "failed"
            bench_error = "{}: {}".format(type(e).__name__, str(e))
        raise
    finally:
        writer.close()
        # benchmark：无论成功/失败/异常都写出 JSON，便于对比脚本读取真实结果
        if is_bench:
            _write_benchmark_json(cfg, bench_epochs, bench_total_time,
                                  bench_best_liver, bench_best_tumor,
                                  bench_final_liver, bench_final_tumor,
                                  bench_status, bench_error)

    print("[完成] 训练结束，最佳验证 mean dice = {:.4f}。".format(best_dice))
    # 明确告知权重路径，便于 GUI / 推理脚本直接取用
    print("[输出] best_model: {}".format(os.path.join(cfg["output_dir"], "best_model.pth")))
    print("[输出] last_model: {}".format(os.path.join(cfg["output_dir"], "last_model.pth")))


def _write_benchmark_json(cfg, epochs, total_time, best_liver, best_tumor,
                          final_liver, final_tumor, status, error):
    """把 benchmark 训练结果写到 cfg['benchmark_output_json']。

    结构：
        {
          "epochs": [ {epoch, train_loss, val_liver_dice, val_tumor_dice,
                       epoch_time_sec, lr, gpu_memory_MB}, ... ],
          "summary": { total_train_time_sec, peak_gpu_memory_MB,
                       best_liver_dice, best_tumor_dice,
                       final_liver_dice, final_tumor_dice,
                       amp_enabled, onecycle_enabled, grad_clip_enabled,
                       cache_strategy, pin_memory, persistent_workers,
                       non_blocking, val_amp, num_epochs_completed,
                       status, error }
        }
    所有数值均来自真实运行（计时 / torch.cuda.max_memory_allocated / 验证指标），
    不含任何理论值或估算。失败时 epochs 仍含已完成的部分，status=failed。
    """
    out_path = cfg.get("benchmark_output_json")
    if not out_path:
        return
    is_cuda = cfg["device"].startswith("cuda")
    # 峰值显存：取 epochs 中最大的 gpu_memory_MB（每 epoch 记录的是累计峰值，末值即全程峰值）
    if is_cuda:
        try:
            peak_gpu = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        except Exception:
            peak_gpu = float("nan")
    else:
        peak_gpu = float("nan")
    # 若 epochs 为空（训练第 1 个 epoch 即崩），用末值兜底
    if epochs and not (peak_gpu == peak_gpu):
        peak_gpu = epochs[-1].get("gpu_memory_MB", float("nan"))

    summary = {
        "total_train_time_sec": float(total_time),
        "peak_gpu_memory_MB": peak_gpu,
        "best_liver_dice": float(best_liver) if best_liver >= 0 else float("nan"),
        "best_tumor_dice": float(best_tumor) if best_tumor >= 0 else float("nan"),
        "final_liver_dice": float(final_liver),
        "final_tumor_dice": float(final_tumor),
        "amp_enabled": bool(cfg.get("use_amp", False)),
        "onecycle_enabled": cfg.get("scheduler", "onecycle") == "onecycle",
        "grad_clip_enabled": bool(cfg.get("grad_clip", True)),
        "cache_strategy": cfg["training"].get("cache", "disk"),
        "pin_memory": bool(cfg.get("pin_memory", False)),
        "persistent_workers": bool(cfg.get("persistent_workers", False)),
        "non_blocking": bool(cfg.get("non_blocking", True)),
        "val_amp": bool(cfg.get("val_amp", False)),
        "num_epochs_completed": len(epochs),
        "status": status,
        "error": error,
    }
    payload = {"epochs": epochs, "summary": summary}
    try:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        print("[benchmark] 结果已写入 {}".format(out_path))
    except Exception as e:
        # 写 JSON 失败不应影响主流程
        print("[benchmark] 写结果 JSON 失败：{}".format(e))


if __name__ == "__main__":
    main()
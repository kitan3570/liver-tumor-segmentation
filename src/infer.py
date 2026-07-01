"""src/infer.py — 肝脏/肝肿瘤 3D 分割推理脚本（单图 / 批量目录版）

文件作用
========
    用训练好的 3D U-Net 权重（best_model.pth），对「单个新 CT NIfTI」或
    「整目录的 CT NIfTI」做推理，输出与输入同名的多类分割预测文件
    （.nii.gz，体素值为类别 id：0=背景 / 1=肝脏 / 2=肿瘤）。这是 tasks.md
    Task 6 的产物，面向医科大学大一学生，因此代码注释格外详尽。

    与 scripts/infer.py 的区别：
        - scripts/infer.py 走的是「数据清单 JSON + DataLoader」的批量流程，
          适合配合训练流程的 test.json 使用；
        - 本脚本面向「拿到一个新 CT 就立刻预测」的轻量场景，支持
          --image_path（单图）或 --image_dir（批量目录），互斥使用。

运行方式
========
    单图推理：
        python src/infer.py \
            --image_path data/raw/Task03_Liver/imagesTs/liver_100.nii.gz \
            --checkpoint outputs/models/best_model.pth \
            --output_dir outputs/predictions \
            --config configs/liver_unet_3d.yaml

    批量推理（处理整个目录下所有 .nii / .nii.gz）：
        python src/infer.py \
            --image_dir data/raw/Task03_Liver/imagesTs \
            --checkpoint outputs/models/best_model.pth \
            --output_dir outputs/predictions \
            --config configs/liver_unet_3d.yaml

    可用 --help 查看所有参数说明；可用 --device cpu/cuda 覆盖自动选择。

输入
====
    --image_path ：单个 CT NIfTI 文件路径（.nii 或 .nii.gz）。与 --image_dir 互斥。
    --image_dir  ：包含若干 CT NIfTI 的目录。与 --image_path 互斥。
    --checkpoint：训练好的模型权重（.pth/.pt），如 outputs/models/best_model.pth。
    --output_dir：预测 .nii.gz 输出目录（必填，不存在会自动创建）。
    --config    ：YAML 配置（必填），提供预处理 / 模型 / 推理超参。
    --device    ：可选，强制指定 cuda 或 cpu；不填则自动选择。

输出
====
    在 --output_dir 下为每张输入 CT 生成同名 .nii.gz 预测文件，体素值为
    类别 id（0/1/2），保留原始空间（affine），可直接用 ITK-SNAP / 3D Slicer
    叠加在原图上查看。终端会打印每例预测的 np.unique 取值与体素计数。

常见错误
========
    1) checkpoint 不存在：报错退出，请确认训练已产出 best_model.pth，
       或用 --checkpoint 指定正确路径。
    2) image_dir 无 NIfTI 文件：报错退出，请确认目录下确实有 .nii/.nii.gz。
    3) 同时指定 --image_path 和 --image_dir：脚本优先使用 --image_path 处理单图，
       以保证行为可预期。
    4) CUDA 显存不足（CUDA out of memory）：
       - 调小 config['inference']['sw_batch_size']（如改为 1）；
       - 或在 --device cpu 下走 CPU（速度慢但不会爆显存）；
       - 或减小 roi_size。
"""

# ===================== 标准库导入 =====================
# os / sys / glob / argparse 用于路径处理、命令行解析与把项目根目录加入 sys.path
import os
import sys
import gc
import glob
import argparse

# numpy 做数组运算（取 argmax 结果、打印 unique 取值计数）
import numpy as np

# ===================== 第三方库导入 =====================
# torch：模型前向与张量搬运到 GPU；推理用 torch.no_grad() 关闭梯度以省显存。
import torch
# nibabel：医学影像 NIfTI 读写，保存预测时需用它写入 affine/header。
import nibabel as nib
# yaml：读取 YAML 配置文件，得到 data/preprocessing/model/inference 子字段。
import yaml

# MONAI：医学影像深度学习框架，提供变换链、3D U-Net、滑动窗口推理。
# 这里用「dict 变换」的 d 后缀版本，便于配合 Invertd 反向还原到原始空间。
import monai.transforms as mt
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet

# ===================== 本项目模块导入 =====================
# 把项目根目录插入 sys.path，使直接 `python src/infer.py ...` 运行时能 import src.*。
# 思路：__file__ 是 src/infer.py，dirname 一次得到 src/，再 dirname 得到项目根 d:\项目1。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 复用 src 中的模型构建与权重加载函数，保证推理用的网络结构与训练完全一致。
from src.models import build_model, load_model


# ============================================================
# 辅助函数：命令行参数解析
# ============================================================
def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    --help 由 argparse 自动提供，无需手动定义；下面为每个参数写中文帮助。

    Returns:
        argparse.Namespace: 含 image_path / image_dir / image_list / checkpoint /
            output_dir / config / device 字段。
    """
    parser = argparse.ArgumentParser(
        description=(
            "用训练好的 3D U-Net（best_model.pth）对新 CT NIfTI 做多类分割推理，"
            "输出与输入同名的 .nii.gz 预测文件。--image_path / --image_dir / "
            "--image_list 三选一（互斥）；未提供则报错。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 单图路径（可选）：与 image_dir / image_list 互斥
    parser.add_argument(
        "--image_path",
        default=None,
        help="单个 CT NIfTI 文件路径（.nii / .nii.gz）。与 --image_dir / --image_list 互斥。",
    )
    # 批量目录（可选）：处理该目录下所有 .nii / .nii.gz
    parser.add_argument(
        "--image_dir",
        default=None,
        help="批量 CT NIfTI 所在目录；脚本会自动扫描其中的 .nii / .nii.gz。",
    )
    # 样本清单 JSON（可选）：正式测试集推理用，读取每条记录的 image 字段。
    # test.json 形如 [{"image": ".../liver_x.nii.gz", "label": "..."}, ...]，
    # 只取 image 做推理，label 不参与推理（评估在 evaluate.py 中单独进行）。
    parser.add_argument(
        "--image_list",
        default=None,
        help="样本清单 JSON 路径（如 data/splits/test.json）；脚本会读取其中"
             "每条记录的 image 字段进行推理。与 --image_path / --image_dir 互斥。",
    )
    # 模型权重（必填）
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="训练好的模型权重路径（.pth/.pt），如 outputs/models/best_model.pth。",
    )
    # 输出目录（必填）
    parser.add_argument(
        "--output_dir",
        required=True,
        help="预测 .nii.gz 输出目录；若不存在会自动创建。",
    )
    # 配置文件（必填）
    parser.add_argument(
        "--config",
        required=True,
        help="YAML 配置文件路径，如 configs/liver_unet_3d.yaml。",
    )
    # 设备覆盖（可选）
    parser.add_argument(
        "--device",
        default=None,
        help="强制指定运行设备（如 cuda / cpu）；不填则自动选择（有 GPU 用 cuda）。",
    )
    return parser.parse_args()


def resolve_image_list(args: argparse.Namespace) -> list:
    """根据命令行参数确定待推理的图像路径列表。

    互斥规则（与题目要求一致）：
        - --image_path / --image_dir / --image_list 三者互斥，必须且只能给一个。
        - 同时给出多个 → 报错退出（避免歧义，正式测试集尤其需要明确）。
        - 三者皆无 → 报错退出。

    各模式说明：
        - image_path：返回 [image_path]，并做 macOS `._` 过滤 + gzip 头校验。
        - image_dir ：扫描目录下所有 .nii / .nii.gz，返回排序后的列表。
        - image_list：读取 JSON 清单，取每条记录的 image 字段，校验存在性。
                      label 字段不参与推理（评估在 evaluate.py 中单独进行）。

    Args:
        args: 已解析的命令行参数。

    Returns:
        list[str]: 待推理图像的路径列表。

    Raises:
        SystemExit: 当三者皆无 / 多于一个 / JSON 不存在 / 缺 image 字段 /
            image 路径不存在 / 目录无 NIfTI 时。
    """
    has_path = args.image_path is not None
    has_dir = args.image_dir is not None
    has_list = args.image_list is not None
    provided = [n for n, f in (("--image_path", has_path),
                               ("--image_dir", has_dir),
                               ("--image_list", has_list)) if f]

    # 情况 1：三者皆无 → 报错
    if len(provided) == 0:
        print("[错误] 必须提供 --image_path / --image_dir / --image_list 之一。"
              "请用 --help 查看用法。")
        sys.exit(1)

    # 情况 2：同时给出多个 → 报错（互斥，避免正式测试集误用）
    if len(provided) > 1:
        print(f"[错误] {provided} 不能同时使用，请只指定其中一个。")
        sys.exit(1)

    # 情况 3：只有 image_path
    if has_path:
        # 校验：跳过 macOS 元数据文件（._ 开头），并确认文件存在且是有效 NIfTI。
        # 这些文件从 macOS 解压/拷贝过来时常见，nibabel 加载会报 `not a gzip file`。
        if os.path.basename(args.image_path).startswith("._"):
            print(f"[错误] 选中的是 macOS 元数据文件，不是有效 NIfTI：{args.image_path}")
            print("       macOS 会为每个真实文件生成一份以 ._ 开头的资源叉文件，"
                  "体积很小且不是图像。")
            print("       请改选同目录下不带 ._ 前缀的同名文件，例如 "
                  f"{os.path.basename(args.image_path).lstrip('._') or 'liver_0.nii.gz'}")
            sys.exit(1)
        if not os.path.isfile(args.image_path):
            print(f"[错误] image_path 不是有效文件：{args.image_path}")
            sys.exit(1)
        try:
            # 轻量探测：读取 gzip 头两个字节（NIfTI gzip 魔数 0x1f 0x8b），避免后续
            # MONAI 加载时报深层 traceback。.nii（未压缩）不做此检查。
            if args.image_path.lower().endswith(".gz"):
                import gzip
                with gzip.open(args.image_path, "rb") as _f:
                    _f.read(2)
        except Exception as _e:
            print(f"[错误] 文件不是有效的 gzip NIfTI：{args.image_path}")
            print(f"       原因：{_e}")
            print("       若该文件以 ._ 开头，是 macOS 元数据；若体积很小，可能下载/解压未完成。")
            sys.exit(1)
        return [args.image_path]

    # 情况 4：只有 image_list → 读取 JSON 清单
    if has_list:
        return _load_image_list_from_json(args.image_list)

    # 情况 5：只有 image_dir → 扫描目录
    if not os.path.isdir(args.image_dir):
        print(f"[错误] --image_dir 不是有效目录：{args.image_dir}")
        sys.exit(1)

    # glob 同时匹配 .nii 与 .nii.gz；注意 .nii.gz 会出现在两次匹配中，用集合去重。
    nii_files = glob.glob(os.path.join(args.image_dir, "*.nii"))
    nii_gz_files = glob.glob(os.path.join(args.image_dir, "*.nii.gz"))

    def _is_valid_nifti(p):
        """过滤掉 macOS 元数据文件（以 ._ 开头）和隐藏文件、非文件项。

        macOS 在非原生文件系统（外置盘、SMB 共享、解压跨平台压缩包）上会为
        每个文件生成一份以 `._` 开头的资源叉文件，例如 `._liver_0.nii.gz`。
        它们体积很小、不是有效的 gzip/NIfTI，glob 会误匹配进来，nibabel 加载
        时报 `not a gzip file`。这里统一过滤所有 `.` 开头的隐藏文件。
        """
        return os.path.isfile(p) and not os.path.basename(p).startswith(".")

    image_list = sorted({p for p in (nii_files + nii_gz_files) if _is_valid_nifti(p)})

    if len(image_list) == 0:
        print(f"[错误] 目录中没有 .nii / .nii.gz 文件：{args.image_dir}")
        sys.exit(1)

    print(f"[信息] 在 {args.image_dir} 中共找到 {len(image_list)} 个 NIfTI 文件。")
    return image_list


def _load_image_list_from_json(image_list_path: str) -> list:
    """从样本清单 JSON 读取待推理的图像路径列表。

    JSON 格式（与 data/splits/test.json 一致）：
        [{"image": ".../liver_x.nii.gz", "label": ".../liver_x.nii.gz"}, ...]

    本函数只取 image 字段做推理；label 不参与推理（评估在 evaluate.py 中单独进行）。

    Args:
        image_list_path: JSON 文件路径。

    Returns:
        list[str]: 待推理图像路径列表（保持 JSON 中的顺序）。

    Raises:
        SystemExit: 当 JSON 不存在 / 不是有效 JSON / 不是 list /
            某条记录缺 image 字段 / image 路径不存在时。
    """
    import json

    # 1) 文件存在性
    if not os.path.isfile(image_list_path):
        print(f"[错误] --image_list 指定的 JSON 不存在：{image_list_path}")
        print("       正式测试集推理请用 data/splits/test.json；"
              "请用 --image_list 指定正确路径。")
        sys.exit(1)

    # 2) 读取并解析 JSON
    try:
        with open(image_list_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[错误] --image_list 不是有效的 JSON：{image_list_path}")
        print(f"       解析失败：{e}")
        sys.exit(1)

    # 3) 必须是 list
    if not isinstance(records, list):
        print(f"[错误] --image_list 的顶层必须是数组（list），"
              f"实际类型：{type(records).__name__}")
        sys.exit(1)

    if len(records) == 0:
        print(f"[错误] --image_list 是空数组，没有待推理样本：{image_list_path}")
        sys.exit(1)

    # 4) 逐条校验：必须有 image 字段，且文件存在
    image_paths = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            print(f"[错误] --image_list 第 {i} 条记录不是字典：{rec}")
            sys.exit(1)
        if "image" not in rec or not rec["image"]:
            print(f"[错误] --image_list 第 {i} 条记录缺少 image 字段：{rec}")
            print("       每条记录形如 "
                  "{\"image\": \".../liver_x.nii.gz\", \"label\": \"...\"}。")
            sys.exit(1)
        img = rec["image"]
        if not os.path.isfile(img):
            print(f"[错误] --image_list 第 {i} 条记录的 image 路径不存在：{img}")
            print("       请检查 JSON 中 image 字段是否指向真实文件。")
            sys.exit(1)
        image_paths.append(img)

    print(f"[信息] 从 {image_list_path} 读取到 {len(image_paths)} 个待推理样本。")
    return image_paths


def load_yaml_config(path: str) -> dict:
    """读取 YAML 配置文件并返回字典。

    Args:
        path: YAML 配置文件路径。

    Returns:
        dict: 解析后的配置字典，顶层键含 data / preprocessing / model / inference。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"[错误] 配置文件不存在：{path}\n请用 --config 指定正确的 YAML 路径。"
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_inference_transforms(config: dict):
    """构建推理所需的两条 MONAI 变换链：前向预处理 + 反向还原。

    【为什么要分两条变换链】
        推理时网络在「预处理后的空间」（统一间距 1.5/1.5/2.0mm、RAS 方向、
        已做 CT 窗归一化）里输出预测；但最终保存的 .nii.gz 需要还原到「原始
        CT 空间」（保留原始 affine / 间距 / 方向），否则在 ITK-SNAP 里无法
        与原图对齐叠加。MONAI 的 Invertd 用同一条「前向变换链」反向应用，
        把预测 warp 回原始空间，并自动管理 affine 与元数据——这是医学影像
        推理「对齐」的关键一步。

    【前向链 pre_transforms（dict 变换，每步都带 'd' 后缀）】
        LoadImaged         : 读取 NIfTI 为 MetaTensor（携带 affine 等 meta）；
        EnsureChannelFirstd: 把 (D,H,W) 调成 (1,D,H,W)，通道维放最前；
        Orientationd       : 统一到 RAS 方向（右-前-上），消除不同扫描方向差异；
        Spacingd            : 三线性插值重采样到 (1.5,1.5,2.0)mm，体素间距统一；
        ScaleIntensityRanged: CT 窗 [-200,250] → [0,1] 截断，与训练归一化一致；
        CropForegroundd     : 裁掉全零外围，聚焦有效区，减小说卷尺寸，加速推理。
        EnsureTyped         : 确保是 MONAI 张量类型，便于 Invertd 解析。

    【Invertd 链】
        pred 在主循环里已用 GPU argmax 得到 (1,D,H,W) 的类别索引（0/1/2），
        这里 **不再** 用 AsDiscreted（避免双重 argmax 把预测全变成 0，同时
        省下 one-hot + CPU argmax 的高内存开销）。
        Invertd(keys=["pred"], transform=pre_transforms, orig_keys="image",
            meta_keys="image_meta_dict", nearest_interp=[True], to_tensor=[False])：
            对 "pred" 反向应用 pre_transforms 的逆操作，用 "image" 当参照得到
            原始空间；插值用最近邻避免类别 id 在重采样时被插值出「1.5」这种非法值；
            to_tensor=False 直接返回 numpy/MetaTensor，方便 nibabel 保存。

    Args:
        config: 顶层配置字典。

    Returns:
        tuple(pre_transforms, post_transforms): 前向预处理链 与 含 Invertd 的后处理链。
    """
    # 取出预处理相关配置
    pp = config["preprocessing"]
    spacing = tuple(pp["spacing"])            # 重采样目标体素间距
    ct_window = pp["ct_window"]               # [-200, 250]
    a_min, a_max = ct_window[0], ct_window[1]

    # —— 前向预处理链 ——
    pre_transforms = mt.Compose([
        # LoadImaged 返回 MetaTensor，自动携带 affine（保留原始空间信息）。
        mt.LoadImaged(keys=["image"], image_only=True),
        # 把 (D,H,W) 调整为 (1,D,H,W)，确保通道维在最前，符合 PyTorch 卷积约定。
        mt.EnsureChannelFirstd(keys=["image"]),
        # 统一到 RAS 方向（x=右 Right，y=前 Anterior，z=上 Superior）。
        mt.Orientationd(keys=["image"], axcodes="RAS"),
        # 三线性(bilinear)重采样到 (1.5,1.5,2.0)mm；CT 是连续标量场，用线性插值合理。
        mt.Spacingd(keys=["image"], pixdim=spacing, mode="bilinear"),
        # CT 窗归一化：把 [-200,250] 的 HU 值线性映射到 [0,1]，并截断窗口外。
        mt.ScaleIntensityRanged(
            keys=["image"], a_min=a_min, a_max=a_max,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        # 以图像自身为参照裁掉全零外围（source_key="image"），减小推理体积。
        mt.CropForegroundd(keys=["image"], source_key="image"),
        # 转成 MONAI 张量类型，便于后续 Invertd 正确解析 meta。
        mt.EnsureTyped(keys=["image"], dtype=torch.float32),
    ])

    # —— 后处理链：反向还原到原始空间 ——
    # 说明：pred 在主循环里已用 outputs.softmax(dim=1).argmax(dim=1, keepdim=True)
    # 在 GPU 上完成 argmax，得到 (1, D', H', W') 的类别索引（0/1/2）。
    # 因此这里 **不再** 用 AsDiscreted(argmax=True, to_onehot=N)：
    #   1) argmax=True 会沿「单通道」再 argmax 一次，结果全 0（背景）——这是
    #      之前预测全是背景 0 的根因；
    #   2) to_onehot=N 会把 (1,D,H,W) 展成 (N,D,H,W) one-hot，体积翻 N 倍，
    #      之后还要 np.argmax 压回，产生 1.67 GiB 的 int64 中间数组，导致 OOM。
    # 现在直接把「单通道类别索引」交给 Invertd，用 nearest_interp=True 反向
    # 重采样，类别 id 不会被插值成非法值，同时省下 one-hot + argmax 的内存。
    post_transforms = mt.Compose([
        # Invertd：把 "pred" 反向 warp 回 "image" 的原始空间。
        mt.Invertd(
            keys=["pred"],
            transform=pre_transforms,        # 用同一套前向变换做反向
            orig_keys="image",               # 用原始 image 作为「目标空间」参照
            meta_keys="image_meta_dict",     # 元数据键，含原始 affine
            nearest_interp=[True],           # 类别标签必须最近邻插值
            to_tensor=[False],              # 输出 numpy / MetaTensor，方便保存
        ),
    ])

    return pre_transforms, post_transforms


def main() -> None:
    """推理主入口：解析参数 → 加载配置/模型 → 逐例预处理/推理/还原/保存。"""
    args = parse_args()

    # ---------- 重定向临时目录到项目盘 ----------
    # 防止 PyTorch/MONAI/numpy 往 C:\Users\...\Temp 写大文件（如滑动窗口推理的
    # 中间张量、numpy argmax 临时数组等），把临时目录设到项目所在盘。
    _proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _tmp_dir = os.path.join(_proj_root, "outputs", ".tmp")
    os.makedirs(_tmp_dir, exist_ok=True)
    os.environ["TEMP"] = _tmp_dir
    os.environ["TMP"] = _tmp_dir

    # —— 读取 YAML 配置 ——
    config = load_yaml_config(args.config)

    # —— 确定运行设备 ——
    if args.device is not None:
        device_str = args.device
        if device_str == "cuda" and not torch.cuda.is_available():
            print("[警告] 请求 cuda 但不可用，自动回退到 cpu。")
            device_str = "cpu"
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"[信息] 运行设备：{device}")

    # —— 校验 checkpoint 存在 ——
    if not os.path.isfile(args.checkpoint):
        print(f"[错误] checkpoint 不存在：{args.checkpoint}\n"
              "请确认训练已产出权重文件，或用 --checkpoint 指定正确路径。")
        sys.exit(1)

    # —— 准备输出目录 ——
    os.makedirs(args.output_dir, exist_ok=True)

    # —— 确定待推理的图像列表 ——
    image_list = resolve_image_list(args)

    # —— 取出推理超参 ——
    roi_size = tuple(config["inference"]["roi_size"])      # 滑动窗口块大小，如 (96,96,96)
    sw_batch_size = config["inference"]["sw_batch_size"]  # 并发窗口数；显存不足调 1

    # —— 给 config['model'] 补 norm 字段，与 train.py 的默认保持一致 ——
    # 说明：train.py 的 build_model 未显式传 norm，走 MONAI UNet 默认 INSTANCE。
    # 推理时必须用相同 norm，否则 state_dict 的归一化层参数键名对不上、
    # load_state_dict(strict=True) 会报错。若 yaml 显式写了 norm 则尊重 yaml。
    if "norm" not in config["model"]:
        config["model"]["norm"] = "INSTANCE"

    # 打印本次推理的关键信息，便于学习者理解每一步在做什么
    print("=" * 60)
    print(f"配置文件     : {args.config}")
    print(f"模型权重     : {args.checkpoint}")
    print(f"输出目录     : {args.output_dir}")
    if args.image_list is not None:
        print(f"样本清单     : {args.image_list}")
    print(f"待推理图像数 : {len(image_list)}")
    print(f"设备         : {device}")
    print(f"滑动窗口     : roi_size={roi_size}, sw_batch_size={sw_batch_size}")
    print(f"类别数       : {config['data']['num_classes']}"
          f"（0=背景 / 1=肝脏 / 2=肿瘤）")
    print("=" * 60)

    # ========== 1. 医学图像预处理 ==========
    # 构建前向预处理链与后处理（Inverdd）链，供每例复用。
    pre_transforms, post_transforms = build_inference_transforms(config)

    # ========== 2. 模型推理 ==========
    # 构造与训练完全一致的 3D UNet 结构，再加载 best_model.pth 权重；
    # build_model 内部会打印参数量；load_model 内部会处理 weights_only 兼容性。
    model = build_model(config)
    model = load_model(model, args.checkpoint, device)
    model.eval()  # 推理必须 eval：BatchNorm 用全局统计量、关闭 Dropout

    # 逐例推理：每张 CT 独立处理（batch=1），用滑动窗口覆盖整卷。
    # success_count 记录成功保存预测的病例数（单例出错不中断整体流程，
    # 便于正式测试集批量推理时部分病例失败仍能产出可用预测）。
    success_count = 0
    for img_idx, image_path in enumerate(image_list, start=1):
        print(f"\n[推理 {img_idx}/{len(image_list)}] {image_path}")
        try:
            success_count += _infer_one_case(
                image_path=image_path,
                pre_transforms=pre_transforms,
                post_transforms=post_transforms,
                model=model,
                device=device,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                output_dir=args.output_dir,
            )
        except Exception as e:
            # 单例失败：打印错误并继续，保证其余病例仍能推理。
            import traceback
            print(f"  [错误] 推理失败：{e}")
            traceback.print_exc()
            # 即便失败也尝试释放显存，避免后续病例因残留显存崩溃
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    # ========== 4. 汇总打印 ==========
    # 正式测试集推理后需明确告知：清单路径、样本总数、成功数、输出目录，
    # 便于核对是否所有测试病例都跑完。
    print("=" * 60)
    print(f"[完成] 推理结束")
    if args.image_list is not None:
        print(f"  样本清单     : {args.image_list}")
    print(f"  样本总数     : {len(image_list)}")
    print(f"  成功预测数量 : {success_count}")
    print(f"  输出目录     : {args.output_dir}")
    print("=" * 60)
    if success_count < len(image_list):
        # 有病例失败：以非 0 退出码提示，便于脚本化调用感知
        sys.exit(1)


def _infer_one_case(image_path, pre_transforms, post_transforms, model,
                    device, roi_size, sw_batch_size, output_dir) -> int:
    """对单张 CT 完成预处理 → 滑动窗口推理 → Invertd 还原 → 保存。

    成功返回 1，失败抛异常（由调用方捕获）。把单例逻辑抽成函数便于：
        1) try/except 包裹单例，单例失败不中断整体批量流程；
        2) 局部变量在函数返回时自动释放，配合 del + gc 更稳妥。

    步骤与原内联逻辑一致，详见行内注释。
    """
    # --- 单例数据是 dict {"image": <路径>}，符合 MONAI dict 变换约定 ---
    data_dict = {"image": image_path}

    # --- 前向预处理：得到 (1,1,D',H',W') 网络输入 + 元数据 ---
    data_dict = pre_transforms(data_dict)
    inputs = data_dict["image"]
    # 增加 batch 维：(1,1,D',H',W') → (1,1,D',H',W') 后搬到目标设备
    inputs = inputs.unsqueeze(0).to(device)

    # --- 滑动窗口推理：把大体积切成 96³ 小块逐块前向，再拼接回整卷 ---
    # torch.no_grad() 关闭梯度，省显存兼加速；softmax→argmax 得到每体素类别。
    with torch.no_grad():
        # outputs 形状 (1, C, D', H', W')，C=3，是各类的 logit 得分。
        outputs = sliding_window_inference(
            inputs,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
        )
        # softmax 把 logit 转成概率（沿类别维归一化），再 argmax 取最大概率类别。
        pred = outputs.softmax(dim=1).argmax(dim=1, keepdim=True)
        # pred 形状 (1,1,D',H',W')，整数类别 0/1/2。

    # --- 反向还原到原始空间：把 "pred" 放回 data_dict，调用后处理链 ---
    data_dict["pred"] = pred[0]  # 取出第 0 个 batch：(1,D',H',W')
    data_dict = post_transforms(data_dict)

    # Invertd 输出是 MetaTensor 或 numpy 数组，提取其 numpy 体素与 affine。
    pred_inv = data_dict["pred"]
    if hasattr(pred_inv, "cpu"):
        pred_np = pred_inv.cpu().numpy()
    else:
        pred_np = np.asarray(pred_inv)

    # ========== 3. 结果保存 ==========
    # pred 已在 GPU 上 argmax 过，Invertd 只是把它 warp 回原始空间，
    # 因此 pred_np 是「单通道类别索引」(0/1/2)，不再是 one-hot。
    # 这里只需 squeeze + astype(uint8)，**不需要** np.argmax——后者会生成
    # (512,512,856) 的 int64 中间数组（1.67 GiB），是之前 OOM 的根因。
    if pred_np.ndim == 4:
        # (1, D, H, W) → 取第 0 通道得 (D, H, W)
        pred_label = pred_np[0].astype(np.uint8)
    elif pred_np.ndim == 3:
        # (D, H, W) 直接转 uint8
        pred_label = pred_np.astype(np.uint8)
    else:
        # 其它奇形：squeeze 掉大小为 1 的维度
        pred_label = np.squeeze(pred_np).astype(np.uint8)

    # --- 取 affine：优先从 Inverdd 输出的 MetaTensor.meta 读取，回退到原始图像 ---
    affine = None
    if hasattr(pred_inv, "meta") and "affine" in pred_inv.meta:
        # Inverdd 把还原后的 affine 写进 .meta['affine']
        affine = pred_inv.meta["affine"]
    elif hasattr(data_dict["image"], "meta") and "affine" in data_dict["image"].meta:
        # 部分版本 affine 仍在 image 的 meta 里保留
        affine = data_dict["image"].meta["affine"]
    else:
        # 兜底：用 nibabel 直接读原始图像的 affine，确保保存时空间对齐
        orig_img = nib.load(image_path)
        affine = orig_img.affine
        print("[警示] 从原始图像直接读取 affine（Invertd 未提供 affine 元数据）。")

    # 把 numpy affine 转成 float32，避免保存时类型不支持
    affine = np.array(affine, dtype=np.float32)

    # --- 输出文件名：与输入同名（含扩展名），放到 output_dir ---
    out_name = os.path.basename(image_path)
    out_path = os.path.join(output_dir, out_name)

    # 用 nibabel 保存：传入 uint8 体素、原 affine，并设 dtype 保持紧凑。
    nii = nib.Nifti1Image(pred_label.astype(np.uint8), affine)
    nib.save(nii, out_path)

    # --- 打印该例预测的 unique 取值与体素计数，便于快速可视化地核对 ---
    uniq, counts = np.unique(pred_label, return_counts=True)
    uniq_summary = ", ".join(
        f"{int(v)}:{int(c)}" for v, c in zip(uniq, counts)
    )
    print(f"  -> 已保存：{out_path}")
    print(f"  -> 预测取值(类别:体素数) -> {uniq_summary}")

    # --- 显式释放本例的大对象，防止累积导致内存膨胀 / C 盘页面文件爆满 ---
    del pred_label, pred_np, pred_inv, pred, outputs, inputs, data_dict
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return 1


if __name__ == "__main__":
    main()
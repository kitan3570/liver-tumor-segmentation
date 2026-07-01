"""src/data.py — 肝脏 3D 分割项目的数据加载与预处理模块

本文件负责把磁盘上的 NIfTI CT 图像与分割标签，经过一系列「变换 (Transform)」
处理成网络可以直接吃的 3D 张量，并封装成 PyTorch DataLoader 供训练 / 验证 /
测试使用。它是整条训练流水线的「数据入口」。

面向大一医学生的概念速览
========================
- **NIfTI (.nii.gz)**：医学影像最常用的三维格式，除体素数据外还携带「仿射矩阵」
  等空间信息（原点、spacing 体素间距、方向）。MONAI 的 LoadImaged 负责读入。
- **Channel First**：深度学习约定张量形状为 (C, D, H, W)，通道维在最前。CT 是
  单模态灰度，C=1；原始 NIfTI 通常没有通道维，需要 EnsureChannelFirstD 补上。
- **Orientation (axcodes='RAS')**：不同扫描仪采集时病人的方向可能不同（有的
  脚朝前、有的头朝前）。统一重定向到 RAS（Right-Anterior-Superior，即左到右、
  后到前、下到上）标准方向，才能让所有数据方向一致、可比较。
- **Spacing 重采样**：不同 CT 的体素间距 (spacing) 不同（如 0.8mm 与 1.5mm），
  同样大小的数组代表的物理体积并不一样。重采样到统一 spacing（本项目 1×1×1mm）
  后，网络看到的「1 个体素 ≈ 1 mm³」一致，避免尺寸 / 比例歧义。图像用双线性
  插值（连续灰度值），标签用最近邻插值（保持 0/1/2 离散值不产生中间值）。
- **CT 窗 (intensity windowing)**：CT 值单位是 HU，范围很宽（-1000~+3000）。
  肝脏软组织常用窗 [-100, 400]：把该范围内的 HU 线性映射到 [0,1]，范围外截断。
  这能让网络聚焦在软组织的灰度差异，而非被骨骼 / 空气的极端值干扰。
- **随机前景裁剪 (RandCropByPosNegLabeld)**：整张 CT 体积很大（如 512×512×100），
  显存放不下也无需整卷训练。按正负样本比例随机裁出一个 96³ 的小块，且要求该
  块里「含前景 (肝脏)」，让网络多见到病灶、少见到纯背景，提升学习效率。
- **标签映射**：原始标签 0=背景、1=肝脏、2=肿瘤。阶段一 (phase1) 做二分类，
  把肿瘤并入肝脏 (label>0 → 1)；阶段二 (phase2) 保留 0/1/2 做三分类。
- **CacheDataset**：把整个数据集的预处理结果缓存在内存里，避免每个 epoch 重复
  读盘与变换，显著加速训练；代价是占用更多内存。

跨模块接口约定（src/trainer.py 已按此调用）
---------------------------------------------
- ``build_dataloader(data_list, config, phase) -> DataLoader``
- ``build_dataset(data_list, config, phase) -> Dataset``
- 返回的 DataLoader 每个 batch 是 dict，含 'image' (B,1,D,H,W) float32 与
  'label' (B,1,D,H,W) int64（值为类别 id 0/1 或 0/1/2，**非 one-hot**）。

模块结构
--------
A. 标签映射：LabelMapTransform (MapTransform 子类)
B. 变换流水线：get_train_transforms / get_val_transforms / get_test_transforms
C. 数据集与加载器：build_dataset / build_dataloader

说明：本模块通过参数接收已由外部加载好的 ``config``（来自 src.utils.load_config）
与 ``data_list``（来自 scripts/split_data.py 生成的 JSON 清单），故无需直接
import src.utils 的函数，保持本文件聚焦于「变换组装与数据加载」。

本文件为「库模块」，不含 if __name__ == "__main__"；由 src/trainer.py 通过
``from src.data import build_dataloader`` 调用。
"""

# ===================== 标准库导入 =====================
# 本模块核心是 MONAI 变换组装，标准库无直接依赖，故此处暂无导入。

# ===================== 第三方库导入 =====================
# torch：提供张量与 dtype（float32/int64），EnsureTyped / LabelMapTransform 会用到。
# numpy：标签映射中对 numpy 数组的兜底类型转换。
# monai.transforms：医学影像专用变换；D 后缀表示「字典版」（对 dict 数据操作）。
# monai.data：CacheDataset 与 DataLoader，用于组织数据集与批量加载。
import numpy as np
import torch
from monai.transforms import (
    Compose,
    MapTransform,
    LoadImaged,
    EnsureChannelFirstD,
    OrientationD,
    SpacingD,
    ScaleIntensityRangeD,
    RandCropByPosNegLabeld,
    RandFlipd,
    EnsureTyped,
    ToTensord,
)
from monai.data import CacheDataset, DataLoader


# ============================================================
# A. 标签映射变换
# ============================================================

class LabelMapTransform(MapTransform):
    """标签映射变换：按训练阶段把原始标签转换为目标类别 id。

    原始数据集 (Task03_Liver) 的标签值为：
        0 = 背景 (background)
        1 = 肝脏 (liver)
        2 = 肿瘤 (tumor)

    为什么要做标签映射？
    --------------------
    本项目分两个阶段：
        - **阶段一 (phase1, num_classes=2)**：先做最基础的「把肝脏整体抠出来」
          的二分类。此时肿瘤体积小、难度大，我们把它**并入肝脏**当作同一个
          前景类 (label>0 → 1)，得到 {0=背景, 1=肝脏}。这样网络先学会分割
          肝脏整体，降低入门难度。
        - **阶段二 (phase2, num_classes>=3)**：在阶段一跑通后，恢复原始三类
          标签 {0=背景, 1=肝脏, 2=肿瘤}，让网络进一步区分肝脏与肝肿瘤，挑战
          小目标分割。

    本变换继承 MONAI 的 ``MapTransform``（字典版变换基类），对 data dict 中
    指定 keys（通常是 ['label']）的张量做映射，再统一转为 int64。
    注意：输出是**类别 id**（0/1/2），**非 one-hot**；one-hot 化由损失函数
    (DiceCELoss, to_onehot_y=True) 内部完成，与 trainer.py 的约定一致。

    Args:
        keys: 要处理的字典键列表，如 ['label']。
        num_classes: 类别数；2 时执行 >0→1 映射，>=3 时保留原始 0/1/2。
    """

    def __init__(self, keys, num_classes: int):
        # MapTransform 基类的 __init__ 会保存 keys 并做基本校验
        super().__init__(keys)
        self.num_classes = num_classes

    def __call__(self, data):
        # 复制一份输入字典再修改，避免污染原始数据（MONAI 变换的通用约定）
        d = dict(data)
        for key in self.keys:
            label = d[key]
            if self.num_classes == 2:
                # 阶段一：把 >0 的体素（肝脏 1 + 肿瘤 2）都映射为 1，做二分类。
                # (label > 0) 得到 bool 数组：True=前景、False=背景；
                # 随后统一转 int64 时 True→1、False→0，正好得到 0/1 标签。
                label = label > 0
            # 阶段二 (num_classes>=3)：保留原始 0/1/2，无需改值，只统一 dtype。
            # 统一转为 int64（网络标签与损失函数期望的整数类别 id）
            if isinstance(label, torch.Tensor):
                # MetaTensor / 普通 tensor：用 .to(torch.int64) 转换，保留 meta
                d[key] = label.to(torch.int64)
            else:
                # 兜底：若上游遗留 numpy 数组，则转成 int64 numpy
                d[key] = np.asarray(label).astype(np.int64)
        return d


# ============================================================
# B. 变换流水线
# ============================================================

def get_train_transforms(config: dict) -> Compose:
    """构建训练阶段的预处理变换流水线（含随机增强）。

    流水线顺序（每一步都用字典版 D 变换，对 {'image':..., 'label':...} 操作）
    --------------------------------------------------------------------
    1. LoadImaged          ：读取 .nii.gz 为 MetaTensor（含空间 meta）。
    2. EnsureChannelFirstD ：补通道维，形状 (D,H,W) → (1,D,H,W)。
    3. OrientationD        ：统一到 RAS 标准方向。
    4. SpacingD            ：重采样到 1×1×1mm；图像双线性、标签最近邻。
    5. ScaleIntensityRangeD：CT 窗 [-100,400] → [0,1] 归一化（只对 image）。
    6. LabelMapTransform   ：标签映射（phase1: >0→1；phase2: 保留 0/1/2）。
    7. RandCropByPosNegLabeld：按正负样本比例随机裁剪含前景的 96³ 块（数据效率核心）。
    8. RandFlipd           ：随机翻转（轻度增强，提升泛化）。
    9. EnsureTyped         ：统一 dtype（image=float32, label=int64）。
    10. ToTensord          ：确保为 torch.Tensor（与 EnsureType 双保险）。

    Args:
        config: 顶层配置字典，需含 data.spacing、data.intensity_clip、
            data.intensity_scale、train.roi_size、num_classes 等字段。

    Returns:
        Compose: 依次执行上述变换的可调用对象。
    """
    # 取出配置中的子字段，提高可读性
    data_cfg = config["data"]
    train_cfg = config["train"]
    spacing = data_cfg["spacing"]                    # 重采样体素间距 [sx, sy, sz]
    a_min, a_max = data_cfg["intensity_clip"]        # CT HU 裁剪范围 [-100, 400]
    b_min, b_max = data_cfg["intensity_scale"]       # 归一化目标范围 [0, 1]
    roi_size = tuple(train_cfg["roi_size"])          # 裁剪块大小 (96, 96, 96)
    num_classes = config["num_classes"]              # 类别数，决定标签映射规则

    train_transforms = Compose([
        # 1) 读取 NIfTI 文件。image_only=True 表示只返回体素数据 (MetaTensor)，
        #    不返回 nibabel Image 对象；空间 meta (affine/spacing) 仍保留在
        #    MetaTensor 的 .meta 中，供后续 Orientation / Spacing 使用。
        LoadImaged(keys=["image", "label"], image_only=True),

        # 2) 确保通道维在最前。NIfTI 3D 数据加载后通常是 (D,H,W) 无通道维，
        #    本变换根据 meta 中的 original_channel_dim 自动补成 (1,D,H,W)。
        EnsureChannelFirstD(keys=["image", "label"]),

        # 3) 统一方向到 RAS (Right-Anterior-Superior)。不同扫描仪 / 摆位会导致
        #    体素坐标轴方向不一致，RAS 是神经影像学公认的标准方向，统一后
        #    所有数据「左到右、后到前、下到上」一致，便于网络学习与推理。
        OrientationD(keys=["image", "label"], axcodes="RAS"),

        # 4) 重采样到统一 spacing (mm)。mode 用元组按 key 顺序指定：
        #    image 用 'bilinear'（连续灰度值，双线性插值更平滑）；
        #    label 用 'nearest'（离散类别 id，最近邻避免插出 0.5 这类非法值）。
        SpacingD(
            keys=["image", "label"],
            pixdim=spacing,
            mode=("bilinear", "nearest"),
        ),

        # 5) CT 强度归一化（只对 image）。把 HU 在 [a_min, a_max] 的范围线性
        #    映射到 [b_min, b_max]，clip=True 表示范围外的值截断到边界。
        #    这是「CT 窗」的本质：聚焦软组织灰度，抑制骨骼 / 空气极端值。
        ScaleIntensityRangeD(
            keys=["image"],
            a_min=a_min,
            a_max=a_max,
            b_min=b_min,
            b_max=b_max,
            clip=True,
        ),

        # 6) 标签映射：phase1 把肿瘤并入肝脏 (>0→1)，phase2 保留 0/1/2。
        #    注意放在 ScaleIntensity 之后、RandCrop 之前——此时 label 已经是
        #    重采样 + 方向统一的整数掩膜，映射结果才正确；且裁剪要用映射后
        #    的前景定位（phase1 前景=liver，确保裁剪块含肝脏）。
        LabelMapTransform(keys=["label"], num_classes=num_classes),

        # 7) 随机前景裁剪：从大体积里按正负样本比例随机裁出一个含前景的
        #    roi_size (96³) 小块。RandCropByPosNegLabeld 会根据 label_key='label'
        #    找出前景与背景体素，按 pos:neg=1:1 的概率选择裁剪中心，使分到的
        #    块以一定概率含前景（phase1 前景=liver(1)，phase2 前景=liver(1)/
        #    tumor(2)），让网络多见病灶、少见纯背景。num_samples=1 表示每个
        #    样本只生成 1 个裁剪块，保持 batch 含义不变（MONAI 的
        #    list_data_collate 会自动 flatten 由 num_samples 产生的 list 层级）。
        #    MONAI 1.6 的 RandCropByPosNegLabeld 不再支持 allow_smallest 参数，
        #    当原图比 roi 小时会自动按较小尺寸裁剪作为兜底。
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi_size,
            num_samples=1,
            pos=1,
            neg=1,
        ),

        # 8) 随机翻转做数据增强。prob=0.5 表示一半概率沿 spatial_axis=0
        #    (第一个空间轴) 翻转。人体近似左右对称，翻转后的肝脏仍是合理的
        #    解剖结构，能让网络见到更多样的数据、减轻过拟合。这里只用一种
        #    翻转做「轻度增强」，保持训练稳定，适合初学者。
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),

        # 9) 统一数据类型：image 用 float32 (网络输入)、label 用 int64 (类别 id)。
        #    dtype 传列表时按 keys 顺序一一对应。
        EnsureTyped(keys=["image", "label"], dtype=[torch.float32, torch.int64]),

        # 10) 确保输出为 torch.Tensor。EnsureTyped 已转成 tensor，这里再做
        #     一次是历史习惯的「双保险」，防止个别变换遗留 numpy 类型。
        ToTensord(keys=["image", "label"]),
    ])
    return train_transforms


def get_val_transforms(config: dict) -> Compose:
    """构建验证阶段的预处理变换流水线（**不含随机增强与随机裁剪**）。

    与训练变换的关键差异
    --------------------
    - **不做随机裁剪**：验证 / 推理时要评估的是**整张 CT 体积**的分割效果，
      而不是某个 96³ 小块。整卷推理由 trainer 的 SlidingWindowInferer 负责
      （把大体积切成多个 96³ 窗分别预测再拼接），因此数据侧只需提供完整
      体积，无需裁剪。
    - **不做随机增强**：增强是为了让训练时数据更多样、提升泛化；而验证要
      「客观、可复现」地衡量模型在真实数据上的表现，任何随机扰动都会引入
      噪声，故一律关闭。翻转 / 旋转等增强在验证集上没有意义。

    保留的确定性预处理（与训练一致）
    --------------------------------
    LoadImage → EnsureChannelFirst → Orientation(RAS) → Spacing(重采样)
    → ScaleIntensityRange(CT 窗) → LabelMapTransform(标签映射)
    → EnsureType → ToTensor

    Args:
        config: 顶层配置字典（字段需求同 get_train_transforms）。

    Returns:
        Compose: 验证用的确定性变换流水线。
    """
    data_cfg = config["data"]
    spacing = data_cfg["spacing"]
    a_min, a_max = data_cfg["intensity_clip"]
    b_min, b_max = data_cfg["intensity_scale"]
    num_classes = config["num_classes"]

    val_transforms = Compose([
        # 读取 NIfTI（同训练）
        LoadImaged(keys=["image", "label"], image_only=True),
        # 补通道维
        EnsureChannelFirstD(keys=["image", "label"]),
        # 统一到 RAS 方向
        OrientationD(keys=["image", "label"], axcodes="RAS"),
        # 重采样到统一 spacing；图像双线性、标签最近邻
        SpacingD(
            keys=["image", "label"],
            pixdim=spacing,
            mode=("bilinear", "nearest"),
        ),
        # CT 窗归一化（只对 image）
        ScaleIntensityRangeD(
            keys=["image"],
            a_min=a_min,
            a_max=a_max,
            b_min=b_min,
            b_max=b_max,
            clip=True,
        ),
        # 标签映射（验证也必须做，保证与训练标签空间一致）
        LabelMapTransform(keys=["label"], num_classes=num_classes),
        # 统一 dtype：image=float32, label=int64
        EnsureTyped(keys=["image", "label"], dtype=[torch.float32, torch.int64]),
        # 确保 tensor 类型
        ToTensord(keys=["image", "label"]),
    ])
    return val_transforms


def get_test_transforms(config: dict) -> Compose:
    """构建测试阶段的变换流水线。

    测试与验证的预处理需求完全一致（都是整卷确定性预处理，交给
    SlidingWindowInferer 做分块推理），因此直接复用 get_val_transforms，
    避免重复代码、保证验证与测试变换始终同步。

    Args:
        config: 顶层配置字典。

    Returns:
        Compose: 测试用的变换流水线（同验证）。
    """
    return get_val_transforms(config)


# ============================================================
# C. 数据集与数据加载器
# ============================================================

def build_dataset(data_list: list, config: dict, phase: str) -> CacheDataset:
    """根据 phase 构建并返回 MONAI 数据集（推荐用 CacheDataset 加速）。

    data_list 的格式
    ----------------
    由 scripts/split_data.py 生成的 JSON 清单
    (outputs/splits/{train, val, test}.json) 解析得到的 list，每个元素是
    dict：{"image": <图像路径>, "label": <标签路径>}。
    MONAI 的 LoadImaged 会按这些路径读取对应 NIfTI 文件。

    CacheDataset vs 普通 Dataset
    ----------------------------
    - **CacheDataset**：在构造时就把「全部样本」的变换结果计算并缓存到内存，
      之后每个 epoch 取数据时无需重复读盘与变换，显著加速（尤其当变换较重
      时，如 Spacing 重采样）。代价是占用更多内存。
    - **普通 Dataset**：懒加载，每次取数据才现场做变换，内存占用小但速度慢。
    本项目标准设置样本量不大（Task03_Liver 训练集约 100+ 例），内存足够，
    故 train / val / test **均用 CacheDataset** 加速；val / test 用
    cache_rate=1.0 全量缓存，使验证 / 推理时数据就绪更快。

    Args:
        data_list: 数据清单 list，每项含 'image' / 'label' 路径。
        config: 顶层配置字典，需含 train.cache_rate、train.num_workers。
        phase: 'train' / 'val' / 'test'，决定使用哪套变换与缓存比例。

    Returns:
        CacheDataset: 构建好的数据集（已缓存预处理结果）。

    Raises:
        ValueError: 当 phase 不是 train / val / test 之一时。
    """
    # 按阶段选择变换流水线与缓存比例
    if phase == "train":
        transform = get_train_transforms(config)
        cache_rate = config["train"]["cache_rate"]   # 训练用配置中的缓存比例 (默认 1.0)
    elif phase in ("val", "test"):
        # val / test 共用同一套确定性变换（get_val_transforms）
        transform = get_val_transforms(config)
        cache_rate = 1.0                             # 验证 / 测试全量缓存，加速评估
    else:
        raise ValueError(
            f"不支持的 phase：{phase}\n"
            "phase 只能取 'train' / 'val' / 'test' 之一。"
        )

    # num_workers 控制「缓存计算」阶段的并行度（多进程同时跑变换）
    num_workers = config["train"]["num_workers"]

    # 用 CacheDataset 构建数据集。
    # 若机器内存吃紧、想降级为普通懒加载 Dataset，可改为：
    #     from monai.data import Dataset
    #     dataset = Dataset(data=data_list, transform=transform)
    dataset = CacheDataset(
        data=data_list,
        transform=transform,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )
    return dataset


def build_dataloader(data_list: list, config: dict, phase: str) -> DataLoader:
    """构建并返回 MONAI / PyTorch DataLoader。

    本函数是 trainer.py 调用的主入口（``from src.data import build_dataloader``），
    内部先 build_dataset 再包成 DataLoader。

    batch 约定（trainer.py 已按此调用）
    -----------------------------------
    返回的 DataLoader 每个 batch 是 dict：
        - 'image': 形状 (B, 1, D, H, W)，dtype=float32；
        - 'label': 形状 (B, 1, D, H, W)，dtype=int64，值为类别 id 0/1 或 0/1/2，
          **非 one-hot**（one-hot 由损失函数内部完成）。
    MONAI 的 DataLoader 默认用 ``list_data_collate``，能把 list[dict] 正确
    拼成 dict[batched_tensor]，无需手动写 collate_fn。

    关于 batch_size
    ---------------
    - 训练用 config['train']['batch_size']（标准设置 2，3D 体素显存占用大）；
    - 验证 / 测试用 **1** 更稳：整卷推理时各样本体积大小不一，batch=1 可避免
      因体积不同导致无法拼批的问题，也便于逐样本计算指标。

    关于 num_workers（Windows 注意事项）
    -----------------------------------
    - num_workers 控制「数据加载」的并行进程数，>0 可加速 IO。
    - **Windows 下多进程数据加载易报错**（如 BrokenPipeError、pickle 序列化
      失败等），因为 Windows 不支持 fork，需 spawn 子进程重新 import 模块。
      若运行时报相关 multiprocessing 错误，请把 config['train']['num_workers']
      改为 0（单进程加载，稳定但较慢）。

    Args:
        data_list: 数据清单 list，每项含 'image' / 'label' 路径。
        config: 顶层配置字典，需含 train.batch_size、train.num_workers。
        phase: 'train' / 'val' / 'test'。

    Returns:
        DataLoader: 可直接迭代的数据加载器（monai.data.DataLoader，
        它是 torch.utils.data.DataLoader 的子类）。
    """
    # 先构建数据集（含变换与缓存）
    dataset = build_dataset(data_list, config, phase)

    # 按阶段设置 batch_size 与 shuffle
    if phase == "train":
        batch_size = config["train"]["batch_size"]
        shuffle = True            # 训练时打乱顺序，减少 epoch 间相关性、提升泛化
    else:  # val / test
        batch_size = 1            # 整卷推理用 1，避免不同体积无法拼批
        shuffle = False           # 评估需确定且可复现的顺序

    num_workers = config["train"]["num_workers"]

    # monai.data.DataLoader 继承自 torch DataLoader，默认 collate 能处理 dict，
    # 无需手动指定 collate_fn。
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return loader

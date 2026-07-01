"""src/utils.py — 肝脏 3D 分割项目的基础工具模块

本文件汇集了项目中各脚本（scripts/*.py）与其他 src 模块（data/models/trainer）
都会用到的「底层工具函数」，按功能分为 5 组：

1. 配置与实验基础：load_config / merge_config / set_seed / get_device / ensure_dir
2. NIfTI 读写：load_nifti / save_nifti
3. JSON / CSV 日志：save_json / load_json / append_csv_row
4. 可视化配色与切片选择：get_class_colors / pick_slice_with_lesion / overlay_mask
5. STL 导出：export_stl / export_label_stl

设计说明
--------
- 面向初学者：每个函数都配中文 docstring，并在关键步骤加行内注释，解释医学影像
  中的概念（如 affine 仿射矩阵、轴向切片、marching cubes 等值面提取、CT 窗等）。
- 延迟导入：第三方依赖中 trimesh 与 scikit-image.measure（用于 STL 导出）在函数
  内部才 import。这样即使没装这两个包，本文件的其他工具函数（配置、NIfTI、JSON
  等）仍可正常使用——缺失 STL 依赖只会在真正调用 STL 导出时报错，而不会在
  import 本模块时就失败。这一点对 scripts/check_env.py 的优雅检测很重要。
- 本文件是「工具库」，不包含 if __name__ == "__main__"；其他模块通过
  ``from src.utils import load_config, ...`` 来调用其中的函数。
"""

# ===================== 标准库导入 =====================
import os
import csv
import json
import random

# ===================== 第三方库导入 =====================
# torch / nibabel / numpy / yaml 是本项目核心依赖，放顶部正常导入；
# 运行时若缺失，由 scripts/check_env.py 统一报告，本文件不为每个导入加 try/except。
import numpy as np
import torch
import nibabel as nib
import yaml


# ============================================================
# 第 1 组：配置与实验基础
# ============================================================

def load_config(path: str) -> dict:
    """读取 YAML 配置文件并返回字典。

    配置为嵌套 dict，顶层键包括：phase、num_classes、classes、seed、device、
    data、train、model、loss、optimizer、scheduler、infer、output
    （与 configs/*.yaml 一致）。

    Args:
        path: YAML 配置文件路径，例如 "configs/phase1_binary.yaml"。

    Returns:
        dict: 解析后的配置字典。

    Raises:
        FileNotFoundError: 当配置文件不存在时抛出，并附带中文提示。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"配置文件未找到：{path}\n"
            "请检查路径是否正确，或先确认 configs/ 目录下的 YAML 文件已生成。"
        )
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)  # safe_load 避免 YAML 中执行任意代码，更安全
    return config


def merge_config(base: dict, override: dict) -> dict:
    """用 override 覆盖 base 的顶层键，返回合并后的新字典（不修改原字典）。

    典型用途：命令行参数（如 --max_epochs、--device）覆盖 YAML 配置中的对应字段。
    本函数只做「顶层浅合并」：若 override 含某顶层键则整体替换 base 中的同名键，
    不深入嵌套子字典逐字段合并——这符合本项目配置覆盖的简单需求。

    约定：override 中值为 ``None`` 的键表示「该参数未在命令行指定，不要覆盖」，
    这样可以方便地把 argparse 的解析结果直接传进来而不影响配置。

    Args:
        base: 基础配置字典（通常来自 load_config）。
        override: 需要覆盖的键值字典（通常来自命令行解析结果）。

    Returns:
        dict: 合并后的新配置字典。
    """
    merged = dict(base)  # 先浅拷贝一份 base，避免污染原始配置
    for key, value in override.items():
        if value is not None:  # None 视为「不覆盖」，其余值（含 0/False/""）均覆盖
            merged[key] = value
    return merged


def set_seed(seed: int) -> None:
    """统一设置 random / numpy / torch（含 CUDA）的随机种子，保证实验可复现。

    深度学习实验中，权重初始化、数据打乱、dropout 等都涉及随机数；固定种子后，
    同一份代码 + 同一台机器上多次运行可得到一致结果，便于调试与结果汇报。

    Args:
        seed: 随机种子整数，本项目默认 42。
    """
    random.seed(seed)                  # Python 内置随机数
    np.random.seed(seed)               # NumPy 随机数
    torch.manual_seed(seed)            # PyTorch CPU 随机数
    torch.cuda.manual_seed_all(seed)   # PyTorch 所有 GPU 的随机数
    # 提示：如需更严格的确定性，可进一步设置
    #   torch.backends.cudnn.deterministic = True
    #   torch.backends.cudnn.benchmark = False
    # 但这通常会牺牲一点训练速度。


def get_device(device: str = "cuda") -> torch.device:
    """根据请求返回 torch.device；若请求 CUDA 但不可用则回退到 CPU 并打印警告。

    Args:
        device: 期望设备字符串，"cuda" 或 "cpu"，默认 "cuda"。

    Returns:
        torch.device: 实际使用的设备对象。
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("[警告] 请求使用 CUDA，但未检测到可用 GPU，自动回退到 CPU。"
              "标准科研设置（96^3、batch 2、300 epoch）在 CPU 上会非常慢，建议使用 GPU。")
        return torch.device("cpu")
    return torch.device(device)


def ensure_dir(path: str) -> None:
    """确保目录存在，不存在则递归创建。

    Args:
        path: 目录路径。若传入的是文件路径，请先取其父目录再调用本函数。
    """
    os.makedirs(path, exist_ok=True)  # exist_ok=True：目录已存在时不报错


# ============================================================
# 第 2 组：NIfTI 读写
# ============================================================

def load_nifti(path: str) -> tuple[np.ndarray, np.ndarray]:
    """加载 NIfTI (.nii / .nii.gz) 文件，返回 (数据数组, 仿射矩阵)。

    NIfTI 是医学影像最常用的格式之一。每个文件除了体素数据外，还包含一个 4x4 的
    「仿射矩阵 affine」，它描述了「体素索引空间 (i, j, k)」到「物理世界坐标
    (x, y, z)」的映射关系（含原点、spacing 体素间距、方向）。重建三维模型或对齐
    不同扫描时，affine 是必不可少的。

    本函数保留磁盘上的原始数据类型（标签常为 int16/uint8，图像为 float/int16），
    既节省内存又保持类型正确；后续 MONAI 变换会按需转换 dtype（如图像转 float32、
    标签转 int）。如需统一的 float，可在调用后自行 ``data.astype(np.float32)``。

    Args:
        path: .nii.gz 或 .nii 文件路径。

    Returns:
        tuple[np.ndarray, np.ndarray]: (data, affine)
            - data: 3D 体素数组（保留磁盘原始 dtype）。
            - affine: 4x4 仿射矩阵（float64）。
    """
    img = nib.load(path)                  # 加载 NIfTI 图像对象（懒加载，不立即读全部数据）
    data = np.asanyarray(img.dataobj)     # 按磁盘原始 dtype 读取体素数据
    affine = img.affine.copy()            # 取出仿射矩阵并复制，避免后续误改原对象
    return data, affine


def save_nifti(data: np.ndarray, affine: np.ndarray, path: str) -> None:
    """将体素数据与仿射矩阵保存为 NIfTI (.nii.gz) 文件。

    Args:
        data: 3D 体素数组（图像或分割预测）。
        affine: 4x4 仿射矩阵，应与数据的空间信息匹配（通常来自 load_nifti）。
        path: 输出路径，建议以 .nii.gz 结尾（nibabel 会自动做 gzip 压缩）。
    """
    ensure_dir(os.path.dirname(os.path.abspath(path)))  # 确保父目录存在
    img = nib.Nifti1Image(data, affine)  # 构建 NIfTI 图像对象
    img.to_filename(path)                # 写入磁盘（扩展名决定是否 gzip 压缩）


# ============================================================
# 第 3 组：JSON / CSV 日志
# ============================================================

def save_json(obj, path: str) -> None:
    """把 Python 对象保存为 JSON 文件（UTF-8、缩进 2 空格、保留中文）。

    Args:
        obj: 可被 json 序列化的对象（dict / list / 数字 / 字符串等）。
        path: 输出 JSON 路径。
    """
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        # ensure_ascii=False：直接输出中文而非 \\uXXXX 转义，便于阅读
        # indent=2：每层缩进 2 空格，格式美观


def load_json(path: str):
    """读取 JSON 文件并返回对应的 Python 对象。

    Args:
        path: JSON 文件路径。

    Returns:
        反序列化后的 Python 对象（通常是 dict 或 list）。
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_csv_row(path: str, row: dict, fieldnames: list = None) -> None:
    """向 CSV 文件追加一行记录；若文件不存在则先写表头。

    常用于记录每个 epoch 的训练/验证指标，逐行追加后可用 Excel 或 pandas 画出
    训练曲线（loss / dice 随 epoch 变化）。

    Args:
        path: CSV 文件路径。
        row: 单行数据，键→值 的字典，如 {"epoch": 1, "val_dice": 0.85}。
        fieldnames: 表头字段顺序列表。若为 None 且文件不存在，则用 row 的键顺序。
    """
    # 确定字段顺序：优先用传入的 fieldnames，否则用 row 的键
    if fieldnames is None:
        fieldnames = list(row.keys())

    file_exists = os.path.isfile(path)
    ensure_dir(os.path.dirname(os.path.abspath(path)))

    # newline="" 是 csv 模块在 Windows 下的推荐写法，避免行间出现多余空行
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()  # 新文件先写表头
        writer.writerow(row)      # 追加一行（row 中多余/缺失的字段会被忽略/留空）


# ============================================================
# 第 4 组：可视化配色与切片选择
# ============================================================

def get_class_colors() -> dict:
    """返回各类别的 RGB 颜色字典（取值 0~1 浮点，便于 matplotlib 叠加显示）。

    约定：
        - 0: 背景，黑色 (0,0,0)（叠加时等同于不画）；
        - 1: 肝脏 liver，红色 (1,0,0)；
        - 2: 肿瘤 tumor，绿色 (0,1,0)。

    Returns:
        dict: {类别id: (R, G, B)}。
    """
    return {
        0: (0.0, 0.0, 0.0),  # 背景（黑/透明）
        1: (1.0, 0.0, 0.0),  # 肝脏——红色
        2: (0.0, 1.0, 0.0),  # 肿瘤——绿色
    }


def pick_slice_with_lesion(label: np.ndarray, num_classes: int) -> int:
    """在轴向切片中选择「前景/病灶像素最多」的那一张切片索引，用于可视化。

    医学 CT 通常沿 Z 轴一层层扫描，每一层叫一个「轴向切片 (axial slice)」。
    本函数假设输入 label 是 3D 数组，且第 3 个维度（axis=2）为轴向方向，即
    ``label.shape = (H, W, D)``，``label[:, :, z]`` 是第 z 张轴向切片。
    （如果你的数据经过 MONAI 变换后空间维度顺序不同，请相应调整求和的轴。）

    选择策略：
        - 三分类 (num_classes>=3)：优先统计肿瘤（类别 2）像素最多的切片；若整卷
          没有肿瘤，则退化为统计所有前景 (label>0) 最多的切片。
        - 二分类 (num_classes==2)：前景即肝脏，统计 label>0 最多的切片。
        - 若整卷完全为空（无任何前景），返回中间切片，保证可视化仍能出图。

    Args:
        label: 3D 标签数组，取值为类别 id（0 背景、1 肝脏、2 肿瘤）。
        num_classes: 类别数（2=二分类，3=三分类），用于决定统计哪种前景。

    Returns:
        int: 选中的轴向切片索引（沿 axis=2）。
    """
    # 三分类优先找肿瘤；二分类直接用肝脏前景
    if num_classes >= 3:
        fg = (label == 2).astype(np.int32)      # 肿瘤前景
        if fg.sum() == 0:
            fg = (label > 0).astype(np.int32)   # 无肿瘤则用全部前景
    else:
        fg = (label > 0).astype(np.int32)       # 二分类前景即肝脏

    # 沿前两轴 (H, W) 求和，得到每张轴向切片的前景像素数，形状 (D,)
    counts = fg.sum(axis=(0, 1))

    if counts.max() == 0:
        # 整卷无任何前景，返回中间切片
        return label.shape[2] // 2
    return int(np.argmax(counts))  # 前景最多的切片索引


def overlay_mask(image_slice: np.ndarray, mask_slice: np.ndarray,
                 color: tuple, alpha: float = 0.5) -> np.ndarray:
    """把单类二值掩膜叠加到灰度 CT 切片上，返回 RGB 图（float32，范围 0~1）。

    可视化脚本 (scripts/visualize.py) 可调用本函数为 liver / tumor 分别叠加
    不同颜色。注意：本函数不做「CT 窗 (windowing)」裁剪，而是把整张切片的灰度
    线性归一化到 [0,1]；如需特定窗宽窗位（如肝窗 W400/L40），请在传入前自行裁剪。

    Args:
        image_slice: 2D 灰度切片数组（CT 值范围任意，函数内部会归一化到 0~1）。
        mask_slice:  2D 二值掩膜（>0 视为该类前景）。
        color:       (R, G, B) 颜色元组，取值 0~1，可用 get_class_colors() 获取。
        alpha:       叠加透明度，0~1，越大颜色越浓，默认 0.5。

    Returns:
        np.ndarray: 形状 (H, W, 3) 的 RGB 图，float32，范围 0~1。
    """
    # 1) 把灰度图线性归一化到 [0,1]，便于与颜色叠加
    img = image_slice.astype(np.float32)
    img_min, img_max = float(img.min()), float(img.max())
    if img_max - img_min > 1e-8:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)  # 常量图直接置零，避免除以 0

    # 2) 扩展为 3 通道灰度 RGB
    rgb = np.stack([img, img, img], axis=-1)  # (H, W, 3)

    # 3) 准备掩膜与颜色
    mask = (mask_slice > 0).astype(np.float32)        # (H, W)
    color_arr = np.array(color, dtype=np.float32)     # (3,)

    # 4) 在掩膜区域按 alpha 混合：out = 灰度*(1-a*mask) + 颜色*(a*mask)
    blend = (alpha * mask)[..., None]  # (H, W, 1)，便于和 (H, W, 3) 广播
    rgb = rgb * (1.0 - blend) + color_arr * blend
    return rgb.astype(np.float32)


# ============================================================
# 第 5 组：STL 导出（scikit-image marching cubes + trimesh）
# ============================================================

def export_stl(mask: np.ndarray, output_path: str,
               spacing: tuple = (1.0, 1.0, 1.0),
               smooth: bool = False) -> None:
    """把 3D 二值掩膜转换为三角网格并导出为 STL 文件。

    原理（面向初学者）：
        - 「marching cubes（移动立方体）」是一种经典的等值面提取算法：它在 3D
          体素网格里逐个小立方体地搜索，找到值为「阈值 (level)」的等值面，并用
          三角片近似该表面。本函数用 level=0.5 在 0/1 掩膜上提取前景的表面。
        - spacing 是体素间距 (mm)，决定输出网格的真实物理尺寸，应与 CT 的
          spacing 一致，否则导出的肝脏会变形。
        - trimesh 把顶点 (vertices) 和三角面 (faces) 组织成网格对象，可做平滑、
          导出 STL/OBJ 等格式。STL 可用 MeshLab / Blender / 在线 3D 查看器打开。

    依赖说明：scikit-image 与 trimesh 在本函数内「延迟导入」，未安装时只有调用
    本函数才会报错，不影响其他工具函数。

    Args:
        mask: 3D 二值 numpy 数组，前景=1、背景=0。
        output_path: 输出 STL 路径，建议以 .stl 结尾。
        spacing: 体素间距 (sx, sy, sz)，单位 mm，默认全 1。
        smooth: 是否对网格做一次 Taubin 平滑（可去锯齿，但会略微改变形状）。
    """
    # —— 延迟导入：仅在使用 STL 功能时才需要这两个包 ——
    from skimage import measure
    import trimesh

    # 若掩膜全空（无前景），打印警告并跳过，避免 marching cubes 报错
    if mask.sum() == 0:
        print(f"[警告] 掩膜无前景体素，跳过 STL 导出：{output_path}")
        return

    # 1) 提取等值面：level=0.5 在 0/1 掩膜上即为前景边界
    #    返回 verts(N,3)、faces(M,3)、normals(N,3)、values(N,)
    verts, faces, normals, values = measure.marching_cubes(
        mask, level=0.5, spacing=spacing
    )

    # 2) 构建三角网格
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    # 3) 可选平滑：Taubin 滤波在保留体积的同时减少表面锯齿
    if smooth:
        try:
            from trimesh.smoothing import filter_taubin
            filter_taubin(mesh, lamb=0.5, mu=-0.53, iterations=10)
        except Exception as e:
            # 平滑失败不应中断导出流程，降级为未平滑网格
            print(f"[警告] 网格平滑失败，将导出未平滑网格：{e}")

    # 4) 保存 STL（确保父目录存在）
    ensure_dir(os.path.dirname(os.path.abspath(output_path)))
    mesh.export(output_path)
    print(f"[STL] 已导出：{output_path}（顶点 {len(verts)}，面 {len(faces)}）")


def export_label_stl(label: np.ndarray, label_id: int, output_path: str,
                     spacing: tuple = (1.0, 1.0, 1.0),
                     smooth: bool = False) -> None:
    """从多类标签中取出指定类别，生成二值掩膜后导出 STL。

    例如 label_id=1 导出肝脏，label_id=2 导出肿瘤。本函数是对 export_stl 的一层
    便捷封装：先 ``(label == label_id)`` 得到该类的 0/1 掩膜，再调用 export_stl。

    Args:
        label: 3D 多类标签数组（0 背景、1 肝脏、2 肿瘤）。
        label_id: 要导出的类别 id。
        output_path: 输出 STL 路径。
        spacing: 体素间距 (mm)，透传给 export_stl。
        smooth: 是否平滑，透传给 export_stl。
    """
    binary_mask = (label == label_id).astype(np.uint8)  # 取出该类二值掩膜
    export_stl(binary_mask, output_path, spacing=spacing, smooth=smooth)

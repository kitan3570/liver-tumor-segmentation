"""src/models.py — 肝脏 3D 分割项目的模型与损失模块

本文件封装了 MONAI 提供的 3D U-Net 网络与医学分割常用的损失函数，供
src/trainer.py 训练循环与 scripts/infer.py 推理脚本调用。包含 4 个函数：

1. build_model : 根据 config['model'] 构建 MONAI 3D U-Net 并打印参数量
2. save_model  : 保存模型权重(state_dict)到磁盘
3. load_model  : 从磁盘加载权重到已有模型结构中
4. build_loss  : 根据 config['loss'] 构建 DiceLoss 或 DiceCELoss

设计说明
--------
- 面向初学者：每个函数配中文 docstring，并在关键步骤加行内注释，解释 3D U-Net
  的「编码器-瓶颈-解码器+跳跃连接」结构、Dice/CE 损失的直觉与各超参含义。
- 模型与损失都「从配置驱动」：所有超参数来自 configs/*.yaml 的 model/loss 字段，
  本模块不硬编码任何数值，方便二分类(phase1)与三分类(phase2)切换。
- save/load 只保存 state_dict（而非整个模型对象），这样更安全、更省空间，且
  不依赖具体代码路径；加载时需先 build_model 构造相同结构再 load_state_dict。
- 本文件是「库模块」，不包含 if __name__ == "__main__"；其他模块通过
  ``from src.models import build_model, build_loss, ...`` 来调用。
"""

# ===================== 标准库导入 =====================
import os

# ===================== 第三方库导入 =====================
# torch 提供张量与模型参数管理；monai 提供医学影像专用的 3D 网络与损失函数。
import torch
from monai.networks.nets import UNet
from monai.losses import DiceLoss, DiceCELoss

# ===================== 本项目模块导入 =====================
from src.utils import ensure_dir


# ============================================================
# 第 1 组：模型构建与权重存取
# ============================================================

def build_model(config: dict) -> torch.nn.Module:
    """根据 config['model'] 构建 MONAI 3D U-Net 并返回，同时打印参数量。

    【3D U-Net 结构直觉（面向初学者）】
        U-Net 是医学影像分割的「经典骨干网络」，因结构图形似字母 U 而得名：
        - 编码器（U 的左半边下行）：由若干「下采样块」组成，每块用步长 2 的卷积
          把空间分辨率减半、同时把特征通道数加倍。它逐步提取从低级(边缘/纹理)
          到高级(器官形状/语义)的特征，但空间分辨率越来越低。
        - 瓶颈（U 的底部）：最深一层，分辨率最小但感受野最大，负责「全局理解」。
        - 解码器（U 的右半边上行）：逐层「上采样」恢复空间分辨率，把小特征图放大
          回原图大小；同时把通道数减半。
        - 跳跃连接(skip connection)：把编码器同层的精细特征直接拼接到解码器对应层，
          让网络在恢复边界细节时「既看得清全局，又画得准边缘」，这是 U-Net 分割
          精度高的关键所在。

    【各参数含义】
        spatial_dims=3：3D 分割，卷积/归一化都用 3D 版本(Conv3d / BatchNorm3d)。
        in_channels：输入通道数，单模态 CT 为 1（灰度体素）。
        out_channels：输出通道数 = 类别数(num_classes)。phase1=2(背景/肝脏)，
            phase2=3(背景/肝脏/肿瘤)。网络对每个体素输出一个 C 维向量，经 softmax
            后得到属于各类的概率。
        channels：各分辨率层的特征通道数列表，如 [16,32,64,128,256]，由浅到深
            通道数递增。列表长度决定 U-Net 的层数。
        strides：每次下采样的步长列表，长度必须 = len(channels) - 1。例如
            [2,2,2,2] 表示 4 次下采样，分辨率会缩小 2^4=16 倍。
        num_res_units：每个卷积块内的残差单元数。残差连接(x + F(x))能缓解深层
            网络的梯度消失，让训练更稳、表达更强；本项目设 2。
        norm：归一化层类型字符串。BATCH=批归一化(BatchNorm)，适合大 batch；
            INSTANCE=实例归一化(InstanceNorm)，对 batch=1~2 的小批量医学影像
            更稳定。本项目标准设置 batch=2，仍用 BATCH。

    Args:
        config: 顶层配置字典，需含 config['model'] 子字典，字段包括
            in_channels、out_channels、channels、strides、num_res_units、norm。

    Returns:
        torch.nn.Module: 构建好的 MONAI 3D U-Net 模型（尚未搬到 GPU，由调用方决定）。
    """
    # 从配置中取出模型超参子字典
    cfg = config["model"]

    # 用 MONAI 的 UNet 构造 3D 网络；所有超参来自配置，本函数不硬编码数值
    model = UNet(
        spatial_dims=3,                     # 3D 体素分割
        in_channels=cfg["in_channels"],     # 输入通道：CT 单模态 = 1
        out_channels=cfg["out_channels"],   # 输出通道 = 类别数(2 或 3)
        channels=cfg["channels"],           # 各层特征通道数列表
        strides=cfg["strides"],             # 下采样步长列表，长度 = len(channels) - 1
        num_res_units=cfg["num_res_units"], # 每个卷积块的残差单元数
        norm=cfg["norm"],                   # 归一化层：BATCH / INSTANCE
    )

    # 统计并打印总参数量，让学习者直观感受模型规模
    # numel() 返回张量元素总数；sum 后得到全部可训练参数个数
    n_params = sum(p.numel() for p in model.parameters())
    # 用千位分隔符让大数字更易读，例如 5,123,456；同时换算成百万(M)单位
    print(f"[模型] 3D U-Net 已构建：out_channels={cfg['out_channels']}，"
          f"参数量 = {n_params:,} ({n_params / 1e6:.2f} M)")

    return model


def save_model(model: torch.nn.Module, path: str) -> None:
    """把模型权重(state_dict)保存到磁盘。

    只保存 state_dict（各层参数张量的有序字典），不保存整个 model 对象。
    原因：保存整个对象会 pickle 模型类定义所在的代码路径，换机器或改代码后
    容易加载失败；而 state_dict 与具体代码解耦，加载时只要网络结构一致即可。

    Args:
        model: 已训练(或部分训练)的模型。
        path:  目标保存路径，通常以 .pt 或 .pth 结尾，位于 outputs/models/ 下。
    """
    # 先确保父目录存在（如 outputs/models/best.pt → 确保 outputs/models 存在）
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    # torch.save 默认用 zip 格式保存 state_dict，体积小、加载快
    torch.save(model.state_dict(), path)
    print(f"[模型] 权重已保存：{path}")


def load_model(model: torch.nn.Module, path: str,
               device: torch.device = None) -> torch.nn.Module:
    """从磁盘加载权重到给定模型结构中，返回加载完毕的模型。

    使用流程（典型）：
        model = build_model(config)                     # 先构造相同结构的空模型
        model = load_model(model, ckpt_path, device)    # 再把权重灌进去

    【weights_only 兼容性处理】
        PyTorch 2.6 起 torch.load 的 weights_only 默认值由 False 改为 True，
        以提升安全性(防止反序列化时执行恶意代码)。本项目只保存纯 state_dict，
        理论上 weights_only=True 总能加载成功；但为兼容旧版 PyTorch(<2.0 该
        参数不存在)以及极少数含辅助对象的旧 checkpoint，这里采用「先试 True，
        失败回退 False」的策略，并捕获 TypeError(旧版无此参数)。

    Args:
        model:  已构造好的、与待加载权重结构一致的空模型。
        path:   权重文件路径(.pt/.pth)，通常由 save_model 生成。
        device: 目标设备；None 时仅按 map_location 放到 CPU，不额外搬运。
            传入 torch.device('cuda') 则加载后搬到 GPU。

    Returns:
        torch.nn.Module: 加载好权重、并已搬到指定设备(若提供)的模型。
    """
    # map_location 决定张量先加载到哪；device 为 None 时退化为字符串 'cpu'
    map_location = device if device is not None else "cpu"

    # —— 兼容多版本 torch 的权重加载 ——
    try:
        # 优先用 weights_only=True（更安全，PyTorch 2.6+ 默认即此）
        state_dict = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # 旧版 torch 的 torch.load 没有 weights_only 参数，去掉它再加载
        state_dict = torch.load(path, map_location=map_location)
    except Exception:
        # 极少数情况：checkpoint 含非权重对象(如旧版保存了整个模型)，
        # weights_only=True 会拒绝加载，此时回退到 False 以兼容历史文件
        state_dict = torch.load(path, map_location=map_location, weights_only=False)

    # strict=True 要求 state_dict 的键与 model 的参数键完全一一对应；
    # 若不匹配会报错，能及时发现「配置改了但用旧权重」等不一致问题
    model.load_state_dict(state_dict, strict=True)

    # 若指定了设备，把整个模型搬到该设备；否则保持原样(已在 CPU 上)
    if device is not None:
        model = model.to(device)

    print(f"[模型] 权重已加载：{path}")
    return model


# ============================================================
# 第 2 组：损失函数构建
# ============================================================

def build_loss(config: dict) -> torch.nn.Module:
    """根据 config['loss'] 构建 DiceLoss 或 DiceCELoss 并返回。

    【损失函数直觉（面向初学者）】
        - Dice 损失：基于「Dice 系数」衡量预测掩膜与真值掩膜的重叠程度。
            Dice = 2|A∩B| / (|A|+|B|)，取值 0~1，越大越重叠。损失用 1 - Dice，
            所以训练时 Dice 损失越小，重叠越好。Dice 对类别不平衡(背景远多于
            前景)不敏感，是医学分割的首选区域损失。
        - 交叉熵(CE) 损失：逐像素做分类，衡量预测概率分布与真值类别的差异。
            它提供「像素级」监督信号，能帮网络学会明确的类别边界，与 Dice 互补。
            DiceCE = lambda_dice * Dice + lambda_ce * CE，兼顾区域重叠与逐像素
            分类，是多类分割的常用组合。

    【关键参数含义】
        include_background：Dice 是否把背景类(0)也算进去。
            False 表示只对前景类(肝脏/肿瘤)计算 Dice——肝脏肿瘤任务中背景占比
            极大，计入背景会让 Dice 虚高、掩盖前景表现，故常用 False。
        softmax：是否对网络输出做 softmax 归一化为概率。多类分割(>2 类)必须
            True，使各通道概率和为 1；本项目二分类/三分类均用多类形式，故 True。
        to_onehot_y：是否把标签转为 one-hot 再与概率比较。多类分割中标签是
            类别 id(0/1/2)，需转成 C 个 0/1 通道才能与 softmax 概率逐通道计算，
            故多类分割必需 True。
        squared_pred：是否在 Dice 分母里对预测取平方。开启后对小目标(如肿瘤)
            的错分惩罚更大，能提升小目标的召回与收敛，本项目设 True。
        lambda_dice / lambda_ce：仅 DiceCELoss 使用，分别是 Dice 与 CE 的权重；
            两者都设 1.0 表示等权组合。

    Args:
        config: 顶层配置字典，需含 config['loss'] 子字典，字段包括
            name、include_background、softmax、to_onehot_y、squared_pred，
            以及(仅 DiceCE 时)lambda_dice、lambda_ce。

    Returns:
        torch.nn.Module: DiceLoss 或 DiceCELoss 实例。

    Raises:
        ValueError: 当 config['loss']['name'] 既不是 'Dice' 也不是 'DiceCE' 时。
    """
    cfg = config["loss"]
    name = cfg["name"]

    # 这四个参数 Dice 与 DiceCE 都支持，统一打包成 kwargs 复用
    common_kwargs = dict(
        include_background=cfg["include_background"],
        softmax=cfg["softmax"],
        to_onehot_y=cfg["to_onehot_y"],
        squared_pred=cfg["squared_pred"],
    )

    if name == "DiceCE":
        # DiceCELoss 额外接受两类损失的权重；lambda_dice / lambda_ce 默认各 1.0
        loss_fn = DiceCELoss(
            **common_kwargs,
            lambda_dice=cfg["lambda_dice"],
            lambda_ce=cfg["lambda_ce"],
        )
        print(f"[损失] 已构建 DiceCELoss：lambda_dice={cfg['lambda_dice']}, "
              f"lambda_ce={cfg['lambda_ce']}, "
              f"include_background={cfg['include_background']}")
    elif name == "Dice":
        # DiceLoss 只用区域重叠，不接受 lambda_dice / lambda_ce
        loss_fn = DiceLoss(**common_kwargs)
        print(f"[损失] 已构建 DiceLoss：include_background={cfg['include_background']}")
    else:
        raise ValueError(
            f"不支持的损失名称：{name}\n"
            "请在 config['loss']['name'] 中指定 'Dice' 或 'DiceCE'。"
        )

    return loss_fn

"""src/trainer.py — 肝脏 3D 分割项目的训练循环与评价指标模块

本文件是项目「跑起来」的核心：它把数据 (src.data)、模型 (src.models)、工具
(src.utils) 串成一条完整的训练—验证流水线，并实现面向初学者的分割评价指标。

面向大一医学生的概念速览
========================
- **3D 语义分割**：网络对每个体素（voxel，3D 像素）输出它属于哪个类别。本项目
  阶段一为二分类 {0=背景, 1=肝脏}，阶段二为三分类 {0=背景, 1=肝脏, 2=肿瘤}。
- **训练循环 (epoch)**：把训练集完整看一遍叫一个 epoch；多次遍历让模型逐步
  学会从 CT 中分割出肝脏/肿瘤。
- **验证 (validate)**：用没参与训练的验证集评估当前模型好坏，防止「死记硬背」
  （过拟合）。我们每隔 ``val_interval`` 个 epoch 验证一次。
- **AMP 混合精度**：用 float16 做前向/反向以省显存、加速，再用 float32 维护
  主权重保持数值稳定；需要 GPU 支持。
- **滑动窗口推理 (SlidingWindowInferer)**：整张 CT 体积（如 512×512×100）远
  大于训练 patch（96³），推理时把大体积切成多个 96³ 小块分别预测，再拼接回
  原尺寸；相邻块重叠 (overlap) 并加权平均，减少拼接处的接缝伪影。
- **评价指标**：Dice、IoU、Precision、Recall（公式见 compute_metrics 注释）。
  对「空真值」（如某验证样本里根本没有肿瘤）单独处理为 NaN，避免分母为 0 给
  出误导性的 0 分或 1 分。

模块结构
--------
A. 指标：compute_metrics / aggregate_metrics
B. 训练 / 验证：train_one_epoch / validate / run_training

本文件为「库模块」，不含 ``if __name__ == "__main__"``；由 scripts/train.py
加载配置后调用 ``run_training(config)`` 启动训练。
"""

# ===================== 标准库导入 =====================
import os

# ===================== 第三方库导入 =====================
# torch：深度学习框架（模型前向、自动求导、AMP 混合精度）
# numpy：指标计算用的数值运算
# tqdm：终端进度条，让训练过程可视化
# monai.inferers.SlidingWindowInferer：MONAI 提供的滑动窗口推理器
import numpy as np
import torch
from tqdm import tqdm
from monai.inferers import SlidingWindowInferer

# ===================== 本项目模块导入 =====================
# 说明：src.data 与 src.models 由其他任务并行实现，此处按约定接口调用。
#   build_dataloader(data_list, config, phase) -> DataLoader
#   build_model(config) -> nn.Module；build_loss(config) -> nn.Module
#   save_model(model, path)；load_model(model, path, device)
from src.utils import (
    set_seed,
    get_device,
    ensure_dir,
    append_csv_row,
    save_json,
    load_json,
)
from src.data import build_dataloader
from src.models import build_model, build_loss, save_model


# ============================================================
# 内部小工具
# ============================================================

def _to_int_3d(arr) -> np.ndarray:
    """把输入规整为 3D、int64 的 numpy 数组，形状 (D, H, W)。

    compute_metrics 接受 tensor 或 numpy，且可能是 (D,H,W) 或带前导 1 维的
    (1,D,H,W)。本函数统一转换，便于后续用 ``arr == c`` 做逐体素类别比较。

    Args:
        arr: torch.Tensor 或 numpy 数组，形状 (D,H,W) 或 (1,D,H,W)，
             取值为类别 id（0/1/2）。

    Returns:
        np.ndarray: 形状 (D,H,W)、dtype=int64 的数组。
    """
    # 1) 若是 torch 张量，先脱离计算图并搬到 CPU 再转 numpy（标签/预测都在 CPU 算指标）
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    # 2) 去掉可能的前导尺寸为 1 的维度：(1,D,H,W) -> (D,H,W)
    if arr.ndim == 4:
        arr = arr[0]
    # 3) 兜底：若维度仍不为 3（如 (1,1,D,H,W)），压缩所有长度为 1 的维度
    if arr.ndim != 3:
        arr = np.squeeze(arr)
    return arr.astype(np.int64)


# ============================================================
# A. 指标计算
# ============================================================

def compute_metrics(pred_seg, label_seg, num_classes: int) -> dict:
    """计算单个 3D 样本的分割指标（Dice / IoU / Precision / Recall）。

    面向初学者的公式说明（对每个类别 c 单独计算）
    ------------------------------------------------
    设预测掩膜 ``pred_c = (pred == c)``，真值掩膜 ``label_c = (label == c)``：

        TP（真正例）= 预测为 c 且真值也是 c 的体素数 = 交集大小
        FP（假正例）= 预测为 c 但真值不是 c 的体素数 = pred_c - label_c
        FN（假负例）= 真值为 c 但预测不是 c 的体素数 = label_c - pred_c

        Dice     = 2·TP / (2·TP + FP + FN)   # 与 F1-score 等价，衡量重叠程度
        IoU/Jacc = TP / (TP + FP + FN)       # 交并比，比 Dice 更严
        Precision= TP / (TP + FP)            # 预测为 c 的体素里，有多少真是 c
        Recall   = TP / (TP + FN)            # 真值为 c 的体素里，有多少被找出来

    上述即「基于混淆矩阵手算」的方式，不依赖 monai.metrics，便于理解与核对。

    空真值处理（关键）
    ------------------
    若某类别在 **真值** 中完全不存在（``label_c.sum() == 0``，例如某样本没有
    肿瘤），则该类的四项指标一律记为 ``NaN``。原因：
        - 此时 Dice 的分母 ``2·TP+FP+FN`` 可能退化为 0（若模型也没预测出该类），
          0/0 无意义；
        - 即便分母不为 0（模型误报了一些 FP），算出的 0 分会拉低均值，误导
          「模型在肿瘤上表现很差」——其实只是这个样本本就没有肿瘤。
    用 NaN 标记后，aggregate_metrics 会「忽略空真值样本」再做平均，更公平。
    返回 dict 的值均为 float 或 nan（不额外塞入布尔标记，保持类型一致）。

    Args:
        pred_seg:  预测分割，3D tensor/数组，值为类别 id（argmax 后的结果）。
        label_seg: 真值分割，3D tensor/数组，值为类别 id。
        num_classes: 类别数；2 时只算 liver，>=3 时额外算 tumor。

    Returns:
        dict: 键为 ``liver_dice/liver_iou/liver_precision/liver_recall``；
              当 ``num_classes >= 3`` 时额外包含 ``tumor_*`` 四项。
              每个值为 float，空真值时为 ``float('nan')``。
    """
    # 统一转成 (D,H,W) 的 int64 数组
    pred = _to_int_3d(pred_seg)
    label = _to_int_3d(label_seg)

    # 需要评估的前景类别：类别 1 = 肝脏 liver；类别 2 = 肿瘤 tumor（仅三分类）
    # 用 (类别id, 类别名) 元组列表，便于生成 'liver_*' / 'tumor_*' 这样的键名
    classes_to_eval = [(1, "liver")]
    if num_classes >= 3:
        classes_to_eval.append((2, "tumor"))

    metrics = {}
    for class_id, name in classes_to_eval:
        # 逐体素判断是否属于当前类别，得到 bool 掩膜
        pred_c = (pred == class_id)
        label_c = (label == class_id)

        # 计算 TP / FP / FN（都用 bool 数组的逻辑运算，再求和得体素数）
        tp = int(np.logical_and(pred_c, label_c).sum())          # 交集：预测=真值=c
        fp = int(np.logical_and(pred_c, np.logical_not(label_c)).sum())  # 预测=c 但真值≠c
        fn = int(np.logical_and(np.logical_not(pred_c), label_c).sum())  # 真值=c 但预测≠c

        # —— 空真值处理：该类在真值中不存在 —— 记为 NaN（见函数 docstring 说明）
        if int(label_c.sum()) == 0:
            dice = iou = precision = recall = float("nan")
        else:
            # 真值存在该类时，分母大多 >0；但对 Precision，当模型完全没有
            # 预测出该类（TP+FP==0）时分母为 0，此时 Precision 无定义，记 NaN。
            denom_dice = 2 * tp + fp + fn
            denom_iou = tp + fp + fn
            denom_prec = tp + fp
            denom_rec = tp + fn  # 注意：denom_rec == label_c.sum() > 0，恒 >0

            dice = (2.0 * tp / denom_dice) if denom_dice > 0 else float("nan")
            iou = (tp / denom_iou) if denom_iou > 0 else float("nan")
            precision = (tp / denom_prec) if denom_prec > 0 else float("nan")
            recall = (tp / denom_rec) if denom_rec > 0 else float("nan")

        # 写入返回字典，键名形如 'liver_dice'、'tumor_recall'
        metrics[f"{name}_dice"] = float(dice)
        metrics[f"{name}_iou"] = float(iou)
        metrics[f"{name}_precision"] = float(precision)
        metrics[f"{name}_recall"] = float(recall)

    return metrics


def aggregate_metrics(metric_dicts: list) -> dict:
    """把多个样本的指标结果汇总成一组均值（与可选标准差）。

    「忽略空真值样本」策略
    ----------------------
    compute_metrics 对空真值类别返回 NaN。本函数对每个指标：
        - 只收集非 NaN 的样本值；
        - 取其均值作为该指标在验证集上的汇总值；
        - 若所有样本都是 NaN（如验证集里没有任何样本含肿瘤），汇总记为 NaN。
    这样做避免了「把没有肿瘤的样本硬记 0 分」拉低肿瘤指标均值的问题，使汇报
    的肿瘤 Dice 等只反映「真正含有肿瘤的样本」上的表现——对医学小目标分割
    尤为重要。

    Args:
        metric_dicts: 各样本经 compute_metrics 得到的 dict 列表。

    Returns:
        dict: 汇总结果。每个原始指标键保留均值（键名不变），并额外附加
              ``<metric>_std`` 标准差（ddof=0）。全部 NaN 时均值与 std 均为 nan。
              若输入为空列表，返回空 dict。
    """
    # 空输入兜底，避免下游除零或取键报错
    if not metric_dicts:
        return {}

    # 以第一个样本的键为基准（所有样本键应一致），逐指标聚合
    keys = list(metric_dicts[0].keys())
    summary = {}
    for k in keys:
        # 收集该指标在所有样本中的值，转 float64 便于 NaN 判断
        vals = np.array([d[k] for d in metric_dicts if k in d], dtype=np.float64)
        valid = vals[~np.isnan(vals)]  # 只保留非 NaN（即非空真值）样本

        if valid.size == 0:
            # 整个验证集都没有该类真值：均值与标准差均记 NaN
            summary[k] = float("nan")
            summary[f"{k}_std"] = float("nan")
        else:
            summary[k] = float(np.mean(valid))
            # ddof=0（总体标准差）：单样本时返回 0 而非 NaN，更直观
            summary[f"{k}_std"] = float(np.std(valid))

    return summary


# ============================================================
# B. 训练与验证
# ============================================================

def train_one_epoch(model, loader, loss_fn, optimizer, device,
                    amp: bool = True, scaler=None) -> float:
    """跑完一个 epoch 的训练，返回该 epoch 的平均 loss。

    AMP 混合精度原理（初学者版）
    ----------------------------
    普通训练用 float32（单精度）做前向/反向，精度高但显存大、速度慢。AMP
    (Automatic Mixed Precision) 在前向时把部分算子自动转成 float16（半精度），
    显存减半、速度提升；而权重的主副本仍用 float32 维护，避免数值溢出/下溢。
    具体流程：
        - ``torch.cuda.amp.autocast(enabled=amp)`` 上下文：内部前向用半精度；
        - ``GradScaler``：把 loss 放大后再 backward，防止小梯度在 float16 下
          变成 0（下溢）；optimizer.step 前再缩放回正确梯度。
    若不在 CUDA 上（如 CPU 自检），AMP 自动降级为普通 float32 训练。

    关于 loss_fn
    ------------
    build_loss 返回的是多类 DiceCE 损失，内部已配置 ``to_onehot_y=True`` 与
    ``softmax=True``。因此这里直接把网络原始输出 (B,C,D,H,W) 和类别 id 标签
    (B,1,D,H,W) 喂给 loss_fn 即可，**无需**手动做 one-hot 或 softmax。

    Args:
        model:     已 ``.to(device)`` 的分割网络（nn.Module）。
        loader:    训练集 DataLoader，每个 batch 是 dict，含 'image' 与 'label'。
        loss_fn:   由 build_loss 构造的多类损失函数。
        optimizer: 优化器（如 Adam）。
        device:    torch.device，模型与数据所在设备。
        amp:       是否启用混合精度；非 CUDA 设备下会自动降级。
        scaler:    可选的 GradScaler；若启用 AMP 但未传入，则内部新建一个。

    Returns:
        float: 该 epoch 所有 batch 的 loss 平均值。
    """
    # 仅在 CUDA 设备上才真正启用 AMP，CPU 下 use_amp=False 走普通路径
    use_amp = amp and (device.type == "cuda")
    if use_amp and scaler is None:
        # 调用方未传 scaler 时自行创建；GradScaler 在 CUDA 不可用时会自动停用
        scaler = torch.cuda.amp.GradScaler()

    model.train()  # 训练模式：启用 Dropout/BN 的训练行为

    running_loss = 0.0
    n_batches = 0

    # tqdm 包裹 loader，显示进度条；dynamic_ncols=True 让宽度自适应终端
    pbar = tqdm(loader, desc="训练 Train", dynamic_ncols=True)
    for batch in pbar:
        # batch 是 dict：'image' (B,1,D,H,W) float、'label' (B,1,D,H,W) int64（类别 id）
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()  # 清空上一步残留梯度

        # autocast 上下文内的前向会自动混合精度；enabled=False 时等同普通前向
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(inputs)        # (B, C, D, H, W)，C = num_classes
            loss = loss_fn(outputs, labels)  # loss_fn 内部做 to_onehot + softmax

        if use_amp:
            # AMP 路径：scale → backward → step(update 内部按需缩放梯度并跳过溢出步)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 普通 float32 路径
            loss.backward()
            optimizer.step()

        # 累计 loss（.item() 取出标量并脱离计算图）
        running_loss += loss.item()
        n_batches += 1
        # 进度条右侧实时显示当前 batch loss 与累计平均 loss
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running_loss / n_batches:.4f}")

    avg_loss = running_loss / n_batches if n_batches > 0 else 0.0
    return float(avg_loss)


def validate(model, loader, device, num_classes: int, inferer) -> dict:
    """在验证集上做滑动窗口推理并汇总指标。

    滑动窗口推理流程（SlidingWindowInferer）
    -----------------------------------------
    1. 整张 CT 体积 (如 512×512×N) 远大于训练 patch 96³，无法一次性送入网络；
    2. inferer 把大体积切成若干 96³ 的窗口，逐个送模型前向，再把各窗口预测
       拼接回原尺寸；``overlap=0.25`` 表示相邻窗口重叠 25%，重叠区做加权平均
       （高斯权重），从而减轻拼接处的接缝/块效应；
    3. ``sw_batch_size`` 控制同时并发的窗口数，显存充裕时可调大加速。
    本函数逐样本推理（验证 batch_size 通常为 1），对每个样本算指标后聚合。

    argmax 解码
    -----------
    网络输出 (B, C, D, H, W) 是每类的「得分/logit」；用 ``argmax(dim=1)`` 取
    得分最大的类别作为最终预测的类别 id，得到 (B,1,D,H,W) 的整数分割图，再与
    真值标签 (B,1,D,H,W) 比较计算指标。

    注意：本函数不做 ``keep_largest_cc`` 后处理（保留最大连通域）。该后处理
    属于推理脚本 (scripts/infer.py) 的职责；这里用原始 argmax 预测评估，能
    更真实地反映模型本身的能力。

    Args:
        model:       分割网络（会在函数内切到 eval 模式）。
        loader:      验证集 DataLoader，每个 batch 是 dict，含 'image' 与 'label'；
                     验证阶段通常 batch_size=1，且不做随机裁剪/翻转等增强，保持
                     全体积原尺寸送入滑动窗口推理。
        num_classes: 类别数；2 时只算 liver，>=3 时额外算 tumor。
        inferer:     monai.inferers.SlidingWindowInferer 实例，负责把大体积切成
                     roi_size 小块分别前向再拼接（滑动窗口推理）。

    Returns:
        dict: 经 aggregate_metrics 汇总后的验证集指标。键形如
              ``liver_dice / liver_iou / liver_precision / liver_recall`` 及其
              ``*_std``；三分类时额外含 ``tumor_*``。空真值样本以 NaN 忽略。
    """
    # —— 推理阶段不更新权重：切到 eval 模式（关闭 Dropout、BN 改用推理统计量），
    #    并用 torch.no_grad() 关闭自动求导图构建，节省显存与计算。
    model.eval()

    sample_metrics = []  # 收集每个样本的指标 dict，最后统一聚合

    with torch.no_grad():
        # tqdm 进度条；验证集虽小，但仍可视化进度
        pbar = tqdm(loader, desc="验证 Val", dynamic_ncols=True)
        for batch in pbar:
            # 'image' 搬到 device 用于前向；'label' 留在 CPU 即可（指标在 CPU 用 numpy 算）
            inputs = batch["image"].to(device)
            labels = batch["label"]  # (B,1,D,H,W) int64，类别 id

            # 滑动窗口推理：inferer 内部把整张 CT 切成若干 roi_size 小块，逐块
            # 前向再拼接回原体积尺寸；输出 (B, C, D, H, W) 是每类的 logit 得分。
            outputs = inferer(inputs, model)

            # argmax 解码：沿类别维 (dim=1) 取得分最大的类别，得到 (B,1,D,H,W)
            # 的整数分割图（类别 id），即可与真值标签逐体素比较计算指标。
            pred_seg = outputs.argmax(dim=1, keepdim=True)

            # 逐样本计算指标：pred_seg[i,0] 与 labels[i,0] 均为 (D,H,W)
            for i in range(pred_seg.shape[0]):
                m = compute_metrics(pred_seg[i, 0], labels[i, 0], num_classes)
                sample_metrics.append(m)

    # 聚合所有样本指标（空真值样本以 NaN 忽略），返回汇总 dict
    return aggregate_metrics(sample_metrics)


def run_training(config: dict) -> dict:
    """完整训练流程：构建数据/模型/优化器/调度器，循环训练并周期性验证，
    保存 best/last checkpoint 与 CSV 指标历史。

    本函数是 scripts/train.py 的直接调用对象：脚本加载 YAML 配置后，把整个
    config dict 传进来，由本函数完成「读划分 → 建 DataLoader → 建模型 →
    训练循环 → 保存」的全过程。

    Args:
        config: 完整配置字典（由 load_config 读自 configs/*.yaml）。需包含：
            seed, device, num_classes, data.splits_dir,
            train.{max_epochs, val_interval, amp},
            optimizer.{lr, weight_decay}, scheduler.T_max,
            infer.{roi_size, sw_batch_size, overlap},
            output.{models_dir, metrics_dir}。

    Returns:
        dict: 训练汇总信息，键为：
            - best_metrics: 最佳 epoch 的验证指标 dict（按 liver_dice 最大）；
            - best_epoch:   最佳 epoch 编号（从 1 开始）；
            - last_epoch:   最后一个 epoch 编号；
            - history_csv:  CSV 指标历史文件路径；
            - best_ckpt:    best.pt 权重路径；
            - last_ckpt:    last.pt 权重路径。
    """
    # 1) 固定随机种子 + 选择设备，保证实验可复现
    set_seed(config["seed"])
    device = get_device(config["device"])

    # 2) 读取 train/val 划分清单（由 scripts/split_data.py 预先生成的 JSON）
    splits_dir = config["data"]["splits_dir"]
    train_list = load_json(os.path.join(splits_dir, "train.json"))
    val_list = load_json(os.path.join(splits_dir, "val.json"))

    # 3) 构建 DataLoader：训练阶段做随机裁剪/增强；验证阶段不做随机增强，保留原尺寸
    train_loader = build_dataloader(train_list, config, "train")
    val_loader = build_dataloader(val_list, config, "val")

    # 4) 模型与损失：模型搬到 device；loss_fn 内部已处理 one-hot 与 softmax
    model = build_model(config).to(device)
    loss_fn = build_loss(config)

    # 5) 优化器：Adam（自适应学习率，医学分割常用）
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
    )

    # 6) 调度器：余弦退火，学习率随训练进度从 lr 平滑降到接近 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["scheduler"]["T_max"]
    )

    # 7) 滑动窗口推理器：验证时用，把大体积切成 roi_size 小块前向再拼接
    inferer = SlidingWindowInferer(
        roi_size=tuple(config["infer"]["roi_size"]),
        sw_batch_size=config["infer"]["sw_batch_size"],
        overlap=config["infer"]["overlap"],
    )

    # 8) AMP 混合精度 scaler：仅当启用 amp 且在 CUDA 上才创建，否则 None 走普通路径
    use_amp = config["train"]["amp"] and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # 9) 训练循环相关参数与产物路径
    max_epochs = config["train"]["max_epochs"]
    # val_interval：每隔几个 epoch 做一次验证并记录指标（验证较慢，不必每轮都做）
    val_interval = config["train"]["val_interval"]
    num_classes = config["num_classes"]
    models_dir = config["output"]["models_dir"]
    metrics_dir = config["output"]["metrics_dir"]
    ensure_dir(models_dir)
    ensure_dir(metrics_dir)
    history_csv = os.path.join(metrics_dir, "train_history.csv")
    best_ckpt = os.path.join(models_dir, "best.pt")
    last_ckpt = os.path.join(models_dir, "last.pt")

    # CSV 字段顺序：phase2(三分类) 额外记录 tumor_* 四项
    fieldnames = ["epoch", "train_loss",
                  "liver_dice", "liver_iou", "liver_precision", "liver_recall"]
    if num_classes >= 3:
        fieldnames += ["tumor_dice", "tumor_iou", "tumor_precision", "tumor_recall"]

    # 跟踪最佳模型：以 liver_dice 最大为准（肝脏是两个阶段的共同主目标）
    best_metrics = None
    best_epoch = 0
    best_liver_dice = -1.0

    # —— 主循环：tqdm 包裹 epoch 序列，显示整体训练进度 ——
    epoch_pbar = tqdm(range(1, max_epochs + 1), desc="Epoch", dynamic_ncols=True)
    for epoch in epoch_pbar:
        # 9.1) 训练一个 epoch，返回本轮平均 loss
        avg_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            amp=config["train"]["amp"], scaler=scaler,
        )

        # 9.2) 调度器步进：CosineAnnealingLR 每个 epoch step 一次，让学习率按
        #      余弦曲线逐步下降。时机：先 train 再 step（与 MONAI 教程一致），
        #      这样 step 用的「当前 epoch」与训练进度对齐。
        scheduler.step()

        # 9.3) 周期性验证：每 val_interval 个 epoch 或最后一个 epoch 做一次
        is_last_epoch = (epoch == max_epochs)
        do_validate = (epoch % val_interval == 0) or is_last_epoch

        if do_validate:
            val_metrics = validate(model, val_loader, device, num_classes, inferer)

            # 打印当前验证指标，便于实时观察收敛情况
            print(f"[Epoch {epoch}/{max_epochs}] train_loss={avg_loss:.4f} | "
                  f"liver_dice={val_metrics.get('liver_dice', float('nan')):.4f}")
            if num_classes >= 3:
                print(f"    tumor_dice={val_metrics.get('tumor_dice', float('nan')):.4f}")

            # 9.4) 写一行 CSV 指标历史（逐行追加，可用 Excel/pandas 画训练曲线）
            row = {
                "epoch": epoch,
                "train_loss": avg_loss,
                "liver_dice": val_metrics.get("liver_dice", float("nan")),
                "liver_iou": val_metrics.get("liver_iou", float("nan")),
                "liver_precision": val_metrics.get("liver_precision", float("nan")),
                "liver_recall": val_metrics.get("liver_recall", float("nan")),
            }
            if num_classes >= 3:
                row["tumor_dice"] = val_metrics.get("tumor_dice", float("nan"))
                row["tumor_iou"] = val_metrics.get("tumor_iou", float("nan"))
                row["tumor_precision"] = val_metrics.get("tumor_precision", float("nan"))
                row["tumor_recall"] = val_metrics.get("tumor_recall", float("nan"))
            append_csv_row(history_csv, row, fieldnames=fieldnames)

            # 9.5) checkpoint 保存策略：
            #   - best.pt：当本轮 liver_dice 超过历史最佳时保存（覆盖写），并记录
            #     best 指标与 best epoch；以 liver_dice 最大为准（肝脏为主目标）。
            #   - last.pt：每次验证都保存（覆盖写），保留最新进度便于断点续训。
            liver_dice = val_metrics.get("liver_dice", float("nan"))
            # NaN 视为不参与 best 比较（NaN 任何比较都为 False，避免逻辑混乱）
            if not np.isnan(liver_dice) and liver_dice > best_liver_dice:
                best_liver_dice = liver_dice
                best_epoch = epoch
                best_metrics = val_metrics
                save_model(model, best_ckpt)
                print(f"    [best] liver_dice 创新高 {liver_dice:.4f}，已保存 {best_ckpt}")

            # last.pt：每次验证都覆盖保存，反映当前最新权重
            save_model(model, last_ckpt)

    # 10) 训练完成汇总
    print("=" * 60)
    print("训练完成 Training finished.")
    print(f"  最佳 epoch: {best_epoch} | 最佳 liver_dice: {best_liver_dice:.4f}")
    print(f"  最终权重: {last_ckpt}")
    print(f"  指标历史: {history_csv}")
    print("=" * 60)

    # 11) 返回训练汇总 dict，供 scripts/train.py 进一步处理/打印
    return {
        "best_metrics": best_metrics,
        "best_epoch": best_epoch,
        "last_epoch": max_epochs,
        "history_csv": history_csv,
        "best_ckpt": best_ckpt,
        "last_ckpt": last_ckpt,
    }
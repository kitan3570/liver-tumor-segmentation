"""scripts/smoke_test.py — 肝脏 3D 分割项目自检脚本（smoke test）

文件作用
========
对应 tasks.md Task 7 SubTask 7.9：自检脚本。在内存中用 numpy 合成少量 3D 图像
与标签，快速跑通「构建数据 → 构建模型 → 1 epoch 训练 → 推理 → 指标 → 可视化
→ STL」整条链路，**无需任何真实数据**，仅用于验证代码链路正常、各模块接口
（src.utils / src.data / src.models / src.trainer）吻合。自检默认在 CPU 上运行
（不求精度，只求跑通），适合在提交作业前做一次端到端冒烟测试：任何一处接口
不匹配或依赖缺失都会立刻暴露，而不是等到下载完真实数据后才发现。

运行方式
========
    # 默认：二分类（phase=1）、1 epoch、CPU
    python scripts/smoke_test.py

    # 显式指定参数（与默认等价）
    python scripts/smoke_test.py --phase 1 --epochs 1 --device cpu

    # 三分类自检（合成数据里会多一个小球肿瘤）
    python scripts/smoke_test.py --phase 2 --epochs 1 --device cpu

输入
====
- 无需真实数据。本脚本用 numpy 合成 3 个 (64,64,64) 的小体积样本：背景为 0，
  中心放一个椭球形「肝脏」(灰度约 100，模拟肝脏软组织 CT 值)；phase2 时再在
  肝内放一个小球状「肿瘤」(灰度约 150)。合成 image/label 会落盘到
  outputs/smoke_data/ 作为临时 NIfTI，供 MONAI 的 LoadImaged 读取，从而**真实地**
  验证数据加载链路（而不是跳过 NIfTI 读取）。

输出
====
- outputs/smoke_data/                    ：合成的临时 NIfTI 图像与标签。
- outputs/smoke_outputs/metrics/smoke_metrics.json ：验证集指标汇总（JSON）。
- outputs/smoke_outputs/figures/smoke.png          ：可视化叠加图（liver 红色叠加在 CT 切片上）。
- outputs/smoke_outputs/stl/smoke_liver.stl        ：预测肝脏的 STL 三维网格。

常见错误
========
1) 依赖缺失 —— monai / torch / nibabel / matplotlib / trimesh / scikit-image 任一
   未安装，会在对应步骤报 ImportError；请先运行 scripts/check_env.py 核对环境。
2) CPU 太慢 —— 自检已尽量缩小（体积 64³、裁剪 32³、channels=[8,16,32]、
   num_workers=0、amp=False），普通笔记本 CPU 数秒~数十秒可完成；若仍嫌慢，可
   把 config 中 train.roi_size 与 infer.roi_size 进一步调小。
3) phase 与 out_channels 不符 —— 模型 out_channels 必须等于 num_classes：
   --phase 1 → 2 类，--phase 2 → 3 类。build_smoke_config 已自动对应，请勿手改
   config 时漏掉这一点，否则损失函数做 one-hot 时维度不匹配会报错。
"""

# ===================== 标准库导入 =====================
import argparse
import os
import sys
import traceback

# ===================== 第三方库导入 =====================
import numpy as np
import torch
# matplotlib 用 Agg 后端（无界面），保证在无显示环境（服务器 / CLI）也能存图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 把项目根目录（本文件所在目录的上一级）加入 sys.path，使得直接运行
# `python scripts/smoke_test.py ...` 时能正确 import src.* 模块。
# 说明：Python 运行脚本时只会把脚本所在目录(scripts/)加入 sys.path[0]，
# 不会自动加入项目根；此处手动补上，保证 src.utils / src.trainer 可被导入。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows 默认控制台编码(cp936/GBK)无法输出 ✓ / ✗ 等 Unicode 符号，结尾的
# 「自检通过 ✓」会因编码错误而崩溃。这里把标准输出重配置为 UTF-8（errors=
# 'replace' 兜底），保证结尾的成功/失败标记能安全打印。Python 3.7+ 支持。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # 极少数环境不支持 reconfigure，忽略即可，不影响主流程

# ===================== 本项目模块导入 =====================
from src.utils import (
    set_seed,
    get_device,
    ensure_dir,
    save_nifti,
    save_json,
    export_label_stl,
    get_class_colors,
    pick_slice_with_lesion,
    overlay_mask,
)
from src.data import build_dataloader
from src.models import build_model, build_loss
from src.trainer import compute_metrics, validate, train_one_epoch
from monai.inferers import SlidingWindowInferer


# ============================================================
# 1. 合成数据生成
# ============================================================

def generate_synthetic_data(n: int = 3, phase: int = 1,
                            out_dir: str = "outputs/smoke_data") -> list:
    """生成 n 个合成 3D 样本（image + label）并保存为 NIfTI，返回 data_list。

    合成策略（面向初学者的医学影像模拟）
    ------------------------------------
    - image：形状 (64,64,64) 的 float32，背景≈0（加少量高斯噪声模拟 CT 噪声），
      在中心放一个椭球形「肝脏」区域，灰度约 100（落在 CT 窗 [-100,400] 内，
      模拟肝脏软组织 CT 值）。
    - label：同形状的 int，背景=0，中心椭球标 1（liver）。
    - phase2（三分类）时，再在肝脏内部放一个小球标 2（tumor），灰度约 150
      （肿瘤常比肝实质稍亮），用以验证多类链路。
    - 给每个样本轻微随机偏移肝脏中心，使 3 个样本略有差异，更接近真实数据的
      多样性；用固定种子保证可复现。
    - affine 用 4x4 单位矩阵 np.eye(4)：1mm 各向同性 spacing、原点在 0、方向为
      RAS，与 config['data']['spacing']=[1,1,1] 一致，故 Spacing 重采样后体积不变。

    Args:
        n:       样本数，默认 3（少量即可，自检只验证链路）。
        phase:   阶段；1=二分类（仅肝脏），2=三分类（肝+肿瘤）。
        out_dir: 合成 NIfTI 的输出目录。

    Returns:
        list: data_list，每项为 {'image': <路径>, 'label': <路径>}，
              可直接喂给 src.data.build_dataloader。
    """
    # 固定本函数内的随机数，保证合成数据可复现（与 run_smoke 里的 set_seed 独立但同种子）
    np.random.seed(42)

    ensure_dir(out_dir)  # 确保合成数据目录存在

    # 体素网格坐标：indexing='ij' 表示第一维对应 z，依次 y、x；
    # zz/yy/xx 形状均为 (64,64,64)，分别给出每个体素的 z/y/x 坐标
    shape = (64, 64, 64)
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    cx, cy, cz = 32.0, 32.0, 32.0  # 肝脏中心默认在体素 (32,32,32)

    affine = np.eye(4)  # 4x4 单位仿射矩阵：1mm spacing、RAS 方向
    data_list = []

    for i in range(n):
        # 轻微随机偏移肝脏中心（±3 体素），让 3 个样本略有差异
        ox = np.random.uniform(-3, 3)
        oy = np.random.uniform(-3, 3)
        oz = np.random.uniform(-3, 3)

        # —— 构造图像：背景 0 + 少量高斯噪声 ——
        image = np.zeros(shape, dtype=np.float32)
        image += np.random.normal(0.0, 2.0, size=shape).astype(np.float32)  # 模拟 CT 噪声

        # —— 构造标签：背景 0 ——
        label = np.zeros(shape, dtype=np.int16)

        # 肝脏椭球：((x-cx)/a)^2 + ((y-cy)/b)^2 + ((z-cz)/c)^2 <= 1
        # 半轴约 14/14/12，覆盖体素约 [18,46]，保证 32³ 随机裁剪块能含到肝脏前景
        a_liver, b_liver, c_liver = 14.0, 14.0, 12.0
        liver_mask = (
            ((xx - (cx + ox)) / a_liver) ** 2
            + ((yy - (cy + oy)) / b_liver) ** 2
            + ((zz - (cz + oz)) / c_liver) ** 2
        ) <= 1.0
        image[liver_mask] = 100.0  # 肝脏灰度
        label[liver_mask] = 1      # liver

        # phase2：在肝脏内部放一个小球肿瘤（label=2，灰度 150）
        if phase == 2:
            a_t, b_t, c_t = 4.0, 4.0, 4.0  # 半径约 4 的小球，位于肝内
            tumor_mask = (
                ((xx - (cx + ox)) / a_t) ** 2
                + ((yy - (cy + oy)) / b_t) ** 2
                + ((zz - (cz + oz)) / c_t) ** 2
            ) <= 1.0
            # 肿瘤覆盖在肝脏之上：先标 liver 再标 tumor，tumor 区域最终为 2
            image[tumor_mask] = 150.0
            label[tumor_mask] = 2

        # 落盘为 NIfTI（.nii.gz），save_nifti 内部会 ensure_dir 父目录
        img_path = os.path.join(out_dir, f"smoke_image_{i:02d}.nii.gz")
        lbl_path = os.path.join(out_dir, f"smoke_label_{i:02d}.nii.gz")
        save_nifti(image, affine, img_path)
        save_nifti(label, affine, lbl_path)

        data_list.append({"image": img_path, "label": lbl_path})

    print(f"[合成数据] 已生成 {n} 个样本到 {out_dir}（phase={phase}）")
    return data_list


# ============================================================
# 2. 临时 config 构造
# ============================================================

def build_smoke_config(phase: int = 1, epochs: int = 1,
                       device: str = "cpu") -> dict:
    """构造一份「缩小版」临时 config，schema 与 configs/phase1/phase2 一致但更省时。

    与正式 config 的关键差异（都为了 CPU 自检能快速跑通）
    ----------------------------------------------------
    - train.roi_size = [32,32,32]（正式为 96³）：小裁剪块，省算力。
    - train.batch_size = 1、num_workers = 0、cache_rate = 0.0：Windows 下最稳。
    - train.amp = False：CPU 不用混合精度。
    - model.channels = [8,16,32]、strides = [2,2]、num_res_units = 1：极小 3D U-Net。
      注意 len(strides) 必须等于 len(channels)-1（=2），MONAI UNet 才能构建。
    - infer.roi_size = [32,32,32]、sw_batch_size = 1、overlap = 0.25：滑动窗口小一点。

    Args:
        phase:  1=二分类(num_classes=2), 2=三分类(num_classes=3)。
        epochs: 训练轮数，自检默认 1。
        device: 设备字符串，自检默认 'cpu'。

    Returns:
        dict: 可直接传给 build_dataloader / build_model / build_loss 等的配置字典。
    """
    num_classes = 2 if phase == 1 else 3
    classes = ["background", "liver"] if phase == 1 else ["background", "liver", "tumor"]

    config = {
        "phase": phase,
        "num_classes": num_classes,
        "classes": classes,
        "seed": 42,
        "device": device,
        "data": {
            "splits_dir": "outputs/smoke_outputs/splits",
            "spacing": [1.0, 1.0, 1.0],
            "intensity_clip": [-100, 400],   # CT 窗：肝脏软组织 HU 范围
            "intensity_scale": [0.0, 1.0],   # 归一化目标范围
        },
        "train": {
            "roi_size": [32, 32, 32],
            "batch_size": 1,
            "num_workers": 0,
            "cache_rate": 0.0,
            "max_epochs": epochs,
            "val_interval": 1,
            "amp": False,
        },
        "model": {
            "in_channels": 1,
            "out_channels": num_classes,
            "channels": [8, 16, 32],
            "strides": [2, 2],
            "num_res_units": 1,
            "norm": "BATCH",
        },
        "loss": {
            "name": "DiceCE",
            "include_background": False,
            "softmax": True,
            "to_onehot_y": True,
            "squared_pred": True,
            "lambda_dice": 1.0,
            "lambda_ce": 1.0,
        },
        "optimizer": {
            "name": "Adam",
            "lr": 1e-4,
            "weight_decay": 1e-5,
        },
        "scheduler": {
            "name": "CosineAnnealingLR",
            "T_max": epochs,
        },
        "infer": {
            "roi_size": [32, 32, 32],
            "sw_batch_size": 1,
            "overlap": 0.25,
            "keep_largest_cc": True,
        },
        "output": {
            "models_dir": "outputs/smoke_outputs/models",
            "predictions_dir": "outputs/smoke_outputs/predictions",
            "metrics_dir": "outputs/smoke_outputs/metrics",
            "figures_dir": "outputs/smoke_outputs/figures",
            "stl_dir": "outputs/smoke_outputs/stl",
        },
    }
    return config


# ============================================================
# 3. 全流程自检
# ============================================================

def run_smoke(config: dict, data_list: list, device: torch.device) -> None:
    """跑通整条链路：数据 → 模型 → 训练 → 推理 → 指标 → 可视化 → STL。

    本函数故意把每一步拆开调用 src.* 的公开接口，目的是「逐环节验证接口吻合」；
    任何一步抛异常都会向上抛出，由 main 统一捕获并打印 traceback，便于定位是
    哪个模块（数据 / 模型 / 训练 / 推理 / 可视化 / STL）出问题。

    Args:
        config:    由 build_smoke_config 构造的临时配置。
        data_list: 由 generate_synthetic_data 构造的数据清单。
        device:    torch.device，自检一般为 CPU。

    Raises:
        任何下游异常都会向上抛出，由 main 统一捕获。
    """
    # 1) 固定随机种子，保证自检可复现
    set_seed(config["seed"])

    num_classes = config["num_classes"]

    # 2) 构建 DataLoader：训练与验证都用同一份 data_list（自检数据量小，不复用划分）
    #    训练阶段会做随机前景裁剪 + 翻转增强；验证阶段保留原尺寸，交给滑动窗口推理。
    print("[自检] 构建数据加载器 ...")
    train_loader = build_dataloader(data_list, config, "train")
    val_loader = build_dataloader(data_list, config, "val")

    # 3) 构建模型 / 损失 / 优化器
    #    build_model 返回的模型尚未搬到设备，此处手动 .to(device)。
    #    build_loss 内部已配置 to_onehot_y 与 softmax，可直接吃 (B,C,...) 与类别 id 标签。
    print("[自检] 构建模型 / 损失 / 优化器 ...")
    model = build_model(config).to(device)
    loss_fn = build_loss(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["lr"])

    # 4) 滑动窗口推理器：把整卷切成 roi_size 小块分别前向再拼接（验证时用）
    inferer = SlidingWindowInferer(
        roi_size=tuple(config["infer"]["roi_size"]),
        sw_batch_size=config["infer"]["sw_batch_size"],
        overlap=config["infer"]["overlap"],
    )

    # 5) 训练 1 个 epoch（amp=False、scaler=None：CPU 走普通 float32 路径）
    print("[自检] 训练 1 epoch ...")
    avg_loss = train_one_epoch(
        model, train_loader, loss_fn, optimizer, device,
        amp=config["train"]["amp"], scaler=None,
    )
    print(f"[自检] 训练完成：avg_loss = {avg_loss:.4f}")

    # 6) 验证一次：滑动窗口推理 + 指标聚合，打印 liver/tumor 指标
    print("[自检] 验证 + 指标 ...")
    val_metrics = validate(model, val_loader, device, num_classes, inferer)
    liver_dice = val_metrics.get("liver_dice", float("nan"))
    print(f"[自检] liver_dice = {liver_dice:.4f}")
    if num_classes >= 3:
        tumor_dice = val_metrics.get("tumor_dice", float("nan"))
        print(f"[自检] tumor_dice = {tumor_dice:.4f}")

    # 把验证指标落盘为 JSON，验证 save_json 链路并留一份小产物
    metrics_dir = config["output"]["metrics_dir"]
    ensure_dir(metrics_dir)
    metrics_path = os.path.join(metrics_dir, "smoke_metrics.json")
    save_json(val_metrics, metrics_path)
    print(f"[自检] 指标已保存：{metrics_path}")

    # 7) 可视化：取验证集第一个样本，推理得到 pred，选含病灶切片，叠加肝脏红色
    print("[自检] 可视化叠加图 ...")
    figures_dir = config["output"]["figures_dir"]
    ensure_dir(figures_dir)
    fig_path = os.path.join(figures_dir, "smoke.png")

    # 取一个验证 batch（batch_size=1），在 no_grad 下做滑动窗口推理
    val_iter = iter(val_loader)
    batch = next(val_iter)
    inputs = batch["image"].to(device)              # (1,1,D,H,W)
    with torch.no_grad():
        outputs = inferer(inputs, model)            # (1,C,D,H,W) 每类 logit
        pred = outputs.argmax(dim=1, keepdim=True)  # (1,1,D,H,W) 类别 id

    # pred[0,0] → (D,H,W) 的类别 id；image 同理取 (D,H,W)
    pred_np = pred[0, 0].detach().cpu().numpy()
    image_np = inputs[0, 0].detach().cpu().numpy()

    # 额外用 compute_metrics 算单样本指标并打印，顺便验证该接口
    single_metrics = compute_metrics(pred_np, batch["label"][0, 0], num_classes)
    print(f"[自检] 单样本指标：{single_metrics}")

    # 选「前景最多」的轴向切片；颜色取肝脏红 (1,0,0)
    slice_idx = pick_slice_with_lesion(pred_np, num_classes)
    liver_color = get_class_colors()[1]
    img_slice = image_np[:, :, slice_idx]
    mask_slice = pred_np[:, :, slice_idx]
    overlay = overlay_mask(img_slice, mask_slice, liver_color, alpha=0.5)

    plt.figure(figsize=(5, 5))
    plt.imshow(overlay)
    plt.title(f"smoke test  slice z={slice_idx}  (red=liver)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"[自检] 可视化已保存：{fig_path}")

    # 8) STL 导出：把预测的肝脏(label==1)导出为三角网格 STL，验证 export_label_stl
    print("[自检] STL 导出 ...")
    stl_dir = config["output"]["stl_dir"]
    ensure_dir(stl_dir)
    stl_path = os.path.join(stl_dir, "smoke_liver.stl")
    export_label_stl(pred_np, 1, stl_path)  # label_id=1 → 肝脏

    # 9) 全流程完成
    print("=" * 60)
    print("自检通过 ✓ 全流程链路正常")
    print("=" * 60)


# ============================================================
# 4. 参数解析与入口
# ============================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 含 phase / epochs / device。
    """
    parser = argparse.ArgumentParser(
        description="肝脏 3D 分割项目自检脚本：用合成 3D 数据跑通全流程链路。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # --help 自动显示默认值
    )
    parser.add_argument(
        "--phase", type=int, default=1, choices=[1, 2],
        help="合成数据类别数：1=二分类(背景/肝脏)，2=三分类(背景/肝脏/肿瘤)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="训练几个 epoch（自检默认 1，只验证链路）",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="自检设备，默认 cpu（不求精度，仅验证链路）；有 GPU 可填 cuda",
    )
    return parser.parse_args()


def main() -> None:
    """脚本主入口：解析参数 → 生成合成数据 + config → run_smoke；异常时打印 traceback。"""
    args = parse_args()

    try:
        # 1) 生成合成数据（落盘到 outputs/smoke_data/）
        data_list = generate_synthetic_data(
            n=3, phase=args.phase, out_dir="outputs/smoke_data"
        )

        # 2) 构造临时 config（缩小版，CPU 友好）
        config = build_smoke_config(
            phase=args.phase, epochs=args.epochs, device=args.device
        )

        # 3) 选择设备（CPU 时直接返回 cpu；填 cuda 但不可用会自动回退并告警）
        device = get_device(args.device)

        # 4) 跑通全流程
        run_smoke(config, data_list, device)

    except Exception:
        # 自检失败：打印完整 traceback，便于定位是数据/模型/训练/推理/可视化/STL 哪一环出错
        print("=" * 60)
        print("自检失败 ✗ 全流程链路异常，traceback 如下：")
        print("-" * 60)
        traceback.print_exc()
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

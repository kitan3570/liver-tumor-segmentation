"""scripts/train.py — 训练入口脚本

文件作用
========
加载 YAML 配置文件（configs/*.yaml），把命令行覆盖项（--max_epochs / --device
/ --amp）合并进配置，然后调用 src.trainer.run_training 启动完整的 3D U-Net
训练流程。训练完成后，把返回的汇总信息（最佳 epoch、最佳指标、checkpoint
路径等）保存为 training_summary.json，并在终端打印关键结果。

本脚本本身很「薄」：随机种子、设备选择、数据加载、模型/损失构建、训练循环、
best/last checkpoint 保存、CSV 指标历史写入都已在 src/trainer.run_training
内部完成；脚本只负责「配置加载 → 命令行覆盖 → 打印摘要 → 调用训练 →
汇总落盘/打印」。

运行方式
========
    # 基本用法：用某份 YAML 配置启动训练
    python scripts/train.py --config configs/phase1_binary.yaml

    # 覆盖训练轮数与设备（如无 GPU 时改用 CPU 自检）
    python scripts/train.py --config configs/phase1_binary.yaml --max_epochs 50 --device cpu

    # 显式启用 / 禁用混合精度（覆盖 YAML 中的 train.amp）
    python scripts/train.py --config configs/phase1_binary.yaml --amp
    python scripts/train.py --config configs/phase1_binary.yaml --no-amp

输入
====
- --config：YAML 配置文件路径，如 configs/phase1_binary.yaml（schema 见该文件）。
- 训练前需先由 scripts/split_data.py 生成 outputs/splits/{train,val}.json 划分清单，
  run_training 内部会读取这两份清单来构建 DataLoader。

输出
====
- outputs/models/best.pt、outputs/models/last.pt：最佳与最新模型权重（由 trainer 写）。
- outputs/metrics/train_history.csv：逐 epoch 的训练/验证指标历史（由 trainer 写）。
- outputs/metrics/training_summary.json：训练汇总（best_epoch/best_metrics/ckpt 路径，
  由本脚本写）。
- 终端打印配置摘要与训练完成信息。

常见错误
========
1) 配置文件不存在 —— 检查 --config 路径，确认 configs/*.yaml 已生成。
2) outputs/splits/{train,val}.json 未生成 —— 先运行 scripts/split_data.py 划分数据。
3) CUDA 不可用 —— 配置 device=cuda 但本机无可用 GPU；改用 --device cpu，
   或安装 CUDA 版 PyTorch。
4) Windows 下 num_workers 报错 —— 多进程数据加载在 Windows 下偶发崩溃；
   可在 YAML 中将 train.num_workers 调小为 0（牺牲速度换稳定）。
"""

# ===================== 标准库导入 =====================
import argparse
import os
import sys

# 把项目根目录（本文件所在目录的上一级）加入 sys.path，使得直接运行
# `python scripts/train.py ...` 时能正确 import src.* 模块。
# 说明：Python 运行脚本时只会把脚本所在目录(scripts/)加入 sys.path[0]，
# 不会自动加入项目根；此处手动补上，保证 src.utils / src.trainer 可被导入。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ===================== 本项目模块导入 =====================
# load_config : 读 YAML 配置（文件不存在时抛 FileNotFoundError 并附中文提示）
# save_json   : 把汇总 dict 写成 UTF-8 JSON（内部已 ensure_dir）
# ensure_dir  : 确保目录存在（此处用于显式确认 metrics_dir 已建）
# run_training: 训练主流程（内部完成种子/设备/数据/模型/训练循环/保存）
from src.utils import load_config, save_json, ensure_dir
from src.trainer import run_training


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    覆盖语义：所有覆盖参数默认 None，表示「不覆盖、沿用 YAML 中的值」；
    一旦在命令行给出，则覆盖 YAML 对应字段。

    Returns:
        argparse.Namespace: 含 config, max_epochs, device, amp。
    """
    parser = argparse.ArgumentParser(
        description="加载 YAML 配置并启动肝脏 3D 分割训练（调用 src.trainer.run_training）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # --help 自动显示默认值
    )
    parser.add_argument(
        "--config", required=True,
        help="YAML 配置文件路径，如 configs/phase1_binary.yaml",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=None,
        help="覆盖 config['train']['max_epochs']；不填则沿用 YAML",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="覆盖 config['device']，如 'cuda' 或 'cpu'；不填则沿用 YAML",
    )
    # --amp / --no-amp 共用同一 dest='amp'，实现「三态」：True / False / None(不覆盖)。
    # 必须显式设 default=None：store_true 默认为 False、store_false 默认为 True，
    # 若不显式设 None，则未传参时会被这些隐式默认值覆盖掉 YAML 里的 amp 设置。
    parser.add_argument(
        "--amp", dest="amp", action="store_true", default=None,
        help="启用 AMP 混合精度训练（覆盖 config['train']['amp']=True）",
    )
    parser.add_argument(
        "--no-amp", dest="amp", action="store_false", default=None,
        help="禁用 AMP 混合精度训练（覆盖 config['train']['amp']=False）",
    )
    return parser.parse_args()


def print_config_summary(config: dict) -> None:
    """打印配置摘要，让同学在训练开始前确认关键设置是否正确。

    Args:
        config: 已合并命令行覆盖项的完整配置字典。
    """
    print("=" * 60)
    print("配置摘要 Config summary")
    print("-" * 60)
    print(f"  phase        : {config['phase']}")                  # 阶段：1=二分类, 2=三分类
    print(f"  num_classes  : {config['num_classes']}")            # 类别数
    print(f"  classes      : {config['classes']}")                # 类别名列表
    print(f"  device       : {config['device']}")                 # 运行设备 cuda/cpu
    print(f"  max_epochs   : {config['train']['max_epochs']}")    # 总训练轮数
    print(f"  batch_size   : {config['train']['batch_size']}")    # 每批样本数
    print(f"  roi_size     : {config['train']['roi_size']}")      # 训练随机裁剪块大小
    print(f"  lr           : {config['optimizer']['lr']}")        # 初始学习率
    print(f"  loss         : {config['loss']['name']}")           # 损失函数名
    print("=" * 60)


def main() -> None:
    """脚本主入口：解析参数 → 加载配置 → 命令行覆盖 → 打印摘要 → 训练 → 落盘汇总。"""
    args = parse_args()

    # 1) 加载 YAML 基础配置
    #    若 --config 指向的文件不存在，load_config 会抛 FileNotFoundError 并附中文提示。
    config = load_config(args.config)

    # 2) 命令行覆盖：手动做嵌套覆盖，而非调用 merge_config。
    #    原因：src.utils.merge_config 只做「顶层浅合并」——若把
    #    {'train': {'max_epochs': 50}} 作为 override 传进去，它会把整个
    #    config['train'] 替换成只含 max_epochs 的小字典，从而丢掉 batch_size、
    #    roi_size、amp、val_interval 等其他训练字段，导致训练报错。
    #    因此这里对需要深入子字典的字段（max_epochs、amp）手动逐字段赋值，
    #    顶层字段（device）也直接赋值，安全且直观。
    if args.max_epochs is not None:
        config["train"]["max_epochs"] = args.max_epochs
    if args.device is not None:
        config["device"] = args.device
    if args.amp is not None:
        config["train"]["amp"] = args.amp

    # 3) 打印配置摘要，便于训练前核对覆盖是否生效
    print_config_summary(config)

    # 4) 调用训练主流程
    #    run_training 内部已完成：set_seed / get_device / build_dataloader /
    #    build_model / build_loss / 训练循环 / best+last checkpoint 保存 /
    #    CSV 指标历史写入；返回训练汇总 dict。
    summary = run_training(config)

    # 5) 把训练汇总保存为 JSON，便于后续写报告或对比不同实验
    metrics_dir = config["output"]["metrics_dir"]
    ensure_dir(metrics_dir)  # 显式确认指标目录存在（save_json 内部也会建）
    summary_path = os.path.join(metrics_dir, "training_summary.json")
    save_json(summary, summary_path)
    print(f"[输出] 训练汇总已保存：{summary_path}")

    # 6) 打印训练完成的关键信息：最佳 epoch、最佳 liver_dice、权重路径
    #    best_metrics 可能含 NaN（某类真值为空时）；用 .get 兜底，避免缺键报错。
    best_metrics = summary.get("best_metrics") or {}
    best_liver_dice = best_metrics.get("liver_dice", float("nan"))
    print("=" * 60)
    print("训练完成 Training finished.")
    print(f"  best_epoch      : {summary.get('best_epoch')}")
    print(f"  best liver_dice : {best_liver_dice:.4f}")
    print(f"  best_ckpt       : {summary.get('best_ckpt')}")
    print(f"  last_ckpt       : {summary.get('last_ckpt')}")
    print(f"  history_csv     : {summary.get('history_csv')}")
    print("=" * 60)


if __name__ == "__main__":
    main()

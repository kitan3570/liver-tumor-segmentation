#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scripts/benchmark_optimizations.py —— 速度优化对比实验

文件作用
========
    自动运行 baseline 与 optimized 两种训练设置的短程对比实验，比较速度、
    显存与初步精度表现。所有数据来自真实运行（src/train.py --benchmark_mode
    产出的 JSON），禁止编造。

    baseline：关闭 AMP、OneCycleLR、梯度裁剪、pin_memory、persistent_workers、
              non_blocking、验证 AMP（恒定 lr 的最朴素设置）。
    optimized：开启上述全部优化（与项目推荐配置一致）。

    两者使用相同的 config / train_json / val_json / max_epochs / batch_size /
    roi_size / seed / device，仅优化开关不同，保证对比公平。

运行方式
========
    在项目根目录 (d:\\项目1) 下执行：

    python scripts/benchmark_optimizations.py --config configs/liver_unet_3d.yaml \
        --train_json data/splits/train.json --val_json data/splits/val.json \
        --max_epochs 10 --batch_size 1 --device cuda

输入
====
    --config      YAML 配置（提供 roi_size / seed / num_workers 等）
    --train_json  训练集划分 JSON
    --val_json    验证集划分 JSON
    --max_epochs  短程训练轮数（默认 10，仅用于速度对比）
    --batch_size  批大小（默认 1）
    --device      cuda / cpu（默认 auto）
    --cache       缓存策略 none/disk/memory（默认 disk，两种模式共用同一策略）
    --output_dir  benchmark 输出根目录（默认 outputs/benchmark）

输出
====
    <output_dir>/baseline/               baseline 训练产物（权重 / 缓存 / benchmark.json）
    <output_dir>/optimized/              optimized 训练产物
    <output_dir>/benchmark_results.csv   逐模式全部指标（含失败原因列）
    <output_dir>/benchmark_results.md    markdown 对比表
    <output_dir>/benchmark_summary.txt   速度/显存/Dice 对比分析与说明

注意
====
    1) 短程 10 epoch benchmark 主要用于速度比较，不代表最终模型性能。
    2) 无 GPU 时 peak_gpu_memory_MB 记为 NaN，且 CPU benchmark 不代表真实训练速度。
    3) 某个模式失败时记录失败原因，不会导致整个脚本崩溃。
"""

import os
import sys
import json
import time
import argparse
import subprocess
import statistics
from collections import deque

# 数学 NaN 统一用 float("nan")；CSV 写入时转 "NaN" 字符串

# Windows 控制台默认 GBK(cp936) 编码，无法写入子进程输出中的 \ufffd 等字符
# (子进程 stdout 以 UTF-8+errors=replace 解码，路径中的中文被替换为 \ufffd)。
# 这里把本脚本自身的 stdout/stderr 重配置为 UTF-8 + errors=replace，
# 保证转发子进程输出时不会 UnicodeEncodeError。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001  某些环境(如重定向到 StringIO)不支持 reconfigure
        pass


# ====================================================================
# 一、命令行参数
# ====================================================================
def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        prog="benchmark_optimizations.py",
        description="对比 baseline 与 optimized 训练设置的速度/显存/精度（真实运行）",
    )
    parser.add_argument("--config", default="configs/liver_unet_3d.yaml",
                        help="YAML 配置文件，默认 configs/liver_unet_3d.yaml")
    parser.add_argument("--train_json", default="data/splits/train.json",
                        help="训练集划分 JSON")
    parser.add_argument("--val_json", default="data/splits/val.json",
                        help="验证集划分 JSON")
    parser.add_argument("--max_epochs", type=int, default=10,
                        help="短程训练轮数，默认 10（仅用于速度对比）")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批大小，默认 1")
    parser.add_argument("--device", default=None,
                        help='cuda / cpu；不填则自动检测')
    parser.add_argument("--cache", default="disk", choices=["none", "disk", "memory"],
                        help="缓存策略，两种模式共用，默认 disk")
    parser.add_argument("--output_dir", default="outputs/benchmark",
                        help="benchmark 输出根目录，默认 outputs/benchmark")
    return parser.parse_args()


# ====================================================================
# 二、构造两种模式的训练命令
# ====================================================================
def _common_args(args):
    """两种模式共享的参数（保证对比公平）。"""
    a = [
        "--config", args.config,
        "--train_json", args.train_json,
        "--val_json", args.val_json,
        "--max_epochs", str(args.max_epochs),
        "--batch_size", str(args.batch_size),
        "--cache_strategy", args.cache,
        "--benchmark_mode",
    ]
    if args.device is not None:
        a += ["--device", args.device]
    return a


def build_baseline_cmd(args, out_dir, bench_json):
    """baseline 命令：关闭全部优化开关（恒定 lr 的最朴素设置）。"""
    py = sys.executable
    script = os.path.join("src", "train.py")
    cmd = [py, script]
    cmd += _common_args(args)
    cmd += ["--output_dir", out_dir]
    cmd += ["--benchmark_output_json", bench_json]
    # —— 显式关闭所有优化 ——
    cmd += ["--no_amp", "--scheduler", "none", "--no_grad_clip",
            "--no_pin_memory", "--no_persistent_workers",
            "--no_non_blocking", "--no_val_amp"]
    return cmd


def build_optimized_cmd(args, out_dir, bench_json):
    """optimized 命令：开启全部优化（项目推荐配置）。"""
    py = sys.executable
    script = os.path.join("src", "train.py")
    cmd = [py, script]
    cmd += _common_args(args)
    cmd += ["--output_dir", out_dir]
    cmd += ["--benchmark_output_json", bench_json]
    # —— 显式开启所有优化 ——
    cmd += ["--amp", "--scheduler", "onecycle", "--grad_clip",
            "--pin_memory", "--persistent_workers",
            "--non_blocking", "--val_amp"]
    return cmd


# ====================================================================
# 三、运行子进程并捕获输出
# ====================================================================
def run_subprocess(cmd, project_root, label):
    """运行训练子进程，实时打印输出并保留末尾若干行用于失败诊断。

    返回 (returncode, tail_text)。returncode==0 视为成功。
    所有训练输出（含 tqdm 进度）直接转发到控制台，便于观察真实进度。
    """
    print("\n" + "=" * 70)
    print("[{}] 开始运行".format(label))
    print("[{}] 命令: {}".format(label, " ".join(cmd)))
    print("=" * 70)
    sys.stdout.flush()
    t0 = time.time()
    tail = deque(maxlen=40)  # 保留末尾 40 行，失败时摘录原因
    try:
        proc = subprocess.Popen(
            cmd, cwd=project_root,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            bufsize=1,
        )
        for line in proc.stdout:
            # 实时转发，让用户看到 tqdm 进度
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            except UnicodeEncodeError:  # noqa: BLE001  某行含当前控制台无法编码的字符
                # 退而求其次：丢弃无法编码的字符再写，保证 benchmark 不被一行日志打断
                try:
                    sys.stdout.write(line.encode("ascii", "replace").decode("ascii"))
                    sys.stdout.flush()
                except Exception:  # noqa: BLE001
                    pass
            tail.append(line.rstrip())
        proc.wait()
        rc = proc.returncode
    except FileNotFoundError as e:
        print("[{}] 启动失败：{}".format(label, e))
        return 127, "启动失败：{}".format(e)
    elapsed = time.time() - t0
    print("[{}] 子进程结束，returncode={}，墙钟耗时 {:.1f}s".format(label, rc, elapsed))
    return rc, "\n".join(tail)


# ====================================================================
# 四、读取 benchmark JSON 并计算派生指标
# ====================================================================
def _safe_float(x, default=float("nan")):
    """安全转 float；None/非法值返回 default。"""
    try:
        if x is None:
            return default
        f = float(x)
        return f
    except (TypeError, ValueError):
        return default


def load_benchmark_result(bench_json_path):
    """读取 train.py 产出的 benchmark JSON，计算派生指标并返回结果 dict。

    派生指标（来自真实 epochs 列表，非理论值）：
        mean_epoch_time_sec   / median_epoch_time_sec
        fastest_epoch_time_sec / slowest_epoch_time_sec
    若 JSON 不存在或 status!=ok，返回带 failed 标记的结果。

    返回字段（与任务要求一一对应）：
        setting, status, error, total_train_time_sec, mean_epoch_time_sec,
        median_epoch_time_sec, fastest_epoch_time_sec, slowest_epoch_time_sec,
        peak_gpu_memory_MB, best_liver_dice, best_tumor_dice,
        final_liver_dice, final_tumor_dice, amp_enabled, onecycle_enabled,
        grad_clip_enabled, cache_strategy, pin_memory, persistent_workers,
        non_blocking, val_amp, num_epochs_completed
    """
    nan = float("nan")
    # 失败占位结果：所有数值为 NaN，status=failed
    fail_result = {
        "setting": "", "status": "failed", "error": "benchmark JSON 缺失",
        "total_train_time_sec": nan, "mean_epoch_time_sec": nan,
        "median_epoch_time_sec": nan, "fastest_epoch_time_sec": nan,
        "slowest_epoch_time_sec": nan, "peak_gpu_memory_MB": nan,
        "best_liver_dice": nan, "best_tumor_dice": nan,
        "final_liver_dice": nan, "final_tumor_dice": nan,
        "amp_enabled": None, "onecycle_enabled": None,
        "grad_clip_enabled": None, "cache_strategy": "",
        "pin_memory": None, "persistent_workers": None,
        "non_blocking": None, "val_amp": None, "num_epochs_completed": 0,
    }

    if not os.path.isfile(bench_json_path):
        return fail_result

    try:
        with open(bench_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        fail_result["error"] = "JSON 解析失败: {}".format(e)
        return fail_result

    summary = payload.get("summary", {})
    epochs = payload.get("epochs", [])

    # 从 epochs 列表计算耗时统计（真实运行数据）
    epoch_times = [_safe_float(e.get("epoch_time_sec")) for e in epochs]
    epoch_times = [t for t in epoch_times if t == t]  # 去掉 NaN
    if epoch_times:
        mean_t = float(statistics.mean(epoch_times))
        median_t = float(statistics.median(epoch_times))
        fastest_t = float(min(epoch_times))
        slowest_t = float(max(epoch_times))
    else:
        mean_t = median_t = fastest_t = slowest_t = nan

    result = {
        "setting": "",
        "status": summary.get("status", "ok" if epochs else "failed"),
        "error": summary.get("error", ""),
        "total_train_time_sec": _safe_float(summary.get("total_train_time_sec")),
        "mean_epoch_time_sec": mean_t,
        "median_epoch_time_sec": median_t,
        "fastest_epoch_time_sec": fastest_t,
        "slowest_epoch_time_sec": slowest_t,
        "peak_gpu_memory_MB": _safe_float(summary.get("peak_gpu_memory_MB")),
        "best_liver_dice": _safe_float(summary.get("best_liver_dice")),
        "best_tumor_dice": _safe_float(summary.get("best_tumor_dice")),
        "final_liver_dice": _safe_float(summary.get("final_liver_dice")),
        "final_tumor_dice": _safe_float(summary.get("final_tumor_dice")),
        "amp_enabled": summary.get("amp_enabled"),
        "onecycle_enabled": summary.get("onecycle_enabled"),
        "grad_clip_enabled": summary.get("grad_clip_enabled"),
        "cache_strategy": summary.get("cache_strategy", ""),
        "pin_memory": summary.get("pin_memory"),
        "persistent_workers": summary.get("persistent_workers"),
        "non_blocking": summary.get("non_blocking"),
        "val_amp": summary.get("val_amp"),
        "num_epochs_completed": int(summary.get("num_epochs_completed", len(epochs))),
    }
    return result


# ====================================================================
# 五、格式化输出（CSV / markdown / summary）
# ====================================================================
def _fmt(x, ndigits=2, nan_str="NaN"):
    """格式化数值：NaN → nan_str，否则保留 ndigits 位字符串。"""
    if x is None:
        return nan_str
    try:
        f = float(x)
    except (TypeError, ValueError):
        return nan_str
    if f != f:  # NaN
        return nan_str
    return "{:.{p}f}".format(f, p=ndigits)


def _bool_str(x):
    """布尔值转 yes/no；None 转 -。"""
    if x is None:
        return "-"
    return "yes" if x else "no"


CSV_COLUMNS = [
    "setting", "status", "num_epochs_completed",
    "total_train_time_sec", "mean_epoch_time_sec", "median_epoch_time_sec",
    "fastest_epoch_time_sec", "slowest_epoch_time_sec", "peak_gpu_memory_MB",
    "best_liver_dice", "best_tumor_dice", "final_liver_dice", "final_tumor_dice",
    "amp_enabled", "onecycle_enabled", "grad_clip_enabled", "cache_strategy",
    "pin_memory", "persistent_workers", "non_blocking", "val_amp", "error",
]


def write_csv(results, csv_path):
    """写 CSV：每个模式一行，列出全部指标 + 失败原因列。"""
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r in results:
            w.writerow([r.get(c, "") for c in CSV_COLUMNS])


def write_markdown(results, md_path):
    """写 markdown 对比表（任务要求的表格格式）。"""
    os.makedirs(os.path.dirname(os.path.abspath(md_path)), exist_ok=True)
    lines = []
    lines.append("# 速度优化对比 (benchmark results)\n")
    lines.append("> 所有数据来自真实运行（src/train.py --benchmark_mode），短程 "
                 "max_epochs 实验，主要用于速度比较，不代表最终模型性能。\n")
    lines.append("")
    # 任务指定表格
    lines.append("| Setting | AMP | Cache | OneCycleLR | Grad Clip | "
                 "Mean Epoch Time (s) | Peak GPU Memory (MB) | "
                 "Best Liver Dice | Best Tumor Dice |")
    lines.append("|---|---|---|---|---|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            "| {setting} | {amp} | {cache} | {oclr} | {gclip} | "
            "{mean} | {peak} | {bl} | {bt} |".format(
                setting=r["setting"],
                amp=_bool_str(r.get("amp_enabled")),
                cache=r.get("cache_strategy", "-"),
                oclr=_bool_str(r.get("onecycle_enabled")),
                gclip=_bool_str(r.get("grad_clip_enabled")),
                mean=_fmt(r.get("mean_epoch_time_sec")),
                peak=_fmt(r.get("peak_gpu_memory_MB")),
                bl=_fmt(r.get("best_liver_dice"), 4),
                bt=_fmt(r.get("best_tumor_dice"), 4),
            )
        )
    lines.append("")
    # 失败模式提示
    for r in results:
        if r.get("status") != "ok":
            lines.append("- **{s} 运行失败**：{e}".format(
                s=r["setting"], e=r.get("error", "未知原因")))
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _pct_change(new, old):
    """计算 (new-old)/old*100，处理 NaN/0。返回 (pct_str, sign)。"""
    try:
        new = float(new); old = float(old)
    except (TypeError, ValueError):
        return "N/A", 0
    if new != new or old != old:  # NaN
        return "N/A", 0
    if old == 0:
        return "N/A", 0
    pct = (new - old) / abs(old) * 100.0
    return "{:+.1f}%".format(pct), pct


def write_summary(results, summary_path, args, has_gpu):
    """写 benchmark_summary.txt：速度提升 / 显存变化 / Dice 差异 / 说明。"""
    os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
    base = results[0] if len(results) > 0 else {}
    opt = results[1] if len(results) > 1 else {}
    lines = []
    lines.append("=" * 70)
    lines.append("速度优化对比报告 (benchmark summary)")
    lines.append("=" * 70)
    lines.append("")
    lines.append("实验设置：")
    lines.append("  config      : {}".format(args.config))
    lines.append("  train_json  : {}".format(args.train_json))
    lines.append("  val_json    : {}".format(args.val_json))
    lines.append("  max_epochs  : {}（短程，仅用于速度对比）".format(args.max_epochs))
    lines.append("  batch_size  : {}".format(args.batch_size))
    lines.append("  device      : {}".format(args.device or "auto"))
    lines.append("  cache       : {}（两种模式共用）".format(args.cache))
    lines.append("  output_dir  : {}".format(args.output_dir))
    lines.append("")

    # —— 1. 平均 epoch 时间提升百分比 ——
    base_mean = _safe_float(base.get("mean_epoch_time_sec"))
    opt_mean = _safe_float(opt.get("mean_epoch_time_sec"))
    lines.append("1) 平均 epoch 时间对比：")
    lines.append("   baseline   : {} s".format(_fmt(base_mean)))
    lines.append("   optimized  : {} s".format(_fmt(opt_mean)))
    if base_mean == base_mean and opt_mean == opt_mean and base_mean > 0:
        pct, _ = _pct_change(opt_mean, base_mean)
        # 负值代表 optimized 更快（时间减少）
        speedup = base_mean / opt_mean if opt_mean > 0 else float("nan")
        if opt_mean < base_mean:
            lines.append("   => optimized 比 baseline 平均每 epoch 快 {pct}（约 {sp:.2f}x 加速）".format(
                pct=pct.replace("+", ""), sp=speedup))
        else:
            lines.append("   => optimized 比 baseline 平均每 epoch 慢 {pct}".format(pct=pct))
    else:
        lines.append("   => 数据缺失（某模式失败或未完成），无法计算提升百分比")
    lines.append("")

    # —— 2. 显存变化 ——
    base_mem = _safe_float(base.get("peak_gpu_memory_MB"))
    opt_mem = _safe_float(opt.get("peak_gpu_memory_MB"))
    lines.append("2) 峰值 GPU 显存对比：")
    if has_gpu:
        lines.append("   baseline   : {} MB".format(_fmt(base_mem)))
        lines.append("   optimized  : {} MB".format(_fmt(opt_mem)))
        if base_mem == base_mem and opt_mem == opt_mem:
            delta = opt_mem - base_mem
            lines.append("   => optimized 显存变化 {:+.0f} MB（{:+.1f}%）".format(
                delta, (delta / base_mem * 100) if base_mem > 0 else 0))
            if opt_mem < base_mem:
                lines.append("      AMP 混合精度使显存下降，符合预期")
            else:
                lines.append("      注意：optimized 显存未下降，可能因 batch/roi 较小未触发 AMP 节省")
    else:
        lines.append("   当前无 GPU，peak_gpu_memory_MB = NaN")
        lines.append("   注意：CPU benchmark 不代表真实训练速度，仅反映 CPU 端耗时差异")
    lines.append("")

    # —— 3. Dice 差异 ——
    base_bl = _safe_float(base.get("best_liver_dice"))
    opt_bl = _safe_float(opt.get("best_liver_dice"))
    base_bt = _safe_float(base.get("best_tumor_dice"))
    opt_bt = _safe_float(opt.get("best_tumor_dice"))
    lines.append("3) Dice 差异（best，短程实验）：")
    lines.append("   liver dice: baseline={} optimized={} 差值={}".format(
        _fmt(base_bl, 4), _fmt(opt_bl, 4),
        _fmt(opt_bl - base_bl, 4) if (base_bl == base_bl and opt_bl == opt_bl) else "N/A"))
    lines.append("   tumor dice: baseline={} optimized={} 差值={}".format(
        _fmt(base_bt, 4), _fmt(opt_bt, 4),
        _fmt(opt_bt - base_bt, 4) if (base_bt == base_bt and opt_bt == opt_bt) else "N/A"))
    lines.append("")

    # —— 4. 说明 ——
    lines.append("4) 说明：")
    lines.append("   - 短程 {} epoch benchmark 主要用于速度比较，不代表最终模型性能。".format(args.max_epochs))
    lines.append("     最终模型精度需训练完整轮数（如 300 epoch）后才能评估。")
    lines.append("   - 两种模式使用相同的 config / train_json / val_json / batch_size / "
                 "roi_size / seed / device，仅优化开关不同，对比公平。")
    lines.append("   - 所有数值均来自真实运行日志（src/train.py --benchmark_mode 产出的 JSON），"
                 "无任何理论值或估算。")
    if not has_gpu:
        lines.append("   - 本次为 CPU 运行，CPU benchmark 不代表真实（GPU）训练速度。")
    # 失败提示
    for r in results:
        if r.get("status") != "ok":
            lines.append("   - {s} 运行失败：{e}".format(s=r["setting"], e=r.get("error", "")))
    lines.append("")
    lines.append("=" * 70)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ====================================================================
# 六、主入口
# ====================================================================
def main():
    args = parse_args()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    out_root = args.output_dir
    base_dir = os.path.join(out_root, "baseline")
    opt_dir = os.path.join(out_root, "optimized")
    base_json = os.path.join(base_dir, "benchmark.json")
    opt_json = os.path.join(opt_dir, "benchmark.json")

    print("[benchmark] 输出根目录: {}".format(out_root))
    print("[benchmark] max_epochs={}, batch_size={}, cache={}".format(
        args.max_epochs, args.batch_size, args.cache))

    # —— 运行 baseline ——
    base_cmd = build_baseline_cmd(args, base_dir, base_json)
    base_rc, base_tail = run_subprocess(base_cmd, project_root, "baseline")
    # —— 运行 optimized ——
    opt_cmd = build_optimized_cmd(args, opt_dir, opt_json)
    opt_rc, opt_tail = run_subprocess(opt_cmd, project_root, "optimized")

    # —— 读取结果 JSON ——
    base_result = load_benchmark_result(base_json)
    opt_result = load_benchmark_result(opt_json)
    base_result["setting"] = "baseline"
    opt_result["setting"] = "optimized"
    # 若 JSON 显示成功但子进程 returncode!=0，标记失败并附 returncode
    for r, rc, tail in [(base_result, base_rc, base_tail),
                        (opt_result, opt_rc, opt_tail)]:
        if rc != 0 and r["status"] == "ok":
            r["status"] = "failed"
            r["error"] = "子进程 returncode={}（详见上方日志）".format(rc)
        elif r["status"] != "ok" and not r.get("error"):
            # 摘录末尾几行作为失败原因
            tail_lines = [l for l in tail.split("\n") if l.strip()][-5:]
            r["error"] = "子进程 returncode={} | {}".format(rc, " / ".join(tail_lines))

    results = [base_result, opt_result]

    # —— 是否有 GPU ——
    has_gpu = _detect_gpu(args.device)

    # —— 写三类输出 ——
    csv_path = os.path.join(out_root, "benchmark_results.csv")
    md_path = os.path.join(out_root, "benchmark_results.md")
    summary_path = os.path.join(out_root, "benchmark_summary.txt")
    write_csv(results, csv_path)
    write_markdown(results, md_path)
    write_summary(results, summary_path, args, has_gpu)

    # —— 控制台汇总 ——
    print("\n" + "=" * 70)
    print("[benchmark] 对比完成，结果文件：")
    print("  CSV   : {}".format(csv_path))
    print("  MD    : {}".format(md_path))
    print("  TXT   : {}".format(summary_path))
    print("=" * 70)
    # 打印 markdown 表格到控制台
    with open(md_path, "r", encoding="utf-8") as f:
        print(f.read())

    # 任一模式失败 → 退出码 1（便于脚本化感知）
    if any(r.get("status") != "ok" for r in results):
        sys.exit(1)


def _detect_gpu(device_arg):
    """判断本次 benchmark 是否在 GPU 上运行。"""
    if device_arg is not None:
        return device_arg.startswith("cuda")
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()

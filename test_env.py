"""
test_env.py —— 肝脏 3D 分割项目环境自检脚本

作用：
    检查当前 Python 环境是否满足项目核心依赖：
      - torch、monai、nibabel 能否正常导入；
      - CUDA 是否可用（仅提示，不作为关键失败项）；
    汇总打印 PASS/FAIL 与最终结论。

运行方式：
    默认运行（仅检测、不影响退出码）：
        python test_env.py
    严格模式（关键依赖缺失时以非零码退出）：
        python test_env.py --strict

输入：
    无需输入文件，仅需命令行可选参数 --strict（store_true）。

输出：
    打印 Python 版本、各包状态(已装/缺失)+版本、CUDA 可用性 + CUDA 版本 + GPU 名称，
    以及最终结论（环境就绪 / 缺失关键依赖）。

常见错误：
    1) torch 导入失败：未安装或 CUDA/驱动问题，参考 README 安装匹配版本。
    2) monai 导入失败：依赖未装齐，执行 pip install -r requirements.txt。
    3) nabibe / SimpleITK 导入失败：医学影像读写库缺失，重新安装对应包。
    4) CUDA 不可用但包齐全：仅属硬件/驱动问题，CPU 仍可训练，不视为关键失败。
    5) --strict 时退出码非 0：仅当 torch/monai/nibabel 任一导入失败才 exit(1)。
"""

import argparse
import importlib
import sys


def probe(name):
    """探测三方包：返回 (是否已装, 版本字符串或空)。缺失时不抛异常。

    使用 importlib.util.find_spec 先探测模块是否存在，再用动态导入取版本，
    避免缺失包时直接崩溃。
    """
    import importlib.util

    spec = importlib.util.find_spec(name)
    if spec is None:
        return False, ""
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "")
        if not ver:
            try:
                import importlib.metadata as md

                ver = md.version(name)
            except Exception:
                ver = ""
        return True, ver
    except Exception as e:
        return False, f"[导入异常: {e}]"


def check_cuda():
    """检查 CUDA：返回 (可用, CUDA版本, GPU名称列表)。torch 不可用时返回 (False, '', [])。"""
    spec = importlib.util.find_spec("torch")
    if spec is None:
        return False, "", []
    try:
        import torch
    except Exception:
        return False, "", []

    try:
        available = bool(torch.cuda.is_available())
    except Exception:
        available = False
    cuda_ver = getattr(torch.version, "cuda", "") or "" if hasattr(torch, "version") else ""
    gpu_names = []
    if available:
        try:
            n = torch.cuda.device_count()
            for i in range(n):
                gpu_names.append(torch.cuda.get_device_name(i))
        except Exception:
            gpu_names = []
    return available, (cuda_ver or ""), gpu_names


def main():
    parser = argparse.ArgumentParser(description="肝脏 3D 分割项目环境自检")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式：torch/monai/nibabel 任一导入失败时以非零码退出",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("肝脏 3D 分割项目 - 环境自检")
    print("=" * 60)
    print(f"Python 版本: {sys.version.split()[0]} ({sys.executable})")
    print("-" * 60)

    key_deps = ["torch", "monai", "nibabel"]
    results = {}
    for name in key_deps:
        ok, ver = probe(name)
        results[name] = ok
        status = "PASS [已装]" if ok else "FAIL [缺失]"
        ver_str = f" 版本={ver}" if ok and ver else (f" {ver}" if not ok else "")
        print(f"{name:<10s} {status}{ver_str}")

    print("-" * 60)
    available, cuda_ver, gpu_names = check_cuda()
    if not available:
        print("CUDA 可用性: 否 (将使用 CPU，仅提示，不计入关键失败)")
    else:
        print("CUDA 可用性: 是")
        print(f"CUDA 版本: {cuda_ver if cuda_ver else '未知'}")
        if gpu_names:
            print("GPU 设备: " + "; ".join(gpu_names))
        else:
            print("GPU 设备: (未能获取名称)")

    print("=" * 60)
    missing = [n for n in key_deps if not results[n]]
    if not missing:
        print("结论: 环境就绪 (关键依赖均已安装)")
        if not available:
            print("提示: CUDA 不可用，可继续 CPU 训练；如需 GPU 请按 README 配置 torch。")
        print("=" * 60)
        sys.exit(0)
    else:
        print(f"结论: 缺失关键依赖 -> {', '.join(missing)}")
        print("建议: 执行 pip install -r requirements.txt 或 conda env create -f environment.yml")
        print("=" * 60)
        if args.strict:
            sys.exit(1)
        else:
            sys.exit(0)


if __name__ == "__main__":
    main()
"""scripts/check_env.py — 肝脏 3D 分割项目环境自检脚本

作用
----
本脚本在正式训练/推理之前，一次性检查本项目（MONAI + PyTorch 的肝脏/肝肿瘤
3D 分割）所需的运行环境是否就绪，并打印一张清晰的终端汇总表，内容包括：

    1. Python 版本，以及是否满足本项目「3.10+」的建议；
    2. PyTorch 是否安装、其版本；
    3. CUDA 是否可用、CUDA 版本、GPU 数量与第 0 块 GPU 名称；
    4. requirements.txt 中列出的全部第三方依赖包的「包名 / 状态 / 版本」；
    5. 缺失关键依赖时给出安装命令提示；
    6. 结尾给出「环境就绪」或「缺失依赖」的结论。

本脚本的核心职责就是「在依赖缺失时也能优雅运行」——因此它不会在文件顶部直接
``import torch`` 之类，而是用 ``importlib.util.find_spec`` 先探测包是否安装，
缺失时只报告而不让 ImportError 中断脚本本身。

运行方式
--------
在项目根目录（``d:\\项目1``）下执行：

    python scripts/check_env.py

可选参数：

    python scripts/check_env.py --strict
        # --strict：严格模式；若缺失关键依赖（torch/monai/nibabel/numpy），
        #           脚本以非零退出码结束，便于在 CI 或后续脚本中串联判断。

输入
----
无外部输入。不读取任何文件，也不需要命令行必选参数；``--strict`` 为唯一可选开关。

输出
----
仅向终端（stdout）打印格式化汇总，不写任何文件。典型输出包含三段：
    - [Python] 版本与建议达标情况；
    - [依赖包] 包名 / 状态[已装|缺失] / 类型[关键|可选] / 版本 的表格；
    - [CUDA / GPU] 可用性、CUDA 版本、GPU 数量与名称；
    - [安装提示]（仅当有缺失时）安装命令；
    - 结尾结论行。

常见错误
--------
1. 缺包：某依赖未安装，表格中状态显示「缺失」。关键依赖缺失时按提示执行
   ``pip install -r requirements.txt`` 补齐；torch 建议按 PyTorch 官网选择
   与本机 CUDA 匹配的安装命令。
2. Python 版本过低：本项目建议 3.10+，低于该版本会打印提示（但不强制报错，
   因为个别较低版本仍可能跑通，只是不作保证）。
3. CUDA 不可用：可能原因——无 NVIDIA GPU、未装显卡驱动、或安装的是 CPU 版
   PyTorch。此时脚本会明确提示「CPU 仍可运行，但标准科研设置（96^3、batch 2、
   300 epoch）在 CPU 上会极慢」，建议使用 GPU。
"""

# ===================== 标准库导入（集中顶部） =====================
# 本文件只导入 Python 标准库，所有第三方依赖都在函数内部按需导入，
# 以保证「缺包时本脚本仍能跑起来并报告缺失」这一核心职责。
import argparse
import sys
import importlib
import importlib.util
import importlib.metadata
import unicodedata


# ===================== 依赖清单配置 =====================
# 每项为四元组：(import 名, 显示名, distribution 名, 是否关键)
#   - import 名：用于 find_spec 探测是否安装（即在代码里 import 时用的模块名）；
#   - 显示名：汇总表里展示给同学的友好名称；
#   - distribution 名：用于 importlib.metadata.version 取版本号（PyPI 包名），
#     多数与 import 名相同，但有两个例外：
#         skimage 的 import 名是 skimage、PyPI 包名是 scikit-image；
#         yaml   的 import 名是 yaml、  PyPI 包名是 PyYAML。
#   - 是否关键：torch/monai/nibabel/numpy 为关键（缺失会触发安装提示，且在
#     --strict 下以非零码退出）；其余为可选（缺失只影响对应功能，不阻断自检）。
REQUIRED_PACKAGES = [
    ("torch",       "torch",        "torch",        True),   # 关键：深度学习框架
    ("monai",       "monai",        "monai",        True),   # 关键：医学影像深度学习框架
    ("nibabel",     "nibabel",      "nibabel",      True),   # 关键：NIfTI (.nii.gz) 读写
    ("numpy",       "numpy",        "numpy",        True),   # 关键：数值计算
    ("scipy",       "scipy",        "scipy",        False),  # 可选：科学计算
    ("skimage",     "scikit-image", "scikit-image", False),  # 可选：marching cubes，用于 STL 导出
    ("matplotlib",  "matplotlib",   "matplotlib",   False),  # 可选：可视化绘图
    ("trimesh",     "trimesh",      "trimesh",      False),  # 可选：STL 三维网格
    ("yaml",        "PyYAML",       "PyYAML",       False),  # 可选：读取 YAML 配置
    ("tqdm",        "tqdm",         "tqdm",         False),  # 可选：进度条
    ("tensorboard", "tensorboard",  "tensorboard",  False),  # 可选：训练日志
]


# ============================================================
# 终端显示宽度辅助
# ============================================================

def _display_width(s: str) -> int:
    """估算字符串在等宽终端里的显示宽度。

    中文/日韩字符（CJK）在终端里占 2 个英文字符宽，而 Python 原生的
    ``f"{s:<8}"`` 是按「字符个数」而非「显示宽度」补空格的，直接用会让含中文
    的表格列对不齐。本函数用 ``unicodedata.east_asian_width`` 把全角/宽字符
    计为 2，其余计为 1，从而得到更接近实际显示的宽度。
    """
    width = 0
    for ch in s:
        # 'W'(Wide)、'F'(Fullwidth) 为明确的双宽；'A'(Ambiguous) 按双宽处理更稳
        width += 2 if unicodedata.east_asian_width(ch) in ("W", "F", "A") else 1
    return width


def _pad(s: str, width: int) -> str:
    """按显示宽度把字符串左侧对齐、右侧补空格，使后续列对齐（兼容中文双宽）。"""
    return s + " " * max(0, width - _display_width(s))


# ============================================================
# 各项检测函数
# ============================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    本脚本没有必选参数，但仍用 argparse 提供统一的 ``--help`` 风格，并支持一个
   可选开关 ``--strict``：开启后，若缺失关键依赖，脚本以非零码退出。
    """
    parser = argparse.ArgumentParser(
        prog="check_env.py",
        description=(
            "肝脏 3D 分割项目环境自检：检查 Python 版本、PyTorch、MONAI、CUDA 可用性"
            "与各依赖包版本，打印汇总；缺失关键依赖时给出安装提示而非崩溃。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式：缺失关键依赖（torch/monai/nibabel/numpy）时以非零码退出。",
    )
    return parser.parse_args()


def check_python_version() -> tuple:
    """检查 Python 版本是否满足 3.10+ 建议。

    Returns:
        tuple: (版本字符串, 是否达标 bool)。达标仅作「建议」提示，不会强制报错。
    """
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    ok = (v.major, v.minor) >= (3, 10)
    return version_str, ok


def _get_version(import_name: str, dist_name: str) -> str:
    """尝试获取已安装包的版本号字符串；获取失败返回 "未知"。

    优先用 ``importlib.metadata.version``（按 PyPI 包名查询，无需真正 import 包，
    不会触发该包的初始化代码，最安全）；若失败再回退到 import 并读取 ``__version__``。
    """
    # 优先：按 distribution 名查元数据（不会执行包代码）
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        pass  # 该 distribution 未注册元数据，走下面的回退方案
    except Exception:
        pass  # 其他异常也回退，保证不中断

    # 回退：真正 import 并读 __version__（个别包版本号藏在子模块里，如老版 tqdm）
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", None)
        if ver is None and hasattr(mod, "version"):
            ver = getattr(mod.version, "__version__", None)
        return str(ver) if ver else "未知"
    except Exception:
        return "未知"


def check_packages() -> list:
    """逐个探测依赖包是否安装并取版本，返回检测结果列表。

    用 ``importlib.util.find_spec`` 探测：它只查找模块位置而不真正执行模块
    代码，缺失时返回 ``None`` 而不抛异常——这正是「缺包也能优雅运行」的关键。

    Returns:
        list[dict]: 每个元素含 import_name / display_name / installed /
        version / is_critical 五个字段。
    """
    results = []
    for import_name, display_name, dist_name, is_critical in REQUIRED_PACKAGES:
        # find_spec 返回 None 表示该包未安装；非 None 表示已安装
        spec = importlib.util.find_spec(import_name)
        installed = spec is not None
        version = _get_version(import_name, dist_name) if installed else "-"
        results.append({
            "import_name": import_name,
            "display_name": display_name,
            "installed": installed,
            "version": version,
            "is_critical": is_critical,
        })
    return results


def check_torch_cuda() -> dict:
    """检测 torch 的 CUDA/GPU 相关信息。仅在 torch 已安装时调用。

    Returns:
        dict: 含 cuda_available / cuda_version / gpu_count / gpu_name，
        检测异常时额外含 error 字段。
    """
    info = {
        "cuda_available": False,
        "cuda_version": None,
        "gpu_count": 0,
        "gpu_name": None,
    }
    try:
        import torch  # torch 已安装才进到这里，安全
        info["cuda_available"] = bool(torch.cuda.is_available())
        # torch.version.cuda 在 CPU 版 PyTorch 下为 None
        info["cuda_version"] = torch.version.cuda
        if info["cuda_available"]:
            info["gpu_count"] = torch.cuda.device_count()
            if info["gpu_count"] > 0:
                info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as e:
        info["error"] = str(e)
    return info


# ============================================================
# 主流程
# ============================================================

def main() -> None:
    """环境自检主流程：解析参数 → 检测 → 打印汇总 → 给出结论与退出码。"""
    args = parse_args()

    # —— 标题 ——
    print("=" * 64)
    print("肝脏 3D 分割项目 —— 环境自检 (check_env.py)")
    print("=" * 64)

    # 1) Python 版本检查
    py_ver, py_ok = check_python_version()
    if py_ok:
        print(f"\n[Python] 版本：{py_ver}  （满足 3.10+ 建议）")
    else:
        print(f"\n[Python] 版本：{py_ver}  （低于 3.10 建议，可能存在兼容问题，建议升级至 3.10+）")

    # 2) 依赖包检测
    results = check_packages()

    # 3) 打印依赖汇总表
    #    列宽按显示宽度设定，配合 _pad 让含中文的状态/类型列也对齐。
    print("\n[依赖包]")
    col_name, col_status, col_kind = 16, 8, 8
    header = "  " + _pad("包名", col_name) + _pad("状态", col_status) + _pad("类型", col_kind) + "版本"
    print(header)
    print("  " + "-" * (_display_width(header) - 2))
    for r in results:
        status = "已装" if r["installed"] else "缺失"
        kind = "关键" if r["is_critical"] else "可选"
        print("  " + _pad(r["display_name"], col_name)
              + _pad(status, col_status) + _pad(kind, col_kind) + r["version"])

    # 4) CUDA / GPU 检测（依赖 torch；torch 缺失则跳过并提示）
    print("\n[CUDA / GPU]")
    torch_installed = any(r["import_name"] == "torch" and r["installed"] for r in results)
    if not torch_installed:
        print("  torch 未安装，无法检测 CUDA。")
        print("  请先安装 torch（建议按 https://pytorch.org 选择与本机 CUDA 匹配的版本）。")
    else:
        cuda_info = check_torch_cuda()
        if cuda_info.get("cuda_available"):
            print(f"  CUDA 可用：是")
            print(f"  CUDA 版本：{cuda_info['cuda_version']}")
            print(f"  GPU 数量：{cuda_info['gpu_count']}")
            print(f"  GPU 名称：{cuda_info['gpu_name']}")
        else:
            print("  CUDA 可用：否")
            print(f"  torch.version.cuda：{cuda_info['cuda_version']}")
            print("  提示：未检测到可用 GPU。可能原因：无 NVIDIA GPU / 未装驱动 / "
                  "安装的是 CPU 版 PyTorch。")
            print("        CPU 仍可运行，但标准科研设置（96^3、batch 2、300 epoch）会极慢，建议使用 GPU。")
        if "error" in cuda_info:
            print(f"  [警告] 检测 CUDA 时发生异常：{cuda_info['error']}")

    # 5) 关键依赖缺失判定与安装提示
    missing_critical = [r for r in results if r["is_critical"] and not r["installed"]]
    missing_optional = [r for r in results if not r["is_critical"] and not r["installed"]]

    if missing_critical or missing_optional:
        print("\n[安装提示]")
        if missing_critical:
            names = "、".join(r["display_name"] for r in missing_critical)
            print(f"  缺失关键依赖：{names}")
            print("  请在项目根目录执行：")
            print("    pip install -r requirements.txt")
            print("  或单独安装关键包：")
            print("    pip install torch monai nibabel numpy")
            print("  （torch 建议按官网选择匹配 CUDA 的版本：https://pytorch.org）")
        if missing_optional:
            names = "、".join(r["display_name"] for r in missing_optional)
            print(f"  缺失可选依赖：{names}（不安装只影响对应功能，不阻断自检与核心训练）")

    # 6) 结论与退出码
    print("\n" + "=" * 64)
    if not missing_critical:
        print("结论：环境就绪（关键依赖齐全）")
        if missing_optional:
            print("（部分可选依赖缺失，对应功能将不可用，详见上方提示）")
    else:
        print("结论：缺失依赖（关键依赖不齐，请按上方提示安装后再运行训练）")
    print("=" * 64)

    # --strict：缺失关键依赖时以非零码退出，便于 CI 或后续脚本串联判断
    if args.strict and missing_critical:
        sys.exit(1)


if __name__ == "__main__":
    main()

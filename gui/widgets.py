# -*- coding: utf-8 -*-
"""
gui/widgets.py
==============

8 个功能页面（QWidget）+ 复用控件。所有页面只负责"采集参数 / 触发命令"，
不直接执行耗时操作；按下"开始 XX"按钮后，通过 `request_run(list[str])` 信号
把命令交给 main_window 的 CommandWorker 后台执行。

布局原则：左侧导航栏 + 右侧 QStackedWidget 切换页面。页面内部用表单式
QFormLayout，控件用 QSpinBox / QDoubleSpinBox / QLineEdit / QComboBox /
QCheckBox，符合医学新手使用习惯。
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


# ============================================================================
# 复用控件
# ============================================================================

class PathSelect(QWidget):
    """路径选择控件：QLineEdit + 「浏览」按钮。

    mode: "dir" 选目录, "open" 选文件, "save" 选保存文件
    """

    def __init__(self, mode: str = "dir", caption: str = "选择", file_filter: str = "", parent=None):
        super().__init__(parent)
        self.mode = mode
        self.caption = caption
        self.file_filter = file_filter
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.line = QLineEdit(self)
        self.btn = QPushButton("浏览…", self)
        self.btn.clicked.connect(self._browse)
        layout.addWidget(self.line)
        layout.addWidget(self.btn)

    def _browse(self) -> None:
        if self.mode == "dir":
            path = QFileDialog.getExistingDirectory(self, self.caption, self.line.text() or ".")
        elif self.mode == "save":
            path, _ = QFileDialog.getSaveFileName(self, self.caption, self.line.text() or "", self.file_filter)
        else:  # open
            path, _ = QFileDialog.getOpenFileName(self, self.caption, self.line.text() or ".", self.file_filter)
        if path:
            self.line.setText(path)

    def text(self) -> str:
        return self.line.text().strip()


class _PageBase(QWidget):
    """所有页面基类：提供统一的 request_run 信号 + 标题。"""

    request_run = Signal(list)  # 发出命令列表（argv）
    log_signal = Signal(str)    # 发出日志行，由 main_window 接入日志窗口

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self._build()

    def _build(self) -> None:  # pragma: no cover - 由子类实现
        raise NotImplementedError

    # 工具方法：参数非空校验，失败弹提示，返回是否通过
    def _require(self, *values: str, hint: str = "请先填写必填项") -> bool:
        for v in values:
            if not v:
                QMessageBox.warning(self, "缺少参数", hint)
                return False
        return True


# ============================================================================
# 1. Environment 环境部署
# ============================================================================

class EnvProbeThread(QThread):
    """后台探测环境（不阻塞 GUI 主线程）：import torch/monai/nibabel 取版本、
    CUDA 与 GPU 名称，并静默检测 pip 是否可用。完成后 `probe_result(dict)` 回传。
    """
    probe_result = Signal(dict)

    def run(self) -> None:  # noqa: D401
        import sys
        import importlib.metadata as md

        r: dict = {}
        r["py_version"] = sys.version.split()[0]
        r["py_exe"] = sys.executable

        # pip 可用性：静默跑 `python -m pip --version`
        try:
            import subprocess
            p = subprocess.run(
                [sys.executable, "-m", "pip", "--version"],
                capture_output=True, text=True, timeout=10,
            )
            r["pip_ok"] = (p.returncode == 0)
            r["pip_info"] = p.stdout.split("\n", 1)[0] if p.stdout else ""
        except Exception as e:  # noqa: BLE001
            r["pip_ok"] = False
            r["pip_info"] = f"检测失败：{e}"

        # 版本：用 importlib.metadata，无需 import 模块本身（快）
        for name, pkg in [("torch", "torch"), ("monai", "monai"), ("nibabel", "nibabel")]:
            try:
                r[name] = md.version(pkg)
            except Exception:  # noqa: BLE001
                r[name] = None

        # CUDA / GPU：必须真 import torch
        r["cuda"] = False
        r["gpu"] = ""
        if r.get("torch"):
            try:
                import torch  # noqa: WPS433  (本线程探测，OK)
                r["cuda"] = bool(torch.cuda.is_available())
                if r["cuda"]:
                    r["gpu"] = torch.cuda.get_device_name(0)
            except Exception:  # noqa: BLE001
                r["cuda"] = False
                r["gpu"] = ""

        self.probe_result.emit(r)


class EnvironmentPage(_PageBase):
    def __init__(self, project_root: str, parent=None):
        self.project_root = project_root
        super().__init__("Environment 环境部署", parent)

    def _build(self) -> None:
        from gui.workers import make_python_cmd, make_pip_install_cmd  # 延迟导入以解耦

        self._make_python_cmd = make_python_cmd
        self._make_pip_cmd = make_pip_install_cmd

        root = QVBoxLayout(self)

        # --- 检测区 ---
        box = QGroupBox("环境检测")
        form = QFormLayout(box)
        self.lbl_py_version = QLabel("未检测")
        self.lbl_py_exe = QLabel("未检测")
        self.lbl_pip = QLabel("未检测")
        self.lbl_torch = QLabel("未检测")
        self.lbl_cuda = QLabel("未检测")
        self.lbl_gpu = QLabel("未检测")
        self.lbl_monai = QLabel("未检测")
        self.lbl_nibabel = QLabel("未检测")
        form.addRow("Python version:", self.lbl_py_version)
        form.addRow("Python 路径:", self.lbl_py_exe)
        form.addRow("pip:", self.lbl_pip)
        form.addRow("torch version:", self.lbl_torch)
        form.addRow("CUDA available:", self.lbl_cuda)
        form.addRow("GPU name:", self.lbl_gpu)
        form.addRow("MONAI version:", self.lbl_monai)
        form.addRow("nibabel version:", self.lbl_nibabel)
        btn_check = QPushButton("一键检查环境（含 import + CUDA + GPU 探测）")
        btn_check.clicked.connect(self._on_check)
        form.addRow(btn_check)
        root.addWidget(box)

        # --- 依赖安装区 ---
        box2 = QGroupBox("依赖安装（PyTorch 需先自行按官网 CUDA 选择安装；本工具不自动装 CUDA 版 PyTorch）")
        v2 = QVBoxLayout(box2)
        hint = QLabel(
            "提示：建议先创建 conda 环境 medseg 并在其中运行本 GUI：\n"
            "    conda create -n medseg python=3.10 -y && conda activate medseg\n"
            "本工具不会自动安装 CUDA 版 PyTorch —— 请从官网选择对应命令粘贴到下方输入框。\n"
            "https://pytorch.org/get-started/locally/"
        )
        hint.setStyleSheet("color: #555;")
        v2.addWidget(hint)
        row = QHBoxLayout()
        btn_install = QPushButton("一键安装依赖 (pip install -r requirements.txt)")
        btn_install.clicked.connect(self._on_install)
        btn_open_req = QPushButton("打开 requirements.txt")
        btn_open_req.clicked.connect(self._on_open_req)
        row.addWidget(btn_install)
        row.addWidget(btn_open_req)
        v2.addLayout(row)
        root.addWidget(box2)

        # --- PyTorch 自定义安装区（torch 未安装时优先使用） ---
        box3 = QGroupBox("安装 PyTorch（自定义命令；请从官网粘贴对应 CUDA 安装命令）")
        v3 = QVBoxLayout(box3)
        torch_hint = QLabel(
            "示例（按你的 CUDA 版本修改）：\n"
            "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118\n"
            "若 torch 已安装但 CUDA 不可用，请卸载后重新按正确 CUDA 安装。"
        )
        torch_hint.setStyleSheet("color: #555;")
        v3.addWidget(torch_hint)
        self.torch_cmd_input = QLineEdit()
        self.torch_cmd_input.setPlaceholderText("粘贴 PyTorch 官网安装命令（pip install ...）")
        v3.addWidget(self.torch_cmd_input)
        row_t = QHBoxLayout()
        btn_install_torch = QPushButton("安装 PyTorch")
        btn_install_torch.clicked.connect(self._on_install_torch)
        btn_open_pytorch = QPushButton("打开 PyTorch 官网")
        btn_open_pytorch.clicked.connect(self._on_open_pytorch_site)
        row_t.addWidget(btn_install_torch)
        row_t.addWidget(btn_open_pytorch)
        v3.addLayout(row_t)
        root.addWidget(box3)

        root.addStretch(1)

    # ---------- 检测 ----------
    def _on_check(self) -> None:
        # 先把检测区置为"正在检测…"，避免误以为旧值
        for lbl in [
            self.lbl_py_version, self.lbl_py_exe, self.lbl_pip, self.lbl_torch,
            self.lbl_cuda, self.lbl_gpu, self.lbl_monai, self.lbl_nibabel,
        ]:
            lbl.setText("正在检测…")
            lbl.setStyleSheet("color: #777;")
        self._probe = EnvProbeThread(self)
        self._probe.probe_result.connect(self._apply_probe)
        self._probe.start()
        # 同时调 test_env.py --strict 走严格的命令行自检（日志可见）
        script = os.path.join(self.project_root, "test_env.py")
        # 注：request_run 会通过 main_window 起后台 worker；此处不阻塞
        self.request_run.emit(self._make_python_cmd(script, ["--strict"]))

    def _apply_probe(self, r: dict) -> None:
        def ok_or_fail(v, ok_text=None):
            if v in (None, "") and ok_text is None:
                return ("✗ 未安装", "#c0392b")
            return (ok_text or str(v), "#27ae60")

        py_t, py_c = ok_or_fail(r.get("py_version"))
        self.lbl_py_version.setText(f"{r.get('py_version', '未知')}")
        self.lbl_py_version.setStyleSheet(f"color: {py_c};")
        self.lbl_py_exe.setText(r.get("py_exe", "未知"))
        self.lbl_py_exe.setStyleSheet("color: #555;")
        pip_t, pip_c = ("✓ 可用", "#27ae60") if r.get("pip_ok") else ("✗ 不可用", "#c0392b")
        self.lbl_pip.setText(pip_t)
        self.lbl_pip.setStyleSheet(f"color: {pip_c};")
        torch_v = r.get("torch")
        self.lbl_torch.setText(torch_v or "✗ 未安装")
        self.lbl_torch.setStyleSheet(f"color: {'#27ae60' if torch_v else '#c0392b'};")
        cuda = bool(r.get("cuda"))
        self.lbl_cuda.setText("✓ 可用" if cuda else "✗ 不可用")
        self.lbl_cuda.setStyleSheet(f"color: {'#27ae60' if cuda else '#c0392b'};")
        gpu = r.get("gpu") or ("N/A（CUDA 不可用）" if not cuda else "N/A")
        self.lbl_gpu.setText(gpu)
        self.lbl_gpu.setStyleSheet(f"color: {'#27ae60' if cuda and r.get('gpu') else '#555'};")
        monai_v = r.get("monai")
        self.lbl_monai.setText(monai_v or "✗ 未安装")
        self.lbl_monai.setStyleSheet(f"color: {'#27ae60' if monai_v else '#c0392b'};")
        nib_v = r.get("nibabel")
        self.lbl_nibabel.setText(nib_v or "✗ 未安装")
        self.lbl_nibabel.setStyleSheet(f"color: {'#27ae60' if nib_v else '#c0392b'};")

        # torch 未安装：提示用户走自定义 PyTorch 安装
        if not torch_v:
            QMessageBox.warning(
                self, "PyTorch 未安装",
                "PyTorch 建议根据你的 CUDA 版本从官网选择安装命令：\n"
                "https://pytorch.org/get-started/locally/\n"
                "请在下方输入框粘贴安装命令并点击「安装 PyTorch」。\n"
                "本工具不会自动安装 CUDA 版 PyTorch。",
            )
            self.torch_cmd_input.setFocus()

    # ---------- 一键安装依赖 ----------
    def _on_install(self) -> None:
        # 安装前弹窗提醒：建议先创建 conda 环境 medseg
        reply = QMessageBox.question(
            self, "一键安装依赖",
            "建议先创建 conda 环境 medseg 再继续，是否现在安装？\n\n"
            "（如已在 medseg / 其它环境中运行本 GUI，点「Yes」继续即可）",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return
        req = os.path.join(self.project_root, "requirements.txt")
        if not os.path.isfile(req):
            QMessageBox.warning(self, "未找到", f"未找到 {req}")
            return
        cmd = self._make_pip_cmd(["-r", req])
        self.request_run.emit(cmd)

    # ---------- 安装 PyTorch（自定义命令） ----------
    def _on_install_torch(self) -> None:
        raw = self.torch_cmd_input.text().strip()
        if not raw:
            QMessageBox.warning(self, "缺少命令", "请粘贴 PyTorch 官网安装命令（如 pip install torch ...）。")
            return
        # 解析用户粘贴的整行命令；支持前缀 "python -m pip install ..." 与 "pip install ..."
        argv = self._parse_user_cmd(raw)
        if not argv:
            QMessageBox.warning(self, "命令无效", "无法解析输入的命令。")
            return
        # 不替用户做 CUDA 校正，原样执行
        self.request_run.emit(argv)

    @staticmethod
    def _parse_user_cmd(raw: str) -> list:
        """把用户粘贴的命令解析为 argv：去掉前导 python / python -m / pip 引导。"""
        import shlex
        toks = shlex.split(raw, posix=(os.name != "nt"))
        if not toks:
            return []
        # 规整：若用户写 "pip install xxx"，转成用当前解释器执行以保证走同一环境
        if toks[0] in ("python", "python3"):
            toks = toks[1:]
        if toks and toks[0] == "-m":
            toks = toks[1:]
        if toks and toks[0] == "pip":
            return [os.environ.get("PYTHON_EXECUTABLE", __import__("sys").executable), "-m", "pip"] + toks[1:]
        # 否则当作直接给出的 install 列表：可能用户也写了 "install ..."，去掉再统一补
        if toks and toks[0] == "install":
            toks = toks[1:]
        return [__import__("sys").executable, "-m", "pip", "install", *toks]

    # ---------- 打开文件 / 官网 ----------
    def _on_open_req(self) -> None:
        req = os.path.join(self.project_root, "requirements.txt")
        if not os.path.isfile(req):
            QMessageBox.warning(self, "未找到", f"未找到 {req}")
            return
        try:
            if os.name == "nt":
                os.startfile(req)  # type: ignore[attr-defined]
            else:
                import subprocess
                subprocess.Popen(["xdg-open", req])  # Linux
        except Exception as e:  # noqa: BLE001
            QMessageBox.information(self, "路径", f"{req}\n打开失败：{e}")

    def _on_open_pytorch_site(self) -> None:
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl("https://pytorch.org/get-started/locally/"))


# ============================================================================
# 2. Dataset 数据管理
# ============================================================================

class CopyToRawThread(QThread):
    """后台把数据目录复制到项目 data/raw/Task03_Liver，按文件数发进度，不阻塞 GUI。

    Signals
    -------
    progress(int, int) : (已复制文件数, 总文件数)
    done(bool, str)    : 完成（成功/失败 + 摘要）
    """
    progress = Signal(int, int)
    done = Signal(bool, str)
    log = Signal(str)

    def __init__(self, src_dir: str, dst_dir: str, parent=None):
        super().__init__(parent)
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:  # noqa: D401
        import shutil
        # 先统计待复制文件数（递归），用于进度条
        src = os.path.abspath(self.src_dir)
        dst = os.path.abspath(self.dst_dir)
        self.log.emit(f"[INFO] 开始复制：{src}  ->  {dst}")
        try:
            files: list[str] = []
            for root_d, _dirs, fs in os.walk(src):
                for f in fs:
                    files.append(os.path.join(root_d, f))
            total = len(files)
            if total == 0:
                self.done.emit(False, "源目录为空，无可复制文件。")
                return
            self.progress.emit(0, total)

            # 清空目标（若已存在）再整体复制，避免残留旧文件
            if os.path.exists(dst):
                shutil.rmtree(dst)
            os.makedirs(dst, exist_ok=True)

            copied = 0
            for fp in files:
                if self._stop:
                    self.log.emit("[INFO] 已请求停止复制。")
                    self.done.emit(False, "用户主动停止复制。")
                    return
                rel = os.path.relpath(fp, src)
                tgt = os.path.join(dst, rel)
                os.makedirs(os.path.dirname(tgt), exist_ok=True)
                shutil.copy2(fp, tgt)
                copied += 1
                if copied % 5 == 0 or copied == total:
                    self.progress.emit(copied, total)
            self.progress.emit(copied, total)
            self.log.emit(f"[OK] 复制完成：共 {copied} 个文件 -> {dst}")
            self.done.emit(True, f"复制完成（{copied} 个文件）")
        except Exception as e:  # noqa: BLE001
            self.log.emit(f"[FAIL] 复制失败：{e}")
            self.done.emit(False, f"复制失败：{e}")


class DatasetPage(_PageBase):
    # 让 main_window 能连接到 stop 信号（与 TrainingPage 同模式）
    request_stop = Signal()

    def __init__(self, project_root: str, parent=None):
        self.project_root = project_root
        # 项目内置数据目录（GUI 默认从这里读数据做后续 inspect/split）
        self._raw_dir = os.path.join(project_root, "data", "raw", "Task03_Liver")
        self._last_cmd: list[str] = []
        self._copy_thread: Optional[CopyToRawThread] = None  # type: ignore[name-defined]
        self._worker_ref = None  # 持有命令 worker，由 main_window 设置，用于停止
        super().__init__("Dataset 数据管理", parent)

    def _build(self) -> None:
        from gui.workers import make_python_cmd
        self._mk = make_python_cmd

        root = QVBoxLayout(self)

        # --- Step 1：选择原始数据目录 + 结构检查 + 导入/复制 ---
        box = QGroupBox("Step 1 选择原始 Task03_Liver 数据目录（应含 imagesTr/labelsTr/dataset.json）")
        v = QVBoxLayout(box)
        self.data_dir = PathSelect("dir", "选择 Task03_Liver 数据根目录")
        self.data_dir.line.textChanged.connect(self._on_data_changed)
        v.addWidget(self.data_dir)

        self.lbl_struct = QLabel("尚未检查")
        self.lbl_struct.setStyleSheet("color: #555;")
        self.lbl_struct.setWordWrap(True)
        v.addWidget(self.lbl_struct)

        self.lbl_counts = QLabel("imagesTr / labelsTr 文件数：未统计")
        self.lbl_counts.setStyleSheet("color: #555;")
        v.addWidget(self.lbl_counts)

        row_btns = QHBoxLayout()
        self.btn_check_struct = QPushButton("检查结构 (imagesTr/labelsTr/dataset.json)")
        self.btn_check_struct.clicked.connect(self._on_check_struct)
        self.btn_import = QPushButton("复制 / 导入到项目 data/raw/Task03_Liver")
        self.btn_import.clicked.connect(self._on_import)
        row_btns.addWidget(self.btn_check_struct)
        row_btns.addWidget(self.btn_import)
        v.addLayout(row_btns)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        v.addWidget(self.progress)
        self.lbl_import = QLabel("")
        self.lbl_import.setStyleSheet("color: #555;")
        self.lbl_import.setWordWrap(True)
        v.addWidget(self.lbl_import)
        root.addWidget(box)

        # --- Step 2：数据集检查 ---
        box2 = QGroupBox("Step 2 数据集检查（运行 src/inspect_dataset.py）")
        v2 = QVBoxLayout(box2)
        self.output_dir = PathSelect("dir", "选择输出目录")
        self.output_dir.line.setText(os.path.join(self.project_root, "outputs"))
        v2.addWidget(QLabel("输出目录（dataset_summary.csv 与 figures/）："))
        v2.addWidget(self.output_dir)

        # --quick 快速模式：只读文件头 shape/spacing，跳过强度统计与画图
        self.chk_quick = QCheckBox(
            "快速模式（--quick）：仅读文件头获取 shape/spacing，跳过强度统计与画图，"
            "大幅加速 27GB 等大数据集检查"
        )
        self.chk_quick.setChecked(True)  # 大数据集默认开启快速模式
        v2.addWidget(self.chk_quick)

        # 检查按钮 + 停止按钮
        row_inspect = QHBoxLayout()
        self.btn_inspect = QPushButton("检查数据集")
        self.btn_inspect.clicked.connect(self._on_inspect)
        self.btn_stop_inspect = QPushButton("停止检查")
        self.btn_stop_inspect.setEnabled(False)
        self.btn_stop_inspect.clicked.connect(self._on_stop)
        row_inspect.addWidget(self.btn_inspect)
        row_inspect.addWidget(self.btn_stop_inspect)
        v2.addLayout(row_inspect)

        # 检查进度条（解析 [进度] i/N 更新）
        self.progress_inspect = QProgressBar()
        self.progress_inspect.setVisible(False)
        v2.addWidget(self.progress_inspect)
        self.lbl_inspect_status = QLabel("")
        self.lbl_inspect_status.setStyleSheet("color: #555;")
        self.lbl_inspect_status.setWordWrap(True)
        v2.addWidget(self.lbl_inspect_status)
        root.addWidget(box2)

        # --- Step 3：划分 ---
        box3 = QGroupBox("Step 3 划分 train / val / test（运行 src/create_splits.py）")
        f3 = QFormLayout(box3)
        self.split_out = PathSelect("dir", "选择 splits 输出目录")
        self.split_out.line.setText(os.path.join(self.project_root, "data/splits"))
        f3.addRow("splits 输出目录:", self.split_out)
        self.train_ratio = QDoubleSpinBox()
        self.train_ratio.setRange(0.0, 1.0)
        self.train_ratio.setSingleStep(0.05)
        self.train_ratio.setValue(0.7)
        self.val_ratio = QDoubleSpinBox()
        self.val_ratio.setRange(0.0, 1.0)
        self.val_ratio.setSingleStep(0.05)
        self.val_ratio.setValue(0.15)
        self.seed = QSpinBox()
        self.seed.setRange(0, 100000)
        self.seed.setValue(42)
        f3.addRow("train_ratio:", self.train_ratio)
        f3.addRow("val_ratio:", self.val_ratio)
        f3.addRow("seed:", self.seed)
        self.btn_split = QPushButton("划分数据集")
        self.btn_split.clicked.connect(self._on_split)
        f3.addRow(self.btn_split)
        self.lbl_splits = QLabel("train.json / val.json / test.json：尚未生成")
        self.lbl_splits.setStyleSheet("color: #555;")
        self.lbl_splits.setWordWrap(True)
        f3.addRow(self.lbl_splits)

        # 校验划分独立性按钮：运行 src/verify_split_integrity.py，
        # 检查 train/val/test 三者 image 与 label 的 basename 是否重叠。
        # 正式测试集推理/评估前应先跑一次，避免测试集病例混入训练集。
        self.btn_verify = QPushButton("校验划分独立性")
        self.btn_verify.clicked.connect(self._on_verify)
        f3.addRow(self.btn_verify)
        self.lbl_verify = QLabel("校验 train/val/test 是否互斥（无病例重叠）。")
        self.lbl_verify.setStyleSheet("color: #555;")
        self.lbl_verify.setWordWrap(True)
        f3.addRow(self.lbl_verify)

        root.addWidget(box3)

        root.addStretch(1)
        # 先预检项目内置目录的 split JSON
        self._check_splits_exist()
        # 默认填项目内置目录（控件均已构造，block 信号避免误触发 _on_data_changed）
        self.data_dir.line.blockSignals(True)
        self.data_dir.line.setText(self._raw_dir)
        self.data_dir.line.blockSignals(False)

    # ---------- Step 1 ----------
    def _on_data_changed(self, _txt: str) -> None:
        self.lbl_struct.setText("尚未检查")
        self.lbl_struct.setStyleSheet("color: #555;")
        self.lbl_counts.setText("imagesTr / labelsTr 文件数：未统计")
        self.lbl_counts.setStyleSheet("color: #555;")

    def _on_check_struct(self) -> None:
        d = self.data_dir.text()
        if not self._require(d, hint="请先选择数据根目录"):
            return
        items = ["imagesTr", "labelsTr", "dataset.json"]
        missing = [x for x in items if not os.path.exists(os.path.join(d, x))]
        if missing:
            self.lbl_struct.setText(f"❌ 缺失：{', '.join(missing)}")
            self.lbl_struct.setStyleSheet("color: #c0392b;")
            self.lbl_counts.setText("imagesTr / labelsTr 文件数：未统计")
            self.lbl_counts.setStyleSheet("color: #555;")
            return
        # 分别统计 imagesTr 与 labelsTr 的 .nii.gz 数量
        n_img = self._count_niigz(os.path.join(d, "imagesTr"))
        n_lbl = self._count_niigz(os.path.join(d, "labelsTr"))
        self.lbl_struct.setText("✓ 结构完整（imagesTr / labelsTr / dataset.json 均存在）")
        self.lbl_struct.setStyleSheet("color: #27ae60;")
        self.lbl_counts.setText(f"imagesTr / labelsTr 文件数：{n_img} / {n_lbl}")
        self.lbl_counts.setStyleSheet("color: #27ae60;" if n_img == n_lbl and n_img > 0 else "color: #c0392b;")
        self.log_signal.emit(f"[INFO] 结构检查通过：imagesTr={n_img}, labelsTr={n_lbl}")

    @staticmethod
    def _count_niigz(d: str) -> int:
        try:
            return sum(1 for f in os.listdir(d) if f.endswith(".nii.gz"))
        except OSError:
            return -1

    def _on_import(self) -> None:
        """把数据导入项目 data/raw/Task03_Liver：可直接记录路径或后台复制。"""
        d = self.data_dir.text()
        if not self._require(d, hint="请先选择数据根目录"):
            return
        src = os.path.abspath(d)
        dst = os.path.abspath(self._raw_dir)
        if os.path.normpath(src) == os.path.normpath(dst):
            QMessageBox.information(self, "无需复制", "所选目录已是项目数据目录 data/raw/Task03_Liver，直接记录路径即可。")
            self.lbl_import.setText("✓ 已记录路径：" + dst)
            self.lbl_import.setStyleSheet("color: #27ae60;")
            return
        # 询问：复制到项目（推荐，便于复现） / 仅记录当前路径
        reply = QMessageBox.question(
            self, "导入数据",
            "是否将所选数据复制到项目目录 data/raw/Task03_Liver？\n"
            "  · 「Yes」：后台复制（推荐，路径固定便于复现）\n"
            "  · 「No」 ：仅记录当前路径，不复制（后续以当前路径为准）",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if reply == QMessageBox.No:
            self._raw_dir = dst  # 保持默认
            self.data_dir.line.setText(src)  # 后续以用户路径为准
            self.lbl_import.setText("✓ 已记录路径（不复制）：" + src)
            self.lbl_import.setStyleSheet("color: #27ae60;")
            self.log_signal.emit(f"[INFO] 仅记录路径（不复制）：{src}")
            return
        # 复制：启动后台线程
        if self._copy_thread and self._copy_thread.isRunning():
            QMessageBox.warning(self, "复制进行中", "已有复制任务在运行，请等待完成。")
            return
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.lbl_import.setText("正在复制...")
        self.lbl_import.setStyleSheet("color: #555;")
        self.btn_import.setEnabled(False)
        self._copy_thread = CopyToRawThread(src, dst, parent=self)
        self._copy_thread.progress.connect(self._on_copy_progress)
        self._copy_thread.done.connect(self._on_copy_done)
        self._copy_thread.log.connect(self.log_signal)
        self._copy_thread.start()

    def _on_copy_progress(self, cur: int, total: int) -> None:
        self.progress.setMaximum(total)
        self.progress.setValue(cur)
        self.lbl_import.setText(f"正在复制... {cur}/{total}")

    def _on_copy_done(self, ok: bool, msg: str) -> None:
        self.btn_import.setEnabled(True)
        if ok:
            self.progress.setValue(self.progress.maximum())
            self.lbl_import.setText(f"✓ {msg}")
            self.lbl_import.setStyleSheet("color: #27ae60;")
            # 复制完成？把数据目录指向项目内置目录，并刷新结构检查
            self.data_dir.line.setText(self._raw_dir)
            self._on_check_struct()
        else:
            self.lbl_import.setText(f"✗ {msg}")
            self.lbl_import.setStyleSheet("color: #c0392b;")
        self.progress.setVisible(False)
        self._copy_thread = None

    # ---------- Step 2 ----------
    def _on_inspect(self) -> None:
        d = self.data_dir.text() or self._raw_dir
        o = self.output_dir.text() or os.path.join(self.project_root, "outputs")
        if not self._require(d, hint="请先选择数据根目录"):
            return
        script = os.path.join(self.project_root, "src", "inspect_dataset.py")
        args = ["--data_dir", d, "--output_dir", o]
        if self.chk_quick.isChecked():
            args.append("--quick")
        cmd = self._mk(script, args)
        self._last_cmd = cmd
        # 重置进度条
        self.progress_inspect.setVisible(True)
        self.progress_inspect.setValue(0)
        self.lbl_inspect_status.setText("正在检查...")
        self.lbl_inspect_status.setStyleSheet("color: #555;")
        self.request_run.emit(cmd)

    def _on_stop(self) -> None:
        """停止当前检查/划分任务（实际停止由 main_window 调 worker.request_stop()）。"""
        self.request_stop.emit()

    # 解析子进程输出中的 [进度] i/N，更新进度条
    # 正则匹配：[进度] 5/131 (liver_42)  或  [进度] 5/131
    _PROGRESS_RE = re.compile(r"\[进度\]\s*(\d+)\s*/\s*(\d+)")

    def parse_progress(self, line: str) -> None:
        """从 worker 日志行中解析 [进度] i/N 并更新进度条。

        被 main_window 连接到 worker.log_signal；非检查命令的输出不会匹配，
        无副作用。
        """
        m = self._PROGRESS_RE.search(line)
        if m:
            cur = int(m.group(1))
            total = int(m.group(2))
            self.progress_inspect.setMaximum(total)
            self.progress_inspect.setValue(cur)
            self.lbl_inspect_status.setText(f"检查进度：{cur}/{total}")

    # ---------- Step 3 ----------
    def _on_split(self) -> None:
        d = self.data_dir.text() or self._raw_dir
        o = self.split_out.text() or os.path.join(self.project_root, "data/splits")
        if not self._require(d, hint="请先选择数据根目录"):
            return
        if self.train_ratio.value() + self.val_ratio.value() >= 1.0:
            QMessageBox.warning(self, "比例错误", "train_ratio + val_ratio 必须 < 1（需留 test）")
            return
        script = os.path.join(self.project_root, "src", "create_splits.py")
        cmd = self._mk(script, [
            "--data_dir", d,
            "--output_dir", o,
            "--train_ratio", f"{self.train_ratio.value()}",
            "--val_ratio", f"{self.val_ratio.value()}",
            "--seed", f"{self.seed.value()}",
        ])
        self._split_out_dir = o  # 记录输出目录，完成后检查
        self._last_cmd = cmd
        self.lbl_splits.setText("train.json / val.json / test.json：正在生成...")
        self.lbl_splits.setStyleSheet("color: #555;")
        self.request_run.emit(cmd)

    # ---------- 校验划分独立性 ----------
    def _on_verify(self) -> None:
        """运行 src/verify_split_integrity.py，检查 train/val/test 是否互斥。

        脚本退出码 0=独立无重叠，1=存在重叠。校验结果写到
        outputs/split_integrity_report.txt，这里读回判断 OK/ERROR。
        """
        o = self.split_out.text() or os.path.join(self.project_root, "data/splits")
        tj = os.path.join(o, "train.json")
        vj = os.path.join(o, "val.json")
        tsj = os.path.join(o, "test.json")
        # 三个 JSON 必须都存在才能校验
        miss = [p for p in (tj, vj, tsj) if not os.path.isfile(p)]
        if miss:
            QMessageBox.warning(
                self, "缺少划分文件",
                "以下划分 JSON 不存在，请先运行「划分数据集」：\n  " +
                "\n  ".join(miss))
            return
        script = os.path.join(self.project_root, "src", "verify_split_integrity.py")
        cmd = self._mk(script, [
            "--train_json", tj,
            "--val_json", vj,
            "--test_json", tsj,
        ])
        self._last_cmd = cmd
        self.lbl_verify.setText("正在校验 train/val/test 是否互斥...")
        self.lbl_verify.setStyleSheet("color: #555;")
        self.request_run.emit(cmd)

    # ---------- main_window 完成钩子 ----------
    def on_cmd_ok(self, cmd: list) -> None:
        cmd_str = " ".join(cmd)
        if "inspect_dataset.py" in cmd_str:
            # 检查完成：进度条满，显示成功
            self.progress_inspect.setValue(self.progress_inspect.maximum())
            self.lbl_inspect_status.setText("✓ 数据集检查完成（见日志与 CSV）")
            self.lbl_inspect_status.setStyleSheet("color: #27ae60;")
        elif "create_splits.py" in cmd_str:
            self._check_splits_exist()
        elif "verify_split_integrity.py" in cmd_str:
            # 校验通过：读回报告文件确认 OK
            self._reflect_verify_result(ok=True)

    def on_cmd_fail(self, cmd: list) -> None:
        cmd_str = " ".join(cmd)
        if "inspect_dataset.py" in cmd_str:
            self.lbl_inspect_status.setText("✗ 数据集检查失败或已停止（见日志）")
            self.lbl_inspect_status.setStyleSheet("color: #c0392b;")
        elif "create_splits.py" in cmd_str:
            self.lbl_splits.setText("train.json / val.json / test.json：生成失败（见日志）")
            self.lbl_splits.setStyleSheet("color: #c0392b;")
        elif "verify_split_integrity.py" in cmd_str:
            # 校验失败 = 存在重叠（脚本以退出码 1 报告重叠）
            self._reflect_verify_result(ok=False)

    def _reflect_verify_result(self, ok: bool) -> None:
        """根据校验结果刷新 lbl_verify，并尝试从报告文件提取关键行。"""
        report_path = os.path.join(self.project_root, "outputs", "split_integrity_report.txt")
        detail = ""
        try:
            if os.path.isfile(report_path):
                with open(report_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line.startswith("OK:") or line.startswith("ERROR:") or "例" in line:
                            detail = line
                            if line.startswith("OK:") or line.startswith("ERROR:"):
                                break
        except OSError:
            detail = ""
        if ok:
            self.lbl_verify.setText("✓ 划分独立：train/val/test 无病例重叠。" +
                                    (f"（{detail}）" if detail else ""))
            self.lbl_verify.setStyleSheet("color: #27ae60;")
        else:
            self.lbl_verify.setText("✗ 划分存在重叠！正式测试集独立性被破坏，请重新划分。" +
                                    (f"（{detail}）" if detail else ""))
            self.lbl_verify.setStyleSheet("color: #c0392b;")

    def _check_splits_exist(self) -> None:
        o = getattr(self, "_split_out_dir", None) or os.path.join(self.project_root, "data/splits")
        names = {"train.json": "train", "val.json": "val", "test.json": "test"}
        flags = {}
        for fn, key in names.items():
            flags[key] = os.path.isfile(os.path.join(o, fn))
        all_ok = all(flags.values())
        if all_ok:
            self.lbl_splits.setText("train.json / val.json / test.json：✓ 均已生成")
            self.lbl_splits.setStyleSheet("color: #27ae60;")
        else:
            miss = [fn for fn, ok in flags.items() if not ok]
            self.lbl_splits.setText("train.json / val.json / test.json：缺失 " + ", ".join(miss))
            self.lbl_splits.setStyleSheet("color: #c0392b;")


# ============================================================================
# 3. Training 模型训练
# ============================================================================

class TrainingPage(_PageBase):
    """训练页：选择 config/train.json/val.json/output_dir，GUI 改参，
    「保存配置」回写 configs/liver_unet_3d.yaml，「开始训练」调 src/train.py，
    支持 binary_liver / multiclass 双模式，实时日志、停止训练、训练完显示 ckpt 路径。
    """

    # 让 main_window 能连接到 stop 信号
    request_stop = Signal()

    def __init__(self, project_root: str, parent=None):
        self.project_root = project_root
        self._worker_ref = None  # 用于持有训练 worker（避免被 GC），由 main_window 设置
        super().__init__("Training 模型训练", parent)

    def _build(self) -> None:
        from gui.workers import make_python_cmd
        self._mk = make_python_cmd

        root = QVBoxLayout(self)

        box = QGroupBox("训练配置（运行 src/train.py）")
        f = QFormLayout(box)

        # ---- 路径区 ----
        self.config = PathSelect("open", "选择 config YAML", "YAML (*.yaml *.yml)")
        self.config.line.setText(os.path.join(self.project_root, "configs", "liver_unet_3d.yaml"))
        self.train_json = PathSelect("open", "选择 train.json", "JSON (*.json)")
        self.val_json = PathSelect("open", "选择 val.json", "JSON (*.json)")
        self.output_dir = PathSelect("dir", "选择模型输出目录")
        self.output_dir.line.setText(os.path.join(self.project_root, "outputs", "models"))
        f.addRow("config YAML:", self.config)
        f.addRow("train.json:", self.train_json)
        f.addRow("val.json:", self.val_json)
        f.addRow("输出目录:", self.output_dir)

        # ---- 训练超参 ----
        self.max_epochs = QSpinBox()
        self.max_epochs.setRange(1, 100000)
        self.max_epochs.setValue(300)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 64)
        self.batch_size.setValue(1)
        self.lr = QDoubleSpinBox()
        self.lr.setRange(1e-8, 1.0)
        self.lr.setDecimals(6)
        self.lr.setSingleStep(1e-5)
        self.lr.setValue(1e-4)
        self.num_workers = QSpinBox()
        self.num_workers.setRange(0, 32)
        self.num_workers.setValue(4)
        self.val_interval = QSpinBox()
        self.val_interval.setRange(1, 100)
        self.val_interval.setValue(2)
        f.addRow("max_epochs:", self.max_epochs)
        f.addRow("batch_size:", self.batch_size)
        f.addRow("learning rate:", self.lr)
        f.addRow("num_workers:", self.num_workers)
        f.addRow("val_interval:", self.val_interval)

        # ---- roi_size（写入 yaml，不进命令行）----
        roi_row = QHBoxLayout()
        self.roi_x = QSpinBox(); self.roi_x.setRange(16, 512); self.roi_x.setValue(96)
        self.roi_y = QSpinBox(); self.roi_y.setRange(16, 512); self.roi_y.setValue(96)
        self.roi_z = QSpinBox(); self.roi_z.setRange(16, 512); self.roi_z.setValue(96)
        roi_row.addWidget(self.roi_x); roi_row.addWidget(self.roi_y); roi_row.addWidget(self.roi_z)
        f.addRow("roi_size (x y z):", roi_row)

        # ---- device ----
        self.device = QComboBox()
        self.device.addItems(["auto", "cuda", "cpu"])
        f.addRow("device:", self.device)

        # ---- cache（缓存策略）----
        # 三档：none=不缓存(最省内存,最慢) / disk=磁盘缓存(省内存,二次训练快) /
        #       memory=内存全缓存(最快,但需~120GB RAM)。27GB 大数据集推荐 disk。
        self.cache = QComboBox()
        self.cache.addItem("磁盘缓存（推荐：省内存，二次训练快）", userData="disk")
        self.cache.addItem("不缓存（最省内存，最慢）", userData="none")
        self.cache.addItem("内存全缓存（最快，需大内存，大数据集慎用）", userData="memory")
        self.cache.setCurrentIndex(0)  # 默认磁盘缓存
        f.addRow("cache:", self.cache)
        cache_hint = QLabel(
            "磁盘缓存：首次训练把重采样后的体素写入项目盘 outputs/.cache/，\n"
            "后续 epoch/训练直接读缓存，不占内存也不重复计算。\n"
            "内存全缓存：全部缓存到 RAM，最快但需 ~120GB 内存（27GB 数据集会爆内存）。"
        )
        cache_hint.setStyleSheet("color: #555;")
        cache_hint.setWordWrap(True)
        f.addRow("", cache_hint)

        # ---- AMP 混合精度 ----
        # fp16 前向 + fp32 反向，GPU 有 Tensor Core 时显存减半、速度提升 1.5-2x
        self.chk_amp = QCheckBox("启用 AMP 混合精度（fp16，显存减半，速度提升 1.5-2x）")
        self.chk_amp.setChecked(True)  # 默认开启（GPU 训练时显著加速）
        f.addRow("AMP:", self.chk_amp)

        # ---- mode（任务模式）----
        # 显示文本 → 实际 task_mode 值；用户看到的 binary_liver / multi_class_liver_tumor
        # 对应 train.py 的 --task_mode binary_liver / multiclass
        self.mode = QComboBox()
        self.mode.addItem("multi_class_liver_tumor", userData="multiclass")
        self.mode.addItem("binary_liver", userData="binary_liver")
        f.addRow("mode:", self.mode)
        mode_hint = QLabel(
            "binary_liver：背景/肝脏 二分类，label>0→1，out_channels=2\n"
            "multi_class_liver_tumor：背景/肝脏/肿瘤 三分类，out_channels=3"
        )
        mode_hint.setStyleSheet("color: #555;")
        mode_hint.setWordWrap(True)
        f.addRow("", mode_hint)

        # ---- 按钮 ----
        btn_row = QHBoxLayout()
        self.btn_save_cfg = QPushButton("保存配置")
        self.btn_save_cfg.clicked.connect(self._on_save_config)
        self.btn_start = QPushButton("开始训练")
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop = QPushButton("停止训练")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        btn_row.addWidget(self.btn_save_cfg)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        f.addRow(btn_row)

        # ---- ckpt 显示 ----
        self.lbl_ckpt = QLabel("训练完成后将在此显示 best_model.pth 与 last_model.pth 路径。")
        self.lbl_ckpt.setStyleSheet("color: #555;")
        self.lbl_ckpt.setWordWrap(True)
        f.addRow("checkpoint:", self.lbl_ckpt)

        root.addWidget(box)
        root.addStretch(1)

    # ------------------------------------------------------ 保存配置
    def _on_save_config(self) -> None:
        """把 GUI 上的参数回写到 config YAML，保留原有注释与结构。
        采用"按 section + key 定位行并替换值"的方式，避免 PyYAML dump 丢注释。
        """
        cfg_path = self.config.text()
        if not cfg_path or not os.path.isfile(cfg_path):
            QMessageBox.warning(self, "配置路径无效", "请先选择有效的 config YAML 文件。")
            return
        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "读取失败", str(e))
            return

        mode = self.mode.currentData()  # "binary_liver" 或 "multiclass"
        num_classes = 2 if mode == "binary_liver" else 3
        roi_str = "[{}, {}, {}]".format(self.roi_x.value(), self.roi_y.value(), self.roi_z.value())
        # (section, key) -> 新值（字符串形式，直接写入 yaml）
        updates = {
            ("data", "task_mode"): str(mode),
            ("data", "num_classes"): str(num_classes),
            ("preprocessing", "roi_size"): roi_str,
            ("training", "max_epochs"): str(self.max_epochs.value()),
            ("training", "batch_size"): str(self.batch_size.value()),
            ("training", "lr"): "{:g}".format(self.lr.value()),
            ("training", "num_workers"): str(self.num_workers.value()),
            ("training", "val_interval"): str(self.val_interval.value()),
        }

        current_section = None
        new_lines = []
        section_pat = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*$")
        kv_pat = re.compile(r"^(\s+)([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$")
        applied = set()
        for line in lines:
            m_sec = section_pat.match(line)
            if m_sec:
                current_section = m_sec.group(1)
                new_lines.append(line)
                continue
            m_kv = kv_pat.match(line)
            if m_kv and current_section is not None:
                indent, key, _ = m_kv.group(1), m_kv.group(2), m_kv.group(3)
                if (current_section, key) in updates:
                    new_lines.append("{}{}: {}\n".format(indent, key, updates[(current_section, key)]))
                    applied.add((current_section, key))
                    continue
            new_lines.append(line)

        # 若某些 key 在 yaml 里不存在（如旧版无 task_mode），追加到对应 section 末尾
        # 简化处理：仅对 data.task_mode 做兜底，避免结构破坏
        missing = set(updates.keys()) - applied
        if missing:
            # 重新扫描，把缺失的 key 追加到对应 section 的最后一行之后
            self._append_missing_keys(new_lines, missing, updates)

        try:
            with open(cfg_path, "w", encoding="utf-8") as fh:
                fh.writelines(new_lines)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "写入失败", str(e))
            return
        msg = "配置已写入：{}\n更新字段：{}".format(
            cfg_path, ", ".join("{}:{}".format(s, k) for (s, k) in sorted(applied | missing))
        )
        self.log_signal.emit("[INFO] " + msg.replace("\n", " | "))
        QMessageBox.information(self, "保存成功", msg)

    @staticmethod
    def _append_missing_keys(lines: list, missing: set, updates: dict) -> None:
        """把缺失的 (section, key) 追加到对应 section 的最后一行之后。"""
        # 收集每个 section 需要补的 key
        by_section: dict = {}
        for (sec, key) in missing:
            by_section.setdefault(sec, []).append(key)
        # 找每个 section 的最后一行位置（最后一个以"  key:"开头的行，或 section 头本身）
        for sec, keys in by_section.items():
            insert_at = None
            for i, line in enumerate(lines):
                if re.match(r"^{}:\s*$".format(re.escape(sec)), line):
                    insert_at = i  # 至少找到 section 头
                # 更新到该 section 内的最后一行（缩进开头的非空行）
                if insert_at is not None and i > insert_at:
                    if re.match(r"^\S", line) and not line.startswith("#"):
                        # 进入下一个 section，回退一位
                        insert_at = i - 1
                        break
                    if re.match(r"^\s+\S", line):
                        insert_at = i
            if insert_at is None:
                continue
            # 在 insert_at+1 位置插入
            for k in keys:
                lines.insert(insert_at + 1, "  {}: {}\n".format(k, updates[(sec, k)]))
                insert_at += 1

    # ------------------------------------------------------ 开始训练
    def _on_start(self) -> None:
        cfg = self.config.text()
        tj = self.train_json.text()
        vj = self.val_json.text()
        od = self.output_dir.text() or os.path.join(self.project_root, "outputs", "models")
        if not self._require(cfg, tj, vj, hint="请填写 config / train.json / val.json"):
            return

        # 提醒：roi_size 与 val_interval 走 yaml，建议先点「保存配置」
        self.log_signal.emit(
            "[提示] roi_size / val_interval 通过 config YAML 生效；"
            "若改动过请先点「保存配置」。"
        )

        script = os.path.join(self.project_root, "src", "train.py")
        # 按需求构造命令：--config/--train_json/--val_json/--output_dir/
        #              --max_epochs/--batch_size/--lr/--num_workers/--device
        cmd = self._mk(script, [
            "--config", cfg,
            "--train_json", tj,
            "--val_json", vj,
            "--output_dir", od,
            "--max_epochs", "{}".format(self.max_epochs.value()),
            "--batch_size", "{}".format(self.batch_size.value()),
            "--lr", "{}".format(self.lr.value()),
            "--num_workers", "{}".format(self.num_workers.value()),
            "--cache", "{}".format(self.cache.currentData()),
        ])
        if self.chk_amp.isChecked():
            cmd.append("--amp")
        if self.device.currentText() != "auto":
            cmd += ["--device", self.device.currentText()]
        # 任务模式：binary_liver → --task_mode binary_liver
        #         multi_class_liver_tumor → --task_mode multiclass
        cmd += ["--task_mode", self.mode.currentData()]

        # 训练前清空 ckpt 提示
        self.lbl_ckpt.setText("训练进行中… 完成后显示 best_model.pth / last_model.pth 路径。")
        self.lbl_ckpt.setStyleSheet("color: #555;")
        self.request_run.emit(cmd)

    # ------------------------------------------------------ 停止训练
    def _on_stop(self) -> None:
        # 实际停止由 main_window 调用 worker.request_stop()
        self.request_stop.emit()

    # ------------------------------------------------------ 完成钩子
    def on_cmd_ok(self, cmd: list) -> None:
        """训练成功完成：显示 best_model.pth 与 last_model.pth 路径。"""
        output_dir = None
        for i, arg in enumerate(cmd):
            if arg == "--output_dir" and i + 1 < len(cmd):
                output_dir = cmd[i + 1]
                break
        if not output_dir:
            output_dir = self.output_dir.text() or os.path.join(self.project_root, "outputs", "models")
        best = os.path.join(output_dir, "best_model.pth")
        last = os.path.join(output_dir, "last_model.pth")
        best_flag = "OK" if os.path.isfile(best) else "MISSING"
        last_flag = "OK" if os.path.isfile(last) else "MISSING"
        self.lbl_ckpt.setText(
            "训练完成！\n"
            "  [{}] best_model: {}\n"
            "  [{}] last_model:  {}".format(best_flag, best, last_flag, last)
        )
        self.lbl_ckpt.setStyleSheet("color: #27ae60;")

    def on_cmd_fail(self, cmd: list) -> None:
        self.lbl_ckpt.setText("训练失败，请查看日志。")
        self.lbl_ckpt.setStyleSheet("color: #c0392b;")


# ============================================================================
# 4. Inference 模型推理
# ============================================================================

class InferencePage(_PageBase):
    """推理页：支持单图(--image_path)、批量(--image_dir)、测试集清单(--image_list)
    三种模式，选择 checkpoint / config / output_dir / device，调用 src/infer.py。
    推理结束后扫描 output_dir 列出预测 .nii.gz，提供「打开结果文件夹」按钮。
    """

    def __init__(self, project_root: str, parent=None):
        self.project_root = project_root
        super().__init__("Inference 模型推理", parent)

    def _build(self) -> None:
        from gui.workers import make_python_cmd
        self._mk = make_python_cmd

        root = QVBoxLayout(self)
        box = QGroupBox("推理配置（运行 src/infer.py）")
        f = QFormLayout(box)

        # ---- 输入模式 ----
        self.mode = QComboBox()
        self.mode.addItem("单个 nii.gz 文件 (--image_path)", userData="single")
        self.mode.addItem("图像文件夹批量 (--image_dir)", userData="batch")
        self.mode.addItem("测试集清单 JSON (--image_list)", userData="list")
        self.mode.currentIndexChanged.connect(self._on_mode_changed)
        f.addRow("输入模式:", self.mode)

        # ---- 图像输入（单图 / 批量 / 清单 三选一，按模式显示）----
        self.image_path = PathSelect("open", "选择 CT NIfTI", "NIfTI (*.nii *.nii.gz)")
        self.image_dir = PathSelect("dir", "选择 CT 图像目录")
        self.image_list = PathSelect("open", "选择测试集清单 JSON", "JSON (*.json)")
        self.image_list.line.setText(os.path.join(self.project_root, "data", "splits", "test.json"))
        f.addRow("image_path:", self.image_path)
        f.addRow("image_dir:", self.image_dir)
        f.addRow("image_list:", self.image_list)
        # 初始为单图模式，隐藏 image_dir 与 image_list
        self.image_dir.hide()
        self.image_list.hide()

        # ---- checkpoint / config / output_dir ----
        self.checkpoint = PathSelect("open", "选择 checkpoint", "Checkpoint (*.pth *.pt)")
        self.checkpoint.line.setText(os.path.join(self.project_root, "outputs", "models", "best_model.pth"))
        self.config = PathSelect("open", "选择 config YAML", "YAML (*.yaml *.yml)")
        self.config.line.setText(os.path.join(self.project_root, "configs", "liver_unet_3d.yaml"))
        self.output_dir = PathSelect("dir", "选择预测输出目录")
        self.output_dir.line.setText(os.path.join(self.project_root, "outputs", "predictions"))
        f.addRow("checkpoint:", self.checkpoint)
        f.addRow("config YAML:", self.config)
        f.addRow("output_dir:", self.output_dir)

        # ---- device ----
        self.device = QComboBox()
        self.device.addItems(["auto", "cuda", "cpu"])
        f.addRow("device:", self.device)

        # ---- 按钮 ----
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("开始推理")
        self.btn_start.clicked.connect(self._on_run)
        self.btn_open_out = QPushButton("打开预测结果文件夹")
        self.btn_open_out.clicked.connect(self._on_open_output)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_open_out)
        f.addRow(btn_row)

        # ---- 预测文件列表显示 ----
        self.lbl_result = QLabel("推理完成后将在此列出预测 .nii.gz 文件。")
        self.lbl_result.setStyleSheet("color: #555;")
        self.lbl_result.setWordWrap(True)
        self.lbl_result.setTextFormat(Qt.PlainText)
        f.addRow("预测结果:", self.lbl_result)

        root.addWidget(box)
        root.addStretch(1)

    # ------------------------------------------------------ 模式切换
    def _on_mode_changed(self, _idx: int) -> None:
        m = self.mode.currentData()
        # 三种输入控件按当前模式只显示一个
        self.image_path.setVisible(m == "single")
        self.image_dir.setVisible(m == "batch")
        self.image_list.setVisible(m == "list")
        # 切到「测试集清单」模式时，把默认输出目录改为 predictions_test，
        # 便于与普通批量推理的输出区分（避免覆盖 outputs/predictions）。
        if m == "list":
            cur = self.output_dir.text()
            default_pred = os.path.join(self.project_root, "outputs", "predictions")
            default_test = os.path.join(self.project_root, "outputs", "predictions_test")
            if not cur or os.path.normpath(cur) == os.path.normpath(default_pred):
                self.output_dir.line.setText(default_test)
        else:
            # 切回其它模式时，若仍是 predictions_test 则还原为 predictions
            cur = self.output_dir.text()
            default_test = os.path.join(self.project_root, "outputs", "predictions_test")
            default_pred = os.path.join(self.project_root, "outputs", "predictions")
            if cur and os.path.normpath(cur) == os.path.normpath(default_test):
                self.output_dir.line.setText(default_pred)

    # ------------------------------------------------------ 开始推理
    def _on_run(self) -> None:
        ck = self.checkpoint.text()
        cfg = self.config.text()
        od = self.output_dir.text() or os.path.join(self.project_root, "outputs", "predictions")
        if not self._require(ck, cfg, hint="请选择 checkpoint 与 config"):
            return

        script = os.path.join(self.project_root, "src", "infer.py")
        # 按需求构造命令骨架：--checkpoint/--output_dir/--config
        cmd = self._mk(script, [
            "--checkpoint", ck,
            "--output_dir", od,
            "--config", cfg,
        ])
        # 输入模式：单图传 --image_path，批量传 --image_dir，清单传 --image_list
        m = self.mode.currentData()
        if m == "single":
            ip = self.image_path.text()
            if not self._require(ip, hint="请选择单个图像文件"):
                return
            cmd += ["--image_path", ip]
        elif m == "batch":
            idir = self.image_dir.text()
            if not self._require(idir, hint="请选择图像目录"):
                return
            cmd += ["--image_dir", idir]
        else:  # list
            il = self.image_list.text()
            if not self._require(il, hint="请选择测试集清单 JSON"):
                return
            cmd += ["--image_list", il]
        # device：auto 时不传（让 infer.py 自动检测），cuda/cpu 显式传
        if self.device.currentText() != "auto":
            cmd += ["--device", self.device.currentText()]

        # 推理前清空旧结果
        self.lbl_result.setText("推理进行中… 完成后列出预测 .nii.gz 文件。")
        self.lbl_result.setStyleSheet("color: #555;")
        self.request_run.emit(cmd)

    # ------------------------------------------------------ 打开结果文件夹
    def _on_open_output(self) -> None:
        od = self.output_dir.text() or os.path.join(self.project_root, "outputs", "predictions")
        if not od or not os.path.isdir(od):
            QMessageBox.warning(self, "目录无效", f"输出目录不存在：{od}\n请先运行一次推理或手动选择。")
            return
        try:
            if os.name == "nt":
                os.startfile(od)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", od])
            else:
                subprocess.Popen(["xdg-open", od])
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "打开失败", str(e))

    # ------------------------------------------------------ 完成钩子
    def on_cmd_ok(self, cmd: list) -> None:
        """推理成功完成：扫描 output_dir 列出预测 .nii.gz 文件。"""
        output_dir = None
        for i, arg in enumerate(cmd):
            if arg == "--output_dir" and i + 1 < len(cmd):
                output_dir = cmd[i + 1]
                break
        if not output_dir:
            output_dir = self.output_dir.text() or os.path.join(self.project_root, "outputs", "predictions")
        self._show_predictions(output_dir)

    def on_cmd_fail(self, cmd: list) -> None:
        self.lbl_result.setText("推理失败，请查看日志。")
        self.lbl_result.setStyleSheet("color: #c0392b;")

    def _show_predictions(self, output_dir: str) -> None:
        """扫描 output_dir 下的 .nii.gz 文件并显示在 lbl_result。"""
        if not os.path.isdir(output_dir):
            self.lbl_result.setText(
                "推理完成，但输出目录不存在：{}\n请查看日志确认。".format(output_dir)
            )
            self.lbl_result.setStyleSheet("color: #c0392b;")
            return
        preds = sorted(
            f for f in os.listdir(output_dir)
            if f.lower().endswith(".nii.gz") or f.lower().endswith(".nii")
        )
        if not preds:
            self.lbl_result.setText(
                "推理完成，但未在输出目录找到预测文件：{}\n请查看日志确认。".format(output_dir)
            )
            self.lbl_result.setStyleSheet("color: #c0392b;")
            return
        lines = ["推理完成！共 {} 个预测文件：".format(len(preds))]
        for name in preds[:50]:  # 最多显示 50 个，避免过长
            lines.append("  - {}".format(os.path.join(output_dir, name)))
        if len(preds) > 50:
            lines.append("  ...（其余 {} 个略，请打开文件夹查看）".format(len(preds) - 50))
        self.lbl_result.setText("\n".join(lines))
        self.lbl_result.setStyleSheet("color: #27ae60;")


# ============================================================================
# 5. Evaluation 结果评估
# ============================================================================

class EvaluationPage(_PageBase):
    """评估页：选 pred_dir / 真值来源（目录 --label_dir 或 清单 --label_json）/
    output_csv，调 src/evaluate.py。
    评估完成后读取 CSV，在 GUI 显示 liver/tumor 的 Dice/IoU mean±std 摘要，
    用 QTableWidget 显示每个病例的指标，提供「打开 CSV」按钮。
    肿瘤 Dice 低时不报错，只给出小目标分割难度提示。
    """

    def __init__(self, project_root: str, parent=None):
        self.project_root = project_root
        super().__init__("Evaluation 结果评估", parent)

    def _build(self) -> None:
        from gui.workers import make_python_cmd
        self._mk = make_python_cmd

        root = QVBoxLayout(self)

        # ---- 评估配置区 ----
        box = QGroupBox("评估配置（运行 src/evaluate.py）")
        f = QFormLayout(box)
        self.pred_dir = PathSelect("dir", "选择预测目录")
        f.addRow("pred_dir:", self.pred_dir)

        # 真值来源：目录 (--label_dir) 或 清单 JSON (--label_json)，二选一
        self.label_mode = QComboBox()
        self.label_mode.addItem("真值目录 (--label_dir)", userData="dir")
        self.label_mode.addItem("测试集清单 JSON (--label_json)", userData="json")
        self.label_mode.currentIndexChanged.connect(self._on_label_mode_changed)
        f.addRow("真值来源:", self.label_mode)

        self.label_dir = PathSelect("dir", "选择真值目录")
        self.label_dir.line.setText(os.path.join(self.project_root, "data", "raw", "Task03_Liver", "labelsTr"))
        self.label_json = PathSelect("open", "选择测试集清单 JSON", "JSON (*.json)")
        self.label_json.line.setText(os.path.join(self.project_root, "data", "splits", "test.json"))
        f.addRow("label_dir:", self.label_dir)
        f.addRow("label_json:", self.label_json)
        # 初始为「目录」模式，隐藏 label_json
        self.label_json.hide()

        self.output_csv = PathSelect("save", "保存为 CSV", "CSV (*.csv)")
        self.output_csv.line.setText(os.path.join(self.project_root, "outputs", "evaluation.csv"))
        f.addRow("output_csv:", self.output_csv)

        # ---- 按钮区 ----
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("开始评估")
        self.btn_start.clicked.connect(self._on_run)
        self.btn_open_csv = QPushButton("打开 CSV")
        self.btn_open_csv.clicked.connect(self._on_open_csv)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_open_csv)
        f.addRow(btn_row)

        root.addWidget(box)

        # ---- 摘要区 ----
        box_summary = QGroupBox("摘要 (mean ± std)")
        vs = QVBoxLayout(box_summary)
        self.lbl_summary = QLabel("评估完成后将在此显示 liver/tumor 的 Dice/IoU 摘要。")
        self.lbl_summary.setStyleSheet("color: #555;")
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setTextFormat(Qt.PlainText)
        vs.addWidget(self.lbl_summary)
        root.addWidget(box_summary)

        # ---- 病例表格区 ----
        box_table = QGroupBox("各病例指标（QTableWidget）")
        vt = QVBoxLayout(box_table)
        self.table = QTableWidget(0, 0)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.table.verticalHeader().setVisible(False)
        vt.addWidget(self.table)
        root.addWidget(box_table)

        root.addStretch(1)

    # ------------------------------------------------------ 真值来源切换
    def _on_label_mode_changed(self, _idx: int) -> None:
        is_json = self.label_mode.currentData() == "json"
        self.label_dir.setVisible(not is_json)
        self.label_json.setVisible(is_json)
        # 切到「清单」模式时，把默认 CSV 改为 evaluation_test.csv，
        # 便于与目录模式的 evaluation.csv 区分。
        if is_json:
            cur = self.output_csv.text()
            default_csv = os.path.join(self.project_root, "outputs", "evaluation.csv")
            default_test = os.path.join(self.project_root, "outputs", "evaluation_test.csv")
            if not cur or os.path.normpath(cur) == os.path.normpath(default_csv):
                self.output_csv.line.setText(default_test)
        else:
            cur = self.output_csv.text()
            default_csv = os.path.join(self.project_root, "outputs", "evaluation.csv")
            default_test = os.path.join(self.project_root, "outputs", "evaluation_test.csv")
            if cur and os.path.normpath(cur) == os.path.normpath(default_test):
                self.output_csv.line.setText(default_csv)

    # ------------------------------------------------------ 开始评估
    def _on_run(self) -> None:
        p = self.pred_dir.text()
        o = self.output_csv.text() or os.path.join(self.project_root, "outputs", "evaluation.csv")
        if not self._require(p, hint="请选择 pred_dir"):
            return
        script = os.path.join(self.project_root, "src", "evaluate.py")
        # 真值来源：目录传 --label_dir，清单传 --label_json（与 src/evaluate.py 参数对应）
        if self.label_mode.currentData() == "json":
            lj = self.label_json.text()
            if not self._require(lj, hint="请选择测试集清单 JSON"):
                return
            cmd = self._mk(script, ["--pred_dir", p, "--label_json", lj, "--output_csv", o])
        else:
            l = self.label_dir.text()
            if not self._require(l, hint="请选择 label_dir"):
                return
            cmd = self._mk(script, ["--pred_dir", p, "--label_dir", l, "--output_csv", o])
        # 评估前清空旧显示
        self.lbl_summary.setText("评估进行中… 完成后显示 liver/tumor Dice/IoU 摘要。")
        self.lbl_summary.setStyleSheet("color: #555;")
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.request_run.emit(cmd)

    # ------------------------------------------------------ 打开 CSV
    def _on_open_csv(self) -> None:
        csv_path = self.output_csv.text() or os.path.join(self.project_root, "outputs", "evaluation.csv")
        if not csv_path or not os.path.isfile(csv_path):
            QMessageBox.warning(self, "文件不存在", f"CSV 文件不存在：{csv_path}\n请先运行一次评估。")
            return
        try:
            if os.name == "nt":
                os.startfile(csv_path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", csv_path])
            else:
                subprocess.Popen(["xdg-open", csv_path])
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "打开失败", str(e))

    # ------------------------------------------------------ 完成钩子
    def on_cmd_ok(self, cmd: list) -> None:
        """评估成功完成：读取 output_csv 并填充摘要与表格。"""
        csv_path = None
        for i, arg in enumerate(cmd):
            if arg == "--output_csv" and i + 1 < len(cmd):
                csv_path = cmd[i + 1]
                break
        if not csv_path:
            csv_path = self.output_csv.text() or os.path.join(self.project_root, "outputs", "evaluation.csv")
        self._load_csv(csv_path)

    def on_cmd_fail(self, cmd: list) -> None:
        self.lbl_summary.setText("评估失败，请查看日志。")
        self.lbl_summary.setStyleSheet("color: #c0392b;")

    # ------------------------------------------------------ 读取 CSV
    def _load_csv(self, csv_path: str) -> None:
        """读取 evaluate.py 输出的 CSV，提取 MEAN/STD 行 + 各病例行。
        CSV 列：case, liver_dice, liver_iou, liver_precision, liver_recall,
                tumor_dice, tumor_iou, tumor_precision, tumor_recall
        最后两行 case 列为 MEAN / STD。
        """
        if not os.path.isfile(csv_path):
            self.lbl_summary.setText("评估完成，但未找到 CSV：{}".format(csv_path))
            self.lbl_summary.setStyleSheet("color: #c0392b;")
            return
        try:
            with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
                fieldnames = reader.fieldnames or []
        except Exception as e:  # noqa: BLE001
            self.lbl_summary.setText("读取 CSV 失败：{}".format(e))
            self.lbl_summary.setStyleSheet("color: #c0392b;")
            return
        if not rows:
            self.lbl_summary.setText("CSV 为空，请查看日志。")
            self.lbl_summary.setStyleSheet("color: #c0392b;")
            return

        # 分离 MEAN / STD 与病例行
        mean_row = None
        std_row = None
        case_rows = []
        for r in rows:
            case = (r.get("case") or "").strip()
            if case.upper() == "MEAN":
                mean_row = r
            elif case.upper() == "STD":
                std_row = r
            else:
                case_rows.append(r)

        # ---- 摘要 ----
        def fmt(v):
            try:
                return "{:.4f}".format(float(v))
            except (TypeError, ValueError):
                return "N/A"

        if mean_row is not None and std_row is not None:
            summary_lines = [
                "liver Dice : {} ± {}".format(fmt(mean_row.get("liver_dice")), fmt(std_row.get("liver_dice"))),
                "liver IoU  : {} ± {}".format(fmt(mean_row.get("liver_iou")),  fmt(std_row.get("liver_iou"))),
                "tumor Dice : {} ± {}".format(fmt(mean_row.get("tumor_dice")), fmt(std_row.get("tumor_dice"))),
                "tumor IoU  : {} ± {}".format(fmt(mean_row.get("tumor_iou")),  fmt(std_row.get("tumor_iou"))),
            ]
        else:
            # 没有 MEAN/STD 行时，从病例行直接计算（兼容旧 CSV）
            import statistics
            def col_stat(col):
                vals = []
                for r in case_rows:
                    try:
                        vals.append(float(r.get(col)))
                    except (TypeError, ValueError):
                        pass
                if not vals:
                    return ("N/A", "N/A")
                m = sum(vals) / len(vals)
                s = statistics.stdev(vals) if len(vals) >= 2 else 0.0
                return ("{:.4f}".format(m), "{:.4f}".format(s))
            ld_m, ld_s = col_stat("liver_dice")
            li_m, li_s = col_stat("liver_iou")
            td_m, td_s = col_stat("tumor_dice")
            ti_m, ti_s = col_stat("tumor_iou")
            summary_lines = [
                "liver Dice : {} ± {}".format(ld_m, ld_s),
                "liver IoU  : {} ± {}".format(li_m, li_s),
                "tumor Dice : {} ± {}".format(td_m, td_s),
                "tumor IoU  : {} ± {}".format(ti_m, ti_s),
            ]

        # 肿瘤 Dice 很低时附提示（< 0.1）
        tumor_dice_mean = None
        if mean_row is not None:
            try:
                tumor_dice_mean = float(mean_row.get("tumor_dice"))
            except (TypeError, ValueError):
                tumor_dice_mean = None
        if tumor_dice_mean is not None and tumor_dice_mean < 0.1:
            summary_lines.append(
                "提示：肿瘤小目标分割难度较高，低 Dice 可能与病灶体积小和类别不平衡有关。"
            )
            self.lbl_summary.setStyleSheet("color: #2980b9;")  # 蓝色提示，非错误
        else:
            self.lbl_summary.setStyleSheet("color: #27ae60;")
        self.lbl_summary.setText("\n".join(summary_lines))
        self.log_signal.emit("[INFO] 评估摘要已更新；CSV: {}".format(csv_path))

        # ---- 表格 ----
        self._fill_table(fieldnames, case_rows, mean_row, std_row)

    def _fill_table(self, fieldnames: list, case_rows: list,
                    mean_row: Optional[dict], std_row: Optional[dict]) -> None:
        """用 QTableWidget 显示各病例指标，末尾附 MEAN / STD 行。"""
        # 用 CSV 的 fieldnames 作为列；若无则用默认列
        cols = list(fieldnames)
        if not cols:
            cols = ["case", "liver_dice", "liver_iou", "liver_precision", "liver_recall",
                    "tumor_dice", "tumor_iou", "tumor_precision", "tumor_recall"]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        # 行：病例 + MEAN + STD
        n_rows = len(case_rows) + (1 if mean_row else 0) + (1 if std_row else 0)
        self.table.setRowCount(n_rows)
        r = 0
        for row in case_rows:
            for c, col in enumerate(cols):
                self.table.setItem(r, c, QTableWidgetItem(str(row.get(col, ""))))
            r += 1
        # MEAN 行
        if mean_row is not None:
            for c, col in enumerate(cols):
                item = QTableWidgetItem(str(mean_row.get(col, "")))
                item.setBackground(Qt.lightGray)
                self.table.setItem(r, c, item)
            r += 1
        # STD 行
        if std_row is not None:
            for c, col in enumerate(cols):
                item = QTableWidgetItem(str(std_row.get(col, "")))
                item.setBackground(Qt.lightGray)
                self.table.setItem(r, c, item)
            r += 1
        # 列宽自适应
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.resizeRowsToContents()


# ============================================================================
# 6. Visualization 可视化
# ============================================================================

class VisualizationPage(_PageBase):
    """可视化页：选 image_path / label_path(可选) / pred_path / output_dir，
    调 src/visualize.py 生成 PNG（GUI 不直接处理 nii.gz）。
    运行完成后扫描 output_dir/figures 下的 PNG，显示缩略图列表；
    点击缩略图弹放大窗口；提供「打开图片文件夹」按钮。
    """

    THUMB_SIZE = 220  # 缩略图边长（像素）

    def __init__(self, project_root: str, parent=None):
        self.project_root = project_root
        super().__init__("Visualization 可视化", parent)

    def _build(self) -> None:
        from gui.workers import make_python_cmd
        self._mk = make_python_cmd

        root = QVBoxLayout(self)

        # ---- 配置区 ----
        box = QGroupBox("可视化配置（运行 src/visualize.py；不在 GUI 内直接处理 nii.gz）")
        f = QFormLayout(box)
        self.image_path = PathSelect("open", "选择 CT NIfTI", "NIfTI (*.nii *.nii.gz)")
        self.label_path = PathSelect("open", "选择 label NIfTI（可选）", "NIfTI (*.nii *.nii.gz)")
        self.pred_path = PathSelect("open", "选择预测 NIfTI", "NIfTI (*.nii *.nii.gz)")
        self.output_dir = PathSelect("dir", "选择输出目录")
        self.output_dir.line.setText(os.path.join(self.project_root, "outputs"))
        f.addRow("image_path:", self.image_path)
        f.addRow("label_path（可选）:", self.label_path)
        f.addRow("pred_path:", self.pred_path)
        f.addRow("output_dir:", self.output_dir)

        self.num_slices = QSpinBox()
        self.num_slices.setRange(1, 50)
        self.num_slices.setValue(4)
        f.addRow("num_slices:", self.num_slices)

        # ---- 按钮 ----
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("生成可视化")
        self.btn_start.clicked.connect(self._on_run)
        self.btn_auto_label = QPushButton("自动匹配 label")
        self.btn_auto_label.clicked.connect(self._auto_match_label)
        self.btn_open_dir = QPushButton("打开图片文件夹")
        self.btn_open_dir.clicked.connect(self._on_open_dir)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_auto_label)
        btn_row.addWidget(self.btn_open_dir)
        f.addRow(btn_row)

        root.addWidget(box)

        # ---- 缩略图区（可滚动）----
        box_thumbs = QGroupBox("缩略图列表（点击放大）")
        vt = QVBoxLayout(box_thumbs)
        self.lbl_thumb_status = QLabel("运行完成后将在此显示 PNG 缩略图。")
        self.lbl_thumb_status.setStyleSheet("color: #555;")
        vt.addWidget(self.lbl_thumb_status)
        # 用 QScrollArea + 内部 QWidget + QGridLayout 放缩略图
        self._thumb_container = QWidget()
        self._thumb_grid = QGridLayout(self._thumb_container)
        self._thumb_grid.setSpacing(8)
        self._thumb_grid.setContentsMargins(4, 4, 4, 4)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._thumb_container)
        vt.addWidget(scroll)
        root.addWidget(box_thumbs, 1)

        self._thumb_paths: list = []  # 缩略图对应的完整路径

    # ------------------------------------------------------ 生成可视化
    def _on_run(self) -> None:
        ip = self.image_path.text()
        pp = self.pred_path.text()
        od = self.output_dir.text() or os.path.join(self.project_root, "outputs")
        if not self._require(ip, pp, hint="请选择 image_path 与 pred_path"):
            return
        # label_path 防呆校验：不通过则禁止运行
        lp = self.label_path.text()
        if lp:
            if not self._validate_label_path(ip, lp):
                return

        script = os.path.join(self.project_root, "src", "visualize.py")
        # 按需求构造命令骨架：--image_path/--pred_path/--output_dir
        cmd = self._mk(script, [
            "--image_path", ip,
            "--pred_path", pp,
            "--output_dir", od,
            "--num_slices", "{}".format(self.num_slices.value()),
        ])
        # label_path 可选：为空则不传
        if lp:
            cmd += ["--label_path", lp]

        # 清空旧缩略图
        self._clear_thumbs()
        self.lbl_thumb_status.setText("可视化生成中… 完成后显示缩略图。")
        self.lbl_thumb_status.setStyleSheet("color: #555;")
        self.request_run.emit(cmd)

    # ------------------------------------------------------ label_path 防呆校验
    def _validate_label_path(self, image_path: str, label_path: str) -> bool:
        """校验 label_path 是否为合法分割标签，不合法返回 False。

        校验规则（按用户要求）：
            1) label_path == image_path：报错并禁止运行。
            2) label_path 含 imagesTr：警告疑似选成 CT 图像，禁止运行。
            3) label_path 不含 labelsTr：弹窗请用户确认。
            4) 读取 NIfTI 检查 unique values 是否 ⊆ {0,1,2}：
               不在范围内 → 警告「不像分割标签」，禁止运行。

        Args:
            image_path: CT 图像路径。
            label_path: 用户填入的 label 路径。

        Returns:
            bool: True 通过可继续运行；False 不通过已弹窗提示。
        """
        # 归一化斜杠便于路径片段比较
        ip_norm = image_path.replace("\\", "/")
        lp_norm = label_path.replace("\\", "/")

        # 1) label_path == image_path
        if os.path.abspath(ip_norm) == os.path.abspath(lp_norm):
            QMessageBox.critical(
                self, "label_path 错误",
                "label_path 与 image_path 相同，请重新选择真值标签（应位于 labelsTr）。",
            )
            return False

        # 2) label_path 位于 imagesTr
        if "imagesTr" in lp_norm:
            QMessageBox.critical(
                self, "label_path 错误",
                "当前 label_path 位于 imagesTr，疑似选择了 CT 图像。\n"
                "真值标签应位于 labelsTr。\n请重新选择 labelsTr 下对应文件。",
            )
            return False

        # 3) label_path 不含 labelsTr：请用户确认
        if "labelsTr" not in lp_norm:
            ret = QMessageBox.warning(
                self, "label_path 确认",
                "当前 label_path 不在 labelsTr 目录下，可能不是标准真值标签。\n"
                "确定要继续吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if ret != QMessageBox.Yes:
                return False

        # 4) 读取 NIfTI 检查 unique values
        if not os.path.isfile(label_path):
            QMessageBox.critical(self, "label_path 错误", "label_path 文件不存在。")
            return False
        try:
            import nibabel as nib
            import numpy as np
            data = np.asanyarray(nib.load(label_path).dataobj)
            uniq = np.unique(data)
            # 合法标签 unique 应 ⊆ {0,1,2}（含 int 与浮点表示）
            allowed = {0, 1, 2, 0.0, 1.0, 2.0}
            if not set(uniq.tolist()).issubset(allowed):
                QMessageBox.critical(
                    self, "label_path 错误",
                    "当前文件不像分割标签（unique 不在 {{0,1,2}} 范围内），"
                    "可能是 CT 图像或错误文件。\n"
                    "实际 unique 值（前 10 个）：{}".format(uniq[:10].tolist()),
                )
                return False
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "label_path 错误", f"读取 label NIfTI 失败：{e}")
            return False
        return True

    # ------------------------------------------------------ 自动匹配 label
    def _auto_match_label(self) -> None:
        """根据 image_path 自动寻找 labelsTr 下同名文件并填入 label_path。

        规则：把 image_path 中的 'imagesTr' 片段替换为 'labelsTr'。
        若替换后文件存在则填入；否则弹窗提示。
        """
        ip = self.image_path.text()
        if not ip:
            QMessageBox.warning(self, "未选 image_path", "请先选择 image_path（CT NIfTI）。")
            return
        ip_norm = ip.replace("\\", "/")
        if "imagesTr" not in ip_norm:
            QMessageBox.warning(
                self, "无法自动匹配",
                "image_path 中未包含 imagesTr，无法自动替换为 labelsTr。\n"
                "请手动选择 label_path。",
            )
            return
        lp_norm = ip_norm.replace("imagesTr", "labelsTr", 1)
        # 还原为系统原生路径分隔
        lp = os.path.normpath(lp_norm)
        if not os.path.isfile(lp):
            QMessageBox.warning(
                self, "自动匹配失败",
                f"自动推导的 label 路径不存在：\n{lp}\n请手动选择 label_path。",
            )
            return
        self.label_path.line.setText(lp)
        self.log_signal.emit(f"[INFO] 自动匹配 label：{lp}")
        QMessageBox.information(self, "自动匹配成功", f"label_path 已填入：\n{lp}")

    # ------------------------------------------------------ 打开图片文件夹
    def _on_open_dir(self) -> None:
        od = self.output_dir.text() or os.path.join(self.project_root, "outputs")
        # visualize.py 把 PNG 写到 <output_dir>/figures
        fig_dir = od if os.path.basename(od) == "figures" else os.path.join(od, "figures")
        if not os.path.isdir(fig_dir):
            # 退而求其次，打开 output_dir 本身
            fig_dir = od if os.path.isdir(od) else None
        if not fig_dir:
            QMessageBox.warning(self, "目录无效", f"输出目录不存在：{od}\n请先运行一次可视化或手动选择。")
            return
        try:
            if os.name == "nt":
                os.startfile(fig_dir)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", fig_dir])
            else:
                subprocess.Popen(["xdg-open", fig_dir])
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "打开失败", str(e))

    # ------------------------------------------------------ 完成钩子
    def on_cmd_ok(self, cmd: list) -> None:
        """可视化成功完成：扫描 output_dir/figures 下的 PNG 显示缩略图。"""
        output_dir = None
        for i, arg in enumerate(cmd):
            if arg == "--output_dir" and i + 1 < len(cmd):
                output_dir = cmd[i + 1]
                break
        if not output_dir:
            output_dir = self.output_dir.text() or os.path.join(self.project_root, "outputs")
        self._show_thumbnails(output_dir)

    def on_cmd_fail(self, cmd: list) -> None:
        self.lbl_thumb_status.setText("可视化失败，请查看日志。")
        self.lbl_thumb_status.setStyleSheet("color: #c0392b;")

    # ------------------------------------------------------ 缩略图显示
    def _show_thumbnails(self, output_dir: str) -> None:
        """扫描 output_dir/figures 下的 PNG 文件，用 QGridLayout 显示缩略图。
        点击缩略图调用 _show_full_image 弹放大窗口。
        """
        # 先清空旧缩略图（无论后续成功/失败，都应清掉旧图）
        self._clear_thumbs()
        # visualize.py 把 PNG 写到 <output_dir>/figures
        fig_dir = output_dir if os.path.basename(output_dir) == "figures" \
            else os.path.join(output_dir, "figures")
        if not os.path.isdir(fig_dir):
            self.lbl_thumb_status.setText(
                "可视化完成，但未找到图片目录：{}\n请查看日志确认。".format(fig_dir)
            )
            self.lbl_thumb_status.setStyleSheet("color: #c0392b;")
            return
        pngs = sorted(
            f for f in os.listdir(fig_dir)
            if f.lower().endswith(".png")
        )
        if not pngs:
            self.lbl_thumb_status.setText(
                "可视化完成，但未在 {} 下找到 PNG 图片。".format(fig_dir)
            )
            self.lbl_thumb_status.setStyleSheet("color: #c0392b;")
            return

        self._thumb_paths = []
        # 每行 4 列
        cols = 4
        for idx, name in enumerate(pngs):
            full = os.path.join(fig_dir, name)
            self._thumb_paths.append(full)
            row, col = divmod(idx, cols)
            thumb = _ThumbLabel(full, self.THUMB_SIZE)
            thumb.clicked_path.connect(self._show_full_image)
            self._thumb_grid.addWidget(thumb, row, col)

        self.lbl_thumb_status.setText(
            "可视化完成！共 {} 张 PNG（点击放大）。目录：{}".format(len(pngs), fig_dir)
        )
        self.lbl_thumb_status.setStyleSheet("color: #27ae60;")
        self.log_signal.emit("[INFO] 缩略图已加载 {} 张，目录：{}".format(len(pngs), fig_dir))

    def _clear_thumbs(self) -> None:
        """清空缩略图网格。"""
        while self._thumb_grid.count():
            item = self._thumb_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._thumb_paths = []

    def _show_full_image(self, path: str) -> None:
        """弹窗显示完整尺寸 PNG。"""
        dlg = QDialog(self)
        dlg.setWindowTitle(os.path.basename(path))
        dlg.setModal(True)
        layout = QVBoxLayout(dlg)
        lbl = QLabel()
        pix = QPixmap(path)
        if pix.isNull():
            lbl.setText("无法加载图片：{}".format(path))
        else:
            # 限制不超过屏幕的 80%，保持比例
            screen = self.screen().availableGeometry() if self.screen() else None
            max_w = int(screen.width() * 0.8) if screen else 1280
            max_h = int(screen.height() * 0.8) if screen else 800
            if pix.width() > max_w or pix.height() > max_h:
                pix = pix.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            lbl.setPixmap(pix)
        layout.addWidget(lbl)
        dlg.setLayout(layout)
        dlg.exec()


class _ThumbLabel(QLabel):
    """可点击的缩略图 QLabel。点击后发出 clicked_path 信号。"""

    clicked_path = Signal(str)

    def __init__(self, image_path: str, thumb_size: int, parent=None):
        super().__init__(parent)
        self._image_path = image_path
        self.setFixedSize(thumb_size, thumb_size + 20)  # 底部留 20px 给文件名
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("QLabel { border: 1px solid #ccc; }"
                           "QLabel:hover { border: 2px solid #2980b9; }")
        pix = QPixmap(image_path)
        if pix.isNull():
            self.setText("无法加载")
        else:
            thumb = pix.scaled(thumb_size, thumb_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(thumb)
        self.setToolTip(os.path.basename(image_path))
        # 把光标设为手型，提示可点击
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self.clicked_path.emit(self._image_path)
        super().mousePressEvent(event)


# ============================================================================
# 7. STL Export 三维导出
# ============================================================================

class StlExportPage(_PageBase):
    """STL 导出页：选 mask_path / output_dir，勾选 liver(1)/tumor(2)，
    设置 smooth / min_component_size，调 src/export_stl.py 生成 STL。
    导出完成后扫描 output_dir 下的 STL 文件并显示路径；提供「打开 STL 文件夹」按钮。
    页面顶部显示醒目的临床免责声明。
    """

    def __init__(self, project_root: str, parent=None):
        self.project_root = project_root
        super().__init__("STL Export 三维导出", parent)

    def _build(self) -> None:
        from gui.workers import make_python_cmd
        self._mk = make_python_cmd

        root = QVBoxLayout(self)

        # ---- 醒目免责声明（置顶）----
        warn = QLabel(
            "⚠ 免责声明：导出的 STL 仅用于科研学习、教学展示和三维可视化，\n"
            "不可直接用于临床诊断、治疗决策或手术导航。"
        )
        warn.setStyleSheet(
            "color: #c0392b; font-weight: bold; background-color: #fdecea;"
            "padding: 8px; border: 1px solid #c0392b; border-radius: 4px;"
        )
        warn.setWordWrap(True)
        root.addWidget(warn)

        # ---- 配置区 ----
        box = QGroupBox("STL 导出配置（运行 src/export_stl.py）")
        f = QFormLayout(box)
        self.mask_path = PathSelect("open", "选择预测 mask .nii.gz", "NIfTI (*.nii *.nii.gz)")
        self.output_dir = PathSelect("dir", "选择 STL 输出目录")
        self.output_dir.line.setText(os.path.join(self.project_root, "outputs", "stl"))
        f.addRow("mask_path:", self.mask_path)
        f.addRow("output_dir:", self.output_dir)

        # ---- 勾选导出标签：liver(1) / tumor(2) ----
        row_labels = QHBoxLayout()
        self.chk_liver = QCheckBox("liver (label=1)")
        self.chk_liver.setChecked(True)
        self.chk_tumor = QCheckBox("tumor (label=2)")
        self.chk_tumor.setChecked(True)
        row_labels.addWidget(self.chk_liver)
        row_labels.addWidget(self.chk_tumor)
        row_labels.addStretch(1)
        f.addRow("导出标签:", row_labels)

        # ---- smooth / min_component_size ----
        self.smooth = QCheckBox("smooth（开启 Taubin 平滑，表面更光滑）")
        f.addRow("smooth:", self.smooth)
        self.min_comp = QSpinBox()
        self.min_comp.setRange(0, 1000000)
        self.min_comp.setValue(100)
        f.addRow("min_component_size:", self.min_comp)

        # ---- 按钮 ----
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("导出 STL")
        self.btn_start.clicked.connect(self._on_run)
        self.btn_open_dir = QPushButton("打开 STL 文件夹")
        self.btn_open_dir.clicked.connect(self._on_open_dir)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_open_dir)
        f.addRow(btn_row)

        root.addWidget(box)

        # ---- 结果显示区 ----
        box_result = QGroupBox("导出结果")
        vr = QVBoxLayout(box_result)
        self.lbl_result = QLabel("导出完成后将在此显示生成的 STL 文件路径。")
        self.lbl_result.setStyleSheet("color: #555;")
        self.lbl_result.setWordWrap(True)
        self.lbl_result.setTextFormat(Qt.PlainText)
        vr.addWidget(self.lbl_result)
        root.addWidget(box_result)

        root.addStretch(1)

    # ------------------------------------------------------ 导出 STL
    def _on_run(self) -> None:
        m = self.mask_path.text()
        od = self.output_dir.text() or os.path.join(self.project_root, "outputs", "stl")
        if not self._require(m, hint="请选择 mask_path"):
            return
        # 勾选标签 -> --labels 列表
        labels = []
        if self.chk_liver.isChecked():
            labels.append("1")
        if self.chk_tumor.isChecked():
            labels.append("2")
        if not labels:
            QMessageBox.warning(self, "标签为空", "请至少勾选 liver 或 tumor 之一。")
            return
        script = os.path.join(self.project_root, "src", "export_stl.py")
        # 按需求构造命令：--mask_path/--output_dir/--labels/--min_component_size
        cmd = self._mk(script, [
            "--mask_path", m,
            "--output_dir", od,
            "--labels", *labels,
            "--min_component_size", "{}".format(self.min_comp.value()),
        ])
        # smooth 开启时附加 --smooth
        if self.smooth.isChecked():
            cmd.append("--smooth")

        # 导出前清空旧结果
        self.lbl_result.setText("STL 导出进行中… 完成后显示生成的 STL 文件路径。")
        self.lbl_result.setStyleSheet("color: #555;")
        self.request_run.emit(cmd)

    # ------------------------------------------------------ 打开 STL 文件夹
    def _on_open_dir(self) -> None:
        od = self.output_dir.text() or os.path.join(self.project_root, "outputs", "stl")
        if not od or not os.path.isdir(od):
            QMessageBox.warning(self, "目录无效", f"输出目录不存在：{od}\n请先运行一次导出或手动选择。")
            return
        try:
            if os.name == "nt":
                os.startfile(od)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", od])
            else:
                subprocess.Popen(["xdg-open", od])
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "打开失败", str(e))

    # ------------------------------------------------------ 完成钩子
    def on_cmd_ok(self, cmd: list) -> None:
        """导出成功完成：扫描 output_dir 下的 STL 文件并显示路径。"""
        output_dir = None
        for i, arg in enumerate(cmd):
            if arg == "--output_dir" and i + 1 < len(cmd):
                output_dir = cmd[i + 1]
                break
        if not output_dir:
            output_dir = self.output_dir.text() or os.path.join(self.project_root, "outputs", "stl")
        self._show_stl_files(output_dir)

    def on_cmd_fail(self, cmd: list) -> None:
        self.lbl_result.setText("STL 导出失败，请查看日志。")
        self.lbl_result.setStyleSheet("color: #c0392b;")

    # ------------------------------------------------------ 扫描 STL 文件
    def _show_stl_files(self, output_dir: str) -> None:
        """扫描 output_dir 下的 .stl 文件并显示在 lbl_result。"""
        if not os.path.isdir(output_dir):
            self.lbl_result.setText(
                "导出完成，但输出目录不存在：{}\n请查看日志确认。".format(output_dir)
            )
            self.lbl_result.setStyleSheet("color: #c0392b;")
            return
        stls = sorted(
            f for f in os.listdir(output_dir)
            if f.lower().endswith(".stl")
        )
        if not stls:
            self.lbl_result.setText(
                "导出完成，但未在输出目录找到 STL 文件：{}\n请查看日志确认（可能 mask 中无对应标签体素）。".format(output_dir)
            )
            self.lbl_result.setStyleSheet("color: #c0392b;")
            return
        lines = ["导出完成！共 {} 个 STL 文件：".format(len(stls))]
        for name in stls:
            lines.append("  - {}".format(os.path.join(output_dir, name)))
        self.lbl_result.setText("\n".join(lines))
        self.lbl_result.setStyleSheet("color: #27ae60;")
        self.log_signal.emit("[INFO] STL 导出完成，共 {} 个文件，目录：{}".format(len(stls), output_dir))


# ============================================================================
# 8. Benchmark 速度对比页
# ============================================================================

class BenchmarkPage(_PageBase):
    """速度优化对比页：调 scripts/benchmark_optimizations.py 自动跑 baseline vs
    optimized 短程对比实验。后台子进程执行，实时日志进 Logs 页，运行中可停止。

    参数面板对应脚本参数：--config / --train_json / --val_json / --max_epochs /
    --batch_size / --device / --cache / --output_dir。
    完成后显示 benchmark_results.md / .csv / _summary.txt 路径，提供「打开结果目录」按钮。
    """

    # 复用 main_window 的通用停止机制（与 TrainingPage / DatasetPage 一致）
    request_stop = Signal()

    def __init__(self, project_root: str, parent=None):
        self.project_root = project_root
        super().__init__("Benchmark 速度对比", parent)

    def _build(self) -> None:
        from gui.workers import make_python_cmd
        self._mk = make_python_cmd

        root = QVBoxLayout(self)

        # ---- 说明区 ----
        hint = QLabel(
            "对比 baseline（关闭全部优化）与 optimized（开启 AMP / OneCycleLR / 梯度裁剪 / "
            "pin_memory / persistent_workers / non_blocking / 验证 AMP）两种训练设置的 "
            "速度、显存与初步精度。\n\n"
            "说明：短程 10 epoch benchmark 主要用于速度比较，不代表最终模型性能；"
            "无 GPU 时 peak_gpu_memory_MB 记为 NaN。"
        )
        hint.setStyleSheet("color: #555;")
        hint.setWordWrap(True)
        root.addWidget(hint)

        # ---- 参数区 ----
        box = QGroupBox("对比实验配置（运行 scripts/benchmark_optimizations.py）")
        f = QFormLayout(box)

        self.config = PathSelect("open", "选择 config YAML", "YAML (*.yaml *.yml)")
        self.config.line.setText(os.path.join(self.project_root, "configs", "liver_unet_3d.yaml"))
        f.addRow("config:", self.config)

        self.train_json = PathSelect("open", "选择训练集 JSON", "JSON (*.json)")
        self.train_json.line.setText(os.path.join(self.project_root, "data", "splits", "train.json"))
        f.addRow("train_json:", self.train_json)

        self.val_json = PathSelect("open", "选择验证集 JSON", "JSON (*.json)")
        self.val_json.line.setText(os.path.join(self.project_root, "data", "splits", "val.json"))
        f.addRow("val_json:", self.val_json)

        # max_epochs / batch_size
        row_ep = QHBoxLayout()
        self.max_epochs = QSpinBox()
        self.max_epochs.setRange(1, 1000)
        self.max_epochs.setValue(10)
        row_ep.addWidget(self.max_epochs)
        row_ep.addWidget(QLabel("（短程默认 10，仅用于速度对比）"))
        f.addRow("max_epochs:", row_ep)

        row_bs = QHBoxLayout()
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 64)
        self.batch_size.setValue(1)
        row_bs.addWidget(self.batch_size)
        f.addRow("batch_size:", row_bs)

        # device：auto / cuda / cpu
        self.device = QComboBox()
        self.device.addItem("auto")
        self.device.addItem("cuda")
        self.device.addItem("cpu")
        f.addRow("device:", self.device)

        # cache：none / disk / memory（两种模式共用同一策略）
        self.cache = QComboBox()
        self.cache.addItem("disk", userData="disk")
        self.cache.addItem("none", userData="none")
        self.cache.addItem("memory", userData="memory")
        f.addRow("cache:", self.cache)

        # output_dir
        self.output_dir = PathSelect("dir", "选择 benchmark 输出目录")
        self.output_dir.line.setText(os.path.join(self.project_root, "outputs", "benchmark"))
        f.addRow("output_dir:", self.output_dir)

        # ---- 按钮区 ----
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("开始 benchmark")
        self.btn_start.clicked.connect(self._on_run)
        self.btn_stop = QPushButton("停止 benchmark")
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_open_dir = QPushButton("打开结果目录")
        self.btn_open_dir.clicked.connect(self._on_open_dir)
        self.btn_open_md = QPushButton("查看对比表 (MD)")
        self.btn_open_md.clicked.connect(self._on_open_md)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_open_dir)
        btn_row.addWidget(self.btn_open_md)
        f.addRow(btn_row)

        root.addWidget(box)

        # ---- 状态区 ----
        self.lbl_status = QLabel("就绪。点击「开始 benchmark」启动对比实验。")
        self.lbl_status.setStyleSheet("color: #555;")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setTextFormat(Qt.PlainText)
        root.addWidget(self.lbl_status)

        root.addStretch(1)

    # ------------------------------------------------------ 开始 benchmark
    def _on_run(self) -> None:
        cfg = self.config.text()
        tj = self.train_json.text()
        vj = self.val_json.text()
        od = self.output_dir.text() or os.path.join(self.project_root, "outputs", "benchmark")
        if not self._require(cfg, tj, vj, hint="请填写 config / train.json / val.json"):
            return

        script = os.path.join(self.project_root, "scripts", "benchmark_optimizations.py")
        if not os.path.isfile(script):
            QMessageBox.warning(self, "脚本不存在", "找不到 benchmark 脚本：{}".format(script))
            return

        # 构造命令（与 README §7C.1 一致）
        cmd = self._mk(script, [
            "--config", cfg,
            "--train_json", tj,
            "--val_json", vj,
            "--max_epochs", "{}".format(self.max_epochs.value()),
            "--batch_size", "{}".format(self.batch_size.value()),
            "--cache", "{}".format(self.cache.currentData()),
            "--output_dir", od,
        ])
        if self.device.currentText() != "auto":
            cmd += ["--device", self.device.currentText()]

        # 运行前清空状态
        self.lbl_status.setText(
            "benchmark 进行中… 完成后在此显示 results.md / .csv / _summary.txt 路径。\n"
            "输出目录：{}".format(od)
        )
        self.lbl_status.setStyleSheet("color: #555;")
        self.request_run.emit(cmd)

    # ------------------------------------------------------ 停止
    def _on_stop(self) -> None:
        # 实际停止由 main_window 调用 worker.request_stop()
        self.request_stop.emit()

    # ------------------------------------------------------ 完成钩子
    def on_cmd_ok(self, cmd: list) -> None:
        output_dir = None
        for i, arg in enumerate(cmd):
            if arg == "--output_dir" and i + 1 < len(cmd):
                output_dir = cmd[i + 1]
                break
        if not output_dir:
            output_dir = self.output_dir.text() or os.path.join(self.project_root, "outputs", "benchmark")
        md_path = os.path.join(output_dir, "benchmark_results.md")
        csv_path = os.path.join(output_dir, "benchmark_results.csv")
        sum_path = os.path.join(output_dir, "benchmark_summary.txt")
        md_flag = "OK" if os.path.isfile(md_path) else "MISSING"
        csv_flag = "OK" if os.path.isfile(csv_path) else "MISSING"
        sum_flag = "OK" if os.path.isfile(sum_path) else "MISSING"
        self.lbl_status.setText(
            "benchmark 完成！\n"
            "  [{}] 对比表 (MD):     {}\n"
            "  [{}] 逐模式指标 (CSV): {}\n"
            "  [{}] 对比说明 (TXT):   {}".format(md_flag, md_path, csv_flag, csv_path, sum_flag, sum_path)
        )
        self.lbl_status.setStyleSheet("color: #27ae60;" if md_flag == "OK" else "color: #e67e22;")

    def on_cmd_fail(self, cmd: list) -> None:
        self.lbl_status.setText("benchmark 失败，请查看日志。")
        self.lbl_status.setStyleSheet("color: #c0392b;")

    # ------------------------------------------------------ 打开结果目录 / MD
    def _on_open_dir(self) -> None:
        od = self.output_dir.text() or os.path.join(self.project_root, "outputs", "benchmark")
        if not od or not os.path.isdir(od):
            QMessageBox.warning(self, "目录无效", "输出目录不存在：{}\n请先运行一次 benchmark。".format(od))
            return
        try:
            if os.name == "nt":
                os.startfile(od)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", od])
            else:
                subprocess.Popen(["xdg-open", od])
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "打开失败", str(e))

    def _on_open_md(self) -> None:
        od = self.output_dir.text() or os.path.join(self.project_root, "outputs", "benchmark")
        md_path = os.path.join(od, "benchmark_results.md")
        if not os.path.isfile(md_path):
            QMessageBox.warning(self, "文件不存在", "对比表不存在：{}\n请先运行一次 benchmark。".format(md_path))
            return
        try:
            if os.name == "nt":
                os.startfile(md_path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", md_path])
            else:
                subprocess.Popen(["xdg-open", md_path])
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "打开失败", str(e))


# ============================================================================
# 9. Logs 日志页
# ============================================================================

class LogsPage(_PageBase):
    """统一日志页：所有页面的任务输出都进入此窗口。

    功能（需求 #4）：
    - 清空日志（btn_clear）
    - 保存日志为 .txt（btn_save，由 main_window._save_log 处理）
    - 自动滚动到底部（chk_autoscroll，默认开启）

    日志级别颜色（需求 #3）：
    - INFO     : 灰色   #555555  信息/成功/子进程原始输出
    - WARNING  : 橙色   #e67e22  警告
    - ERROR    : 红色   #c0392b  错误/失败
    - COMMAND  : 蓝色   #2980b9  命令行（加粗）

    每行格式：[HH:MM:SS] [LEVEL] message
    """

    # 级别 -> 颜色（CSS）
    LOG_COLORS = {
        "INFO": "#555555",
        "WARNING": "#e67e22",
        "ERROR": "#c0392b",
        "COMMAND": "#2980b9",
    }

    def __init__(self, parent=None):
        super().__init__("Logs 日志", parent)

    def _build(self) -> None:
        from PySide6.QtWidgets import QTextBrowser
        root = QVBoxLayout(self)

        # 顶部说明
        hint = QLabel(
            "统一日志窗口：所有页面的任务输出（INFO/WARNING/ERROR/COMMAND）都汇聚于此。"
        )
        hint.setStyleSheet("color: #555;")
        root.addWidget(hint)

        # 日志显示区：用 QTextBrowser 支持 HTML 彩色显示
        self.log = QTextBrowser()
        self.log.setReadOnly(True)
        self.log.setOpenExternalLinks(False)
        # 等宽字体便于对齐日志
        self.log.setStyleSheet(
            "font-family: Consolas, 'Courier New', monospace; font-size: 12px;"
        )
        root.addWidget(self.log, 1)

        # 按钮区
        row = QHBoxLayout()
        self.btn_clear = QPushButton("清空日志")
        self.btn_save = QPushButton("保存日志")
        self.chk_autoscroll = QCheckBox("自动滚动到底部")
        self.chk_autoscroll.setChecked(True)
        row.addWidget(self.btn_clear)
        row.addWidget(self.btn_save)
        row.addStretch(1)
        row.addWidget(self.chk_autoscroll)
        root.addLayout(row)

    # ------------------------------------------------------ 追加日志（统一入口）
    def append_log(self, level: str, ts: str, text: str) -> None:
        """按级别追加一行彩色日志。

        参数：
            level: "INFO" / "WARNING" / "ERROR" / "COMMAND"
            ts   : 时间戳字符串，如 "12:34:56"
            text : 日志正文（已转义会在此处理）
        """
        from html import escape
        color = self.LOG_COLORS.get(level, "#555555")
        safe_text = escape(text)
        bold = "font-weight:bold;" if level == "COMMAND" else ""
        # 时间戳灰色、级别标签彩色加粗、正文同色
        html = (
            f'<span style="color:#888;">[{ts}]</span> '
            f'<span style="color:{color};font-weight:bold;">[{level}]</span> '
            f'<span style="color:{color};{bold}">{safe_text}</span>'
        )
        self.log.append(html)
        # 自动滚动到底部
        if self.chk_autoscroll.isChecked():
            sb = self.log.verticalScrollBar()
            sb.setValue(sb.maximum())

    # ------------------------------------------------------ 纯文本（供保存）
    def toPlainText(self) -> str:  # noqa: N802  与 Qt API 同名便于替换
        """返回日志纯文本（带时间戳与级别，无 HTML 标签），用于保存为 .txt。"""
        return self.log.toPlainText()
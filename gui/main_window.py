# -*- coding: utf-8 -*-
"""
gui/main_window.py
==================

主窗口：左侧导航栏 (QListWidget) + 右侧 QStackedWidget（8 个页面）。

负责：
1) 把各页面的 `request_run(list)` 信号接到 `_run_command`，起一个 CommandWorker
   后台执行；运行中禁用当前页面的按钮，结束恢复。
2) 统一日志系统：把 worker 的 `log_signal` 信号接到 `_append_log`，按行级别
   （INFO/WARNING/ERROR/COMMAND）分类、加时间戳、彩色显示到 LogsPage。
3) 训练页特有：连接 `request_stop` 信号到 worker.request_stop()，实现"停止训练"。
4) 训练命令发出时切换按钮启用/禁用态。
5) 命令执行前输出完整命令（COMMAND）；成功显示退出码（INFO）；失败显示错误
   提示（ERROR）但 GUI 不崩溃（所有回调 try/except 包裹）。

不实现任何训练/推理逻辑，全部通过 src/*.py 完成。
"""

from __future__ import annotations

import os
import shlex
import time
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QWidget,
)

from gui.widgets import (
    BenchmarkPage,
    DatasetPage,
    EnvironmentPage,
    EvaluationPage,
    InferencePage,
    LogsPage,
    StlExportPage,
    TrainingPage,
    VisualizationPage,
)
from gui.workers import CommandWorker


class MainWindow(QMainWindow):
    def __init__(self, project_root: str):
        super().__init__()
        self.project_root = os.path.abspath(project_root)
        self._worker: Optional[CommandWorker] = None
        # 记录当前发出命令的页面与命令本身，便于结束后回调页面钩子
        self._current_page: Optional[QWidget] = None
        self._current_cmd: list = []

        self.setWindowTitle("Liver Tumor Segmentation Toolkit")
        self.resize(1040, 720)

        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        self.nav = QListWidget()
        self.nav.setFixedWidth(200)
        self.stack = QStackedWidget()

        # 创建 9 个页面，顺序对应左侧导航
        self.page_env = EnvironmentPage(self.project_root)
        self.page_dataset = DatasetPage(self.project_root)
        self.page_train = TrainingPage(self.project_root)
        self.page_infer = InferencePage(self.project_root)
        self.page_eval = EvaluationPage(self.project_root)
        self.page_vis = VisualizationPage(self.project_root)
        self.page_stl = StlExportPage(self.project_root)
        self.page_bench = BenchmarkPage(self.project_root)
        self.page_logs = LogsPage()

        pages = [
            ("1 Environment 环境部署", self.page_env),
            ("2 Dataset 数据管理", self.page_dataset),
            ("3 Training 模型训练", self.page_train),
            ("4 Inference 模型推理", self.page_infer),
            ("5 Evaluation 结果评估", self.page_eval),
            ("6 Visualization 可视化", self.page_vis),
            ("7 STL Export 三维导出", self.page_stl),
            ("8 Benchmark 速度对比", self.page_bench),
            ("9 Logs 日志", self.page_logs),
        ]
        for name, w in pages:
            self.nav.addItem(QListWidgetItem(name))
            self.stack.addWidget(w)

        self.nav.setCurrentRow(0)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)

        splitter.addWidget(self.nav)
        splitter.addWidget(self.stack)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("就绪。")

    # ------------------------------------------------------------ 信号接线
    def _connect_signals(self) -> None:
        for page in [
            self.page_env,
            self.page_dataset,
            self.page_train,
            self.page_infer,
            self.page_eval,
            self.page_vis,
            self.page_stl,
            self.page_bench,
        ]:
            page.request_run.connect(self._run_command)
            page.log_signal.connect(self._append_log)  # 页面自发日志也进统一日志窗口

        # 训练页停止按钮
        self.page_train.request_stop.connect(self._stop_training)
        # 数据集页停止按钮（检查/划分任务）
        self.page_dataset.request_stop.connect(self._stop_training)
        # benchmark 页停止按钮
        self.page_bench.request_stop.connect(self._stop_training)

        # 日志页按钮（清空/保存由 LogsPage 自身处理，这里只接保存到默认路径的提示）
        self.page_logs.btn_clear.clicked.connect(self._on_clear_log)
        self.page_logs.btn_save.clicked.connect(self._save_log)

    # ------------------------------------------------------------ 后台执行
    def _run_command(self, cmd: list) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(self, "任务进行中", "请等待当前任务结束，或先停止。")
            return
        # 记录发起页面（点击按钮的页面）与本次命令，便于完成后回调
        self._current_page = self.sender() if isinstance(self.sender(), QWidget) else None
        self._current_cmd = list(cmd)
        self._set_running(True)

        # 需求 #5：每次执行命令前，日志输出完整命令（COMMAND 级别）
        self._log("COMMAND", self._pretty_cmd(cmd))

        self._worker = CommandWorker(cmd, cwd=self.project_root)
        # 修复：worker 的真实信号名为 log_signal / finished_signal / error_signal
        self._worker.log_signal.connect(self._append_log)
        self._worker.log_signal.connect(self.page_dataset.parse_progress)
        self._worker.finished_signal.connect(self._on_finished_ok)
        self._worker.error_signal.connect(self._on_finished_fail)
        # 关键：持有引用否则被 GC
        self._worker.start()

    def _stop_training(self) -> None:
        # 通用停止：对所有正在运行的 worker 调用 request_stop
        if self._worker and self._worker.isRunning():
            self._log("WARNING", "正在请求停止当前任务...")
            self._worker.request_stop()

    # ---- worker 回调 ----
    def _on_finished_ok(self, rc: int) -> None:
        # 需求 #6：命令执行成功后显示退出码；try/except 保证 GUI 不崩
        try:
            self._log("INFO", "任务完成 (returncode={})".format(rc))
            self.statusBar().showMessage("任务完成。")
            self._notify_page_done(ok=True)
        except Exception as e:  # noqa: BLE001  GUI 不崩溃
            self._log("ERROR", "完成回调异常：{}".format(e))
        finally:
            self._set_running(False)
            self._worker = None

    def _on_finished_fail(self, summary: str) -> None:
        # 需求 #7：命令失败时显示错误提示（ERROR 级别），但 GUI 不崩溃
        try:
            self._log("ERROR", "任务失败：{}".format(summary))
            self.statusBar().showMessage("任务失败，请查看日志。")
            self._notify_page_done(ok=False)
        except Exception as e:  # noqa: BLE001  GUI 不崩溃
            self._log("ERROR", "失败回调异常：{}".format(e))
        finally:
            self._set_running(False)
            self._worker = None

    def _notify_page_done(self, ok: bool) -> None:
        """命令结束后，若发起页实现了 on_cmd_ok/on_cmd_fail 钩子则回调。"""
        page = self._current_page
        cmd = list(self._current_cmd)
        if page is None:
            return
        hook = getattr(page, "on_cmd_ok" if ok else "on_cmd_fail", None)
        if callable(hook):
            try:
                hook(cmd)
            except Exception as e:  # noqa: BLE001  钩子异常不影响主流程
                self._log("WARNING", "页面完成钩子异常：{}".format(e))

    # ------------------------------------------------------------ 统一日志
    def _log(self, level: str, text: str) -> None:
        """统一日志入口：加时间戳后交 LogsPage 彩色显示，并刷新状态栏。

        level ∈ {"INFO", "WARNING", "ERROR", "COMMAND"}（需求 #3）。
        """
        ts = time.strftime("%H:%M:%S")
        self.page_logs.append_log(level, ts, text)
        # 状态栏只显示关键级别，避免每行刷新噪声
        if level in ("COMMAND", "ERROR"):
            self.statusBar().showMessage("[{}] {}".format(level, text))

    def _append_log(self, text: str) -> None:
        """接收 worker/page 发来的原始日志行，按前缀分类后进入统一日志系统。

        worker 已在内部对关键事件加了前缀（$ / [OK] / [FAIL] / [INFO] / [WARN] / [ERROR]），
        这里据此映射到 4 个统一级别；无前缀的子进程原始输出（如训练进度行）归为 INFO。
        注意：worker 的 "$ <cmd>" 回显已被 _run_command 中的 _log("COMMAND", ...) 取代，
        为避免重复，这里跳过以 "$ " 开头的行。
        """
        if not text:
            return
        # 跳过 worker 的命令回显（已在 _run_command 中以 COMMAND 级别记录过）
        if text.startswith("$ "):
            return
        if text.startswith("[OK]"):
            level = "INFO"
            body = text
        elif text.startswith("[FAIL]"):
            level = "ERROR"
            body = text
        elif text.startswith("[WARN]") or text.startswith("[WARNING]"):
            level = "WARNING"
            body = text
        elif text.startswith("[ERROR]"):
            level = "ERROR"
            body = text
        elif text.startswith("[INFO]"):
            level = "INFO"
            body = text
        else:
            # 子进程原始输出（如 Epoch 1/100 loss=0.5）—— 归为 INFO
            level = "INFO"
            body = text
        self._log(level, body)

    @staticmethod
    def _pretty_cmd(cmd: list) -> str:
        """把 argv 列表渲染为可读的命令字符串（带引号，便于日志回看）。"""
        try:
            return " ".join(shlex.quote(str(c)) for c in cmd)
        except Exception:  # noqa: BLE001
            return " ".join(str(c) for c in cmd)

    # ---- 清空日志 ----
    def _on_clear_log(self) -> None:
        self.page_logs.log.clear()
        self._log("INFO", "日志已清空。")

    # ---- 按钮启用/禁用 ----
    def _set_running(self, running: bool) -> None:
        # 训练页：开始/停止按钮切换
        self.page_train.btn_start.setEnabled(not running)
        self.page_train.btn_stop.setEnabled(running)
        # 数据集页：检查/划分/结构/导入/校验 按钮切换；停止按钮反向
        self.page_dataset.btn_inspect.setEnabled(not running)
        self.page_dataset.btn_stop_inspect.setEnabled(running)
        self.page_dataset.btn_split.setEnabled(not running)
        self.page_dataset.btn_check_struct.setEnabled(not running)
        self.page_dataset.btn_import.setEnabled(not running)
        self.page_dataset.btn_verify.setEnabled(not running)
        # benchmark 页：开始/停止按钮切换
        self.page_bench.btn_start.setEnabled(not running)
        self.page_bench.btn_stop.setEnabled(running)

    # ---- 保存日志 ----
    def _save_log(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "保存日志", os.path.join(self.project_root, "outputs", "gui_log.txt"),
            "Text (*.txt);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.page_logs.toPlainText())
            self._log("INFO", "日志已保存到 {}".format(path))
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "保存失败", str(e))
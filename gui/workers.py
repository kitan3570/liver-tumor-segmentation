# -*- coding: utf-8 -*-
"""
gui/workers.py
==============

PySide6 后台命令执行器。所有耗时命令（训练 / 推理 / 评估 / 可视化 / STL 导出 /
环境检查 / pip 安装等）在 QThread 中以 subprocess 运行，stdout/stderr 实时按行
通过 Qt Signal 回传 GUI 日志窗口，主线程不阻塞。

对外提供两个类：
    CommandWorker  —— QThread 子类，执行一条命令，发出日志/完成/失败信号。
    CommandRunner  —— QObject 管理器，封装"发命令 / 停止 / 信号转接"的便捷接口，
                      便于多个页面共用同一个后台执行器。

跨平台兼容 Windows / Linux：
    - Windows 用 CREATE_NEW_PROCESS_GROUP 便于停任务时整组终止；
    - 停止优先用 terminate()，超时再 kill()；
    - 文本模式 UTF-8 + replace，避免中文/异常字符崩溃。

设计原则：本文件只负责"跑命令 + 回传输出"，不实现任何训练/推理逻辑；真正业务
逻辑仍在 src/*.py 中，GUI 仅作为调用入口。
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from typing import Optional, Union

from PySide6.QtCore import QObject, QThread, Signal


# ============================================================================
# CommandWorker —— QThread 子类，后台执行一条命令
# ============================================================================

class CommandWorker(QThread):
    """在后台线程运行一条 shell 命令，实时回传输出。

    Parameters
    ----------
    command : list[str] | str
        命令。推荐 list 形式（参数无需手拼空格转义）；
        若给 str，则按 platform 用 shlex.split 解析（Windows 下需 posix=False）。
    cwd : str, optional
        子进程工作目录，默认项目根（os.getcwd()）。
    env : dict, optional
        额外环境变量；会与 os.environ 合并（不替换原环境）。

    Signals
    -------
    log_signal(str)        : 每读到一个新行就发一次（stdout+stderr 合并，按行）。
    finished_signal(int)   : 进程正常结束（returncode==0），参数为 returncode。
    error_signal(str)      : 进程失败（returncode!=0 或被停止），参数为失败摘要。
    """

    # 三个对外信号（命名按需求要求）
    log_signal = Signal(str)
    finished_signal = Signal(int)
    error_signal = Signal(str)

    def __init__(
        self,
        command: Union[list[str], str],
        cwd: Optional[str] = None,
        env: Optional[dict] = None,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        # 统一把 command 转成 list[str]
        self.cmd: list[str] = self._normalize_command(command)
        self.cwd: str = cwd or os.getcwd()
        # 合并环境：以当前进程环境为底，叠加用户传入的键
        self.env: dict = os.environ.copy()
        if env:
            self.env.update({k: str(v) for k, v in env.items()})

        self._proc: Optional[subprocess.Popen] = None
        self._stop_requested: bool = False

    # ------------------------- 对外接口 -------------------------
    def request_stop(self) -> None:
        """请求停止：先 terminate() 子进程；若调用者希望更强杀，run() 结束判断中处理。"""
        self._stop_requested = True
        if self._proc and self._proc.poll() is None:
            self._terminate_subprocess()

    # ------------------------- 内部工具 -------------------------
    @staticmethod
    def _normalize_command(command: Union[list[str], str]) -> list[str]:
        """把 str/list 统一成 list[str]，跨平台 shlex 解析。"""
        if isinstance(command, (list, tuple)):
            return [str(c) for c in command]
        # str → shlex.split；Windows 路径含反斜杠需 posix=False
        if os.name == "nt":
            return shlex.split(command, posix=False)
        return shlex.split(command)

    def _terminate_subprocess(self) -> None:
        """安全终止子进程：先 terminate()，超时再 kill()。"""
        if not self._proc:
            return
        try:
            self._proc.terminate()  # 优雅终止（SIGTERM / TerminateProcess）
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()  # 强杀
                self._proc.wait(timeout=3)
        except Exception as e:  # noqa: BLE001
            # 终止失败不应让 GUI 崩，仅日志提示
            self.log_signal.emit(f"[WARN] 终止子进程时出错：{e}")

    # ------------------------- 线程主循环 -------------------------
    def run(self) -> None:  # noqa: D401  (QThread override)
        # Windows 下建独立进程组，便于 stop 时整组终止
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        # 先回显可读命令，便于日志回看（shlex.quote 在 Windows 上转义不完美，仅作展示）
        pretty = " ".join(shlex.quote(c) for c in self.cmd)
        self.log_signal.emit(f"$ {pretty}")

        try:
            self._proc = subprocess.Popen(
                self.cmd,
                cwd=self.cwd,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,        # 合并 stderr 到 stdout，统一按行
                bufsize=1,                        # 行缓冲
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creationflags,
            )
        except FileNotFoundError as e:
            msg = f"[ERROR] 命令未找到：{self.cmd[0]}（{e}）"
            self.log_signal.emit(msg)
            self.error_signal.emit(msg)
            return
        except Exception as e:  # noqa: BLE001
            msg = f"[ERROR] 启动子进程失败：{e}"
            self.log_signal.emit(msg)
            self.error_signal.emit(msg)
            return

        # 按行读取实时输出；保留最近若干行便于失败摘要
        tail: list[str] = []
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            text = line.rstrip("\r\n")
            if text:
                self.log_signal.emit(text)
                tail.append(text)
                if len(tail) > 20:
                    tail.pop(0)

        returncode = self._proc.wait()

        if self._stop_requested:
            msg = "用户主动停止"
            self.log_signal.emit(f"[INFO] {msg}（returncode={returncode}）")
            self.error_signal.emit(msg)
        elif returncode == 0:
            self.log_signal.emit(f"[OK] 任务完成（returncode={returncode}）")
            self.finished_signal.emit(returncode)
        else:
            err_summary = "\n".join(tail[-5:]) or "（无输出）"
            self.log_signal.emit(f"[FAIL] 任务失败（returncode={returncode}）")
            self.error_signal.emit(f"returncode={returncode} | {err_summary}")


# ============================================================================
# CommandRunner —— QObject 管理器，封装"发命令 / 停止 / 信号转接"
# ============================================================================

class CommandRunner(QObject):
    """多个页面可共用的后台命令管理器。

    - `start(command, cwd, env)`：启动一条命令（若已有任务在跑会拒绝并返回 False）。
    - `stop()`：对当前 worker 调 request_stop()。
    - 转发底层 worker 的 log_signal / finished_signal / error_signal 给上层，
      并提供 `is_running()` 与状态信号，便于主窗口切换按钮启用态。

    Signals
    -------
    log_signal(str)         : 透传 worker 的日志行。
    finished_signal(int)    : 透传 worker 的成功完成。
    error_signal(str)       : 透传 worker 的失败/停止摘要。
    started()               : 任务开始（worker 启动后发出）。
    stopped()               : 任务结束（无论成功/失败/停止）后发出。
    """

    log_signal = Signal(str)
    finished_signal = Signal(int)
    error_signal = Signal(str)
    started = Signal()
    stopped = Signal()

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._worker: Optional[CommandWorker] = None

    # ------------------------- 对外接口 -------------------------
    def start(
        self,
        command: Union[list[str], str],
        cwd: Optional[str] = None,
        env: Optional[dict] = None,
    ) -> bool:
        """启动一条后台命令。返回 True 表示成功启动，False 表示已有任务在跑。"""
        if self.is_running():
            # 已有任务在跑，拒绝重复启动
            self.log_signal.emit("[WARN] 已有任务在运行，请等待结束或先停止。")
            return False

        self._worker = CommandWorker(command, cwd=cwd, env=env, parent=self)
        # 转接信号
        self._worker.log_signal.connect(self.log_signal)
        self._worker.finished_signal.connect(self.finished_signal)
        self._worker.error_signal.connect(self.error_signal)
        # 任务结束后统一清理 + 发 stopped
        self._worker.finished_signal.connect(lambda _rc: self._on_done())
        self._worker.error_signal.connect(lambda _msg: self._on_done())

        self._worker.start()
        self.started.emit()
        return True

    def stop(self) -> bool:
        """停止当前任务。返回 True 表示发出了停止请求。"""
        if self._worker and self._worker.isRunning():
            self._worker.request_stop()
            return True
        return False

    def is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    # ------------------------- 内部回调 -------------------------
    def _on_done(self) -> None:
        # 释放对 worker 的引用，便于 GC；先断信号避免重复触发
        worker = self._worker
        if worker is not None:
            try:
                worker.finished_signal.disconnect()
                worker.error_signal.disconnect()
            except RuntimeError:
                # 信号已断开则忽略
                pass
        self._worker = None
        self.stopped.emit()


# ============================================================================
# 便捷工厂：构造 python <script> arg... 与 pip install ... 命令
# ============================================================================

def make_python_cmd(script_path: str, args: Optional[list[str]] = None) -> list[str]:
    """构造 `python <script> arg1 arg2 ...`，跨平台使用同一 Python 解释器。"""
    return [sys.executable, script_path, *(args or [])]


def make_pip_install_cmd(targets: list[str]) -> list[str]:
    """构造 `python -m pip install <targets...>` 命令。"""
    return [sys.executable, "-m", "pip", "install", *targets]
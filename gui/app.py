# -*- coding: utf-8 -*-
"""
gui/app.py
==========

Liver Tumor Segmentation Toolkit —— 桌面前端入口。

运行：
    python gui/app.py

依赖：PySide6（pip install PySide6）。
本入口仅创建 QApplication 与 MainWindow，不实现任何训练/推理逻辑；所有业务
通过子进程调用项目根目录下的 src/*.py 完成，保证 GUI 与核心代码解耦。

兼容 Windows / Linux。
"""

from __future__ import annotations

import os
import sys


def _project_root() -> str:
    """返回项目根目录（gui/app.py 的上一级）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def main() -> int:
    # 让子进程能 `from gui...` 与 `from src...` 导入（src 自包含脚本本身不依赖）
    root = _project_root()
    if root not in sys.path:
        sys.path.insert(0, root)

    # 屏蔽 Qt 无害警告：
    #   - qt.text.font.db: OpenType support missing for "...", script 9
    #     （系统字体缺 Devanagari 脚本 OpenType 表，不影响中英文显示）
    #   - QWindowsContext: OleInitialize failed（偶发 COM 初始化提示）
    # 通过环境变量 QT_LOGGING_RULES 关闭 qt.* 分类日志。
    os.environ.setdefault("QT_LOGGING_RULES", "qt.*.warning=false;qt.text.font.db=false")

    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        sys.stderr.write(
            "[ERROR] 未安装 PySide6。请先安装：\n"
            "    pip install PySide6\n"
        )
        return 2

    from gui.main_window import MainWindow

    # 高 DPI 适配（Qt6 默认已开启，这里显式设置以兼容旧式环境）
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Liver Tumor Segmentation Toolkit")

    win = MainWindow(project_root=root)
    win.show()

    # 启动时在日志页给一句欢迎与免责
    win._append_log("Liver Tumor Segmentation Toolkit 已启动。")
    win._append_log("⚠ 本工具仅用于科研/教学，不可用于临床诊断或治疗决策。")
    win._append_log(f"项目根目录：{root}")
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
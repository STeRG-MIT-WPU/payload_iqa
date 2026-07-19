#!/usr/bin/env python3
"""Launch the PyQt5 monitoring dashboard.

    python3 run_dashboard.py

Everything is controlled from the window: pick the source (USB camera or the
test-images folder), toggle Edge TPU vs CPU, set the capture interval, then
press Run. The same pipeline can be exercised without any GUI through
run_headless.py.
"""
import multiprocessing
import sys


def main():
    from PyQt5.QtWidgets import QApplication
    from dashboard.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

"""PyQt5 dashboard for the two-Coral screening pipeline.

Press Run: the camera (or test folder) starts feeding frames, Coral 1 runs
IQA + cloud segmentation, Coral 2 runs face detection, and every event is
reflected live: per-image table, rejected panel with reasons, live previews,
CPU/RAM usage and per-TPU duty cycle.
"""
import time

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout,
                             QLabel, QListWidget, QListWidgetItem, QMainWindow,
                             QPlainTextEdit, QPushButton, QSplitter,
                             QTableWidget, QTableWidgetItem, QTabWidget,
                             QVBoxLayout, QWidget)

from pipeline.config import Config
from pipeline.monitor import ResourceMonitor
from pipeline.orchestrator import Pipeline

from .widgets import ImagePane, MeterBlock

TABLE_COLS = ["Image", "IQA", "Cloud %", "Faces", "IQA ms", "Cloud ms",
              "Face ms", "Status"]
STATUS_COLORS = {
    "captured": QColor("#555555"),
    "processing": QColor("#8d6e00"),
    "PASSED": QColor("#1b5e20"),
    "CROPPED+PASSED": QColor("#33691e"),
    "REJECTED (iqa)": QColor("#7f1d1d"),
    "REJECTED (cloud)": QColor("#7f1d1d"),
}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CM5 Coral Pipeline — IQA / Cloud / Faces")
        self.resize(1360, 820)

        self.pipeline = None
        self.monitor = ResourceMonitor()
        self.rows = {}          # image name -> table row
        self.counters = dict(captured=0, iqa_rej=0, cloud_rej=0,
                             passed=0, faces=0)

        self._build_ui()

        self.timer = QTimer(self)
        self.timer.setInterval(150)
        self.timer.timeout.connect(self._tick)
        self._last_res_sample = 0.0

    # --- UI construction ----------------------------------------------------

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)

        # top bar
        bar = QHBoxLayout()
        self.run_btn = QPushButton("▶  Run")
        self.run_btn.setCheckable(True)
        self.run_btn.setFixedWidth(120)
        self.run_btn.setStyleSheet("font-weight:bold; font-size:14px;")
        self.run_btn.toggled.connect(self._on_run_toggled)
        bar.addWidget(self.run_btn)

        bar.addWidget(QLabel("Source:"))
        self.source_box = QComboBox()
        self.source_box.addItems(["camera", "folder (test-images)"])
        bar.addWidget(self.source_box)

        self.tpu_check = QCheckBox("Use Edge TPUs")
        self.tpu_check.setChecked(True)
        bar.addWidget(self.tpu_check)

        bar.addWidget(QLabel("Interval s:"))
        self.interval_box = QDoubleSpinBox()
        self.interval_box.setRange(0.1, 60.0)
        self.interval_box.setSingleStep(0.5)
        self.interval_box.setValue(2.0)
        bar.addWidget(self.interval_box)

        bar.addStretch(1)
        self.status_lbl = QLabel("idle")
        self.status_lbl.setStyleSheet("color:#888;")
        bar.addWidget(self.status_lbl)
        outer.addLayout(bar)

        # main splitter: previews | tabs | resources
        split = QSplitter(Qt.Horizontal)
        outer.addWidget(split, 1)

        # left: previews
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(QLabel("Latest capture"))
        self.capture_pane = ImagePane("waiting for camera…")
        ll.addWidget(self.capture_pane, 1)
        ll.addWidget(QLabel("Latest face detection"))
        self.result_pane = ImagePane("waiting for results…")
        ll.addWidget(self.result_pane, 1)
        split.addWidget(left)

        # center: tabs (table / rejected / log)
        self.tabs = QTabWidget()
        self.table = QTableWidget(0, len(TABLE_COLS))
        self.table.setHorizontalHeaderLabels(TABLE_COLS)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 320)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.tabs.addTab(self.table, "All images")

        self.rejected_list = QListWidget()
        self.rejected_list.setIconSize(QPixmap(96, 54).size())
        self.rejected_list.itemClicked.connect(
            lambda it: self.result_pane.set_image(it.data(Qt.UserRole)))
        self.tabs.addTab(self.rejected_list, "Rejected (0)")

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.tabs.addTab(self.log, "Log")
        split.addWidget(self.tabs)

        # right: stats + resources
        right = QWidget()
        rl = QVBoxLayout(right)
        self.stats_lbl = QLabel()
        self.stats_lbl.setStyleSheet("font-size:13px;")
        self.stats_lbl.setTextFormat(Qt.RichText)
        rl.addWidget(self.stats_lbl)

        self.cpu_meter = MeterBlock("CPU", "#4fc3f7")
        self.ram_meter = MeterBlock("RAM", "#ba68c8")
        self.tpu1_meter = MeterBlock("Coral 1 — IQA + cloud seg", "#81c784")
        self.tpu2_meter = MeterBlock("Coral 2 — face detection", "#ffb74d")
        for m in (self.cpu_meter, self.ram_meter, self.tpu1_meter,
                  self.tpu2_meter):
            rl.addWidget(m)
        if not self.monitor.available:
            self.cpu_meter.detail.setText("psutil not installed")
        note = QLabel("TPU load = inference duty cycle\n"
                      "(share of time inside invoke())")
        note.setStyleSheet("color:#777; font-size:10px;")
        rl.addWidget(note)
        rl.addStretch(1)
        split.addWidget(right)
        split.setSizes([380, 640, 280])

        self._update_stats()

    # --- run / stop ---------------------------------------------------------

    def _on_run_toggled(self, checked):
        if checked:
            self._start()
        else:
            self._stop()

    def _start(self):
        cfg = Config(
            source="folder" if self.source_box.currentIndex() == 1 else "camera",
            use_edgetpu=self.tpu_check.isChecked(),
            capture_interval=self.interval_box.value(),
            loop_folder=False,
        )
        self.pipeline = Pipeline(cfg)
        self.pipeline.start()
        self.run_btn.setText("■  Stop")
        self.status_lbl.setText(
            f"running — {cfg.source}, "
            f"{'Edge TPU' if cfg.use_edgetpu else 'CPU'}")
        self._log(f"pipeline started (source={cfg.source}, "
                  f"edgetpu={cfg.use_edgetpu})")
        for w in (self.source_box, self.tpu_check, self.interval_box):
            w.setEnabled(False)
        self.timer.start()

    def _stop(self):
        self.timer.stop()
        if self.pipeline:
            self.pipeline.stop()
            for ev in self.pipeline.poll_events():
                self._dispatch(ev)
            self.pipeline = None
        self.run_btn.blockSignals(True)
        self.run_btn.setChecked(False)
        self.run_btn.blockSignals(False)
        self.run_btn.setText("▶  Run")
        self.status_lbl.setText("stopped")
        self._log("pipeline stopped")
        for w in (self.source_box, self.tpu_check, self.interval_box):
            w.setEnabled(True)

    def closeEvent(self, ev):
        self._stop()
        super().closeEvent(ev)

    # --- event handling -----------------------------------------------------

    def _tick(self):
        if self.pipeline:
            for ev in self.pipeline.poll_events():
                self._dispatch(ev)
        now = time.monotonic()
        if now - self._last_res_sample >= 1.0:
            self._last_res_sample = now
            s = self.monitor.sample()
            if s:
                self.cpu_meter.update_value(
                    s["cpu"], "per core: " + " ".join(
                        f"{c:.0f}" for c in s["per_core"]))
                self.ram_meter.update_value(
                    s["ram"], f"{s['ram_used_gb']:.1f} / "
                              f"{s['ram_total_gb']:.1f} GB")

    def _dispatch(self, ev):
        t = ev.get("type")
        if t == "capture":
            self.counters["captured"] += 1
            self.capture_pane.set_image(ev["path"])
            self._row(ev["image"], status="captured")
        elif t == "iqa":
            self._row(ev["image"], iqa=f"{ev['score']:.1f}",
                      iqa_ms=f"{ev['ms']:.0f}",
                      status="processing" if ev["ok"] else "REJECTED (iqa)")
        elif t == "cloud":
            self._row(ev["image"], cloud=f"{ev['coverage']:.1f}",
                      cloud_ms=f"{ev['ms']:.0f}",
                      status={"reject": "REJECTED (cloud)",
                              "crop": "processing",
                              "pass": "processing"}[ev["action"]])
        elif t == "face":
            self.result_pane.set_image(ev["annotated"])
            self._row(ev["image"], faces=str(ev["count"]),
                      face_ms=f"{ev['ms']:.0f}")
            self.counters["faces"] += ev["count"]
        elif t == "passed":
            self.counters["passed"] += 1
            self._row(ev["image"],
                      status="CROPPED+PASSED" if ev.get("cropped") else "PASSED")
        elif t == "rejected":
            self.counters["iqa_rej" if ev["stage"] == "iqa"
                          else "cloud_rej"] += 1
            self._add_rejected(ev)
        elif t == "util":
            meter = self.tpu1_meter if ev["tpu"] == "coral1" else self.tpu2_meter
            meter.update_value(ev["duty"], f"device: {ev['device']}")
        elif t == "status":
            self._log(ev["msg"])
        elif t == "error":
            self._log(f"ERROR ({ev.get('stage', '?')}): {ev['msg']}")
            self.tabs.setCurrentWidget(self.log)
        elif t == "done":
            self._log("source exhausted — stopping")
            self._stop()
        self._update_stats()

    # --- widgets updates ----------------------------------------------------

    def _row(self, image, iqa=None, cloud=None, faces=None, iqa_ms=None,
             cloud_ms=None, face_ms=None, status=None):
        base = image.replace("_cropped", "")
        if base not in self.rows:
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.rows[base] = r
            self.table.setItem(r, 0, QTableWidgetItem(image))
            for c in range(1, len(TABLE_COLS)):
                self.table.setItem(r, c, QTableWidgetItem(""))
            self.table.scrollToBottom()
        r = self.rows[base]
        for col, val in ((1, iqa), (2, cloud), (3, faces), (4, iqa_ms),
                         (5, cloud_ms), (6, face_ms), (7, status)):
            if val is not None:
                self.table.item(r, col).setText(val)
        if status is not None:
            color = STATUS_COLORS.get(status)
            if color:
                for c in range(len(TABLE_COLS)):
                    self.table.item(r, c).setBackground(color)

    def _add_rejected(self, ev):
        details = []
        if ev.get("iqa") is not None:
            details.append(f"IQA {ev['iqa']:.1f}")
        if ev.get("cloud") is not None:
            details.append(f"cloud {ev['cloud']:.1f}%")
        item = QListWidgetItem(
            f"{ev['image']}\n  rejected by {ev['stage'].upper()}: "
            f"{ev['reason']}" + (f"   [{', '.join(details)}]" if details else ""))
        item.setData(Qt.UserRole, ev["path"])
        pix = QPixmap(ev["path"])
        if not pix.isNull():
            item.setIcon(QIcon(pix.scaled(96, 54, Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation)))
        self.rejected_list.addItem(item)
        self.rejected_list.scrollToBottom()
        n = self.rejected_list.count()
        self.tabs.setTabText(self.tabs.indexOf(self.rejected_list),
                             f"Rejected ({n})")

    def _update_stats(self):
        c = self.counters
        self.stats_lbl.setText(
            "<b>Session</b><br>"
            f"captured: <b>{c['captured']}</b><br>"
            f"rejected by IQA: <b style='color:#e57373'>{c['iqa_rej']}</b><br>"
            f"rejected by cloud: <b style='color:#e57373'>{c['cloud_rej']}</b><br>"
            f"passed: <b style='color:#81c784'>{c['passed']}</b><br>"
            f"faces found: <b>{c['faces']}</b>")

    def _log(self, msg):
        self.log.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {msg}")

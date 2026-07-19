"""Small custom widgets for the dashboard."""
from collections import deque

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QColor, QPainter, QPen, QPixmap, QPolygonF
from PyQt5.QtCore import QPointF
from PyQt5.QtWidgets import (QFrame, QLabel, QProgressBar, QVBoxLayout,
                             QWidget)


class ImagePane(QLabel):
    """A QLabel that keeps its pixmap aspect-scaled to the widget size."""

    def __init__(self, placeholder="--"):
        super().__init__(placeholder)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 130)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("background:#181818; color:#666;")
        self._pix = None

    def set_image(self, path):
        pix = QPixmap(str(path))
        if pix.isNull():
            return
        self._pix = pix
        self._rescale()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._rescale()

    def _rescale(self):
        if self._pix is not None:
            self.setPixmap(self._pix.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class Sparkline(QWidget):
    """Tiny scrolling line chart for 0-100 values (CPU %, TPU duty...)."""

    def __init__(self, color="#4fc3f7", maxlen=120):
        super().__init__()
        self._vals = deque(maxlen=maxlen)
        self._color = QColor(color)
        self.setMinimumHeight(42)

    def add(self, v):
        self._vals.append(max(0.0, min(100.0, float(v))))
        self.update()

    def sizeHint(self):
        return QSize(160, 42)

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#181818"))
        if len(self._vals) < 2:
            return
        w, h = self.width(), self.height()
        n = self._vals.maxlen
        pts = QPolygonF([
            QPointF(w * (n - len(self._vals) + i) / (n - 1),
                    h - 2 - (h - 4) * v / 100.0)
            for i, v in enumerate(self._vals)
        ])
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(self._color, 1.5))
        p.drawPolyline(pts)


class MeterBlock(QWidget):
    """Title + progress bar + sparkline + detail label, for one resource."""

    def __init__(self, title, color="#4fc3f7"):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 4)
        lay.setSpacing(2)
        self.title = QLabel(title)
        self.title.setStyleSheet("font-weight:bold;")
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(True)
        self.bar.setFixedHeight(16)
        self.spark = Sparkline(color)
        self.detail = QLabel("")
        self.detail.setStyleSheet("color:#888; font-size:11px;")
        for w in (self.title, self.bar, self.spark, self.detail):
            lay.addWidget(w)

    def update_value(self, pct, detail=""):
        self.bar.setValue(int(round(pct)))
        self.spark.add(pct)
        if detail:
            self.detail.setText(detail)

"""
EyeMouse overlay: extends the base GazeOverlay with a dwell progress ring,
click-flash animations, and a drag indicator.
"""

import sys
import os
import time
from collections import deque
from typing import Tuple

from PyQt6.QtCore import Qt, QPointF, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QGuiApplication, QFont
from PyQt6.QtWidgets import QWidget

# Reuse the base overlay from GazeOverlay/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "GazeOverlay"))
from overlay import GazeOverlay  # noqa: E402


class DwellOverlay(GazeOverlay):
    """GazeOverlay + dwell ring + click-flash visualization."""

    _dwell_received = pyqtSignal(float)
    _flash_received = pyqtSignal(str, int, int)  # kind, x, y
    _drag_received = pyqtSignal(bool)
    _mode_received = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self._dwell_progress = 0.0
        self._is_dragging = False
        self._eye_mouse_active = False
        self._flashes: deque = deque()  # (ts, kind, x, y)

        self._dwell_received.connect(self._on_dwell)
        self._flash_received.connect(self._on_flash)
        self._drag_received.connect(self._on_drag)
        self._mode_received.connect(self._on_mode)

    # ---- thread-safe API ----

    def update_dwell(self, progress: float):
        self._dwell_received.emit(max(0.0, min(1.0, progress)))

    def show_click_flash(self, kind: str, x: int, y: int):
        self._flash_received.emit(kind, x, y)

    def set_dragging(self, dragging: bool):
        self._drag_received.emit(dragging)

    def set_eye_mouse_active(self, active: bool):
        self._mode_received.emit(active)

    # ---- slots ----

    def _on_dwell(self, p: float):
        self._dwell_progress = p

    def _on_flash(self, kind: str, x: int, y: int):
        self._flashes.append((time.time(), kind, x, y))

    def _on_drag(self, dragging: bool):
        self._is_dragging = dragging
        self.update()

    def _on_mode(self, active: bool):
        self._eye_mouse_active = active
        self.update()

    # ---- rendering: extend parent's paintEvent ----

    def paintEvent(self, event):
        super().paintEvent(event)

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        pt = self._smoothed_xy

        # Dwell progress ring around the gaze dot
        if self._eye_mouse_active and self._dwell_progress > 0.02 and self._has_gaze:
            self._draw_dwell_ring(p, pt, self._dwell_progress)

        # Click flashes
        self._draw_flashes(p)

        # Drag indicator
        if self._is_dragging and self._has_gaze:
            self._draw_drag_indicator(p, pt)

        # EyeMouse status badge (top-center)
        self._draw_mode_badge(p)

        p.end()

    def _draw_dwell_ring(self, p: QPainter, pt: QPointF, progress: float):
        # Background ring
        p.setPen(QPen(QColor(255, 255, 255, 60), 3))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(pt, 32.0, 32.0)

        # Progress arc — clockwise from 12 o'clock
        p.setPen(QPen(QColor(0, 220, 255, 240), 4, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap))
        rect_size = 32.0
        arc_rect = (int(pt.x() - rect_size), int(pt.y() - rect_size),
                    int(rect_size * 2), int(rect_size * 2))
        # Qt angles are in 1/16 degree, 0 = 3 o'clock, positive = CCW.
        # Start at 12 o'clock = 90, go clockwise (negative span).
        start = 90 * 16
        span = -int(360 * 16 * progress)
        from PyQt6.QtCore import QRectF
        p.drawArc(QRectF(arc_rect[0], arc_rect[1], arc_rect[2], arc_rect[3]),
                  start, span)

    def _draw_flashes(self, p: QPainter):
        now = time.time()
        # Drop expired flashes
        while self._flashes and (now - self._flashes[0][0]) > 0.6:
            self._flashes.popleft()

        for ts, kind, x, y in self._flashes:
            age = now - ts
            t = age / 0.6  # 0..1
            radius = 18 + t * 70
            alpha = int(255 * (1 - t))
            if kind == "left":
                color = QColor(0, 255, 136, alpha)
            elif kind == "right":
                color = QColor(255, 80, 80, alpha)
            else:
                color = QColor(255, 200, 0, alpha)
            p.setPen(QPen(color, 3))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(x, y), radius, radius)

    def _draw_drag_indicator(self, p: QPainter, pt: QPointF):
        # Yellow square outline pulses around the cursor
        pulse = (int(time.time() * 4) % 2) == 0
        color = QColor(255, 200, 0, 220 if pulse else 140)
        p.setPen(QPen(color, 3))
        p.setBrush(Qt.BrushStyle.NoBrush)
        size = 40
        p.drawRect(int(pt.x() - size), int(pt.y() - size), size * 2, size * 2)

        font = QFont("Segoe UI", 9, QFont.Weight.DemiBold)
        p.setFont(font)
        p.setPen(QPen(color))
        p.drawText(int(pt.x() + size + 6), int(pt.y() - size + 12), "DRAGGING")

    def _draw_mode_badge(self, p: QPainter):
        text = "  EyeMouse: ON  " if self._eye_mouse_active else "  EyeMouse: OFF (F8)  "
        font = QFont("Segoe UI", 10, QFont.Weight.DemiBold)
        p.setFont(font)
        fm = p.fontMetrics()
        w = fm.horizontalAdvance(text) + 8
        h = fm.height() + 6
        x = (self.width() - w) // 2
        y = 12

        bg = QColor(0, 100, 60, 180) if self._eye_mouse_active else QColor(80, 80, 80, 160)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(bg))
        p.drawRoundedRect(x, y, w, h, 6, 6)

        fg = QColor("#00ff88") if self._eye_mouse_active else QColor("#cccccc")
        p.setPen(QPen(fg))
        p.drawText(x, y, w, h, Qt.AlignmentFlag.AlignCenter, text)

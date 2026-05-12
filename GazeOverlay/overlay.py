"""
Transparent, click-through, always-on-top fullscreen overlay for the gaze dot.

The overlay window covers the entire primary monitor but does not capture mouse
input — clicks pass straight through to whatever game/app is underneath. The
`update_gaze(x, y)` method is thread-safe (it dispatches to the Qt event loop
via a queued signal) so the gaze engine thread can call it directly.
"""

import time
from collections import deque

from PyQt6.QtCore import Qt, QPointF, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QGuiApplication, QFont
from PyQt6.QtWidgets import QWidget


class GazeOverlay(QWidget):
    _gaze_received = pyqtSignal(int, int)
    _status_received = pyqtSignal(str, str)  # (text, color_hex)
    _learn_received = pyqtSignal(int, int, str)  # (x, y, kind) for +1 / X animations

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool                 # don't show in taskbar
            | Qt.WindowType.WindowTransparentForInput  # click-through
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        screen = QGuiApplication.primaryScreen().geometry()
        self.setGeometry(screen)

        self._gaze_xy = QPointF(screen.width() / 2, screen.height() / 2)
        self._smoothed_xy = QPointF(self._gaze_xy)
        self._has_gaze = False
        self._status_text = "Initializing..."
        self._status_color = QColor("#ffaa00")
        self._recording = False

        # Floating "+1" feedback events: list of (ts, x, y, kind)
        # kind = "accept" (green +1) or "reject" (faint X)
        self._learn_events: deque = deque(maxlen=24)
        self._learn_count = 0       # session-total accepted samples

        self._gaze_received.connect(self._on_gaze)
        self._status_received.connect(self._on_status)
        self._learn_received.connect(self._on_learn)

        # 60 fps redraw — decoupled from gaze sample rate (~30 fps)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(16)

    # ---- thread-safe public API ----

    def update_gaze(self, x: int, y: int):
        """Safe to call from any thread."""
        self._gaze_received.emit(x, y)

    def set_status(self, text: str, color: str = "#00ff88"):
        self._status_received.emit(text, color)

    def set_recording(self, recording: bool):
        self._recording = recording
        self.update()

    def show_learn_event(self, x: int, y: int, kind: str = "accept"):
        """Trigger a floating '+1' (kind='accept') or rejection 'X' animation
        at the given screen position. Thread-safe."""
        self._learn_received.emit(x, y, kind)

    # ---- Qt slots ----

    def _on_gaze(self, x: int, y: int):
        self._gaze_xy = QPointF(x, y)
        self._has_gaze = True

    def _on_status(self, text: str, color: str):
        self._status_text = text
        self._status_color = QColor(color)

    def _on_learn(self, x: int, y: int, kind: str):
        self._learn_events.append((time.time(), x, y, kind))
        if kind == "accept":
            self._learn_count += 1

    # ---- rendering ----

    def paintEvent(self, _event):
        # Exponential smoothing for the rendered dot (visually nicer than the
        # raw 30 fps samples; engine already does its own 10-frame average)
        a = 0.35
        self._smoothed_xy = QPointF(
            self._smoothed_xy.x() * (1 - a) + self._gaze_xy.x() * a,
            self._smoothed_xy.y() * (1 - a) + self._gaze_xy.y() * a,
        )

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Status badge (top-left)
        self._draw_status(p)

        # Recording indicator (top-right)
        if self._recording:
            self._draw_rec_indicator(p)

        # Gaze dot
        if self._has_gaze:
            self._draw_gaze_dot(p, self._smoothed_xy)

        # Floating learn-events (+1 / X) — drawn last so they're on top
        self._draw_learn_events(p)

        p.end()

    def _draw_learn_events(self, p: QPainter):
        now = time.time()
        # Drop expired events
        while self._learn_events and (now - self._learn_events[0][0]) > 1.4:
            self._learn_events.popleft()

        font = QFont("Segoe UI", 14, QFont.Weight.Bold)
        p.setFont(font)
        for ts, x, y, kind in self._learn_events:
            age = now - ts
            t = age / 1.4                    # 0..1
            alpha = int(255 * (1 - t * t))   # ease-out fade
            float_up = int(t * 60)           # drifts upward 60 px

            if kind == "accept":
                color = QColor(120, 255, 180, alpha)
                text = "+1"
                # Subtle ring at origin to anchor it visually
                ring_alpha = int(160 * (1 - t))
                p.setPen(QPen(QColor(0, 255, 136, ring_alpha), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                r = 14 + t * 18
                p.drawEllipse(QPointF(x, y), r, r)
            else:
                color = QColor(255, 130, 130, max(60, alpha // 2))
                text = "x"

            p.setPen(QPen(color))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawText(x + 14, y - float_up, text)

    def _draw_status(self, p: QPainter):
        text = f"  GazeOverlay: {self._status_text}  "
        font = QFont("Segoe UI", 10, QFont.Weight.DemiBold)
        p.setFont(font)
        fm = p.fontMetrics()
        w = fm.horizontalAdvance(text) + 8
        h = fm.height() + 6

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(0, 0, 0, 160)))
        p.drawRoundedRect(12, 12, w, h, 6, 6)

        p.setPen(QPen(self._status_color))
        p.drawText(12, 12, w, h, Qt.AlignmentFlag.AlignCenter, text)

    def _draw_rec_indicator(self, p: QPainter):
        screen_w = self.width()
        x = screen_w - 130
        y = 12

        font = QFont("Segoe UI", 10, QFont.Weight.DemiBold)
        p.setFont(font)
        fm = p.fontMetrics()
        h = fm.height() + 6

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(0, 0, 0, 160)))
        p.drawRoundedRect(x, y, 110, h, 6, 6)

        # Red blinking circle
        from time import time
        blink_on = (int(time() * 2) % 2) == 0
        if blink_on:
            p.setBrush(QBrush(QColor(255, 60, 60)))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(x + 10, y + 8, 12, 12)

        p.setPen(QPen(QColor("#ffffff")))
        p.drawText(x + 28, y, 80, h, Qt.AlignmentFlag.AlignVCenter, "REC")

    def _draw_gaze_dot(self, p: QPainter, pt: QPointF):
        # Outer glow ring
        p.setPen(QPen(QColor(0, 255, 136, 120), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(pt, 22.0, 22.0)

        # Inner solid dot
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(0, 255, 136, 220)))
        p.drawEllipse(pt, 8.0, 8.0)

        # Crosshair
        p.setPen(QPen(QColor(0, 255, 136, 180), 1))
        p.drawLine(QPointF(pt.x() - 14, pt.y()), QPointF(pt.x() - 26, pt.y()))
        p.drawLine(QPointF(pt.x() + 14, pt.y()), QPointF(pt.x() + 26, pt.y()))
        p.drawLine(QPointF(pt.x(), pt.y() - 14), QPointF(pt.x(), pt.y() - 26))
        p.drawLine(QPointF(pt.x(), pt.y() + 14), QPointF(pt.x(), pt.y() + 26))

"""
Fullscreen 9-point calibration overlay.

Walks the user through a 3x3 grid of calibration points. At each point a
pulsing target appears for a brief settle period, then samples the engine's
latest (yaw, pitch) for ~30 frames, then advances. After the 9th point we
fit the CalibrationModel and emit `finished`.

Usage:
    win = NinePointCalibration(engine, parent=None)
    win.finished.connect(on_done)        # signature: on_done(success: bool)
    win.start()                          # shows fullscreen, runs the sequence
"""

import math
import time
from typing import List, Tuple

from PyQt6.QtCore import Qt, QPointF, QTimer, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QGuiApplication
from PyQt6.QtWidgets import QWidget

from calibration import CalibrationModel, CalibSample, nine_point_targets


SETTLE_MS = 600        # delay before we start sampling at each point
SAMPLE_MS = 1000       # how long to sample at each point
PER_POINT_MS = SETTLE_MS + SAMPLE_MS


class NinePointCalibration(QWidget):
    finished = pyqtSignal(bool)   # True = model fit succeeded

    def __init__(self, engine, model: CalibrationModel):
        super().__init__()
        self.engine = engine
        self.model = model

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        primary = QGuiApplication.primaryScreen()
        screen = primary.geometry()
        self._dpr = primary.devicePixelRatio()
        self.setGeometry(screen)
        # Logical (Qt paint coords)
        self.screen_w = screen.width()
        self.screen_h = screen.height()
        # Physical (matches model + pynput coords)
        self.phys_w = int(round(self.screen_w * self._dpr))
        self.phys_h = int(round(self.screen_h * self._dpr))

        # Targets are stored in PHYSICAL pixels (so they go straight into the
        # CalibrationModel) and converted to logical for paint.
        self.targets: List[Tuple[int, int]] = nine_point_targets(self.phys_w, self.phys_h)
        self.current_idx = -1               # -1 = intro screen
        self.point_start_ts: float = 0.0
        self.collected_samples: List[Tuple[float, float]] = []  # (yaw, pitch) for current point
        self._animate_phase = 0.0

        # Repaint at 60 fps for the pulse animation
        self._paint_timer = QTimer(self)
        self._paint_timer.timeout.connect(self.update)
        self._paint_timer.start(16)

        # Sampling tick at 50 Hz
        self._sample_timer = QTimer(self)
        self._sample_timer.timeout.connect(self._sample_tick)

    # ---- public ----

    def start(self):
        """Show fullscreen and begin the sequence after a 1-second intro.
        Wipes any existing samples — they're from a previous eye-sphere lock
        and won't match the new one. The fresh 9-point will repopulate the
        anchor set; continuous calibration takes over from there."""
        self.current_idx = -1
        self.collected_samples.clear()
        self.model.samples.clear()
        self.model.coef_x = None
        self.model.coef_y = None
        self.showFullScreen()
        self.raise_()
        self.activateWindow()
        QTimer.singleShot(1500, self._next_point)

    def _next_point(self):
        # Finalize previous point if any
        if self.current_idx >= 0 and self.collected_samples:
            avg_yaw = sum(y for y, _ in self.collected_samples) / len(self.collected_samples)
            avg_pitch = sum(p for _, p in self.collected_samples) / len(self.collected_samples)
            tx, ty = self.targets[self.current_idx]
            self.model.add_calibration_sample(tx, ty, avg_yaw, avg_pitch,
                                              weight=1.0, is_anchor=True)
        self.collected_samples.clear()
        self.current_idx += 1

        if self.current_idx >= len(self.targets):
            self._finish()
            return

        self.point_start_ts = time.time()
        self._sample_timer.start(20)  # 50 Hz

    def _sample_tick(self):
        now = time.time()
        elapsed_ms = (now - self.point_start_ts) * 1000.0
        if elapsed_ms < SETTLE_MS:
            return  # still settling, don't collect yet
        if elapsed_ms >= PER_POINT_MS:
            self._sample_timer.stop()
            self._next_point()
            return
        # Collect raw angles
        if self.engine.eye_locked:
            yaw, pitch = self.engine.latest_angles()
            self.collected_samples.append((yaw, pitch))

    def _finish(self):
        self._sample_timer.stop()
        ok = self.model.fit()
        self.hide()
        self.finished.emit(ok)

    def cancel(self):
        self._sample_timer.stop()
        self.hide()
        self.finished.emit(False)

    # ---- input ----

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self.cancel()

    # ---- painting ----

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Solid dark backdrop
        p.fillRect(self.rect(), QColor(15, 17, 22))

        if self.current_idx == -1:
            self._draw_intro(p)
        else:
            self._draw_progress_bar(p)
            self._draw_target(p)
            self._draw_help(p)

        p.end()

    def _draw_intro(self, p: QPainter):
        font = QFont("Segoe UI", 24, QFont.Weight.DemiBold)
        p.setFont(font)
        p.setPen(QPen(QColor("#00ff88")))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                   "9-point calibration\n\nFollow the dot with your eyes.\nKeep your head still.")

    def _draw_progress_bar(self, p: QPainter):
        # Top progress bar showing point N of 9
        bar_y = 16
        bar_h = 6
        margin = 80
        track_w = self.screen_w - 2 * margin
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(255, 255, 255, 40)))
        p.drawRoundedRect(margin, bar_y, track_w, bar_h, 3, 3)

        progress = self.current_idx / max(1, len(self.targets))
        p.setBrush(QBrush(QColor("#00ff88")))
        p.drawRoundedRect(margin, bar_y, int(track_w * progress), bar_h, 3, 3)

        font = QFont("Segoe UI", 11)
        p.setFont(font)
        p.setPen(QPen(QColor("#7a8a98")))
        p.drawText(margin, bar_y + 28,
                   f"Point {self.current_idx + 1} of {len(self.targets)}")

    def _draw_target(self, p: QPainter):
        if self.current_idx < 0 or self.current_idx >= len(self.targets):
            return
        phys_tx, phys_ty = self.targets[self.current_idx]
        # Convert physical -> logical for Qt paint
        tx = phys_tx / self._dpr
        ty = phys_ty / self._dpr
        elapsed_ms = (time.time() - self.point_start_ts) * 1000.0
        in_settle = elapsed_ms < SETTLE_MS
        sample_progress = max(0.0, min(1.0, (elapsed_ms - SETTLE_MS) / SAMPLE_MS))

        # Pulse radius animation
        pulse = (math.sin(elapsed_ms / 1000.0 * 6.0) + 1) * 0.5  # 0..1
        outer_r = 36 + pulse * 14

        # Outer ring (color: amber while settling, green while sampling)
        ring_color = QColor("#ffaa33") if in_settle else QColor("#00ff88")
        ring_color.setAlpha(180)
        p.setPen(QPen(ring_color, 3))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(tx, ty), outer_r, outer_r)

        # Sampling progress arc (clockwise from 12 o'clock)
        if not in_settle and sample_progress > 0:
            from PyQt6.QtCore import QRectF
            p.setPen(QPen(QColor("#00ddff"), 5, Qt.PenStyle.SolidLine,
                          Qt.PenCapStyle.RoundCap))
            r = 28
            arc_rect = QRectF(tx - r, ty - r, r * 2, r * 2)
            start = 90 * 16
            span = -int(360 * 16 * sample_progress)
            p.drawArc(arc_rect, start, span)

        # Inner solid dot
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor("#00ff88")))
        p.drawEllipse(QPointF(tx, ty), 8.0, 8.0)

        # Crosshair
        p.setPen(QPen(QColor(0, 255, 136, 200), 1))
        p.drawLine(QPointF(tx - 22, ty), QPointF(tx - 14, ty))
        p.drawLine(QPointF(tx + 14, ty), QPointF(tx + 22, ty))
        p.drawLine(QPointF(tx, ty - 22), QPointF(tx, ty - 14))
        p.drawLine(QPointF(tx, ty + 14), QPointF(tx, ty + 22))

    def _draw_help(self, p: QPainter):
        font = QFont("Segoe UI", 11)
        p.setFont(font)
        p.setPen(QPen(QColor("#7a8a98")))
        text = "Look at the green dot.   Esc = cancel."
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(text)
        p.drawText((self.screen_w - tw) // 2, self.screen_h - 30, text)

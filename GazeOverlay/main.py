"""
GazeOverlay — entry point.

Flow:
  1. Show a small calibration window (webcam preview + instructions).
  2. User looks at screen center, presses C → eye-sphere lock.
  3. User looks at screen center again, presses S → screen mapping zeroed.
  4. Calibration window hides; transparent gaze overlay takes over.

Hotkeys (global, work even when game is focused):
  F8  — toggle overlay visibility
  F9  — start / stop session recording (CSV)
  F10 — quit
  F11 — show calibration window again

Sessions are saved to: ./sessions/session_<timestamp>.csv
After F10 quit, you can run:  python report.py sessions/session_<ts>.csv
"""

import os
import sys
import time
from pathlib import Path

import cv2
import keyboard
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap, QGuiApplication
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame,
)

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gaze_engine import GazeEngine, GazeSample
from metrics import MetricsLogger, summarize
from overlay import GazeOverlay


SESSIONS_DIR = Path(__file__).resolve().parent / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)


class CalibrationWindow(QWidget):
    """Small window with webcam preview + calibration instructions."""

    started_recording = pyqtSignal()  # not currently used; reserved

    def __init__(self, engine: GazeEngine):
        super().__init__()
        self.engine = engine
        self.setWindowTitle("GazeOverlay — Calibration")
        self.setFixedSize(720, 560)
        self.setStyleSheet("""
            QWidget { background: #1a1d24; color: #e0e6ed; font-family: 'Segoe UI'; }
            QLabel#title { font-size: 18px; font-weight: bold; color: #00ff88; }
            QLabel#step { font-size: 13px; color: #aac0d0; }
            QLabel#hint { font-size: 11px; color: #7a8a98; }
            QPushButton {
                background: #2a3142; border: 1px solid #3a4458; padding: 8px 16px;
                border-radius: 4px; font-size: 12px; color: #e0e6ed;
            }
            QPushButton:hover { background: #3a4458; }
            QPushButton#primary { background: #00aa55; border: 1px solid #00ff88; color: white; }
            QPushButton#primary:hover { background: #00cc66; }
            QFrame#sep { background: #2a3142; max-height: 1px; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        title = QLabel("GazeOverlay")
        title.setObjectName("title")
        layout.addWidget(title)

        self.preview = QLabel()
        self.preview.setFixedSize(680, 380)
        self.preview.setStyleSheet("background: #000; border: 1px solid #3a4458;")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setText("Starting webcam...")
        layout.addWidget(self.preview)

        self.step_label = QLabel("Step 1: Look at the CENTER of your screen, then press the button below.")
        self.step_label.setObjectName("step")
        self.step_label.setWordWrap(True)
        layout.addWidget(self.step_label)

        sep = QFrame(); sep.setObjectName("sep"); sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)

        btn_row = QHBoxLayout()
        self.calib_btn = QPushButton("1. Lock eye spheres  (C)")
        self.calib_btn.setObjectName("primary")
        self.calib_btn.clicked.connect(self._do_eye_calib)
        btn_row.addWidget(self.calib_btn)

        self.screen_btn = QPushButton("2. Calibrate screen center  (S)")
        self.screen_btn.clicked.connect(self._do_screen_calib)
        self.screen_btn.setEnabled(False)
        btn_row.addWidget(self.screen_btn)

        self.done_btn = QPushButton("3. Start overlay  (Enter)")
        self.done_btn.clicked.connect(self._finish)
        self.done_btn.setEnabled(False)
        btn_row.addWidget(self.done_btn)
        layout.addLayout(btn_row)

        hint = QLabel(
            "Hotkeys after calibration:  "
            "F8 = toggle overlay   F9 = start/stop recording   "
            "F10 = quit   F11 = recalibrate"
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # Refresh preview from engine ~30 fps
        self._refresh = QTimer(self)
        self._refresh.timeout.connect(self._tick_preview)
        self._refresh.start(33)

        self.engine.preview_enabled = True

        # Allow keyboard shortcuts inside the calibration window
        self._kb_timer = QTimer(self)
        self._kb_timer.timeout.connect(self._poll_keys)
        self._kb_timer.start(50)

    def _tick_preview(self):
        frame = self.engine.get_preview_frame()
        if frame is None:
            return
        # Mirror so it feels like a mirror
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        img = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setPixmap(pix)

    def _poll_keys(self):
        # In-window only; we do not consume global F-keys here (those run outside)
        if self.isActiveWindow():
            try:
                if keyboard.is_pressed('c') and self.calib_btn.isEnabled():
                    self._do_eye_calib()
                elif keyboard.is_pressed('s') and self.screen_btn.isEnabled():
                    self._do_screen_calib()
                elif keyboard.is_pressed('enter') and self.done_btn.isEnabled():
                    self._finish()
            except Exception:
                pass

    def _do_eye_calib(self):
        self.engine.request_eye_calibration()
        QTimer.singleShot(150, self._after_eye_calib)

    def _after_eye_calib(self):
        if self.engine.is_calibrated:
            self.calib_btn.setText("[OK] Eye spheres locked")
            self.calib_btn.setEnabled(False)
            self.screen_btn.setEnabled(True)
            self.screen_btn.setObjectName("primary")
            self.screen_btn.setStyle(self.screen_btn.style())
            self.step_label.setText(
                "Step 2: Keep looking at the screen CENTER, then press the next button."
            )

    def _do_screen_calib(self):
        self.engine.request_screen_calibration()
        QTimer.singleShot(150, self._after_screen_calib)

    def _after_screen_calib(self):
        self.screen_btn.setText("[OK] Screen mapping zeroed")
        self.screen_btn.setEnabled(False)
        self.done_btn.setEnabled(True)
        self.done_btn.setObjectName("primary")
        self.done_btn.setStyle(self.done_btn.style())
        self.step_label.setText("All set — press Start to launch the overlay.")

    def _finish(self):
        self.engine.preview_enabled = False
        self.hide()

    def closeEvent(self, e):
        self.engine.preview_enabled = False
        super().closeEvent(e)


class AppController(QObject):
    """Coordinates engine, overlay, calibration window, and hotkeys."""

    sample_signal = pyqtSignal(object)   # GazeSample (queued, thread-safe)

    def __init__(self):
        super().__init__()
        self.app = QApplication.instance() or QApplication(sys.argv)

        screen = QGuiApplication.primaryScreen().geometry()
        self.engine = GazeEngine(
            screen_w=screen.width(),
            screen_h=screen.height(),
            on_sample=self._on_sample_thread,
        )
        self.overlay = GazeOverlay()
        self.calib_window = CalibrationWindow(self.engine)

        self.recording = False
        self.logger: MetricsLogger = None
        self.current_csv: Path = None

        self.sample_signal.connect(self._on_sample_main, type=Qt.ConnectionType.QueuedConnection)

        # Engine starts immediately; user calibrates via the window.
        self.engine.start()
        self.overlay.set_status("Calibrate first  (see window)", "#ffaa00")
        self.overlay.show()
        self.calib_window.show()
        self.calib_window.raise_()
        self.calib_window.activateWindow()

        # Global hotkey poller (Qt timer instead of `keyboard.add_hotkey` so we
        # don't need elevated perms on Linux/macOS — works on Windows fine)
        self._hotkey_timer = QTimer()
        self._hotkey_timer.timeout.connect(self._poll_hotkeys)
        self._hotkey_timer.start(80)
        self._hotkey_cooldown = {}

    # ---- engine callback (runs on engine thread) ----

    def _on_sample_thread(self, sample: GazeSample):
        # Marshal to Qt main thread
        self.sample_signal.emit(sample)

    # ---- main thread ----

    def _on_sample_main(self, sample: GazeSample):
        if sample.face_visible and sample.calibrated:
            self.overlay.update_gaze(sample.screen_x, sample.screen_y)
            self.overlay.set_status("Tracking", "#00ff88")
        elif sample.face_visible and not sample.calibrated:
            self.overlay.set_status("Calibrate first  (see window)", "#ffaa00")
        else:
            self.overlay.set_status("No face detected", "#ff5577")

        if self.recording and self.logger is not None:
            self.logger.add_sample(sample)

    def _poll_hotkeys(self):
        now = time.time()

        def fire(name, fn):
            last = self._hotkey_cooldown.get(name, 0)
            if now - last < 0.4:
                return
            try:
                if keyboard.is_pressed(name):
                    self._hotkey_cooldown[name] = now
                    fn()
            except Exception:
                pass

        fire('f8', self._toggle_overlay)
        fire('f9', self._toggle_recording)
        fire('f10', self._quit)
        fire('f11', self._show_calibration)

    def _toggle_overlay(self):
        if self.overlay.isVisible():
            self.overlay.hide()
        else:
            self.overlay.show()

    def _toggle_recording(self):
        if not self.engine.is_calibrated:
            self.overlay.set_status("Cannot record — not calibrated", "#ff5577")
            return
        if not self.recording:
            ts = time.strftime("%Y%m%d_%H%M%S")
            self.current_csv = SESSIONS_DIR / f"session_{ts}.csv"
            self.logger = MetricsLogger(str(self.current_csv))
            self.recording = True
            self.overlay.set_recording(True)
            self.overlay.set_status(f"Recording -> {self.current_csv.name}", "#00ff88")
        else:
            stats = self.logger.close()
            summary = summarize(stats)
            self.recording = False
            self.logger = None
            self.overlay.set_recording(False)
            self.overlay.set_status(
                f"Saved {self.current_csv.name}  "
                f"(stress={summary['stress_score']}, blinks={summary['blink_count']})",
                "#00ff88",
            )
            print(f"\n[Session saved] {self.current_csv}")
            print(f"  Run: python report.py \"{self.current_csv}\"")

    def _show_calibration(self):
        self.engine.reset_calibration()
        self.calib_window.calib_btn.setText("1. Lock eye spheres  (C)")
        self.calib_window.calib_btn.setEnabled(True)
        self.calib_window.screen_btn.setText("2. Calibrate screen center  (S)")
        self.calib_window.screen_btn.setEnabled(False)
        self.calib_window.done_btn.setEnabled(False)
        self.calib_window.step_label.setText(
            "Step 1: Look at the CENTER of your screen, then press the button below."
        )
        self.engine.preview_enabled = True
        self.calib_window.show()
        self.calib_window.raise_()
        self.calib_window.activateWindow()

    def _quit(self):
        if self.recording and self.logger:
            stats = self.logger.close()
            print(f"[Auto-saved on quit] {self.current_csv}")
        self.engine.stop()
        self.app.quit()

    def run(self):
        return self.app.exec()


def main():
    ctrl = AppController()
    sys.exit(ctrl.run())


if __name__ == "__main__":
    main()

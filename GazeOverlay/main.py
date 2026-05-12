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
from calibration import CalibrationModel
from calibration_window import NinePointCalibration
from continuous_calibration import ContinuousCalibrator, ClickVerdict


def _app_dir() -> Path:
    """Folder where the EXE (or main.py) lives — sessions go next to it.
    Under PyInstaller --onefile, __file__ points at a temp unpack dir that
    gets wiped on exit, so we use sys.executable instead when frozen."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


SESSIONS_DIR = _app_dir() / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)
LOG_PATH = _app_dir() / "app.log"


def _log(msg: str, exc: bool = False):
    """Append a line to app.log next to the EXE. Silent on failure."""
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
            if exc:
                import traceback
                f.write(traceback.format_exc())
                f.write("\n")
    except OSError:
        pass


class CalibrationWindow(QWidget):
    """Small window with webcam preview. The user confirms framing and clicks
    'Lock eye spheres'; the controller then launches the fullscreen 9-point
    sequence."""

    eye_lock_requested = pyqtSignal()
    skipped = pyqtSignal()

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

        self.step_label = QLabel(
            "Make sure your face is centered and well lit, then look at the\n"
            "CENTER of your screen and press 'Lock eye spheres'."
        )
        self.step_label.setObjectName("step")
        self.step_label.setWordWrap(True)
        layout.addWidget(self.step_label)

        sep = QFrame(); sep.setObjectName("sep"); sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)

        btn_row = QHBoxLayout()
        self.calib_btn = QPushButton("Lock eye spheres  (C)")
        self.calib_btn.setObjectName("primary")
        self.calib_btn.clicked.connect(self._do_eye_calib)
        btn_row.addWidget(self.calib_btn)

        self.skip_btn = QPushButton("Use saved calibration")
        self.skip_btn.clicked.connect(lambda: self.skipped.emit())
        btn_row.addWidget(self.skip_btn)
        layout.addLayout(btn_row)

        hint = QLabel(
            "After lock, a 9-point fullscreen calibration runs (~15 sec).\n"
            "Hotkeys later:  F8 toggle overlay  -  F9 record  -  F10 quit  -  F11 recalibrate"
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
        # Heartbeat: print once per second so we can see the QTimer is alive
        now = time.time()
        if not hasattr(self, "_last_tick_hb"):
            self._last_tick_hb = 0.0
            self._tick_count = 0
        self._tick_count += 1
        if now - self._last_tick_hb >= 1.0:
            print(f"[preview-tick] {self._tick_count} tick/sec, "
                  f"engine.preview_enabled={self.engine.preview_enabled}",
                  flush=True)
            self._tick_count = 0
            self._last_tick_hb = now

        try:
            frame = self.engine.get_preview_frame()
            if frame is None:
                return
            # Mirror horizontally so it feels like a mirror
            frame = cv2.flip(frame, 1)
            # IMPORTANT: must be contiguous in memory for QImage; the engine
            # produces fresh arrays so this is usually true, but cv2.flip
            # and cv2.cvtColor return contiguous arrays anyway.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            # Wrap and IMMEDIATELY copy — QImage(ndarray.data, ...) is a view
            # that goes invalid when `rgb` goes out of scope. .copy() detaches.
            img = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
            pix = QPixmap.fromImage(img).scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.preview.setPixmap(pix)
        except Exception as e:
            print(f"[preview] tick failed: {e!r}", flush=True)

    def _poll_keys(self):
        if self.isActiveWindow():
            try:
                if keyboard.is_pressed('c') and self.calib_btn.isEnabled():
                    self._do_eye_calib()
            except Exception:
                pass

    def _do_eye_calib(self):
        self.engine.request_eye_calibration()
        QTimer.singleShot(200, self._after_eye_calib)

    def _after_eye_calib(self):
        if self.engine.eye_locked:
            self.calib_btn.setText("[OK] Eye spheres locked")
            self.calib_btn.setEnabled(False)
            self.step_label.setText("Starting 9-point calibration...")
            self.eye_lock_requested.emit()

    def set_skip_enabled(self, enabled: bool):
        """Show / hide the 'Use saved calibration' button based on whether
        a fitted model was loaded from disk."""
        self.skip_btn.setVisible(enabled)

    def hide_for_9point(self):
        """Hide preview + stop the engine's annotation overhead during the
        fullscreen calibration."""
        self.engine.preview_enabled = False
        self.hide()

    def closeEvent(self, e):
        self.engine.preview_enabled = False
        super().closeEvent(e)


class AppController(QObject):
    """Coordinates engine, overlay, calibration window, and hotkeys."""

    sample_signal = pyqtSignal(object)   # GazeSample (queued, thread-safe)
    click_signal = pyqtSignal(int, int)  # mouse click on main thread

    def __init__(self):
        super().__init__()
        self.app = QApplication.instance() or QApplication(sys.argv)

        # Single coord system throughout: Qt LOGICAL pixels (whatever Qt
        # reports for geometry, e.g. 1536x864 on a 1.25x scaled 1920x1080).
        # The only place we touch physical pixels is when pynput delivers a
        # raw mouse click — we divide by devicePixelRatio there and treat
        # everything else (model, 9-point targets, overlay paint) in logical
        # space. This keeps the engine + calibration code agnostic of DPI.
        primary = QGuiApplication.primaryScreen()
        geo = primary.geometry()
        self._dpr = primary.devicePixelRatio()
        sw = geo.width()
        sh = geo.height()
        _log(f"Screen geometry: {sw}x{sh} logical, dpr={self._dpr}")

        # Try to restore a saved calibration model
        saved_model = CalibrationModel.load(sw, sh)
        self.has_saved_model = saved_model is not None
        model = saved_model if saved_model is not None else CalibrationModel(
            screen_w=sw, screen_h=sh,
        )

        self.engine = GazeEngine(
            screen_w=sw, screen_h=sh,
            on_sample=self._on_sample_thread,
            model=model,
        )
        self.overlay = GazeOverlay()
        self.calib_window = CalibrationWindow(self.engine)
        self.calib_window.set_skip_enabled(self.has_saved_model)
        self.calib_window.eye_lock_requested.connect(self._start_9point)
        self.calib_window.skipped.connect(self._use_saved_model)

        self.nine_point: NinePointCalibration = None  # created lazily

        # F2 burst click-to-correct state
        self._click_correct_active = False
        self._click_correct_until = 0.0
        self._mouse_listener = None

        # Continuous (always-on) calibration. Default ON.
        self.continuous = ContinuousCalibrator(
            self.engine.model, sw, sh,
            on_accept=self._on_continuous_accept,
            on_reject=self._on_continuous_reject,
        )
        self.continuous.set_enabled(True)

        self.recording = False
        self.logger: MetricsLogger = None
        self.current_csv: Path = None

        self.sample_signal.connect(self._on_sample_main, type=Qt.ConnectionType.QueuedConnection)
        self.click_signal.connect(self._on_click_main, type=Qt.ConnectionType.QueuedConnection)

        # Engine starts immediately; user calibrates via the window.
        self.engine.start()
        self._start_global_click_listener()
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
            self._update_tracking_status()
        elif sample.face_visible and not sample.calibrated:
            self.overlay.set_status("Calibrate first  (see window)", "#ffaa00")
        else:
            self.overlay.set_status("No face detected", "#ff5577")

        # Feed the continuous calibrator's fixation buffer
        self.continuous.feed_sample(
            sample.timestamp, sample.screen_x, sample.screen_y,
            sample.yaw_deg, sample.pitch_deg, sample.calibrated,
        )

        if self.recording and self.logger is not None:
            try:
                self.logger.add_sample(sample)
            except Exception as e:
                _log(f"add_sample failed: {e!r}", exc=True)
                self.overlay.set_status("Logger error — see log", "#ff5577")

    def _update_tracking_status(self):
        """Compose the tracking status text including continuous-learning stats."""
        if self._click_correct_active:
            return  # F2 burst handles its own status
        if self.continuous.enabled:
            n = self.overlay._learn_count if hasattr(self.overlay, "_learn_count") else 0
            self.overlay.set_status(f"Tracking  (learning +{n})", "#00ff88")
        else:
            self.overlay.set_status("Tracking  (learning paused — F3)", "#cccccc")

    # ---- continuous-calibration callbacks ----

    def _on_continuous_accept(self, v: ClickVerdict):
        self.overlay.show_learn_event(v.mx, v.my, "accept")
        if v.refit_ran:
            # Persist periodically so we don't lose progress on crash
            try:
                self.engine.model.save()
            except Exception as e:
                _log(f"continuous save failed: {e!r}")

    def _on_continuous_reject(self, v: ClickVerdict):
        # Silent unless face was visible — avoids spam on background clicks
        # while user isn't even at the screen.
        if v.reason not in ("disabled", "off-screen"):
            self.overlay.show_learn_event(v.mx, v.my, "reject")

    # ---- global mouse listener (always-on) ----

    def _start_global_click_listener(self):
        try:
            from pynput import mouse  # type: ignore
        except ImportError:
            _log("pynput not installed — continuous calibration disabled. "
                 "pip install pynput to enable.")
            self.continuous.set_enabled(False)
            return

        def on_click(x, y, button, pressed):
            if not pressed:
                return
            # pynput on Windows reports PHYSICAL pixel coords. Everything
            # else in this app works in Qt LOGICAL pixels. Convert here at
            # the single boundary where physical coords enter the system.
            lx = int(x / self._dpr)
            ly = int(y / self._dpr)
            # Marshal to Qt main thread; pynput runs in its own thread
            self.click_signal.emit(lx, ly)

        self._mouse_listener = mouse.Listener(on_click=on_click)
        self._mouse_listener.start()
        _log("Global click listener started for continuous calibration")

    def _on_click_main(self, mx: int, my: int):
        """Runs on Qt main thread for every real mouse click anywhere."""
        # F2 burst takes priority if active (different gating logic)
        if self._click_correct_active:
            yaw, pitch = self.engine.latest_angles()
            self.engine.model.add_click_correction(mx, my, yaw, pitch)
            ok = self.engine.model.fit()
            if ok:
                self.engine.model.save()
                self.overlay.show_learn_event(mx, my, "accept")
            return
        # Otherwise feed the continuous calibrator (its own gating decides)
        self.continuous.on_click(mx, my)

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

        fire('f2',  self._toggle_click_correct)
        fire('f3',  self._toggle_continuous)
        fire('f8',  self._toggle_overlay)
        fire('f9',  self._toggle_recording)
        fire('f10', self._quit)
        fire('f11', self._show_calibration)

        # Auto-disable F2 burst after timeout
        if self._click_correct_active and now > self._click_correct_until:
            self._click_correct_active = False
            self.overlay.set_status("F2 burst timed out", "#cccccc")

    def _toggle_continuous(self):
        new = not self.continuous.enabled
        self.continuous.set_enabled(new)
        if new:
            self.overlay.set_status("Continuous calibration: ON", "#00ff88")
        else:
            self.overlay.set_status("Continuous calibration: PAUSED  (F3)", "#cccccc")
        _log(f"Continuous calibration {'enabled' if new else 'disabled'}")

    def _toggle_overlay(self):
        if self.overlay.isVisible():
            self.overlay.hide()
        else:
            self.overlay.show()

    def _toggle_recording(self):
        if not self.engine.is_calibrated:
            self.overlay.set_status("Cannot record — not calibrated", "#ff5577")
            return
        try:
            if not self.recording:
                ts = time.strftime("%Y%m%d_%H%M%S")
                self.current_csv = SESSIONS_DIR / f"session_{ts}.csv"
                self.logger = MetricsLogger(str(self.current_csv))
                self.recording = True
                self.overlay.set_recording(True)
                self.overlay.set_status(f"Recording -> {self.current_csv.name}", "#00ff88")
                _log(f"Recording started: {self.current_csv}")
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
                _log(f"Session saved: {self.current_csv} (samples={summary['samples']})")
                print(f"\n[Session saved] {self.current_csv}")
                print(f"  Run: python report.py \"{self.current_csv}\"")
        except Exception as e:
            self.recording = False
            self.logger = None
            self.overlay.set_recording(False)
            self.overlay.set_status(f"Recording error — see log", "#ff5577")
            _log(f"Recording toggle failed: {e!r}", exc=True)

    def _show_calibration(self):
        # Wipe the current model — full re-calibration from scratch
        self.engine.reset_calibration()
        self.has_saved_model = False
        self.calib_window.calib_btn.setText("Lock eye spheres  (C)")
        self.calib_window.calib_btn.setEnabled(True)
        self.calib_window.set_skip_enabled(False)
        self.calib_window.step_label.setText(
            "Look at the CENTER of your screen and press 'Lock eye spheres'."
        )
        self.engine.preview_enabled = True
        self.calib_window.show()
        self.calib_window.raise_()
        self.calib_window.activateWindow()

    # ---- 9-point flow ----

    def _start_9point(self):
        """Called after eye-sphere lock succeeds in the small preview window."""
        self.calib_window.hide_for_9point()
        self.overlay.hide()
        if self.nine_point is None:
            self.nine_point = NinePointCalibration(self.engine, self.engine.model)
            self.nine_point.finished.connect(self._on_9point_done)
        self.nine_point.start()

    def _on_9point_done(self, success: bool):
        self.overlay.show()
        if success:
            self.engine.model.save()
            _log(f"9-point calibration done; saved {len(self.engine.model.samples)} samples")
            self.overlay.set_status("Calibrated  (F2 = click-to-correct)", "#00ff88")
        else:
            _log("9-point calibration failed or cancelled")
            self.overlay.set_status("Calibration cancelled — F11 to retry", "#ff5577")

    def _use_saved_model(self):
        """User clicked 'Use saved calibration'. We still need eye-spheres
        locked at the current head pose, so request a quick lock and skip
        straight to tracking."""
        self.engine.request_eye_calibration()
        QTimer.singleShot(250, self.calib_window.hide_for_9point)
        QTimer.singleShot(300, lambda: self.overlay.set_status(
            "Using saved calibration  (F2 = click-to-correct)", "#00ff88"))
        _log("Reused saved calibration model")

    # ---- click-to-correct ----

    def _toggle_click_correct(self):
        """F2: enable a 10-second 'burst' mode where every click is treated
        as a higher-weight calibration sample (weight=3.0 instead of the
        continuous 1.5). Useful when you've just sat down and want fast
        refinement before continuous learning kicks in.

        The global click listener is always-on; F2 just changes the routing
        in _on_click_main."""
        if self._click_correct_active:
            self._click_correct_active = False
            self.overlay.set_status("F2 burst OFF", "#cccccc")
            return
        if not self.engine.is_calibrated:
            self.overlay.set_status("Calibrate first (F11)", "#ff5577")
            return
        self._click_correct_active = True
        self._click_correct_until = time.time() + 10.0
        self.overlay.set_status("F2 burst ON (10s) — click where you're looking", "#00ddff")

    def _stop_global_click_listener(self):
        if self._mouse_listener is not None:
            try:
                self._mouse_listener.stop()
            except Exception:
                pass
            self._mouse_listener = None

    def _quit(self):
        if self.recording and self.logger:
            stats = self.logger.close()
            print(f"[Auto-saved on quit] {self.current_csv}")
        self._stop_global_click_listener()
        if self.engine.model.is_fitted:
            self.engine.model.save()
        self.engine.stop()
        self.app.quit()

    def run(self):
        return self.app.exec()


def main():
    _log(f"GazeOverlay starting (frozen={getattr(sys, 'frozen', False)}, app_dir={_app_dir()})")
    try:
        ctrl = AppController()
        sys.exit(ctrl.run())
    except Exception as e:
        _log(f"Fatal startup error: {e!r}", exc=True)
        raise


if __name__ == "__main__":
    main()

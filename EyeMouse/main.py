"""
EyeMouse — entry point.

Reuses GazeOverlay's gaze_engine + calibration window; swaps in DwellOverlay
and adds the click engine and settings window.

Hotkeys:
  F8  - toggle EyeMouse on/off  (cursor control + clicks)
  F9  - show/hide settings window
  F11 - recalibrate
  F12 - PANIC OFF  (force-disable everything; use if cursor goes wild)
  F10 - quit

Safety: starts DISABLED. You must press F8 to activate after calibrating.
"""

import os
import sys
import time
from pathlib import Path

import keyboard
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QApplication

# Reuse modules from GazeOverlay/
HERE = os.path.dirname(os.path.abspath(__file__))
GAZE_OVERLAY_DIR = os.path.join(os.path.dirname(HERE), "GazeOverlay")
sys.path.insert(0, GAZE_OVERLAY_DIR)
sys.path.insert(0, HERE)

from gaze_engine import GazeEngine, GazeSample  # noqa: E402
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location("gaze_overlay_main", os.path.join(GAZE_OVERLAY_DIR, "main.py"))
_gov = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_gov)
CalibrationWindow = _gov.CalibrationWindow
from input_backend import make_backend  # noqa: E402
from click_engine import ClickEngine, ClickEvent  # noqa: E402
from dwell_overlay import DwellOverlay  # noqa: E402
from settings_window import SettingsWindow, load_config, save_config  # noqa: E402


class EyeMouseController(QObject):
    sample_signal = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.app = QApplication.instance() or QApplication(sys.argv)

        self.cfg = load_config()
        self.cfg.enabled = False  # always start safe

        screen = QGuiApplication.primaryScreen().geometry()
        self.engine = GazeEngine(
            screen_w=screen.width(),
            screen_h=screen.height(),
            on_sample=self._on_sample_thread,
        )
        self.overlay = DwellOverlay()
        self.overlay.set_eye_mouse_active(False)

        self.backend_name = "sendinput" if sys.platform == "win32" else "pyautogui"
        self.backend = make_backend(self.backend_name)
        self.click_engine = ClickEngine(self.backend, self.cfg, self._on_click_event)

        self.calib_window = CalibrationWindow(self.engine)
        self.settings_window = SettingsWindow(
            self.cfg,
            on_change=self._on_settings_change,
            backend_name=self.backend_name,
            on_backend_change=self._switch_backend,
        )

        self.sample_signal.connect(self._on_sample_main, type=Qt.ConnectionType.QueuedConnection)

        self.engine.start()
        self.overlay.set_status("Calibrate first  (see window)", "#ffaa00")
        self.overlay.show()
        self.calib_window.show()
        self.calib_window.raise_()
        self.calib_window.activateWindow()

        # Hotkey poller
        self._hotkey_timer = QTimer()
        self._hotkey_timer.timeout.connect(self._poll_hotkeys)
        self._hotkey_timer.start(80)
        self._hotkey_cooldown = {}

    # ---- engine sample callback ----

    def _on_sample_thread(self, sample: GazeSample):
        self.sample_signal.emit(sample)

    def _on_sample_main(self, sample: GazeSample):
        # Always update the visual gaze dot
        if sample.face_visible and sample.calibrated:
            self.overlay.update_gaze(sample.screen_x, sample.screen_y)
        # Run click logic (no-op if disabled)
        self.click_engine.feed_sample(sample)
        # Push dwell progress for the ring
        self.overlay.update_dwell(self.click_engine.dwell_progress)
        self.overlay.set_dragging(self.click_engine.is_dragging)

        # Status badge text
        if not self.cfg.enabled:
            self.overlay.set_status("EyeMouse OFF — press F8 to enable", "#cccccc")
        elif not sample.face_visible:
            self.overlay.set_status("No face — paused", "#ff5577")
        elif not sample.calibrated:
            self.overlay.set_status("Not calibrated  (F11)", "#ffaa00")
        else:
            mode = "DRAG" if self.click_engine.is_dragging else "ACTIVE"
            self.overlay.set_status(f"EyeMouse {mode}", "#00ff88")

    # ---- click events ----

    def _on_click_event(self, ev: ClickEvent):
        self.overlay.show_click_flash(ev.kind, ev.x, ev.y)
        print(f"[EyeMouse] {ev.kind} at ({ev.x}, {ev.y})")

    # ---- settings ----

    def _on_settings_change(self):
        # cfg is mutated in place by SettingsWindow; nothing to do here besides
        # save (we save on slider release; for now save on every change is fine).
        save_config(self.cfg)

    def _switch_backend(self, name: str):
        self.backend_name = name
        self.backend = make_backend(name)
        self.click_engine.backend = self.backend
        print(f"[EyeMouse] click backend -> {name}")

    # ---- hotkeys ----

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

        fire('f8',  self._toggle_eye_mouse)
        fire('f9',  self._toggle_settings)
        fire('f10', self._quit)
        fire('f11', self._show_calibration)
        fire('f12', self._panic_off)

    def _toggle_eye_mouse(self):
        if not self.engine.is_calibrated:
            self.overlay.set_status("Cannot enable — calibrate first (F11)", "#ff5577")
            return
        new_state = not self.cfg.enabled
        self.click_engine.set_enabled(new_state)
        self.overlay.set_eye_mouse_active(new_state)
        print(f"[EyeMouse] {'ENABLED' if new_state else 'DISABLED'}")

    def _toggle_settings(self):
        if self.settings_window.isVisible():
            self.settings_window.hide()
        else:
            self.settings_window.show()
            self.settings_window.raise_()
            self.settings_window.activateWindow()

    def _show_calibration(self):
        # Disable EyeMouse during calibration for safety
        self.click_engine.set_enabled(False)
        self.overlay.set_eye_mouse_active(False)
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

    def _panic_off(self):
        """F12: force-stop everything — release any held drag, disable clicks."""
        if self.click_engine.is_dragging:
            self.backend.left_up()
        self.click_engine.set_enabled(False)
        self.overlay.set_eye_mouse_active(False)
        print("[EyeMouse] PANIC OFF — all input disabled")

    def _quit(self):
        self._panic_off()
        save_config(self.cfg)
        self.engine.stop()
        self.app.quit()

    def run(self):
        return self.app.exec()


def main():
    print("=" * 60)
    print("EyeMouse — eye-controlled cursor + click")
    print("=" * 60)
    print("Hotkeys: F8 toggle | F9 settings | F10 quit | F11 recalibrate | F12 panic")
    print()
    ctrl = EyeMouseController()
    sys.exit(ctrl.run())


if __name__ == "__main__":
    main()

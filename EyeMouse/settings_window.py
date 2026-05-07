"""
Settings window for EyeMouse: tune dwell, blink, wink thresholds at runtime.
Saves to ~/.eye_mouse_config.json so settings persist between sessions.
"""

import json
import os
from pathlib import Path
from dataclasses import asdict
from typing import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QCheckBox,
    QComboBox, QPushButton, QFrame, QGroupBox,
)

from click_engine import EyeMouseConfig


CONFIG_PATH = Path.home() / ".eye_mouse_config.json"


def load_config() -> EyeMouseConfig:
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text())
            return EyeMouseConfig(**{k: v for k, v in data.items()
                                     if k in EyeMouseConfig.__annotations__})
        except (json.JSONDecodeError, TypeError):
            pass
    return EyeMouseConfig()


def save_config(cfg: EyeMouseConfig):
    try:
        CONFIG_PATH.write_text(json.dumps(asdict(cfg), indent=2))
    except OSError:
        pass


class _LabeledSlider(QWidget):
    def __init__(self, label: str, vmin: int, vmax: int, value: int,
                 suffix: str = "", on_change: Callable[[int], None] = None):
        super().__init__()
        self.on_change = on_change
        self.suffix = suffix

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        row = QHBoxLayout()
        self.title = QLabel(label)
        self.value_label = QLabel(f"{value}{suffix}")
        self.value_label.setStyleSheet("color: #00ff88; font-weight: bold;")
        row.addWidget(self.title)
        row.addStretch()
        row.addWidget(self.value_label)
        layout.addLayout(row)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(vmin, vmax)
        self.slider.setValue(value)
        self.slider.valueChanged.connect(self._changed)
        layout.addWidget(self.slider)

    def _changed(self, v: int):
        self.value_label.setText(f"{v}{self.suffix}")
        if self.on_change:
            self.on_change(v)


class SettingsWindow(QWidget):
    def __init__(self, cfg: EyeMouseConfig, on_change: Callable[[], None],
                 backend_name: str, on_backend_change: Callable[[str], None]):
        super().__init__()
        self.cfg = cfg
        self.on_change = on_change
        self.on_backend_change = on_backend_change

        self.setWindowTitle("EyeMouse — Settings")
        self.setFixedSize(420, 640)
        self.setStyleSheet("""
            QWidget { background: #1a1d24; color: #e0e6ed; font-family: 'Segoe UI'; font-size: 12px; }
            QGroupBox { border: 1px solid #2a3142; border-radius: 4px; margin-top: 14px; padding-top: 8px; }
            QGroupBox::title { color: #00ff88; subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QSlider::groove:horizontal { background: #2a3142; height: 4px; border-radius: 2px; }
            QSlider::handle:horizontal {
                background: #00ff88; width: 14px; margin: -6px 0; border-radius: 7px;
            }
            QSlider::sub-page:horizontal { background: #00aa55; border-radius: 2px; }
            QCheckBox { spacing: 8px; }
            QComboBox {
                background: #2a3142; border: 1px solid #3a4458; padding: 4px 8px;
                border-radius: 3px;
            }
            QPushButton {
                background: #2a3142; border: 1px solid #3a4458; padding: 6px 12px;
                border-radius: 3px;
            }
            QPushButton:hover { background: #3a4458; }
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)

        title = QLabel("EyeMouse Settings")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #00ff88;")
        root.addWidget(title)

        # ---- Movement ----
        gm = QGroupBox("Cursor movement")
        ml = QVBoxLayout(gm)
        self._add_slider(ml, "Smoothing  (lower = snappier)", 5, 80,
                         int(cfg.move_smooth * 100), "%",
                         lambda v: self._set("move_smooth", v / 100.0))
        root.addWidget(gm)

        # ---- Dwell ----
        gd = QGroupBox("Dwell click  (look at one spot)")
        dl = QVBoxLayout(gd)
        self._add_slider(dl, "Dwell radius", 20, 150, cfg.dwell_radius_px, " px",
                         lambda v: self._set("dwell_radius_px", v))
        self._add_slider(dl, "Dwell time", 200, 2000, cfg.dwell_ms, " ms",
                         lambda v: self._set("dwell_ms", v))
        self._add_slider(dl, "Cooldown after click", 200, 2000, cfg.cooldown_ms, " ms",
                         lambda v: self._set("cooldown_ms", v))
        cb_smart = QCheckBox("Smart dwell  (pause when eyes closed)")
        cb_smart.setChecked(cfg.smart_dwell)
        cb_smart.toggled.connect(lambda v: self._set("smart_dwell", v))
        dl.addWidget(cb_smart)
        root.addWidget(gd)

        # ---- Blink / wink ----
        gb = QGroupBox("Blink + wink gestures")
        bl = QVBoxLayout(gb)
        self._add_slider(bl, "Blink threshold (EAR x100)", 8, 30,
                         int(cfg.blink_left_threshold * 100), "",
                         lambda v: (self._set("blink_left_threshold", v / 100.0),
                                    self._set("blink_right_threshold", v / 100.0)))
        self._add_slider(bl, "Long-blink right-click", 200, 1500,
                         cfg.long_blink_ms, " ms",
                         lambda v: self._set("long_blink_ms", v))
        self._add_slider(bl, "Wink drag toggle", 150, 1000, cfg.wink_ms, " ms",
                         lambda v: self._set("wink_ms", v))
        root.addWidget(gb)

        # ---- Behaviour ----
        gx = QGroupBox("Behaviour")
        xl = QVBoxLayout(gx)
        cb_pause = QCheckBox("Pause when no face detected")
        cb_pause.setChecked(cfg.pause_when_no_face)
        cb_pause.toggled.connect(lambda v: self._set("pause_when_no_face", v))
        xl.addWidget(cb_pause)

        be_row = QHBoxLayout()
        be_row.addWidget(QLabel("Click backend:"))
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["sendinput  (Windows, recommended)", "pyautogui  (fallback)"])
        self.backend_combo.setCurrentIndex(0 if backend_name == "sendinput" else 1)
        self.backend_combo.currentIndexChanged.connect(
            lambda i: self.on_backend_change("sendinput" if i == 0 else "pyautogui"))
        be_row.addWidget(self.backend_combo)
        xl.addLayout(be_row)
        root.addWidget(gx)

        # ---- Buttons ----
        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save settings")
        save_btn.clicked.connect(lambda: save_config(self.cfg))
        btn_row.addWidget(save_btn)
        reset_btn = QPushButton("Reset to defaults")
        reset_btn.clicked.connect(self._reset)
        btn_row.addWidget(reset_btn)
        root.addLayout(btn_row)

        hint = QLabel(
            "Tip: enable EyeMouse with F8 only when needed. F12 = panic OFF.\n"
            "Cursor jitter? Lower the smoothing slider. Accidental clicks? "
            "Increase Dwell time."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #7a8a98; font-size: 10px;")
        root.addWidget(hint)
        root.addStretch()

    def _add_slider(self, layout, label, lo, hi, val, suf, cb):
        w = _LabeledSlider(label, lo, hi, val, suf, cb)
        layout.addWidget(w)

    def _set(self, key: str, value):
        setattr(self.cfg, key, value)
        if self.on_change:
            self.on_change()

    def _reset(self):
        defaults = EyeMouseConfig()
        # Update fields in place so the engine sees changes
        for k in EyeMouseConfig.__annotations__.keys():
            setattr(self.cfg, k, getattr(defaults, k))
        save_config(self.cfg)
        # Rebuild UI
        self.close()

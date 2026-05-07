"""
Click decision engine. Consumes GazeSamples, decides when to:

- move the cursor (always, when active)
- left-click via dwell  (gaze stays inside DWELL_RADIUS for DWELL_MS)
- right-click via long blink  (both eyes closed for LONG_BLINK_MS)
- toggle drag via wink  (one eye closed > WINK_MS while the other stays open)

The "Midas touch" problem (every glance triggers a click) is real, so this
module enforces:
  - clicks only when gaze has stabilized (low velocity over the dwell window)
  - cooldown after every click
  - blink/wink must end before another can be detected
  - all thresholds are configurable from the settings UI
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

from input_backend import MouseBackend


# Defaults — overridable from EyeMouseConfig
@dataclass
class EyeMouseConfig:
    enabled: bool = False                  # F8 toggle starts OFF for safety
    dwell_radius_px: int = 50              # gaze must stay inside this circle
    dwell_ms: int = 700                    # ...for this long to fire a click
    cooldown_ms: int = 600                 # post-click silence
    move_smooth: float = 0.35              # exponential smoothing for cursor (0..1)

    blink_left_threshold: float = 0.18     # EAR below this = closed
    blink_right_threshold: float = 0.18
    long_blink_ms: int = 600               # both eyes closed > this = right-click
    wink_ms: int = 350                     # one eye closed > this while other open = drag toggle
    wink_max_other_ear: float = 0.30       # the "open" eye must be above this

    smart_dwell: bool = True               # only dwell when both eyes are open & face visible
    pause_when_no_face: bool = True


@dataclass
class ClickEvent:
    kind: str                # "left", "right", "drag_start", "drag_end"
    x: int
    y: int
    ts: float = field(default_factory=time.time)


class ClickEngine:
    def __init__(self, backend: MouseBackend, config: EyeMouseConfig,
                 on_event: Optional[Callable[[ClickEvent], None]] = None):
        self.backend = backend
        self.cfg = config
        self.on_event = on_event or (lambda e: None)

        self._smoothed_x: Optional[float] = None
        self._smoothed_y: Optional[float] = None

        # Dwell state
        self._dwell_window: deque = deque()  # (ts, x, y)
        self._dwell_progress: float = 0.0    # 0..1 for visualizer
        self._last_click_ts: float = 0.0

        # Drag state
        self._dragging: bool = False
        self._wink_left_start: Optional[float] = None
        self._wink_right_start: Optional[float] = None

        # Blink state
        self._both_closed_start: Optional[float] = None

    # ---- public API ----

    def set_enabled(self, on: bool):
        self.cfg.enabled = on
        if not on and self._dragging:
            self._end_drag(0, 0)
        self._reset_state()

    def set_config(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)

    @property
    def dwell_progress(self) -> float:
        """0..1, exposed for the visualizer."""
        return self._dwell_progress

    @property
    def is_dragging(self) -> bool:
        return self._dragging

    def feed_sample(self, sample) -> None:
        """Called from the main thread for every GazeSample."""
        if not self.cfg.enabled:
            self._dwell_progress = 0.0
            return

        if not sample.face_visible:
            if self.cfg.pause_when_no_face:
                self._reset_state()
            return

        if not sample.calibrated:
            return

        x, y = sample.screen_x, sample.screen_y
        sx, sy = self._smooth(x, y)
        self.backend.move_to(int(sx), int(sy))

        now = sample.timestamp
        ear_l = sample.ear_left
        ear_r = sample.ear_right
        l_closed = ear_l < self.cfg.blink_left_threshold
        r_closed = ear_r < self.cfg.blink_right_threshold

        # ---- 1. Long-blink right-click (both eyes) ----
        if l_closed and r_closed:
            if self._both_closed_start is None:
                self._both_closed_start = now
        else:
            if self._both_closed_start is not None:
                dur_ms = (now - self._both_closed_start) * 1000.0
                self._both_closed_start = None
                if dur_ms >= self.cfg.long_blink_ms and self._cooldown_ok(now):
                    self.backend.right_click()
                    self._last_click_ts = now
                    self.on_event(ClickEvent("right", int(sx), int(sy), now))
                    self._dwell_window.clear()
                    return

        # ---- 2. Wink drag toggle (one eye closed, other open) ----
        if l_closed and not r_closed and ear_r > self.cfg.wink_max_other_ear:
            if self._wink_left_start is None:
                self._wink_left_start = now
        else:
            if self._wink_left_start is not None:
                dur_ms = (now - self._wink_left_start) * 1000.0
                self._wink_left_start = None
                if dur_ms >= self.cfg.wink_ms and self._cooldown_ok(now):
                    if self._dragging:
                        self._end_drag(int(sx), int(sy))
                    else:
                        self._start_drag(int(sx), int(sy))
                    self._last_click_ts = now
                    return

        if r_closed and not l_closed and ear_l > self.cfg.wink_max_other_ear:
            if self._wink_right_start is None:
                self._wink_right_start = now
        else:
            if self._wink_right_start is not None:
                # Right wink reserved (could be another action; for now ignored)
                self._wink_right_start = None

        # ---- 3. Dwell click ----
        if self.cfg.smart_dwell and (l_closed or r_closed):
            # Don't accumulate dwell while either eye is closed
            self._dwell_window.clear()
            self._dwell_progress = 0.0
            return

        self._dwell_window.append((now, sx, sy))
        # Drop samples older than dwell_ms
        cutoff = now - (self.cfg.dwell_ms / 1000.0)
        while self._dwell_window and self._dwell_window[0][0] < cutoff:
            self._dwell_window.popleft()

        # Compute dispersion within the window
        if len(self._dwell_window) >= 3:
            xs = [p[1] for p in self._dwell_window]
            ys = [p[2] for p in self._dwell_window]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            max_d = max(math.hypot(px - cx, py - cy) for _, px, py in self._dwell_window)
            window_span_ms = (self._dwell_window[-1][0] - self._dwell_window[0][0]) * 1000.0

            if max_d <= self.cfg.dwell_radius_px:
                self._dwell_progress = min(1.0, window_span_ms / self.cfg.dwell_ms)
                if window_span_ms >= self.cfg.dwell_ms and self._cooldown_ok(now):
                    self.backend.left_click()
                    self._last_click_ts = now
                    self.on_event(ClickEvent("left", int(cx), int(cy), now))
                    self._dwell_window.clear()
                    self._dwell_progress = 0.0
            else:
                # Movement broke the dwell — start fresh
                self._dwell_window.clear()
                self._dwell_window.append((now, sx, sy))
                self._dwell_progress = 0.0
        else:
            self._dwell_progress = 0.0

    # ---- internals ----

    def _smooth(self, x: int, y: int):
        a = self.cfg.move_smooth
        if self._smoothed_x is None:
            self._smoothed_x = float(x)
            self._smoothed_y = float(y)
        else:
            self._smoothed_x = self._smoothed_x * (1 - a) + x * a
            self._smoothed_y = self._smoothed_y * (1 - a) + y * a
        return self._smoothed_x, self._smoothed_y

    def _cooldown_ok(self, now: float) -> bool:
        return (now - self._last_click_ts) * 1000.0 >= self.cfg.cooldown_ms

    def _start_drag(self, x: int, y: int):
        self.backend.left_down()
        self._dragging = True
        self.on_event(ClickEvent("drag_start", x, y))

    def _end_drag(self, x: int, y: int):
        self.backend.left_up()
        self._dragging = False
        self.on_event(ClickEvent("drag_end", x, y))

    def _reset_state(self):
        self._dwell_window.clear()
        self._dwell_progress = 0.0
        self._both_closed_start = None
        self._wink_left_start = None
        self._wink_right_start = None

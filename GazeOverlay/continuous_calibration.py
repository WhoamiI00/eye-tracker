"""
ContinuousCalibrator — always-on background self-calibration.

Treats every real mouse click as a "the user was probably looking near here"
signal. Gates aggressively to avoid poisoning the model:

  - Must have a recent gaze fixation (so we have a stable angle).
  - Gaze must be reasonably close to the click (rejects muscle-memory or
    accidental clicks where the eyes were already elsewhere).
  - Must be inside the primary monitor.
  - Spatial-bin LRU keeps the sample pool balanced across the screen.
  - Throttled refit (max 1/sec) plus outlier pruning after each refit.

The hook is a single method `on_click(mx, my)` invoked by the global mouse
listener in main.py. Returns a verdict describing what happened, so the UI
can show a "+1" / "rejected" animation.
"""

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

from calibration import CalibrationModel


@dataclass
class ClickVerdict:
    accepted: bool
    mx: int
    my: int
    reason: str            # "ok" or rejection reason
    refit_ran: bool = False
    samples_dropped: int = 0


# ---- gating constants ----
FIXATION_WINDOW_S = 0.25       # gaze must have been stable in the last X sec
FIXATION_DISPERSION_PX = 80    # ...within this radius
MAX_GAZE_CLICK_DIST_PX = 350   # reject if gaze is too far from click
REFIT_MIN_INTERVAL_S = 0.8     # throttle refits
PRUNE_EVERY_N_REFITS = 5       # run outlier prune every Nth refit


class ContinuousCalibrator:
    """
    Drop-in for the existing CalibrationModel: keeps a short rolling buffer
    of recent gaze samples (to test fixation), and on_click decides whether
    to add a sample + refit. Thread-safety: methods are called from the Qt
    main thread (sample feed via signal, click via pynput marshalled into Qt).
    """

    def __init__(self, model: CalibrationModel, screen_w: int, screen_h: int,
                 on_accept: Optional[Callable[[ClickVerdict], None]] = None,
                 on_reject: Optional[Callable[[ClickVerdict], None]] = None):
        self.model = model
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.on_accept = on_accept or (lambda v: None)
        self.on_reject = on_reject or (lambda v: None)

        self.enabled = True
        self._gaze_buffer: deque = deque(maxlen=60)  # (ts, gx, gy, yaw, pitch)
        self._last_refit_ts: float = 0.0
        self._refit_count = 0

        # Stats
        self.accepted_count = 0
        self.rejected_count = 0

    # ---- per-frame gaze feed ----

    def feed_sample(self, ts: float, screen_x: int, screen_y: int,
                    yaw: float, pitch: float, calibrated: bool):
        """Called for every GazeSample. Maintains the fixation buffer."""
        if not calibrated:
            return
        self._gaze_buffer.append((ts, screen_x, screen_y, yaw, pitch))
        # Trim old entries beyond the fixation window
        cutoff = ts - FIXATION_WINDOW_S
        while self._gaze_buffer and self._gaze_buffer[0][0] < cutoff:
            self._gaze_buffer.popleft()

    # ---- click hook ----

    def on_click(self, mx: int, my: int) -> ClickVerdict:
        if not self.enabled:
            return self._reject(mx, my, "disabled")

        # Reject clicks outside primary monitor (multi-monitor case)
        if not (0 <= mx < self.screen_w and 0 <= my < self.screen_h):
            return self._reject(mx, my, "off-screen")

        if not self.model.is_fitted:
            return self._reject(mx, my, "not calibrated yet")

        # Need a recent fixation to extract a stable (yaw, pitch)
        if len(self._gaze_buffer) < 5:
            return self._reject(mx, my, "no recent gaze")

        # Was the gaze actually stable? Check dispersion in the buffer.
        xs = [p[1] for p in self._gaze_buffer]
        ys = [p[2] for p in self._gaze_buffer]
        disp = max(max(xs) - min(xs), max(ys) - min(ys))
        if disp > FIXATION_DISPERSION_PX:
            return self._reject(mx, my, "gaze unstable")

        # Mean gaze position over the fixation window
        gx = sum(xs) / len(xs)
        gy = sum(ys) / len(ys)

        # Mean raw angles (these are what we feed to the model)
        yaw = sum(p[3] for p in self._gaze_buffer) / len(self._gaze_buffer)
        pitch = sum(p[4] for p in self._gaze_buffer) / len(self._gaze_buffer)

        # Reject if gaze is wildly far from the click — almost certainly bad.
        dist = math.hypot(gx - mx, gy - my)
        if dist > MAX_GAZE_CLICK_DIST_PX:
            return self._reject(mx, my, f"too far ({int(dist)}px)")

        # All checks passed — add and refit
        self.model.add_continuous_sample(mx, my, yaw, pitch)

        now = time.time()
        refit_ran = False
        dropped = 0
        if now - self._last_refit_ts >= REFIT_MIN_INTERVAL_S:
            self._last_refit_ts = now
            ok = self.model.fit()
            if ok:
                refit_ran = True
                self._refit_count += 1
                if self._refit_count % PRUNE_EVERY_N_REFITS == 0:
                    dropped = self.model.prune_outliers()
                    if dropped:
                        # Refit one more time after pruning
                        self.model.fit()

        self.accepted_count += 1
        verdict = ClickVerdict(True, mx, my, "ok",
                               refit_ran=refit_ran, samples_dropped=dropped)
        self.on_accept(verdict)
        return verdict

    def set_enabled(self, enabled: bool):
        self.enabled = enabled

    def _reject(self, mx: int, my: int, reason: str) -> ClickVerdict:
        self.rejected_count += 1
        v = ClickVerdict(False, mx, my, reason)
        self.on_reject(v)
        return v

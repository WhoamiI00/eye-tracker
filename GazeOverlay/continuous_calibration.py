"""
ContinuousCalibrator — always-on background self-calibration.

Treats every real mouse click as a "user was probably looking near here"
signal. Gates aggressively:

  - face must be visible
  - gaze must have been a stable fixation in the last 250 ms
  - gaze must be within ~350 px of the click
  - click must be on the primary monitor

On accept: add a moderate-weight sample, refit (throttled), prune outliers
every 5 refits. Anchor (9-point) samples are protected from eviction.
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
    reason: str
    refit_ran: bool = False
    samples_dropped: int = 0


FIXATION_WINDOW_S = 0.25
# Pre-One-Euro screen samples can wobble ~100-130 px on a noisy webcam
# even when the user's gaze is genuinely fixed. Was 80 (too strict —
# rejected most clicks). Raised to 140.
FIXATION_DISPERSION_PX = 140
# Webcam baseline error is ~50-100 px and people naturally look slightly
# ahead/below of where they click. Was 350 (still rejected lots). 500 is
# permissive but the outlier-prune step removes truly bad samples.
MAX_GAZE_CLICK_DIST_PX = 500
REFIT_MIN_INTERVAL_S = 0.8
PRUNE_EVERY_N_REFITS = 5


class ContinuousCalibrator:
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
        self.accepted_count = 0
        self.rejected_count = 0

    def feed_sample(self, ts: float, screen_x: int, screen_y: int,
                    yaw: float, pitch: float, calibrated: bool):
        if not calibrated:
            return
        self._gaze_buffer.append((ts, screen_x, screen_y, yaw, pitch))
        cutoff = ts - FIXATION_WINDOW_S
        while self._gaze_buffer and self._gaze_buffer[0][0] < cutoff:
            self._gaze_buffer.popleft()

    def on_click(self, mx: int, my: int) -> ClickVerdict:
        if not self.enabled:
            return self._reject(mx, my, "disabled")
        if not (0 <= mx < self.screen_w and 0 <= my < self.screen_h):
            return self._reject(mx, my, "off-screen")
        if not self.model.is_fitted:
            return self._reject(mx, my, "not calibrated yet")
        if len(self._gaze_buffer) < 5:
            return self._reject(mx, my, "no recent gaze")

        xs = [p[1] for p in self._gaze_buffer]
        ys = [p[2] for p in self._gaze_buffer]
        disp = max(max(xs) - min(xs), max(ys) - min(ys))
        if disp > FIXATION_DISPERSION_PX:
            return self._reject(mx, my, "gaze unstable")

        gx = sum(xs) / len(xs)
        gy = sum(ys) / len(ys)
        yaw = sum(p[3] for p in self._gaze_buffer) / len(self._gaze_buffer)
        pitch = sum(p[4] for p in self._gaze_buffer) / len(self._gaze_buffer)

        dist = math.hypot(gx - mx, gy - my)
        if dist > MAX_GAZE_CLICK_DIST_PX:
            return self._reject(mx, my, f"too far ({int(dist)}px)")

        self.model.add_continuous_sample(mx, my, yaw, pitch)

        now = time.time()
        refit_ran = False
        dropped = 0
        if now - self._last_refit_ts >= REFIT_MIN_INTERVAL_S:
            self._last_refit_ts = now
            if self.model.fit():
                refit_ran = True
                self._refit_count += 1
                if self._refit_count % PRUNE_EVERY_N_REFITS == 0:
                    dropped = self.model.prune_outliers()
                    if dropped:
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

"""
CalibrationModel: maps raw gaze angles (yaw, pitch) to screen pixels.

Replaces the old single-point + hardcoded-FOV approach with a least-squares
polynomial fit. Standard approach in commercial eye trackers.

Two phases:
  1. **Bulk calibration**: user looks at N points, we collect ~30 frames of
     (yaw, pitch) per point and fit a polynomial mapping.
  2. **Click-to-correct refinement**: each real mouse click adds a weighted
     sample (assuming the user was looking near where they clicked) and we
     re-solve the least-squares problem incrementally.

Polynomial features (degree 2): [1, y, p, y*p, y^2, p^2]
where y = yaw_deg, p = pitch_deg. Six terms is enough for a smooth screen
warp; more invites overfitting on 9 points.

Persistence: saves to ~/.gaze_overlay_calibration.json so users don't
recalibrate every launch.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
import time as _time
from typing import List, Optional, Tuple

import numpy as np


CALIB_PATH = Path.home() / ".gaze_overlay_calibration.json"

# Each calibration point is a target screen position (sx, sy) and the average
# observed (yaw, pitch) while the user was looking at it.
@dataclass
class CalibSample:
    sx: float                # target screen x (pixels)
    sy: float                # target screen y (pixels)
    yaw: float               # observed yaw (degrees)
    pitch: float             # observed pitch (degrees)
    weight: float = 1.0      # >1 = trust more (e.g. click-to-correct fresh sample)
    is_anchor: bool = False  # 9-point calibration samples — never evicted
    bin_key: Optional[Tuple[int, int]] = None  # spatial grid cell (col, row) for LRU
    added_at: float = 0.0    # for LRU within a bin


def _features(yaw: float, pitch: float) -> np.ndarray:
    """Polynomial feature row for one (yaw, pitch). Keep in sync with model order."""
    return np.array([1.0, yaw, pitch, yaw * pitch, yaw * yaw, pitch * pitch], dtype=float)


@dataclass
class CalibrationModel:
    """Fitted (yaw, pitch) -> (sx, sy) mapping plus a sample buffer for refits."""

    samples: List[CalibSample] = field(default_factory=list)
    coef_x: Optional[np.ndarray] = None   # shape (6,) — [c0..c5] for sx
    coef_y: Optional[np.ndarray] = None   # shape (6,) — [c0..c5] for sy
    screen_w: int = 1920
    screen_h: int = 1080

    # F2 burst click-to-correct
    click_weight: float = 3.0             # fresh F2 burst clicks weigh more

    # Continuous (always-on) calibration knobs
    continuous_weight: float = 1.5        # less than F2 burst, more than anchors
    bins_x: int = 4                       # spatial grid: 4 cols x 3 rows = 12 bins
    bins_y: int = 3
    max_per_bin: int = 8                  # LRU cap per bin (anchors don't count)

    @property
    def is_fitted(self) -> bool:
        return self.coef_x is not None and self.coef_y is not None

    # ---- adding samples ----

    def _bin_for(self, sx: float, sy: float) -> Tuple[int, int]:
        col = max(0, min(self.bins_x - 1, int(sx / max(1, self.screen_w) * self.bins_x)))
        row = max(0, min(self.bins_y - 1, int(sy / max(1, self.screen_h) * self.bins_y)))
        return (col, row)

    def add_calibration_sample(self, sx: float, sy: float, yaw: float, pitch: float,
                               weight: float = 1.0, is_anchor: bool = False):
        """Used during the 9-point bulk calibration. is_anchor=True means
        this sample is never evicted by the LRU."""
        self.samples.append(CalibSample(
            sx=sx, sy=sy, yaw=yaw, pitch=pitch,
            weight=weight, is_anchor=is_anchor,
            bin_key=self._bin_for(sx, sy), added_at=_time.time(),
        ))

    def add_click_correction(self, sx: float, sy: float, yaw: float, pitch: float):
        """F2 burst-mode click. Higher weight, evicted by LRU like continuous samples."""
        self._add_evictable(sx, sy, yaw, pitch, self.click_weight)

    def add_continuous_sample(self, sx: float, sy: float, yaw: float, pitch: float):
        """Continuous-calibration sample (every real click that passes gating).
        Lower weight than F2 burst, also evicted by LRU."""
        self._add_evictable(sx, sy, yaw, pitch, self.continuous_weight)

    def _add_evictable(self, sx: float, sy: float, yaw: float, pitch: float, weight: float):
        bin_key = self._bin_for(sx, sy)
        sample = CalibSample(
            sx=sx, sy=sy, yaw=yaw, pitch=pitch,
            weight=weight, is_anchor=False,
            bin_key=bin_key, added_at=_time.time(),
        )
        self.samples.append(sample)
        # LRU per bin (only over non-anchor samples in this bin)
        in_bin = [i for i, s in enumerate(self.samples)
                  if not s.is_anchor and s.bin_key == bin_key]
        excess = len(in_bin) - self.max_per_bin
        if excess > 0:
            # Drop oldest non-anchor samples in this bin
            in_bin_sorted = sorted(in_bin, key=lambda i: self.samples[i].added_at)
            for i in in_bin_sorted[:excess]:
                self.samples[i] = None
            self.samples = [s for s in self.samples if s is not None]

    def prune_outliers(self, sigma: float = 2.5) -> int:
        """After a fit, drop non-anchor samples whose residual is > sigma * MAD
        from the median. Returns number dropped. Anchors are preserved."""
        if not self.is_fitted or len(self.samples) < 8:
            return 0
        F = np.stack([_features(s.yaw, s.pitch) for s in self.samples], axis=0)
        pred_x = F @ self.coef_x
        pred_y = F @ self.coef_y
        actual_x = np.array([s.sx for s in self.samples])
        actual_y = np.array([s.sy for s in self.samples])
        residuals = np.hypot(pred_x - actual_x, pred_y - actual_y)
        med = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - med))) or 1.0
        threshold = med + sigma * 1.4826 * mad   # 1.4826 -> MAD to stddev for normal dist
        dropped = 0
        kept = []
        for s, r in zip(self.samples, residuals):
            if s.is_anchor or r <= threshold:
                kept.append(s)
            else:
                dropped += 1
        self.samples = kept
        return dropped

    def stats(self) -> dict:
        anchors = sum(1 for s in self.samples if s.is_anchor)
        clicks = len(self.samples) - anchors
        bins_used = len({s.bin_key for s in self.samples if not s.is_anchor})
        return {
            "anchors": anchors,
            "click_samples": clicks,
            "bins_used": bins_used,
            "bins_total": self.bins_x * self.bins_y,
        }

    def reset(self):
        self.samples.clear()
        self.coef_x = None
        self.coef_y = None

    # ---- fitting ----

    def fit(self) -> bool:
        """Solve weighted least squares for both axes. Returns True on success.
        Needs at least 6 distinct samples (matches the number of poly features)."""
        if len(self.samples) < 6:
            return False

        F = np.stack([_features(s.yaw, s.pitch) for s in self.samples], axis=0)  # (N, 6)
        sx = np.array([s.sx for s in self.samples], dtype=float)
        sy = np.array([s.sy for s in self.samples], dtype=float)
        w = np.array([s.weight for s in self.samples], dtype=float)
        # Sqrt-weight rows so the unweighted lstsq solves the weighted problem.
        W = np.sqrt(w)[:, None]
        Fw = F * W
        sx_w = sx * W[:, 0]
        sy_w = sy * W[:, 0]

        try:
            cx, *_ = np.linalg.lstsq(Fw, sx_w, rcond=None)
            cy, *_ = np.linalg.lstsq(Fw, sy_w, rcond=None)
        except np.linalg.LinAlgError:
            return False

        self.coef_x = cx
        self.coef_y = cy
        return True

    # ---- prediction ----

    def predict(self, yaw: float, pitch: float) -> Tuple[int, int]:
        """Apply the fit. Falls back to screen center if not fitted yet."""
        if not self.is_fitted:
            return self.screen_w // 2, self.screen_h // 2
        f = _features(yaw, pitch)
        sx = float(self.coef_x @ f)
        sy = float(self.coef_y @ f)
        sx = max(0, min(self.screen_w - 1, int(round(sx))))
        sy = max(0, min(self.screen_h - 1, int(round(sy))))
        return sx, sy

    # ---- persistence ----

    def save(self, path: Path = CALIB_PATH) -> bool:
        if not self.is_fitted:
            return False
        try:
            data = {
                "screen_w": self.screen_w,
                "screen_h": self.screen_h,
                "coef_x": self.coef_x.tolist(),
                "coef_y": self.coef_y.tolist(),
                "samples": [
                    {"sx": s.sx, "sy": s.sy, "yaw": s.yaw, "pitch": s.pitch,
                     "weight": s.weight, "is_anchor": s.is_anchor,
                     "bin_key": list(s.bin_key) if s.bin_key else None,
                     "added_at": s.added_at}
                    for s in self.samples
                ],
            }
            path.write_text(json.dumps(data, indent=2))
            return True
        except OSError:
            return False

    @classmethod
    def load(cls, screen_w: int, screen_h: int, path: Path = CALIB_PATH) -> Optional["CalibrationModel"]:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

        # Only reuse if the screen size matches — otherwise the px coords are wrong
        if int(data.get("screen_w", -1)) != screen_w or int(data.get("screen_h", -1)) != screen_h:
            return None

        m = cls(screen_w=screen_w, screen_h=screen_h)
        samples = []
        for s in data.get("samples", []):
            bk = s.get("bin_key")
            samples.append(CalibSample(
                sx=s["sx"], sy=s["sy"], yaw=s["yaw"], pitch=s["pitch"],
                weight=s.get("weight", 1.0),
                is_anchor=bool(s.get("is_anchor", False)),
                bin_key=tuple(bk) if bk else None,
                added_at=float(s.get("added_at", 0.0)),
            ))
        m.samples = samples
        try:
            m.coef_x = np.array(data["coef_x"], dtype=float)
            m.coef_y = np.array(data["coef_y"], dtype=float)
        except (KeyError, ValueError):
            return None
        if m.coef_x.shape != (6,) or m.coef_y.shape != (6,):
            return None
        return m


def nine_point_targets(screen_w: int, screen_h: int, margin_pct: float = 0.08) -> List[Tuple[int, int]]:
    """3x3 grid in screen space, with a margin so points aren't exactly at the edge."""
    mx = int(screen_w * margin_pct)
    my = int(screen_h * margin_pct)
    xs = [mx, screen_w // 2, screen_w - mx]
    ys = [my, screen_h // 2, screen_h - my]
    # Center first so a single-point fallback (if user aborts) still has the center.
    pts = [(xs[1], ys[1])]
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            if (i, j) == (1, 1):
                continue
            pts.append((x, y))
    return pts

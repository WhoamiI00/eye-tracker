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

    # Click-to-correct controls
    max_click_samples: int = 30           # cap recent click samples to avoid drift
    click_weight: float = 3.0             # fresh clicks weigh more than initial calib

    @property
    def is_fitted(self) -> bool:
        return self.coef_x is not None and self.coef_y is not None

    # ---- adding samples ----

    def add_calibration_sample(self, sx: float, sy: float, yaw: float, pitch: float, weight: float = 1.0):
        """Used during the 9-point bulk calibration."""
        self.samples.append(CalibSample(sx, sy, yaw, pitch, weight))

    def add_click_correction(self, sx: float, sy: float, yaw: float, pitch: float):
        """Used at runtime when the user physically clicks somewhere.
        We assume their gaze was near that click and add it as a high-weight
        sample. We also bound the number of click samples so the model
        doesn't drift unboundedly if the user clicks while looking elsewhere."""
        self.samples.append(CalibSample(sx, sy, yaw, pitch, self.click_weight))
        # Trim oldest CLICK-weighted samples beyond the cap (preserve bulk-calib ones)
        click_idxs = [i for i, s in enumerate(self.samples) if s.weight >= self.click_weight - 1e-9]
        excess = len(click_idxs) - self.max_click_samples
        if excess > 0:
            for i in click_idxs[:excess]:
                self.samples[i] = None
            self.samples = [s for s in self.samples if s is not None]

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
                    {"sx": s.sx, "sy": s.sy, "yaw": s.yaw, "pitch": s.pitch, "weight": s.weight}
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
        m.samples = [
            CalibSample(s["sx"], s["sy"], s["yaw"], s["pitch"], s.get("weight", 1.0))
            for s in data.get("samples", [])
        ]
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

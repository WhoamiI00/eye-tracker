"""
Reusable gaze-tracking engine extracted from Webcam3DTracker/MonitorTracking.py.

Runs the webcam + MediaPipe pipeline on a background thread and emits
GazeSample objects (screen position, yaw/pitch, eye-aspect-ratios) via a
thread-safe callback. The overlay app and (later) the eye-mouse app both
consume this same engine.
"""

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation as Rscipy

from calibration import CalibrationModel
from one_euro import OneEuro2D


# Nose landmark indices (stable for head pose PCA) — same as MonitorTracking.py
NOSE_INDICES = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241,
                461, 125, 354, 218, 438, 195, 167, 393, 165, 391,
                3, 248]

# MediaPipe iris indices
LEFT_IRIS_IDX = 468
RIGHT_IRIS_IDX = 473

# Eye Aspect Ratio landmark indices (MediaPipe Face Mesh, refined eyes)
# Format: (outer_corner, top1, top2, inner_corner, bottom1, bottom2)
LEFT_EYE_EAR = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_EAR = (362, 385, 387, 263, 373, 380)


@dataclass
class GazeSample:
    """One frame of gaze data."""
    timestamp: float          # seconds since epoch
    screen_x: int             # pixels on primary monitor
    screen_y: int
    yaw_deg: float            # raw gaze yaw (camera-relative)
    pitch_deg: float
    ear_left: float           # eye aspect ratio, left eye
    ear_right: float
    calibrated: bool          # True only after eye-sphere lock + screen calibration
    face_visible: bool        # True if MediaPipe found a face this frame


def _compute_scale(points_3d: np.ndarray) -> float:
    n = len(points_3d)
    if n < 2:
        return 1.0
    diff = points_3d[:, None, :] - points_3d[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(n, k=1)
    vals = dists[iu]
    return float(vals.mean()) if vals.size else 1.0


def _ear(face_landmarks, indices, w: int, h: int) -> float:
    """Eye Aspect Ratio: average vertical eyelid distance / horizontal eye width.
    Lower values mean a more closed eye. Typical open ~0.3, closed ~0.1."""
    p = [(face_landmarks[i].x * w, face_landmarks[i].y * h) for i in indices]
    p0, p1, p2, p3, p4, p5 = p
    v1 = math.hypot(p1[0] - p5[0], p1[1] - p5[1])
    v2 = math.hypot(p2[0] - p4[0], p2[1] - p4[1])
    horiz = math.hypot(p0[0] - p3[0], p0[1] - p3[1])
    if horiz < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * horiz)


class GazeEngine:
    """
    Headless gaze pipeline. Call start() to spin up a background thread that
    reads webcam frames, runs MediaPipe, computes gaze, and invokes
    on_sample(GazeSample) for every processed frame.

    Calibration is now in two layers:
      1. request_eye_calibration() — user looks at screen center; this locks
         both eye spheres so the gaze direction can be derived. From this
         point on, every frame produces raw (yaw, pitch) in degrees.
      2. CalibrationModel (in calibration.py) — separately, a 9-point
         polynomial fit maps (yaw, pitch) -> (screen_x, screen_y). The
         engine holds a model reference but does not run the fitting itself
         (the calibration window owns that flow).
    """

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        on_sample: Callable[[GazeSample], None],
        camera_index: int = 0,
        filter_length: int = 18,    # was 10 — more averaging of the raw direction vector
        model: Optional[CalibrationModel] = None,
    ):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.on_sample = on_sample
        self.camera_index = camera_index
        self.filter_length = filter_length
        self.model = model if model is not None else CalibrationModel(
            screen_w=screen_w, screen_h=screen_h,
        )

        # Adaptive smoothing on the final screen-space (sx, sy). Tuned for
        # ~30 fps gaze: heavy smoothing when still, light during saccades.
        self._euro = OneEuro2D(min_cutoff=0.8, beta=0.012, d_cutoff=1.0)

        # Fixation-snap state: if gaze stays inside a small radius for a
        # short time, lock the rendered position to the fixation centroid
        # so micro-jitter at rest is completely killed.
        self._fix_buffer: deque = deque(maxlen=10)   # (ts, sx, sy)
        self._fix_locked_xy: Optional[tuple] = None

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._calib_eye_request = False

        # State (mirrors MonitorTracking.py globals)
        self._left_locked = False
        self._right_locked = False
        self._left_offset = None
        self._right_offset = None
        self._left_calib_scale = None
        self._right_calib_scale = None
        self._R_ref_nose = [None]
        self._dir_buffer: deque = deque(maxlen=filter_length)

        # Latest raw angles, exposed for the calibration window to sample
        self._latest_yaw: float = 0.0
        self._latest_pitch: float = 0.0
        self._latest_lock = threading.Lock()

        # For external preview windows (None unless preview_enabled=True)
        self.preview_enabled = False
        self._latest_preview_frame = None
        self._preview_lock = threading.Lock()

    # ---- public control API ----

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def request_eye_calibration(self):
        """Lock eye spheres on the next frame (user must be looking at screen center)."""
        self._calib_eye_request = True

    def reset_calibration(self):
        self._left_locked = False
        self._right_locked = False
        self._dir_buffer.clear()
        self._fix_buffer.clear()
        self._fix_locked_xy = None
        self._euro.reset()
        self.model.reset()

    @property
    def eye_locked(self) -> bool:
        """Eye spheres locked? Required before yaw/pitch is meaningful."""
        return self._left_locked and self._right_locked

    @property
    def is_calibrated(self) -> bool:
        """Fully usable: eyes locked AND screen-mapping model fitted."""
        return self.eye_locked and self.model.is_fitted

    def latest_angles(self) -> Tuple[float, float]:
        """Most recent (yaw, pitch) in degrees. Used by the calibration window
        to grab samples while the user looks at each target point."""
        with self._latest_lock:
            return self._latest_yaw, self._latest_pitch

    def get_preview_frame(self):
        """Returns the most recent annotated webcam frame (BGR), or None."""
        with self._preview_lock:
            return None if self._latest_preview_frame is None else self._latest_preview_frame.copy()

    # ---- internal ----

    def _run(self):
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam index {self.camera_index}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        base_radius = 20

        try:
            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                ts = time.time()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame_rgb)

                if not results.multi_face_landmarks:
                    self.on_sample(GazeSample(ts, self.screen_w // 2, self.screen_h // 2,
                                              0.0, 0.0, 0.0, 0.0,
                                              calibrated=self.is_calibrated, face_visible=False))
                    self._update_preview(frame, "no face")
                    continue

                lm = results.multi_face_landmarks[0].landmark

                # --- head pose via PCA on nose landmarks ---
                nose_pts = np.array([[lm[i].x * w, lm[i].y * h, lm[i].z * w] for i in NOSE_INDICES])
                head_center = nose_pts.mean(axis=0)
                centered = nose_pts - head_center
                cov = np.cov(centered.T)
                _, eigvecs = np.linalg.eigh(cov)
                eigvecs = eigvecs[:, ::-1]
                if np.linalg.det(eigvecs) < 0:
                    eigvecs[:, 2] *= -1
                r = Rscipy.from_matrix(eigvecs)
                roll, pitch, yaw = r.as_euler('zyx', degrees=False)
                R_final = Rscipy.from_euler('zyx', [roll, pitch, yaw]).as_matrix()
                if self._R_ref_nose[0] is None:
                    self._R_ref_nose[0] = R_final.copy()
                else:
                    R_ref = self._R_ref_nose[0]
                    for i in range(3):
                        if np.dot(R_final[:, i], R_ref[:, i]) < 0:
                            R_final[:, i] *= -1

                # --- iris positions (3D-ish) ---
                left_iris = lm[LEFT_IRIS_IDX]
                right_iris = lm[RIGHT_IRIS_IDX]
                iris_l = np.array([left_iris.x * w, left_iris.y * h, left_iris.z * w])
                iris_r = np.array([right_iris.x * w, right_iris.y * h, right_iris.z * w])

                # --- handle calibration requests ---
                if self._calib_eye_request:
                    self._calib_eye_request = False
                    cur_scale = _compute_scale(nose_pts)
                    cam_dir_local = R_final.T @ np.array([0, 0, 1])
                    self._left_offset = R_final.T @ (iris_l - head_center) + base_radius * cam_dir_local
                    self._right_offset = R_final.T @ (iris_r - head_center) + base_radius * cam_dir_local
                    self._left_calib_scale = cur_scale
                    self._right_calib_scale = cur_scale
                    self._left_locked = True
                    self._right_locked = True
                    self._euro.reset()
                    self._fix_buffer.clear()
                    self._fix_locked_xy = None
                    self._dir_buffer.clear()

                # --- compute gaze if eyes locked ---
                screen_x = self.screen_w // 2
                screen_y = self.screen_h // 2
                yaw_deg = 0.0
                pitch_deg = 0.0

                if self._left_locked and self._right_locked:
                    cur_scale = _compute_scale(nose_pts)
                    sl_ratio = cur_scale / self._left_calib_scale if self._left_calib_scale else 1.0
                    sr_ratio = cur_scale / self._right_calib_scale if self._right_calib_scale else 1.0
                    sphere_l = head_center + R_final @ (self._left_offset * sl_ratio)
                    sphere_r = head_center + R_final @ (self._right_offset * sr_ratio)

                    lg = iris_l - sphere_l
                    rg = iris_r - sphere_r
                    if np.linalg.norm(lg) > 1e-9 and np.linalg.norm(rg) > 1e-9:
                        lg /= np.linalg.norm(lg)
                        rg /= np.linalg.norm(rg)
                        combined = (lg + rg) / 2.0
                        combined /= np.linalg.norm(combined)
                        self._dir_buffer.append(combined)
                        avg = np.mean(self._dir_buffer, axis=0)
                        avg /= np.linalg.norm(avg)

                        yaw_deg, pitch_deg = self._direction_to_angles(avg)

                        with self._latest_lock:
                            self._latest_yaw = yaw_deg
                            self._latest_pitch = pitch_deg

                        if self.model.is_fitted:
                            raw_sx, raw_sy = self.model.predict(yaw_deg, pitch_deg)
                            screen_x, screen_y = self._smooth_and_snap(raw_sx, raw_sy, ts)

                # --- blink detection (works without calibration) ---
                ear_l = _ear(lm, LEFT_EYE_EAR, w, h)
                ear_r = _ear(lm, RIGHT_EYE_EAR, w, h)

                self.on_sample(GazeSample(
                    timestamp=ts,
                    screen_x=int(screen_x),
                    screen_y=int(screen_y),
                    yaw_deg=float(yaw_deg),
                    pitch_deg=float(pitch_deg),
                    ear_left=float(ear_l),
                    ear_right=float(ear_r),
                    calibrated=self.is_calibrated,
                    face_visible=True,
                ))

                if self.preview_enabled:
                    self._draw_preview(frame, lm, w, h, iris_l, iris_r, head_center,
                                       R_final, screen_x, screen_y)
        finally:
            cap.release()
            face_mesh.close()

    def _direction_to_angles(self, direction: np.ndarray) -> Tuple[float, float]:
        """Convert a 3D gaze direction vector to (yaw, pitch) in degrees.
        Sign convention preserved from the original tracker: yaw is negative
        across the whole horizontal range (the polynomial fit handles the
        actual mapping)."""
        ref = np.array([0, 0, -1])
        d = direction / np.linalg.norm(direction)

        xz = np.array([d[0], 0, d[2]])
        xz /= np.linalg.norm(xz) + 1e-9
        yaw_rad = math.acos(np.clip(np.dot(ref, xz), -1.0, 1.0))
        if d[0] < 0:
            yaw_rad = -yaw_rad

        yz = np.array([0, d[1], d[2]])
        yz /= np.linalg.norm(yz) + 1e-9
        pitch_rad = math.acos(np.clip(np.dot(ref, yz), -1.0, 1.0))
        if d[1] > 0:
            pitch_rad = -pitch_rad

        yaw_deg = math.degrees(yaw_rad)
        pitch_deg = math.degrees(pitch_rad)
        yaw_deg = -yaw_deg if yaw_deg > 0 else -yaw_deg
        return yaw_deg, pitch_deg

    def _smooth_and_snap(self, raw_sx: float, raw_sy: float, ts: float) -> Tuple[int, int]:
        """Two-stage smoothing of the predicted screen position:

        1. One-Euro filter — adaptive low-pass; aggressive smoothing when
           gaze is slow, light during saccades.
        2. Fixation snap — if the smoothed signal stays inside a small
           radius for ~200 ms, lock to the centroid until motion resumes.
           This kills residual micro-jitter during fixations entirely.
        """
        # Stage 1: One-Euro
        sx_f, sy_f = self._euro.filter(raw_sx, raw_sy, ts)

        # Stage 2: fixation snap. Maintain a short trailing window.
        self._fix_buffer.append((ts, sx_f, sy_f))
        cutoff = ts - 0.30  # 300 ms window
        while self._fix_buffer and self._fix_buffer[0][0] < cutoff:
            self._fix_buffer.popleft()

        # Compute window dispersion
        if len(self._fix_buffer) >= 6:
            xs = [p[1] for p in self._fix_buffer]
            ys = [p[2] for p in self._fix_buffer]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            max_d = max(math.hypot(px - cx, py - cy) for _, px, py in self._fix_buffer)

            FIX_LOCK_RADIUS = 45.0    # px — within this, treat as a fixation
            FIX_BREAK_RADIUS = 70.0   # px — leave the lock if we drift outside

            if self._fix_locked_xy is None:
                if max_d <= FIX_LOCK_RADIUS:
                    self._fix_locked_xy = (cx, cy)
            else:
                lx, ly = self._fix_locked_xy
                dist_from_lock = math.hypot(sx_f - lx, sy_f - ly)
                if dist_from_lock > FIX_BREAK_RADIUS:
                    self._fix_locked_xy = None  # saccade — release
                else:
                    # Slowly drift the lock toward the new centroid (handles
                    # genuine slow shifts without re-acquiring instantly)
                    a = 0.05
                    self._fix_locked_xy = (lx * (1 - a) + cx * a,
                                           ly * (1 - a) + cy * a)

        if self._fix_locked_xy is not None:
            sx_f, sy_f = self._fix_locked_xy

        sx_i = max(0, min(self.screen_w - 1, int(round(sx_f))))
        sy_i = max(0, min(self.screen_h - 1, int(round(sy_f))))
        return sx_i, sy_i

    def _draw_preview(self, frame, lm, w, h, iris_l, iris_r, head_center, R_final, sx, sy):
        # Cheap overlay: iris dots + status text
        cv2.circle(frame, (int(iris_l[0]), int(iris_l[1])), 3, (255, 255, 25), -1)
        cv2.circle(frame, (int(iris_r[0]), int(iris_r[1])), 3, (25, 255, 255), -1)
        cv2.circle(frame, (int(head_center[0]), int(head_center[1])), 4, (255, 0, 255), -1)
        status = "CALIBRATED" if self.is_calibrated else "NOT CALIBRATED"
        color = (0, 255, 0) if self.is_calibrated else (0, 0, 255)
        cv2.putText(frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if self.is_calibrated:
            cv2.putText(frame, f"Gaze -> ({sx}, {sy})", (10, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        with self._preview_lock:
            self._latest_preview_frame = frame

    def _update_preview(self, frame, status):
        if not self.preview_enabled:
            return
        cv2.putText(frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        with self._preview_lock:
            self._latest_preview_frame = frame

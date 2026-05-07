"""
Eye-tracking metrics: blink detection, fixation/saccade segmentation, CSV logging.

Algorithms:
- Blink: EAR (Eye Aspect Ratio) below threshold for >= MIN_BLINK_MS, then recovery.
- Fixation (I-DT): consecutive samples within DISPERSION_PX radius for >= MIN_FIX_MS.
- Saccade: any non-fixation, non-blink movement between fixations.

CSV schema (one row per sample, plus event rows):
    timestamp, screen_x, screen_y, yaw, pitch, ear_left, ear_right, event
where `event` is one of: "", "fixation_start", "fixation_end",
"saccade", "blink_start", "blink_end".
"""

import csv
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

from gaze_engine import GazeSample


# Tuning knobs (sensible defaults; can be exposed in UI later)
EAR_BLINK_THRESHOLD = 0.18      # eye closed if avg EAR < this
MIN_BLINK_MS = 80               # < this is noise, not a blink
MAX_BLINK_MS = 500              # > this is probably an extended close, not a normal blink

DISPERSION_PX = 60              # I-DT spatial threshold (gaze "stays" within this radius)
MIN_FIX_MS = 100                # fixation must last at least this long
SACCADE_MIN_DIST_PX = 80        # ignore micro-movements between fixations


@dataclass
class FixationEvent:
    start: float
    end: float
    x: float
    y: float
    duration_ms: float


@dataclass
class SaccadeEvent:
    start: float
    end: float
    from_xy: tuple
    to_xy: tuple
    distance_px: float
    duration_ms: float
    velocity_px_s: float


@dataclass
class BlinkEvent:
    start: float
    end: float
    duration_ms: float


@dataclass
class SessionStats:
    started_at: float = 0.0
    ended_at: float = 0.0
    sample_count: int = 0
    fixations: List[FixationEvent] = field(default_factory=list)
    saccades: List[SaccadeEvent] = field(default_factory=list)
    blinks: List[BlinkEvent] = field(default_factory=list)


class MetricsLogger:
    """
    Consumes GazeSamples in real-time, segments them into fixations/saccades/blinks,
    and writes everything to a CSV file. Thread-safe.
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)

        self._file = open(csv_path, "w", newline="", buffering=1)
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "timestamp", "screen_x", "screen_y", "yaw_deg", "pitch_deg",
            "ear_left", "ear_right", "event",
        ])

        # Fixation / saccade state
        self._fix_buffer: deque = deque()   # GazeSamples currently considered as a fixation
        self._last_fixation: Optional[FixationEvent] = None

        # Blink state
        self._blink_start_ts: Optional[float] = None
        self._blink_in_progress = False

        self.stats = SessionStats(started_at=time.time())

    # ---- public API ----

    def add_sample(self, s: GazeSample):
        self.stats.sample_count += 1
        events: List[str] = []

        # Blink detection from EAR
        ear_avg = (s.ear_left + s.ear_right) / 2.0
        if s.face_visible:
            if ear_avg < EAR_BLINK_THRESHOLD and not self._blink_in_progress:
                self._blink_in_progress = True
                self._blink_start_ts = s.timestamp
                events.append("blink_start")
            elif ear_avg >= EAR_BLINK_THRESHOLD and self._blink_in_progress:
                self._blink_in_progress = False
                dur_ms = (s.timestamp - self._blink_start_ts) * 1000.0
                if MIN_BLINK_MS <= dur_ms <= MAX_BLINK_MS:
                    self.stats.blinks.append(BlinkEvent(
                        start=self._blink_start_ts, end=s.timestamp, duration_ms=dur_ms
                    ))
                    events.append("blink_end")

        # Fixation detection (I-DT) — only if calibrated and not currently blinking
        if s.calibrated and s.face_visible and not self._blink_in_progress:
            self._update_fixation(s, events)

        # Write the sample row (one row per event tag, or one with empty tag)
        if not events:
            self._write_row(s, "")
        else:
            for ev in events:
                self._write_row(s, ev)

    def close(self) -> SessionStats:
        # Flush any in-progress fixation
        if self._fix_buffer:
            self._finalize_fixation()
        self.stats.ended_at = time.time()
        self._file.flush()
        self._file.close()
        return self.stats

    # ---- internal ----

    def _write_row(self, s: GazeSample, event: str):
        self._writer.writerow([
            f"{s.timestamp:.6f}", s.screen_x, s.screen_y,
            f"{s.yaw_deg:.3f}", f"{s.pitch_deg:.3f}",
            f"{s.ear_left:.4f}", f"{s.ear_right:.4f}",
            event,
        ])

    def _update_fixation(self, s: GazeSample, events: List[str]):
        if not self._fix_buffer:
            self._fix_buffer.append(s)
            return

        # Compute dispersion of buffer + this new sample
        xs = [p.screen_x for p in self._fix_buffer] + [s.screen_x]
        ys = [p.screen_y for p in self._fix_buffer] + [s.screen_y]
        dispersion = max(max(xs) - min(xs), max(ys) - min(ys))

        if dispersion <= DISPERSION_PX:
            self._fix_buffer.append(s)
            # Emit fixation_start once buffer crosses MIN_FIX_MS
            dur_ms = (self._fix_buffer[-1].timestamp - self._fix_buffer[0].timestamp) * 1000.0
            if dur_ms >= MIN_FIX_MS and not self._last_fixation:
                events.append("fixation_start")
                # Mark a sentinel so we don't keep emitting starts
                self._last_fixation = FixationEvent(
                    start=self._fix_buffer[0].timestamp,
                    end=self._fix_buffer[-1].timestamp,
                    x=sum(xs) / len(xs),
                    y=sum(ys) / len(ys),
                    duration_ms=dur_ms,
                )
        else:
            # Movement broke the fixation
            self._finalize_fixation(emit_to=events, next_sample=s)
            self._fix_buffer.clear()
            self._fix_buffer.append(s)

    def _finalize_fixation(self, emit_to: Optional[List[str]] = None, next_sample: Optional[GazeSample] = None):
        if not self._fix_buffer:
            self._last_fixation = None
            return

        first = self._fix_buffer[0]
        last = self._fix_buffer[-1]
        dur_ms = (last.timestamp - first.timestamp) * 1000.0

        if dur_ms >= MIN_FIX_MS:
            xs = [p.screen_x for p in self._fix_buffer]
            ys = [p.screen_y for p in self._fix_buffer]
            fix = FixationEvent(
                start=first.timestamp, end=last.timestamp,
                x=sum(xs) / len(xs), y=sum(ys) / len(ys),
                duration_ms=dur_ms,
            )
            self.stats.fixations.append(fix)
            if emit_to is not None:
                emit_to.append("fixation_end")

            # Saccade from previous fixation to this one's end → next sample
            if next_sample is not None and self.stats.fixations and len(self.stats.fixations) >= 1:
                prev = self.stats.fixations[-1]
                dx = next_sample.screen_x - prev.x
                dy = next_sample.screen_y - prev.y
                dist = (dx * dx + dy * dy) ** 0.5
                if dist >= SACCADE_MIN_DIST_PX:
                    s_dur_s = max(1e-3, next_sample.timestamp - prev.end)
                    sac = SaccadeEvent(
                        start=prev.end, end=next_sample.timestamp,
                        from_xy=(prev.x, prev.y),
                        to_xy=(next_sample.screen_x, next_sample.screen_y),
                        distance_px=dist, duration_ms=s_dur_s * 1000.0,
                        velocity_px_s=dist / s_dur_s,
                    )
                    self.stats.saccades.append(sac)
                    if emit_to is not None:
                        emit_to.append("saccade")
        self._last_fixation = None


def summarize(stats: SessionStats) -> dict:
    """Compute session-level summary numbers used by the report."""
    duration_s = max(1e-3, stats.ended_at - stats.started_at)
    fixations = stats.fixations
    saccades = stats.saccades
    blinks = stats.blinks

    avg_fix_ms = sum(f.duration_ms for f in fixations) / len(fixations) if fixations else 0.0
    avg_sac_vel = sum(s.velocity_px_s for s in saccades) / len(saccades) if saccades else 0.0
    blink_rate_per_min = len(blinks) / (duration_s / 60.0) if duration_s > 0 else 0.0

    # Composite stress score (0-100, heuristic — needs validation):
    #   high blink rate, short fixations, fast saccades all push it up.
    #   Normal blink rate is ~15/min; fixations ~250 ms; saccades ~500 px/s.
    blink_factor = min(1.0, blink_rate_per_min / 30.0)        # 30/min = max stress
    fix_factor = 1.0 - min(1.0, avg_fix_ms / 400.0)           # shorter = more stress
    sac_factor = min(1.0, avg_sac_vel / 1500.0)               # faster = more stress
    stress_score = round(100.0 * (0.4 * blink_factor + 0.3 * fix_factor + 0.3 * sac_factor), 1)

    # Accuracy proxy: gaze-to-screen-center distance during fixations
    # (only meaningful if user was supposed to look at the center; otherwise
    # treat as "stability" — lower = more locked in)
    if fixations:
        dists = [((f.x - 0.5) ** 2 + (f.y - 0.5) ** 2) ** 0.5 for f in fixations]
        avg_drift = sum(dists) / len(dists)
    else:
        avg_drift = 0.0

    return {
        "duration_s": duration_s,
        "samples": stats.sample_count,
        "fixation_count": len(fixations),
        "saccade_count": len(saccades),
        "blink_count": len(blinks),
        "avg_fixation_ms": avg_fix_ms,
        "avg_saccade_velocity_px_s": avg_sac_vel,
        "blink_rate_per_min": blink_rate_per_min,
        "stress_score": stress_score,
        "fixation_drift_norm": avg_drift,
    }

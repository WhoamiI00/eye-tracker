"""
Post-session report viewer. Reads a CSV produced by metrics.MetricsLogger
and renders charts: gaze heatmap, fixation duration histogram, blink rate
over time, saccade velocity distribution, and a summary panel.

Usage:
    python report.py path/to/session.csv
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: str):
    samples = []
    fixations = []   # list of (start_ts, end_ts) — pairs derived from fixation_start/end
    blinks = []
    saccades = []

    fix_starts = {}  # ts -> True

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        last_fix_start = None
        last_blink_start = None
        for row in reader:
            try:
                ts = float(row["timestamp"])
                sx = int(row["screen_x"])
                sy = int(row["screen_y"])
                ev = row["event"]
            except (ValueError, KeyError):
                continue

            samples.append((ts, sx, sy,
                            float(row.get("ear_left", 0) or 0),
                            float(row.get("ear_right", 0) or 0)))

            if ev == "fixation_start":
                last_fix_start = (ts, sx, sy)
            elif ev == "fixation_end" and last_fix_start is not None:
                fixations.append((last_fix_start[0], ts, last_fix_start[1], last_fix_start[2]))
                last_fix_start = None
            elif ev == "blink_start":
                last_blink_start = ts
            elif ev == "blink_end" and last_blink_start is not None:
                blinks.append((last_blink_start, ts))
                last_blink_start = None
            elif ev == "saccade":
                saccades.append((ts, sx, sy))

    return samples, fixations, blinks, saccades


def render_report(csv_path: str):
    samples, fixations, blinks, saccades = load_csv(csv_path)
    if not samples:
        print(f"No samples found in {csv_path}")
        return

    t0 = samples[0][0]
    duration_s = max(1e-3, samples[-1][0] - t0)

    sx = np.array([s[1] for s in samples])
    sy = np.array([s[2] for s in samples])
    times = np.array([s[0] - t0 for s in samples])

    fix_durations_ms = [(end - start) * 1000.0 for start, end, _, _ in fixations]
    blink_durations_ms = [(e - s) * 1000.0 for s, e in blinks]
    blink_count = len(blinks)
    blink_rate = blink_count / (duration_s / 60.0)

    avg_fix = np.mean(fix_durations_ms) if fix_durations_ms else 0.0

    # Heuristic stress score (matches metrics.summarize)
    blink_factor = min(1.0, blink_rate / 30.0)
    fix_factor = 1.0 - min(1.0, avg_fix / 400.0)
    stress = round(100.0 * (0.4 * blink_factor + 0.3 * fix_factor + 0.3 * 0.5), 1)

    # ---- Build the figure ----
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(f"GazeOverlay Session Report — {Path(csv_path).name}",
                 fontsize=14, fontweight="bold")

    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.3)

    # --- 1) Gaze heatmap ---
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    if len(sx) > 0:
        screen_w = max(1920, int(sx.max()) + 50)
        screen_h = max(1080, int(sy.max()) + 50)
        heat, xedges, yedges = np.histogram2d(
            sx, sy, bins=[60, 35],
            range=[[0, screen_w], [0, screen_h]],
        )
        ax1.imshow(heat.T, origin="upper", aspect="auto",
                   extent=[0, screen_w, screen_h, 0],
                   cmap="hot", interpolation="bilinear")
        ax1.set_title("Gaze Heatmap (where your eyes were on screen)")
        ax1.set_xlabel("Screen X (px)")
        ax1.set_ylabel("Screen Y (px)")
        # Overlay fixation centers
        if fixations:
            fx = [f[2] for f in fixations]
            fy = [f[3] for f in fixations]
            ax1.scatter(fx, fy, c="cyan", s=30, alpha=0.6,
                        edgecolors="black", linewidths=0.5, label="Fixations")
            ax1.legend(loc="upper right")

    # --- 2) Summary panel ---
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    summary_lines = [
        f"Duration:        {duration_s:.1f} s",
        f"Samples:         {len(samples)}",
        f"Fixations:       {len(fixations)}",
        f"Saccades:        {len(saccades)}",
        f"Blinks:          {blink_count}",
        f"Blink rate:      {blink_rate:.1f} / min",
        f"Avg fixation:    {avg_fix:.0f} ms",
        "",
        f"Stress score:    {stress} / 100",
    ]
    ax2.text(0.0, 1.0, "\n".join(summary_lines),
             family="monospace", fontsize=11, va="top", ha="left")
    ax2.set_title("Session Summary", fontweight="bold")

    # --- 3) Fixation duration histogram ---
    ax3 = fig.add_subplot(gs[1, 2])
    if fix_durations_ms:
        ax3.hist(fix_durations_ms, bins=20, color="#3aa9ff", edgecolor="black")
        ax3.axvline(np.mean(fix_durations_ms), color="red", linestyle="--",
                    label=f"mean={np.mean(fix_durations_ms):.0f}ms")
        ax3.legend()
    ax3.set_title("Fixation Durations")
    ax3.set_xlabel("Duration (ms)")
    ax3.set_ylabel("Count")

    # --- 4) Blink rate over time ---
    ax4 = fig.add_subplot(gs[2, 0])
    if blinks:
        bin_size_s = max(5.0, duration_s / 20.0)
        n_bins = max(1, int(duration_s / bin_size_s))
        blink_times = np.array([s - t0 for s, _ in blinks])
        counts, edges = np.histogram(blink_times, bins=n_bins,
                                     range=(0, duration_s))
        rate = counts / (bin_size_s / 60.0)
        centers = (edges[:-1] + edges[1:]) / 2.0
        ax4.plot(centers, rate, marker="o", color="#ff5577")
        ax4.fill_between(centers, rate, alpha=0.3, color="#ff5577")
    ax4.set_title("Blink Rate Over Time (fatigue indicator)")
    ax4.set_xlabel("Session time (s)")
    ax4.set_ylabel("Blinks / min")

    # --- 5) EAR (eyelid openness) trace ---
    ax5 = fig.add_subplot(gs[2, 1])
    if len(samples) > 0:
        ear_avg = np.array([(s[3] + s[4]) / 2.0 for s in samples])
        ax5.plot(times, ear_avg, color="#9966ff", linewidth=0.8)
        ax5.axhline(0.18, color="red", linestyle=":", label="blink threshold")
        ax5.legend(loc="lower right", fontsize=8)
    ax5.set_title("Eyelid Openness (EAR)")
    ax5.set_xlabel("Session time (s)")
    ax5.set_ylabel("EAR")
    ax5.set_ylim(0, 0.5)

    # --- 6) Gaze position over time (X) ---
    ax6 = fig.add_subplot(gs[2, 2])
    if len(samples) > 0:
        ax6.plot(times, sx, color="#3aa9ff", linewidth=0.6, label="X")
        ax6.plot(times, sy, color="#ff9933", linewidth=0.6, label="Y")
        ax6.legend(loc="upper right", fontsize=8)
    ax6.set_title("Gaze Position Over Time")
    ax6.set_xlabel("Session time (s)")
    ax6.set_ylabel("Screen px")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = Path(csv_path).with_suffix(".png")
    plt.savefig(out_png, dpi=110)
    print(f"Saved chart image: {out_png}")
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python report.py <session.csv>")
        sys.exit(1)
    render_report(sys.argv[1])


if __name__ == "__main__":
    main()

# GazeOverlay

Transparent, click-through gaze overlay for gaming + post-session eye-tracking metrics (fixations, saccades, blinks, stress score).

Phase 1 of the desktop eye-tracking app series. Reuses the gaze pipeline from [`Webcam3DTracker/MonitorTracking.py`](../Webcam3DTracker/MonitorTracking.py) — refactored into a reusable `gaze_engine` module.

## What it does

- Runs a fullscreen, transparent, **click-through** overlay on top of any game/app — your inputs go straight to the game, the overlay just paints a green gaze dot where you're looking.
- Logs every gaze sample, fixation, saccade, and blink to a CSV during a session.
- Generates a **post-session report** (heatmap, fixation histogram, blink rate over time, stress score) you can review after a match.

## Hotkeys

Global — work even when a game has focus:

| Key | Action |
|-----|--------|
| `F2`  | Toggle **click-to-correct** mode (10 sec) — clicks teach the tracker where you were looking |
| `F8`  | Toggle overlay visibility |
| `F9`  | Start / stop session recording |
| `F10` | Quit |
| `F11` | Recalibrate from scratch |

## Setup (Python 3.11 required)

```powershell
cd GazeOverlay
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

## Calibration flow

1. Look at the center of your screen and click **"Lock eye spheres"** (or press `C`).
2. A **fullscreen 9-point sequence** runs (~15 sec). Follow the green dot with your eyes; keep your head still.
3. The fit is saved to `~/.gaze_overlay_calibration.json` — next launch you can reuse it.
4. (Optional) Press `F2` and click on a few real UI items to refine the fit while you use the app.
5. Press `F9` to start recording a session.
6. Press `F9` again to stop. The CSV is written to `sessions/`.

### Why the 9-point fit
The previous single-point version hardcoded "screen spans 25° horizontally" and got worse the further you looked from center. The new version fits a degree-2 polynomial `(yaw, pitch) → (x, y)` via least squares — each region of the screen gets its own correction. Click-to-correct (F2) adds high-weight samples on the fly without restarting calibration.

## Viewing a session report

```powershell
python report.py sessions/session_20260507_143012.csv
```

A matplotlib window opens with a heatmap, fixation histogram, blink rate, EAR trace, and a summary panel including the heuristic **stress score** (0–100, composite of blink rate, fixation duration, saccade velocity).

## Building a standalone .exe

```powershell
.\build_exe.bat
```

Output: `dist/GazeOverlay.exe` (~80–100 MB; MediaPipe is large).

## Caveats

- **Fullscreen-exclusive games hide overlays.** Use Borderless Windowed mode in the game's video settings — same constraint Discord/Steam overlays have.
- **Anti-cheat games**: this only reads the webcam and paints pixels. It does not inject input or read game memory, so it should be safe with most anti-cheats. Still, on competitive games (Valorant, CS2 with VAC, etc.) test in casual modes first. We're not lawyers.
- **Webcam accuracy is ~1–2°** ≈ 30–50 px on 1080p. The overlay is for awareness/training, not for aiming.
- **Stress score is heuristic**, not clinically validated. Useful for spotting trends across sessions, not for absolute claims.

## File layout

```
GazeOverlay/
├── gaze_engine.py   # webcam + MediaPipe + gaze math (reusable module)
├── overlay.py       # PyQt6 transparent click-through window
├── metrics.py       # blink/fixation/saccade detection + CSV logger
├── report.py        # post-session matplotlib report viewer
├── main.py          # entry point: calibration UI, hotkeys, glue
├── requirements.txt
├── build_exe.bat
└── sessions/        # CSV files written here (gitignored)
```

## What's next

- **Phase 2** (`eye-mouse` branch): turn this into a system-wide input device — dwell-click, blink-click, drag — for accessibility and slow games.
- **Phase 3**: real-time stress dashboard during gameplay (live overlay metrics).

# EyeMouse

Phase 2 of the desktop eye-tracking app series. **Turns your eyes into a cursor.**

Built on top of [GazeOverlay](../GazeOverlay/) — reuses the gaze engine and calibration UI, adds a click engine and a dwell visualizer.

## What it does

- Moves the OS cursor to wherever you're looking.
- **Dwell-click**: keep your gaze inside a small radius for ~700 ms → left-click.
- **Long blink**: close both eyes for ~600 ms → right-click.
- **Wink drag**: close left eye while keeping the right eye open for ~350 ms → toggle drag mode (look to destination, wink again to drop).

## Hotkeys

| Key | Action |
|-----|--------|
| `F8`  | **Toggle EyeMouse on/off** (starts OFF for safety) |
| `F9`  | Open / close settings window (live sliders) |
| `F10` | Quit |
| `F11` | Recalibrate |
| `F12` | **PANIC OFF** — force-disable everything if cursor goes wild |

## Setup

```powershell
cd EyeMouse
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

## Calibration

Same two-step flow as GazeOverlay (look at center → C → S → Start). After calibration, the overlay status badge says `EyeMouse OFF — press F8 to enable`. Press F8 only when you're ready.

## Settings window (F9)

Live-tunable sliders:
- **Cursor smoothing** — higher = smoother but laggy.
- **Dwell radius / time / cooldown** — bigger radius and longer time make accidental clicks rarer but feel sluggish.
- **Blink threshold (EAR)** — lower if your blinks aren't being detected; raise if natural blinks trigger right-clicks.
- **Long-blink time** for right-click.
- **Wink time** for drag toggle.
- **Click backend**: `sendinput` (Windows, deeper input — works in more games) or `pyautogui` (cross-platform fallback).

Settings save to `~/.eye_mouse_config.json`.

## How the click engine works

The "Midas touch problem" — every glance triggering a click — is the central design challenge. Mitigations:

1. **Spatial dispersion check** — dwell only counts when all samples within the window stay inside `dwell_radius_px`. Movement resets the window.
2. **Smart dwell** — if either eye is closing/closed, dwell pauses (so you don't click while blinking).
3. **Cooldown** — minimum gap between clicks (default 600 ms).
4. **Blink/wink end before another fires** — the engine waits for the eye(s) to reopen before re-arming.
5. **Pause when no face** — webcam loses your face → cursor freezes (configurable).

## Caveats

- Webcam tracking is ~1–2° accurate. **Don't expect aim-assist precision.** Good for accessibility, slow games, browsing, productivity. Not good for shooters.
- **Anti-cheat**: `sendinput` is system-level synthetic input. Most kernel-level anti-cheats (Vanguard, EAC, BattlEye) flag this kind of automation. **Do NOT enable EyeMouse in competitive matches.** Test in custom/casual modes only.
- **Fullscreen exclusive games** may capture mouse independently — borderless windowed mode is safer.
- **Always keep F12 in mind.** If your eyes leave the screen and the cursor freezes weirdly, F12 panic-off.

## Building a .exe

```powershell
.\build_exe.bat
```

Output: `dist/EyeMouse.exe`.

## Files

```
EyeMouse/
├── input_backend.py    # Win32 SendInput + pyautogui fallback
├── click_engine.py     # dwell + blink + wink decision logic
├── dwell_overlay.py    # extends GazeOverlay with progress ring + flashes
├── settings_window.py  # live-tunable sliders, persists to ~/.eye_mouse_config.json
├── main.py             # entry point, hotkeys, glue
├── requirements.txt
├── build_exe.bat
└── README.md
```

Reuses from `GazeOverlay/`: `gaze_engine.py`, `overlay.py`, `main.py` (CalibrationWindow only).

## What's next (Phase 3 ideas)

- **Hybrid mode**: physical mouse button clicks at gaze position (most ergonomic for productivity).
- **Eye scrolling** (look to top/bottom edge → scroll).
- **Live stress dashboard** (combine GazeOverlay metrics + EyeMouse usage).
- **Glasses-grade hardware** (separate near-IR build).

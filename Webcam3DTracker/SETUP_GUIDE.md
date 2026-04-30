# Webcam 3D Eye Tracker — Setup Guide

This guide walks you through getting the **Webcam3DTracker** running on your machine after cloning the repo. Follow the steps in order — most issues come from skipping one.

Tested on **Windows 11** with **Python 3.11**. Should also work on Windows 10. Linux/macOS notes at the end.

---

## 0. What you need before starting

- A working **webcam** (built-in laptop cam or USB).
- **Python 3.11** specifically — *not* 3.12 or 3.13. The `mediapipe` library does not yet ship reliable wheels for newer Python versions on all platforms, so 3.11 is the safest choice.
- Internet connection (the first run downloads the MediaPipe FaceMesh model automatically — ~10 MB).

---

## 1. Clone the repo

```bash
git clone https://github.com/Sumityvd/webcam-eye-tracker.git
cd webcam-eye-tracker
```

You only care about the [Webcam3DTracker/](Webcam3DTracker/) folder. Everything else in the repo is for other tracker variants and you can ignore them.

---

## 2. Install Python 3.11 (if you don't have it)

Check what you currently have:

```bash
py -0
```

If you see `-V:3.11` in the list, you're good — skip to step 3.

If not:

1. Go to https://www.python.org/downloads/release/python-3119/ (or any 3.11.x release).
2. Download the **Windows installer (64-bit)**.
3. During install, tick **"Add Python to PATH"** and **"Install launcher for all users"**.
4. After install, re-run `py -0` to confirm 3.11 is listed.

---

## 3. (Recommended) Create a virtual environment

This keeps the eye tracker's dependencies isolated from your system Python.

```bash
py -3.11 -m venv .venv
```

Activate it:

- **Windows (cmd / PowerShell):**
  ```bash
  .venv\Scripts\activate
  ```
- **Git Bash / WSL / Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

You'll know it's active when your shell prompt shows `(.venv)` at the start. From here on, just use `python` — it now points to the 3.11 inside `.venv`.

> **If you skip the venv:** use `py -3.11` everywhere instead of `python` so you hit the right interpreter.

---

## 4. Install the dependencies

From the repo root:

```bash
pip install -r Webcam3DTracker/requirements.txt
```

This installs:

| Package | What it's for |
|---|---|
| `opencv-python` | Webcam capture + drawing the visualization windows |
| `numpy` | Math |
| `mediapipe` | Face & iris landmark detection (the core tracking engine) |
| `scipy` | Rotation / orientation math |
| `pyautogui` | Moves your mouse cursor when you toggle mouse-control on (F7) |
| `keyboard` | Reads key presses for the orbit / calibration hotkeys |

Install takes 1–3 minutes. `mediapipe` is the biggest download (~50 MB).

> **Heads up — `keyboard` on Linux:** the `keyboard` package needs root on Linux (`sudo python MonitorTracking.py`). On Windows and macOS it works without admin rights.

---

## 5. Run it

```bash
cd Webcam3DTracker
python MonitorTracking.py
```

(Or `py -3.11 MonitorTracking.py` if you skipped the venv.)

**First run only:** there will be a short pause while MediaPipe downloads its FaceMesh model. You'll also see some `TensorFlow Lite` / `XNNPACK` startup messages and a `protobuf` deprecation warning — these are normal, not errors.

Two windows should open:

1. **Integrated Eye Tracking** — your webcam feed with face mesh, eye landmarks, and gaze rays drawn on it.
2. **Head/Eye Debug** — a black 3D orbit view. *This stays mostly blank until you calibrate.*

---

## 6. Calibrate

1. Sit roughly where you normally sit at your monitor.
2. **Look directly at the center of your screen.**
3. Press **`c`** (with one of the OpenCV windows focused).

You should now see:
- Cyan circles around your eyes in the live view.
- A green/orange virtual monitor rectangle appear in the 3D debug view.
- Live coordinates `Screen: (x, y)` printed at the top of the live view as you move your eyes.

If the gaze direction feels off, see "Tuning" below.

---

## 7. Controls

All hotkeys work when **either** OpenCV window has focus.

| Key | Action |
|---|---|
| `c` | Calibrate (look at screen center first — do this once at start) |
| `F7` | Toggle mouse control on/off (off by default — your gaze will move the cursor when on) |
| `j` / `l` | Orbit the 3D debug view left / right |
| `i` / `k` | Orbit pitch up / down |
| `[` / `]` | Zoom debug view out / in |
| `r` | Reset orbit view |
| `x` | Drop a green marker where you're currently looking on the virtual monitor |
| `s` | Re-center screen calibration (look at screen center, then press) |
| `q` | Quit |

---

## 8. Tuning (optional — only if accuracy feels wrong)

Open [Webcam3DTracker/MonitorTracking.py](Webcam3DTracker/MonitorTracking.py) and find these lines (around 408–410):

```python
yawDegrees = 5 * 3      # how far left/right you turn your eyes to span the full screen width
pitchDegrees = 2.0 * 2.5  # how far up/down to span the full screen height
```

- If the cursor reaches the screen edge **too easily** (small eye movement = big cursor jump): **increase** these numbers (try `yawDegrees = 25`, `pitchDegrees = 15`).
- If you have to roll your eyes very far to reach the edges: **decrease** them.

Other knobs:

- **Jittery gaze?** Increase `filter_length = 10` (line 18) to `15` or `20` — more smoothing, slightly more lag.
- **Wrong webcam?** Change `cv2.VideoCapture(0)` (line 73) to `1`, `2`, etc.

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` | Wrong Python. Either activate your venv, or use `py -3.11` instead of `python`. |
| `ModuleNotFoundError: No module named 'mediapipe'` | `pip install mediapipe` — and double-check you're on Python 3.11, not 3.12+. |
| Black/empty webcam window | Another app (Zoom, Teams, OBS) is holding the camera. Close it. Or change `cv2.VideoCapture(0)` to a different index. |
| `[Mouse Control]` toggle doesn't work | The `keyboard` library needs root on Linux. On Windows, make sure the OpenCV window has focus. |
| Debug view stays black | You haven't pressed `c` yet to calibrate, or your face isn't detected — check the live view first. |
| Gaze rays point wildly off | Lighting matters. Sit with even, front-facing light. Avoid backlight from a window. |
| `pyautogui` fail-safe error when mouse moves to corner | That's a built-in safety feature. Move the mouse manually back to center, or set `pyautogui.FAILSAFE = False` after the import. |

---

## 10. Linux / macOS notes

The script *should* work but isn't tested by me. Differences:

- **Linux:** the `keyboard` library requires root, so run as `sudo python MonitorTracking.py`. Alternative: replace `keyboard` calls with `pynput` (small code change).
- **macOS:** you'll need to grant the terminal app **Accessibility** and **Camera** permissions in System Settings → Privacy & Security. `pyautogui` mouse control also needs Accessibility permission.

---

## Quick reference (TL;DR)

```bash
git clone https://github.com/Sumityvd/webcam-eye-tracker.git
cd webcam-eye-tracker
py -3.11 -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -r Webcam3DTracker/requirements.txt
cd Webcam3DTracker
python MonitorTracking.py
# look at screen center, press c
```

That's it — happy tracking.

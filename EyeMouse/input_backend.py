"""
Low-level mouse input backends. Two implementations:

- WinSendInputBackend (default on Windows): uses the Win32 SendInput API via
  ctypes. Works inside most games because it produces events that look
  identical to a real USB mouse to the OS input queue.

- PyAutoGuiBackend (fallback): cross-platform but slower and ignored by
  some fullscreen games.

All backends expose move_to(x, y), left_click(), right_click(),
left_down(), left_up().
"""

import ctypes
import sys
import time
from ctypes import wintypes
from typing import Protocol


class MouseBackend(Protocol):
    def move_to(self, x: int, y: int) -> None: ...
    def left_click(self) -> None: ...
    def right_click(self) -> None: ...
    def left_down(self) -> None: ...
    def left_up(self) -> None: ...


class PyAutoGuiBackend:
    """Fallback. Simple, cross-platform, but blocked by some games."""

    def __init__(self):
        import pyautogui
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        self._g = pyautogui

    def move_to(self, x, y):
        self._g.moveTo(x, y, _pause=False)

    def left_click(self):
        self._g.click(_pause=False)

    def right_click(self):
        self._g.rightClick(_pause=False)

    def left_down(self):
        self._g.mouseDown(button="left", _pause=False)

    def left_up(self):
        self._g.mouseUp(button="left", _pause=False)


# ---- Windows SendInput backend ----

if sys.platform == "win32":
    INPUT_MOUSE = 0
    MOUSEEVENTF_MOVE = 0x0001
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_RIGHTDOWN = 0x0008
    MOUSEEVENTF_RIGHTUP = 0x0010
    MOUSEEVENTF_ABSOLUTE = 0x8000

    user32 = ctypes.WinDLL("user32", use_last_error=True)

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
        ]

    class _INPUT_UNION(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]

    class INPUT(ctypes.Structure):
        _anonymous_ = ("u",)
        _fields_ = [("type", wintypes.DWORD), ("u", _INPUT_UNION)]

    SCREEN_W = user32.GetSystemMetrics(0)
    SCREEN_H = user32.GetSystemMetrics(1)


    class WinSendInputBackend:
        """Windows SendInput backend. Goes deeper into the input stack
        than pyautogui — most games + apps see these events as real mouse."""

        def __init__(self):
            self._screen_w = max(1, SCREEN_W)
            self._screen_h = max(1, SCREEN_H)

        def _send(self, dx, dy, flags):
            inp = INPUT()
            inp.type = INPUT_MOUSE
            inp.mi = MOUSEINPUT(dx, dy, 0, flags, 0, None)
            user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

        def move_to(self, x, y):
            # SendInput absolute coords are normalized to 0..65535 on the
            # virtual screen. For multi-monitor support, callers can pass
            # the absolute screen coord; we convert.
            ax = int(x * 65535 / self._screen_w)
            ay = int(y * 65535 / self._screen_h)
            self._send(ax, ay, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE)

        def left_click(self):
            self._send(0, 0, MOUSEEVENTF_LEFTDOWN)
            time.sleep(0.012)
            self._send(0, 0, MOUSEEVENTF_LEFTUP)

        def right_click(self):
            self._send(0, 0, MOUSEEVENTF_RIGHTDOWN)
            time.sleep(0.012)
            self._send(0, 0, MOUSEEVENTF_RIGHTUP)

        def left_down(self):
            self._send(0, 0, MOUSEEVENTF_LEFTDOWN)

        def left_up(self):
            self._send(0, 0, MOUSEEVENTF_LEFTUP)


def make_backend(name: str = "auto") -> MouseBackend:
    """name: 'auto' | 'sendinput' | 'pyautogui'"""
    name = (name or "auto").lower()
    if name in ("auto", "sendinput") and sys.platform == "win32":
        return WinSendInputBackend()
    return PyAutoGuiBackend()

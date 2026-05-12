"""
One-Euro filter — adaptive low-pass for noisy pointer signals.

Heavy smoothing at low speeds (kills jitter when the user is fixating);
light smoothing at high speeds (no lag during saccades). Reference:
Casiez, Roussel & Vogel (CHI 2012).

Tuning:
  - min_cutoff: lower => smoother but more lag (Hz)
  - beta:       higher => more responsive at speed
  - d_cutoff:   derivative cutoff (Hz); rarely needs changing
"""

import math
import time


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007,
                 d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: float = 0.0
        self._dx_prev: float = 0.0
        self._t_prev: float = 0.0
        self._initialized = False

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def reset(self):
        self._initialized = False

    def filter(self, x: float, t: float = None) -> float:
        if t is None:
            t = time.time()
        if not self._initialized:
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = t
            self._initialized = True
            return x

        dt = max(1e-3, t - self._t_prev)

        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat


class OneEuro2D:
    """Two independent OneEuroFilters for (x, y)."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007,
                 d_cutoff: float = 1.0):
        self.fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def reset(self):
        self.fx.reset()
        self.fy.reset()

    def filter(self, x: float, y: float, t: float = None) -> tuple:
        return self.fx.filter(x, t), self.fy.filter(y, t)

"""Small numeric helpers shared by the scene and the map grid."""

from __future__ import annotations

import math

import numpy as np


def nice_step(span: float, target_lines: int = 6) -> float:
    """A round grid interval (1, 2, 2.5 or 5 x 10^k) giving about ``target_lines`` lines."""
    if span <= 0:
        return 1.0
    raw = span / target_lines
    mag = 10.0 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= m * mag * (1 + 1e-12):
            return m * mag
    return 10.0 * mag


def nice_grid(lo: np.ndarray, hi: np.ndarray, target_lines: int = 6):
    """Snap bounds outward to round grid intervals.

    Returns (lo, hi, n_labels per axis) so that grid lines fall on round numbers.
    """
    lo2, hi2, n = np.empty(3), np.empty(3), []
    for i in range(3):
        step = nice_step(float(hi[i] - lo[i]), target_lines if i < 2 else 4)
        lo2[i] = math.floor(lo[i] / step) * step
        hi2[i] = math.ceil(hi[i] / step) * step
        if hi2[i] <= lo2[i]:
            hi2[i] = lo2[i] + step
        n.append(int(round((hi2[i] - lo2[i]) / step)) + 1)
    return lo2, hi2, n

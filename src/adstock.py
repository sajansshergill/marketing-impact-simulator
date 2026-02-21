from __future__ import annotations
import numpy as np


def adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """
    Classic geometric adstock:
      a[t] = x[t] + decay * a[t-1]
    decay in [0,1).
    """
    if not (0.0 <= decay < 1.0):
        raise ValueError("decay must be in [0, 1).")
    x = np.asarray(x, dtype=float)
    a = np.zeros_like(x)
    for t in range(len(x)):
        a[t] = x[t] + (decay * a[t - 1] if t > 0 else 0.0)
    return a
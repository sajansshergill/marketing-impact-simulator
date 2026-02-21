from __future__ import annotations
import numpy as np


def hill_saturation(x: np.ndarray, alpha: float = 1.0, gamma: float = 1.0) -> np.ndarray:
    """
    Simple Hill / saturation curve:
      f(x) = x^gamma / (alpha^gamma + x^gamma)
    Output in [0,1)
    """
    x = np.asarray(x, dtype=float)
    if alpha <= 0 or gamma <= 0:
        raise ValueError("alpha and gamma must be > 0")
    num = np.power(x, gamma)
    den = np.power(alpha, gamma) + num
    return np.divide(num, den, out=np.zeros_like(num), where=den != 0)
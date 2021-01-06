import numba
import numpy as np


@numba.njit
def linear_interp(d1: float, d2: float, x1: float, x2: float) -> float:
    return (x1 * d2 + x2 * d1) / (d1 + d2)

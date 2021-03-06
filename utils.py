import numpy as np
from conditions import GRID, R_CC, M_cc, KQ


def CFL(v, x, tmp, param=1.0):
    assert x.ndim == 1, "x must be one dimentional array."
    assert x.shape[0] > 1, "x must be array."
    assert param > 0, "param must be positive."
    x_diff = np.diff(x)
    t = x_diff / np.maximum(np.abs(v[:GRID]), get_cs(tmp))
    """
    dx_min = x[1] - x[0]
    for i in range(x.shape[0] - 1):
        dx_min = min(dx_min, x[i + 1] - x[i])
    dt = dx_min / get_cs(tmp)
    """
    dt = np.min(t)
    return dt * param


def L(x):
    return np.diff(x) * KQ


def vstack_n(v, n):
    x = np.copy(v)
    for _ in range(n - 1):
        x = np.vstack([x, v])
    return x


def get_cs(tmp):
    # 10K 0.3km/s
    base = 0.3 * 100 * 1000
    coef = np.sqrt(tmp / 10)
    return base * coef


def m_init(r):
    res = np.power(r / R_CC, 3) * M_cc
    return res


def r_init():
    res = np.linspace(0, R_CC, GRID + 1)
    res[0] = 1
    return res

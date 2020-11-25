import numpy as np

from conditions import T_ORDER, GRID, R_cc, M_cc, AU


def CFL(x, tmp=10.0, param=1.0):
    assert x.ndim == 1, "x must be one dimentional array."
    assert x.shape[0] > 1, "x must be array."
    assert param > 0, "param must be positive."
    dx_min = x[1] - x[0]
    for i in range(x.shape[0] - 1):
        dx_min = min(dx_min, x[i + 1] - x[i])
    print("min", dx_min)
    dt = dx_min / get_cs(tmp)
    return dt * param


def vstack_n(v, n):
    x = np.copy(v)
    for _ in range(n - 1):
        x = np.vstack([x, v])
    return x


def get_cs(tmp):
    # 10K 0.3km/s
    base = 0.3 * 100 * 1000
    coef = np.sqrt(tmp / 10)
    return base * coef * T_ORDER / AU


def r_init():
    res = np.linspace(0, R_cc, GRID + 1)
    return res


def m_init():
    res = np.linspace(0, M_cc, GRID + 1)
    return res

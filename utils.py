import numpy as np


def CFL(x):
    return 0.01


def vstack_n(v, n):
    x = np.copy(v)
    for i in range(n - 1):
        x = np.vstack([x, v])
    return x
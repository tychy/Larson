import numpy as np
from conditions import GRID, R_cc, M_cc


def CFL(x, tmp=10.0, param=1.0):
    assert x.ndim == 1, "x must be one dimentional array."
    assert x.shape[0] > 1, "x must be array."
    assert param > 0, "param must be positive."
    dx_min = x[1] - x[0]
    for i in range(x.shape[0] - 1):
        dx_min = min(dx_min, x[i + 1] - x[i])
    dt = dx_min / get_cs(tmp)
    return dt * param


def L(x, param=2):
    dx_min = x[1] - x[0]
    for i in range(x.shape[0] - 1):
        dx_min = min(dx_min, x[i + 1] - x[i])

    return dx_min * param


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
    res = np.power(r / R_cc, 3) * M_cc
    return res


def r_init():
    res = np.linspace(0, R_cc, GRID + 1)
    res[0] = 1
    return res


def save(base_dir, idx, t_h, deltat, v, r, rho, p, tmp, r_h, r_l, p_l, t):
    np.save(base_dir + "/step_{}_t_h.npy".format(idx), t_h)
    np.save(base_dir + "/step_{}_deltat.npy".format(idx), deltat)
    np.save(base_dir + "/step_{}_v.npy".format(idx), v)
    np.save(base_dir + "/step_{}_r.npy".format(idx), r)
    np.save(base_dir + "/step_{}_rho.npy".format(idx), rho)
    np.save(base_dir + "/step_{}_p.npy".format(idx), p)
    np.save(base_dir + "/step_{}_tmp.npy".format(idx), tmp)
    np.save(base_dir + "/step_{}_r_h.npy".format(idx), r_h)
    np.save(base_dir + "/step_{}_r_l.npy".format(idx), r_l)
    np.save(base_dir + "/step_{}_p_l.npy".format(idx), p_l)
    np.save(base_dir + "/step_{}_t.npy".format(idx), t)
    return


def save_with_energy(base_dir, idx, v, r, rho, p, tmp, r_h, t, Q, e, gamma, x):
    np.save(base_dir + "/step_{}_v.npy".format(idx), v)
    np.save(base_dir + "/step_{}_r.npy".format(idx), r)
    np.save(base_dir + "/step_{}_rho.npy".format(idx), rho)
    np.save(base_dir + "/step_{}_p.npy".format(idx), p)
    np.save(base_dir + "/step_{}_tmp.npy".format(idx), tmp)
    np.save(base_dir + "/step_{}_r_h.npy".format(idx), r_h)
    np.save(base_dir + "/step_{}_t.npy".format(idx), t)
    np.save(base_dir + "/step_{}_Q.npy".format(idx), Q)
    np.save(base_dir + "/step_{}_e.npy".format(idx), e)
    np.save(base_dir + "/step_{}_gamma.npy".format(idx), gamma)
    np.save(base_dir + "/step_{}_x.npy".format(idx), x)

    return
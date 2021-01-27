import numpy as np
from utils import CFL, L
from conditions import CFL_CONST, planck, m_p, m_e, xi_h, xi_d, kb, R


def calc_t(idx, v, r, t, t_h, deltat, tmp):
    t_diff = CFL(v[idx - 1], r[idx], tmp, CFL_CONST)
    t = np.append(t, t[idx] + t_diff)
    t_h = np.append(t_h, t_diff)
    deltat = np.append(deltat, (t_diff + t_h[idx - 1]) / 2)
    if False:
        print("t_diff:", t_diff)
        print("t:", t)
        print("t_h:", t_h)
        print("deltat:", deltat)
    return t, t_h, deltat


def calc_lambda(idx, v, r, p, t_h, r_l, p_l):
    t_diff = t_h[idx] - t_h[idx - 1]
    t_coef = t_diff / t_h[idx - 1]
    r_res = r[idx] + t_diff * v[idx - 1] / 4
    p_res = p[idx] + t_coef * (p[idx] - p[idx - 1]) / 4

    r_l = np.vstack((r_l, r_res))
    p_l = np.vstack((p_l, p_res))
    return r_l, p_l


def calc_deltam(m):
    m_res = np.zeros(m.shape[0] - 1)
    for i in range(m_res.shape[0]):
        m_res[i] = m[i + 1] - m[i]
    return m_res


def calc_half(idx, r, r_h):
    r_res = np.zeros(r.shape[1] - 1)
    for i in range(r_res.shape[0] - 1):
        r_res[i] = (r[idx][i] ** 3 + r[idx][i + 1] ** 3) ** (1 / 3)
    r_res[-1] = r[idx][-1]
    r_h = np.vstack((r_h, r_res))
    return r_h


def calc_Q(idx, v, r, rho, t_h, deltat, Q):
    mu_res = L(r[idx]) ** 2 * (rho[idx + 1] - rho[idx]) / t_h[idx]
    for i in range(mu_res.shape[0]):
        if mu_res[i] < 0:
            mu_res[i] = 0  # todo np where

    Q_first = (np.diff(v[idx])) / (np.diff(r[idx + 1] + r[idx]) / 2)
    Q_second = (np.log(rho[idx + 1]) - np.log(rho[idx])) / t_h[idx] / 3
    Q_res = Q_first + Q_second
    Q_res = -2 * mu_res * Q_res
    Q = np.vstack((Q, Q_res))
    return Q


def calc_gamma(fht):
    molecular = 7 / 5
    monoatomic = 5 / 3
    gamma_res = molecular * fht * 2 + monoatomic * (0.5 - fht) * 2
    return gamma_res


def calc_fh(tmp, p):
    coef_b = ((2 * np.pi * m_e) ** 0.5 / planck) ** 3
    coef_a = ((2 * np.pi * m_p) ** 0.5 / planck) ** 3

    kbt = kb * tmp
    kbt_quad = kbt ** 2.5

    exp_xi_d = np.exp(-xi_d / kbt)
    exp_xi_h = np.exp(-xi_h / kbt)
    kh_d = coef_a / p * kbt_quad * exp_xi_d
    kh_h = coef_b / p * kbt_quad * exp_xi_h

    fh_d = np.sqrt(kh_d / (1 + kh_d))
    fh_h = np.sqrt(kh_h / (1 + kh_h))

    fht = (1 - fh_d) / 2
    fh = fh_d * (1 - fh_h)
    fion = fh_d * fh_h

    return fh, fht, fion


def calc_fh_rho(tmp, rho):
    p = rho * R * tmp
    return calc_fh(tmp, p)

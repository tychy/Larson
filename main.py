import numpy as np

from conditions import M_cc, G, Rho_init, R_cc
from conditions import DT, TMP_init, AU, GRID, T_END
from utils import CFL, vstack_n


def next(idx, t_h, deltat, v, r, rho, p, tmp, m, deltam, r_h, r_l, p_l, Q):
    # v, r, rho, p, tmp
    v_res = v[idx - 1] - deltat[idx] * G * m[idx] / (r_l[idx] * r_l[idx])
    for i in range(1, v_res.shape[0]):
        v_res[i] -= (
            4
            * np.pi
            * deltat[idx]
            * (
                np.power(r_l[idx][i], 2) * (p_l[idx][i] - p_l[idx][i - 1])
                + (
                    np.power(r_h[idx][i]) * Q[idx - 1][i]
                    - np.power(r_h[idx][i - 1], 3) * Q[idx - 1][i - 1]
                )
                / r[idx][i]
            )
            / deltam[i]
        )
    v_res[0] = 0
    v_res[v_res.shape[0] - 1] = 0
    v.vstack(v_res)

    r_res = r[idx] + v_res * t_h[idx]
    r.vstack(r_res)
    rho_res = np.zeros(r_res.shape[0] - 1)
    for i in range(rho_res.shape[0]):
        rho_res[i] = (
            (3 / 4)
            * deltam[idx][i]
            / (np.power(r_res[idx][i + 1], 3) - np.power(r_res[idx][i], 3))
        )
    rho.vstack(rho_res)
    return


def calc_lambda(idx, v, r, p, t_h, r_l, p_l):
    r_res = r[idx] + (ht[idx] - t_h[idx - 1]) * v[idx] / 4
    p_res = p[idx] + (ht[idx + 1] - t_h[idx - 1])(p[idx] - p[idx - 1]) / (
        4 * t_h[idx - 1]
    )
    r_l.vstack(r_res)
    p_l.vstack(p_res)
    return


def calc_deltam(m, deltam):
    m_res = np.zeros(m.shape[1] - 1)
    for i in range(m_res.shape[0]):
        m_res[i] = m[i + 1] - m[i]
    deltam.vstack(m_res)
    return


def calc_half(r, r_h):
    r_res = np.zeros(r.shape[1] - 1)
    for i in range(r_res[0]):
        r_res[i] = np.power(np.power(r[i], 3) + np.power(r[i], 3), 1 / 3)
    r_h.vstack(r_res)
    return


def calc_Q(idx, l_const, t_h, v, r, rho, Q):
    mu = l_const * l_const * (rho[idx] - rho[idx - 1]) / t_h[idx]
    Q_res = (1 / 3)(np.log(rho[idx]) - np.log(rho[idx - 1])) / t_h[idx]
    for i in range(Q_res.shape[0]):
        Q_res[i] += (v[idx - 1][i + 1] - v[idx - 1][i]) / (
            r[idx - 1][i + 1] - r[idx - 1][i]
        )
    Q_res = -2 * mu * Q_res
    Q.vstack(Q_res)
    return


def main():
    # v_i+\half = idx[i]
    # 初期化
    t = np.arange(0, T_END, DT)

    t_h = np.zeros(t.shape - 1)
    for idx in range(t.shape - 1):
        t_h[idx] = t[idx + 1] - t[idx]
    deltat = np.zeros(t.shape)
    for idx in range(t.shape - 1):
        deltat[idx + 1] = (t_h[idx + 1] + t_h[idx]) / 2
    m = vstack_n((np.arange(0, M_cc + GRID, M_cc / GRID),3)
    v = np.zeros([2, GRID + 1])
    r = vstack_n((np.arange(0, R_cc + GRID, R_cc / GRID),3)
    p = np.zeros([3, GRID])
    rho = np.ones([3, GRID]) * 1.0 / GRID
    tmp = np.ones([3, GRID]) * 10

    # 中間生成物
    l_const = 1
    r_l = np.zeros([3, GRID])
    r_h = np.zeros([3, GRID])
    p_l = np.zeros([3, GRID])
    Q = np.zeros([2, GRID])  # わざと2
    deltam = np.zeros([3, GRID])

    # main loop
    counter = 0
    while t < T_END:
        print("counter:",counter)
        # debug
        if counter == 2:
            break
        calc_lambda(counter, v, r, p, t_h, r_l, p_l)
        calc_deltam(m, deltam)
        calc_half(r, r_h)
        calc_Q(counter, l_const, t_h, v, r, rho, Q)
        next(counter, t_h, deltat, v, r, rho, p, tmp, m, deltam, r_h, r_l, p_l, Q)
        counter += 1


if __name__ == "__main__":
    main()

import numpy as np

from conditions import M_cc, G, Rho_init, R_cc
from conditions import DT, TMP_init, AU, GRID, T_END
from utils import CFL, vstack_n, get_cs


def next(idx, t_h, deltat, v, r, lnrho, p, tmp, m, deltam, r_h, r_l, p_l, Q):
    # v, r, rho, p, tmp
    v_res = v[idx - 1] - deltat[idx] * G * m[idx] / (r_l[idx] * r_l[idx])
    for i in range(1, v_res.shape[0] - 1):
        coef = 4 * np.pi * deltat[idx] / deltam[i]
        v_res[i] -= (
            np.power(r_l[idx][i], 2) * (p_l[idx][i] - p_l[idx][i - 1])
            + (
                np.power(r_h[idx][i], 3) * Q[idx - 1][i]
                - np.power(r_h[idx][i - 1], 3) * Q[idx - 1][i - 1]
            )
            / r[idx][i]
        ) * coef
    v_res[0] = 0
    v_res[v_res.shape[0] - 1] = 0
    v = np.vstack((v, v_res))
    r_res = r[idx] + v_res * t_h[idx]
    r = np.vstack((r, r_res))
    rho_res = np.zeros(r_res.shape[0] - 1)
    p_res = np.zeros(p.shape[1])
    for i in range(rho_res.shape[0]):
        coef = np.power(r_res[i + 1], 3) - np.power(r_res[i], 3)
        rho_res[i] = np.log((3 / 4) * deltam[i] / coef)
        p_res[i] = 1  # change here
    lnrho = np.vstack((lnrho, np.log(rho_res)))
    p = np.vstack((p, p_res))

    return v, r, lnrho, p, tmp


def calc_t(idx, r, t, t_h, deltat, tmp):
    t_diff = CFL(r[idx], tmp.mean(), 0.5)
    t = np.append(t, t[idx] + t_diff)
    t_h = np.append(t_h, t_diff)
    deltat = np.append(deltat, (t_diff + t_h[idx - 1]) / 2)
    if True:
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
    for i in range(r_res.shape[0]):
        r_res[i] = np.power(np.power(r[idx][i], 3) + np.power(r[idx][i], 3), 1 / 3)
    r_h = np.vstack((r_h, r_res))
    return r_h


def calc_Q(idx, l_const, t_h, v, r, lnrho, Q):
    mu = l_const * l_const * (lnrho[idx] - lnrho[idx - 1]) / t_h[idx]
    Q_res = (lnrho[idx] - lnrho[idx - 1]) / (t_h[idx] * 3)
    for i in range(Q_res.shape[0]):
        Q_res[i] += (v[idx - 1][i + 1] - v[idx - 1][i]) / (
            r[idx - 1][i + 1] - r[idx - 1][i]
        )
    Q_res = -2 * mu * Q_res
    Q = np.vstack((Q, Q_res))
    return Q


def main():
    # v_i+\half = idx[i]
    # 初期化
    t = np.array([0, 0.000001, 0.000002])
    t_h = np.zeros(t.shape[0] - 1)
    for idx in range(t.shape[0] - 1):
        t_h[idx] = t[idx + 1] - t[idx]
    deltat = np.zeros(t_h.shape[0])
    for idx in range(1, deltat.shape[0]):
        deltat[idx] = (t_h[idx] + t_h[idx - 1]) / 2
    m = np.arange(0, M_cc + M_cc / GRID, M_cc / GRID)
    v = np.zeros([2, GRID + 1])
    r = vstack_n(np.arange(0, R_cc + R_cc / GRID, R_cc / GRID), 3)
    p = np.ones([3, GRID])
    rho = np.ones([3, GRID]) * 1.0 / GRID
    tmp = np.ones([3, GRID]) * 10

    # 中間生成物
    l_const = 1
    r_l = np.zeros([2, GRID + 1])
    r_h = np.zeros([2, GRID])
    p_l = np.zeros([2, GRID])
    Q = np.zeros([2, GRID])  # わざと2
    deltam = calc_deltam(m)

    # main loop
    counter = 2
    cur_t = 0.0
    while cur_t < T_END:
        print("counter:", counter)
        print("cur_t:{:.8}".format(cur_t))

        # debug
        if counter == 6:
            break
        t, t_h, deltat = calc_t(counter, r, t, t_h, deltat, tmp)
        r_l, p_l = calc_lambda(counter, v, r, p, t_h, r_l, p_l)
        r_h = calc_half(counter, r, r_h)
        Q = calc_Q(counter, l_const, t_h, v, r, rho, Q)

        v, r, rho, p, tmp = next(
            counter, t_h, deltat, v, r, rho, p, tmp, m, deltam, r_h, r_l, p_l, Q
        )
        print(r[idx])
        counter += 1
        cur_t += t_h[idx]


if __name__ == "__main__":
    main()

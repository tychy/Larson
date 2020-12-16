import os
import numpy as np

from conditions import M_cc, G, R_cc
from conditions import TMP_init, AU, GRID, T_END, R, AVG
from conditions import KQ, kb, ki_h
from utils import vstack_n, get_cs, r_init, m_init, save_with_energy
from file_operator import read_json
from calc_operator import calc_t, calc_lambda, calc_deltam, calc_half, calc_Q


eps = 0.0000000001


def next(idx, t_h, deltat, v, r, rho, p, tmp, m, deltam, r_h, r_l, p_l, Q, e, gamma, x):
    # v, r, rho, p, tmp
    v_res_a = v[idx - 1] - deltat[idx] * G * m / (r_l[idx] * r_l[idx])
    v_res_b = np.zeros_like(v_res_a)
    p_diff = np.diff(p_l[idx])
    p_diff = np.insert(p_diff, [0, p_diff.shape[0]], [0, 0])
    m_cur = np.zeros_like(m)
    for i in range(1, m_cur.shape[0] - 1):
        m_cur[i] = (deltam[i - 1] + deltam[i]) / 2
    v_res_b = -(4 * np.pi * deltat[idx] / (m_cur + eps)) * (
        np.power(r_l[idx], 2) * p_diff
    )
    r_three = np.zeros(r[idx].shape[0] - 1)
    for i in range(r_three.shape[0]):
        r_three[i] = (r[idx][i] ** 3 + r[idx][i + 1] ** 3) / 2
    Q_r_three = r_three / 2 * Q[idx - 1]
    Q_diff = np.diff(Q_r_three)
    Q_diff = np.insert(Q_diff, [0, Q_diff.shape[0]], [0, 0])
    v_res_c = -(4 * np.pi * deltat[idx] / (m_cur + eps)) * Q_diff / r[idx]

    v_res = v_res_a + v_res_b + v_res_c
    v_res[0] = 0
    v_res[v_res.shape[0] - 1] = 0
    v = np.vstack((v, v_res.astype(np.float64)))
    print("v:", v_res)
    print("v from g:", v_res_a)
    print("v from p:", v_res_b)
    print("v from q:", v_res_c)

    r_res = r[idx] + v_res * t_h[idx]
    r = np.vstack((r, r_res))
    rho_res = np.zeros(rho.shape[1])
    rho_res = deltam / ((4 / 3) * np.pi * (np.diff(np.power(r_res, 3))))
    rho = np.vstack((rho, rho_res.astype(np.float64)))
    Q = calc_Q(idx, v, r, rho, t_h, deltat, Q)

    e_res = (
        e[idx]
        - p[idx] * (1 / rho[idx] - 1 / rho[idx - 1]) / 2
        + deltat[idx] * (-3 / 2 * Q[idx] + 0)
    )
    e = np.vstack((e, e_res))
    tmp_res = (e_res - x[idx] * xi_h) * 2 / 3 / R * AVG / (1 + x[idx])
    tmp = np.vstack((tmp, tmp_res))
    gamma_res = 
    gamma = np.vstack((gamma, gamma_res))

    p_res = np.zeros(p.shape[1])
    p_res = rho_res * R * tmp[idx] / AVG
    p = np.vstack((p, p_res))

    u_zero =
    u_one =
    k_h = u_one/u_zero*2/p_res
    x_res = np.sqrt(k_h/(1+k_h))
    x = np.vstack((x, x_res))
    return v, r, rho, p, tmp, Q, e, gamma, x


def main():
    config = read_json()
    base_dir = os.path.join("data", str(config["tag"]))
    os.makedirs(base_dir, exist_ok=True)
    # v_i+\half = idx[i]
    # 初期化
    t = np.array([0, 0.000001, 0.000002])
    t_h = np.zeros(t.shape[0] - 1)
    for idx in range(t.shape[0] - 1):
        t_h[idx] = t[idx + 1] - t[idx]
    deltat = np.zeros(t_h.shape[0])
    for idx in range(1, deltat.shape[0]):
        deltat[idx] = (t_h[idx] + t_h[idx - 1]) / 2
    r = vstack_n(r_init(), 3)
    m = m_init(r_init())
    v = np.zeros([2, GRID + 1])
    # 中間生成物
    r_l = np.zeros([2, GRID + 1])
    r_h = np.zeros([2, GRID])
    p_l = np.zeros([2, GRID])
    deltam = calc_deltam(m)

    p = np.zeros([3, GRID])
    Q = np.zeros([2, GRID])
    rho = vstack_n(deltam / ((4 / 3) * np.pi * (np.diff(np.power(r[2], 3)))), 3)
    tmp = np.ones([3, GRID]) * 10
    e = 2.5 * kb * tmp
    gamma = np.ones([3, GRID])
    x = np.zeros([3, GRID])
    # main loop
    counter = 2
    cur_t = 0.0
    while cur_t < T_END:

        t, t_h, deltat = calc_t(counter, r, t, t_h, deltat, tmp)
        r_l, p_l = calc_lambda(counter, v, r, p, t_h, r_l, p_l)
        r_h = calc_half(counter, r, r_h)
        v, r, rho, p, tmp, Q, e, gamma, x = next(
            counter,
            t_h,
            deltat,
            v,
            r,
            rho,
            p,
            tmp,
            m,
            deltam,
            r_h,
            r_l,
            p_l,
            Q,
            e,
            gamma,
            x,
        )
        if counter % 100 == 0:
            print("counter:", counter)
            print("cur_t:{:.8}".format(cur_t))
            save_with_energy(
                base_dir, counter, v, r, rho, p, tmp, r_h, t, Q, e, gamma, x
            )

        cur_t += t_h[counter]
        counter += 1
    save_with_energy(base_dir, counter, v, r, rho, p, tmp, r_h, t, Q, e, gamma, x)


if __name__ == "__main__":
    main()

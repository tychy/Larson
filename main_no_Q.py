import json
import os
import numpy as np
import matplotlib.pyplot as plt

from conditions import M_cc, G, R_cc
from conditions import DT, TMP_init, AU, GRID, T_END, R_LOG, AVG
from conditions import KQ, CFL_CONST
from utils import CFL, vstack_n, get_cs, r_init, m_init, save

eps = 0.0000000001


def next(idx, t_h, deltat, v, r, rho, p, tmp, m, deltam, r_h, r_l, p_l):
    # v, r, rho, p, tmp
    v_res_a = v[idx - 1] - deltat[idx] * G * m / (r_l[idx] * r_l[idx] + eps)
    v_res_b = np.zeros_like(v_res_a)
    p_diff = np.diff(p_l[idx])
    p_diff = np.insert(p_diff, [0, p_diff.shape[0]], [0, 0])
    m_cur = np.zeros_like(m)
    for i in range(1, m_cur.shape[0] - 1):
        m_cur[i] = (deltam[i - 1] + deltam[i]) / 2
    v_res_b = -(4 * np.pi * deltat[idx] / (m_cur + eps)) * (
        np.power(r_l[idx], 2) * p_diff
    )

    v_res = v_res_a + v_res_b
    v_res[0] = 0
    v_res[v_res.shape[0] - 1] = 0
    v = np.vstack((v, v_res))
    r_res = r[idx] + v_res * t_h[idx]
    r = np.vstack((r, r_res))
    tmp = np.vstack((tmp, tmp[0]))
    rho_res = np.zeros(rho.shape[1])
    rho_res = deltam / ((4 / 3) * np.pi * (np.diff(np.power(r_res, 3))))

    p_res = np.zeros(p.shape[1])
    p_res = rho_res * np.power(10, R_LOG) * tmp[idx] / AVG
    rho = np.vstack((rho, rho_res))
    p = np.vstack((p, p_res))
    return v, r, rho, p, tmp


def calc_t(idx, r, t, t_h, deltat, tmp):
    t_diff = CFL(r[idx], tmp.mean(), CFL_CONST)
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
    for i in range(r_res.shape[0]):
        r_res[i] = np.power(np.power(r[idx][i], 3) + np.power(r[idx][i], 3), 1 / 3)
    r_h = np.vstack((r_h, r_res))
    return r_h


def main():
    with open("configs.json", "r") as f:
        json_open = json.load(f)
    base_dir = os.path.join("data", str(json_open["tag"]))
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
    m = m_init()
    v = np.zeros([2, GRID + 1])
    r = vstack_n(r_init(m), 3)
    # 中間生成物
    r_l = np.zeros([2, GRID + 1])
    r_h = np.zeros([2, GRID])
    p_l = np.zeros([2, GRID])
    deltam = calc_deltam(m)

    p = np.ones([3, GRID]) / np.power(10, 5)
    rho = vstack_n(deltam / ((4 / 3) * np.pi * (np.diff(np.power(r[2], 3)))), 3)
    tmp = np.ones([3, GRID]) * 10

    # main loop
    counter = 2
    cur_t = 0.0
    while cur_t < T_END:

        t, t_h, deltat = calc_t(counter, r, t, t_h, deltat, tmp)
        r_l, p_l = calc_lambda(counter, v, r, p, t_h, r_l, p_l)
        r_h = calc_half(counter, r, r_h)
        v, r, rho, p, tmp = next(
            counter, t_h, deltat, v, r, rho, p, tmp, m, deltam, r_h, r_l, p_l
        )
        if counter % 50000 == 0:
            print("counter:", counter)
            print("cur_t:{:.8}".format(cur_t))
            plt.plot(
                np.log(r_h[counter]),
                np.log(rho[counter]),
                label="{}".format(t[counter]),
            )
            save(base_dir, counter, t_h, deltat, v, r, rho, p, tmp, r_h, r_l, p_l, t)
        cur_t += t_h[counter]
        counter += 1
    save(base_dir, counter, t_h, deltat, v, r, rho, p, tmp, r_h, r_l, p_l, t)
    plt.legend()
    plt.savefig("results/step_{}_noQ.png".format(counter))


if __name__ == "__main__":
    main()

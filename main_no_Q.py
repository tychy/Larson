import os
import numpy as np

from conditions import M_cc, G, R_CC
from conditions import TMP_INIT, AU, GRID, T_END, R, AVG
from conditions import KQ
from utils import vstack_n, get_cs, r_init, m_init
from file_operator import read_json, copy_json, save
from calc_operator import calc_t, calc_lambda, calc_deltam, calc_half


eps = 0.0000000001


def next(idx, t_h, deltat, v, r, rho, p, tmp, m, deltam, r_h, r_l, p_l):
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

    v_res = v_res_a + v_res_b
    v_res[0] = 0
    v_res[v_res.shape[0] - 1] = 0
    v = np.vstack((v, v_res.astype(np.float64)))
    print("v:", v_res)
    print("v from g:", v_res_a)
    print("v from p:", v_res_b)

    r_res = r[idx] + v_res * t_h[idx]
    r = np.vstack((r, r_res))
    tmp = np.vstack((tmp, tmp[0]))
    rho_res = np.zeros(rho.shape[1])
    rho_res = deltam / ((4 / 3) * np.pi * (np.diff(np.power(r_res, 3))))
    p_res = np.zeros(p.shape[1])
    p_res = rho_res * R * tmp[idx] / AVG
    rho = np.vstack((rho, rho_res.astype(np.float64)))
    p = np.vstack((p, p_res))
    return v, r, rho, p, tmp


def main():
    config = read_json()
    base_dir = os.path.join("data", str(config["tag"]))
    os.makedirs(base_dir, exist_ok=True)
    copy_json(base_dir)

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
    rho = vstack_n(deltam / ((4 / 3) * np.pi * (np.diff(np.power(r[2], 3)))), 3)
    tmp = np.ones([3, GRID]) * TMP_INIT

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
        if counter % 100 == 0:
            print("counter:", counter)
            print("cur_t:{:.8}".format(cur_t))
            save(base_dir, counter, t_h, deltat, v, r, rho, p, tmp, r_h, r_l, p_l, t)
        cur_t += t_h[counter]
        counter += 1
    save(base_dir, counter, t_h, deltat, v, r, rho, p, tmp, r_h, r_l, p_l, t)


if __name__ == "__main__":
    main()

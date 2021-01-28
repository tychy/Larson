import os
import numpy as np

from conditions import M_cc, G, R_CC
from conditions import TMP_INIT, AU, GRID, T_END, R, AVG
from conditions import KQ, kb, Kapper, SB, xi_d
from utils import vstack_n, get_cs, r_init, m_init
from file_operator import read_json, copy_json, save_with_ionization
from calc_operator import calc_t, calc_lambda, calc_deltam, calc_half, calc_Q
from calc_operator import calc_gamma, calc_fh, calc_fh_rho
from logging import getLogger, StreamHandler, DEBUG
import os
from contextlib import redirect_stdout

eps = 0.0000000001


def next(
    idx, t_h, deltat, v, r, rho, p, tmp, m, deltam, r_h, r_l, p_l, Q, e, fh, fht, fion
):
    # v, r, rho, p, tmp
    v_res_a = v[idx - 1] - deltat[idx] * G * m / (r_l[idx] * r_l[idx])
    v_res_b = np.zeros_like(v_res_a)
    print("p_l:", p_l[idx])
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
    print("idx", idx)
    print("v:", v_res)
    print("v from g:", v_res_a)
    print("pdiff", p_diff)
    print("v from p:", v_res_b)
    print("v from q:", v_res_c)

    r_res = r[idx] + v_res * t_h[idx]
    r = np.vstack((r, r_res))
    r_h = calc_half(idx + 1, r, r_h)

    rho_res = np.zeros(rho.shape[1])
    rho_res = deltam / ((4 / 3) * np.pi * (np.diff(np.power(r_res, 3))))
    print("rho_res", rho_res)
    rho = np.vstack((rho, rho_res.astype(np.float64)))

    gamma = calc_gamma(fht[idx])
    print("gamma", gamma)
    Q = calc_Q(idx, v, r, rho, t_h, deltat, Q)
    efromq = deltat[idx] * (-3 / 2 * Q[idx])

    # 計算を先に済ませておく
    coef_inv_rho = (1 / rho[idx + 1] - 1 / rho[idx]) / 2
    coef_a = 4 / 3 * SB / Kapper
    coef_b = coef_a / rho_res / (r_h[idx] ** 2)
    coef_c = coef_a / (rho_res ** 2)
    coef_d = r_h[idx] ** 2 / rho_res
    tmp_two = tmp[idx] ** 2
    tmp_three = tmp[idx] ** 3

    deltar = np.diff(r_h[idx])
    deltar_res = np.diff(r_h[idx + 1])

    # T^nの位置微分
    pder = np.zeros_like(tmp[idx])
    for j in range(1, pder.shape[0] - 1):
        pder[j] = (tmp[idx][j + 1] - tmp[idx][j - 1]) / deltar[j] / 2
    pder[-1] = pder[-2]  # b.c.
    pder[0] = pder[1]
    ppder = np.zeros_like(tmp[idx])
    for j in range(1, ppder.shape[0] - 1):
        ppder[j] = (
            (tmp[idx][j + 1] - tmp[idx][j]) / deltar[j]
            - (tmp[idx][j] - tmp[idx][j - 1]) / deltar[j - 1]
        ) / deltar[j]
    ppder[0] = ppder[1]
    ppder[-1] = ppder[-2]

    # TMPの更新
    d = np.zeros_like(tmp[idx])
    f = np.zeros_like(tmp[idx])
    tmp_res = np.ones_like(tmp[idx]) * TMP_INIT
    deltatmp = np.zeros_like(tmp[idx])
    t_n = deltat[idx]
    if idx <= 10:
        pderfht = np.zeros_like(tmp[idx])
    else:
        pderfht = (
            calc_fh(tmp[idx] * 1.001, p[idx])[1] - calc_fh(tmp[idx], p[idx])[1]
        ) / (0.001 * tmp[idx])
    print("pderfht", pderfht)
    na = 6 * 10 ** 23
    fht_rho = (
        calc_fh_rho(tmp[idx], rho[idx + 1])[1] - calc_fh_rho(tmp[idx], rho[idx])[1]
    )

    for j in range(1, tmp[idx].shape[0] - 1):
        cur_a = t_n * coef_c[j] * 4 * tmp_three[j] / (deltar_res[j] ** 2)
        cur_b = t_n * coef_b[j] * (coef_d[j + 1] - coef_d[j - 1]) / 2 / deltar_res[j]
        cur_c = t_n * coef_c[j] * 24 * tmp_two[j] * pder[j] / 2 / deltar_res[j]
        cur_d = t_n * coef_c[j] * (pder[j] ** 2)

        a_j = cur_a - cur_b * 4 * tmp_three[j] / 2 / deltar_res[j] - cur_c

        b_j = (
            +R / (gamma[j] - 1)
            + R * rho_res[j] * coef_inv_rho[j]
            + 2 * cur_a
            - 24 * cur_d * tmp[idx][j]
            - cur_b * 12 * tmp_two[j] * pder[j]
            - xi_d * pderfht[j] * na / 2  # 2は修正必要かも
        )
        c_j = cur_b * 4 * tmp_three[j] / 2 / deltar_res[j] + cur_a + cur_c

        r_j = (
            -R * (tmp[idx][j] * (rho[idx][j] + rho_res[j])) * coef_inv_rho[j]
            + efromq[j]
            + 12 * cur_d * tmp_two[j]
            + t_n * coef_c[j] * 4 * tmp_three[j] * ppder[j]
            + cur_b * 4 * tmp_three[j] * pder[j]
            + xi_d * fht_rho[j] * na / 2
        )
        d[j] = c_j / (b_j - a_j * d[j - 1])
        f[j] = (r_j + a_j * f[j - 1]) / (b_j - a_j * d[j - 1])
    d[0] = d[1]
    f[0] = f[1]
    for j in reversed(range(tmp[idx].shape[0])):
        if j == tmp[idx].shape[0] - 1:
            deltatmp[j] = 10 * d[j] + f[j]
        else:
            deltatmp[j] = deltatmp[j + 1] * d[j] + f[j]
    tmp_res = tmp[idx] + deltatmp
    print("delta_tmp:", deltatmp)
    print("tmp:", tmp_res)
    tmp = np.vstack((tmp, tmp_res))

    e_res = tmp_res * R / (gamma[j] - 1)
    print("e:", e_res)
    e = np.vstack((e, e_res))
    p_res = (gamma[j] - 1) * rho_res * e_res
    print("p", p_res)
    p = np.vstack((p, p_res))
    fh_res, fht_res, fion_res = calc_fh(tmp_res, p_res)
    fh = np.vstack((fh, fh_res))
    fht = np.vstack((fht, fht_res))
    fion = np.vstack((fion, fion_res))
    return v, r, rho, p, tmp, Q, e, fh, fht, fion


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
    print("deltam", deltam)
    p = np.zeros([3, GRID])
    Q = np.zeros([2, GRID])
    rho = vstack_n(deltam / ((4 / 3) * np.pi * (np.diff(np.power(r[2], 3)))), 3)
    tmp = np.ones([3, GRID]) * 10
    e = vstack_n(tmp[-1] * R / 0.4, 3)
    fh = np.zeros([3, GRID])
    fht = np.zeros([3, GRID])
    fion = np.zeros([3, GRID])

    # main loop
    counter = 2
    cur_t = 0.0
    while cur_t < T_END:

        t, t_h, deltat = calc_t(counter, r, t, t_h, deltat, tmp[counter])
        r_l, p_l = calc_lambda(counter, v, r, p, t_h, r_l, p_l)
        r_h = calc_half(counter, r, r_h)
        v, r, rho, p, tmp, Q, e, fh, fht, fion = next(
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
            fh,
            fht,
            fion,
        )
        if counter % 500 == 0:
            print("counter:", counter)
            print("cur_t:{:.8}".format(cur_t))
            save_with_ionization(
                base_dir, counter, v, r, rho, p, tmp, r_h, t, Q, e, fh, fht, fion
            )

        cur_t += t_h[counter]
        counter += 1
    save_with_ionization(
        base_dir, counter, v, r, rho, p, tmp, r_h, t, Q, e, fh, fht, fion
    )


if __name__ == "__main__":
    with redirect_stdout(open(os.devnull, "w")):
        main()
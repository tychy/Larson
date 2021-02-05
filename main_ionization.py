import os
import numpy as np

from conditions import M_cc, G, RCC, DISPLAY
from conditions import TMP_INIT, AU, GRID, MAXSTEP, R, AVG
from conditions import KQ, kb, Kapper, SB, xi_d, xi_h, NA
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
    t_n = deltat[idx]
    # v, r, rho, p, tmp
    v_res_a = v[idx - 1] - t_n * G * m / (r_l[idx] * r_l[idx])
    v_res_b = np.zeros_like(v_res_a)
    if DISPLAY:
        print("p_l:", p_l[idx])
    p_diff = np.diff(p_l[idx])
    p_diff = np.insert(p_diff, [0, p_diff.shape[0]], [0, 0])
    m_cur = np.zeros_like(m)
    for i in range(1, m_cur.shape[0] - 1):
        m_cur[i] = (deltam[i - 1] + deltam[i]) / 2
    v_res_b = -(4 * np.pi * t_n / (m_cur + eps)) * (np.power(r_l[idx], 2) * p_diff)
    r_three = np.zeros(r[idx].shape[0] - 1)
    for i in range(r_three.shape[0]):
        r_three[i] = (r[idx][i] ** 3 + r[idx][i + 1] ** 3) / 2
    Q_r_three = r_three / 2 * Q[idx - 1]
    Q_diff = np.diff(Q_r_three)
    Q_diff = np.insert(Q_diff, [0, Q_diff.shape[0]], [0, 0])
    v_res_c = -(4 * np.pi * t_n / (m_cur + eps)) * Q_diff / r[idx]

    v_res = v_res_a + v_res_b + v_res_c
    v_res[v_res.shape[0] - 1] = 0
    v_res[0] = 0
    v = np.vstack((v, v_res.astype(np.float64)))

    r_res = r[idx] + v_res * t_h[idx]
    r = np.vstack((r, r_res))
    if DISPLAY:
        print("idx", idx)
        print("r", r_res)
        print("v:", v_res)
        print("v from g:", v_res_a)
        print("pdiff", p_diff)
        print("v from p:", v_res_b)
        print("v from q:", v_res_c)
    r_h = calc_half(idx + 1, r, r_h)

    rho_res = np.zeros(rho.shape[1])
    rho_res = deltam / ((4 / 3) * np.pi * (np.diff(np.power(r_res, 3))))
    rho = np.vstack((rho, rho_res.astype(np.float64)))

    gamma = calc_gamma(fh[idx], fht[idx], fion[idx])

    Q, Phi_res = calc_Q(idx, v, r, rho, t_h, deltat, Q)
    efromq = t_n * Phi_res
    if DISPLAY:
        print("rho_res", rho_res)
        print("gamma", gamma)

    # 計算を先に済ませておく
    coef_inv_rho = (1 / rho[idx + 1] - 1 / rho[idx]) / 2
    deltar_mid = np.diff(r_res)
    deltar_res = np.zeros_like(r_res)
    deltar_res[0] = deltar_mid[0]
    for i in range(1, len(deltar_res) - 1):
        deltar_res[i] = (deltar_mid[i - 1] + deltar_mid[i]) / 2
    coef_base = 4 * np.pi / 3 * SB / Kapper
    tmp_three = tmp[idx] ** 3
    tmp_four = tmp[idx] ** 4
    # TMPの更新

    a = np.zeros_like(tmp[idx])
    b = np.zeros_like(tmp[idx])
    c = np.zeros_like(tmp[idx])
    r_ls = np.zeros_like(tmp[idx])
    d = np.zeros_like(tmp[idx])
    f = np.zeros_like(tmp[idx])
    tmp_res = np.ones_like(tmp[idx]) * TMP_INIT
    deltatmp = np.zeros_like(tmp[idx])
    xmu = 1.4 / (0.1 + fh[idx] + fht[idx] + 2 * fion[idx])
    if idx <= 10:
        pderfht = np.zeros_like(tmp[idx])
    else:
        dtmp = 0.001
        pderfht = (
            calc_fh(tmp[idx] * (1 + dtmp), rho[idx + 1], xmu)[1]
            - calc_fh(tmp[idx], rho[idx + 1], xmu)[1]
        ) / (dtmp * tmp[idx])
    if idx <= 10:
        pderfion = np.zeros_like(tmp[idx])
    else:
        dtmp = 0.001
        pderfion = (
            calc_fh(tmp[idx] * (1 + dtmp), rho[idx + 1], xmu)[2]
            - calc_fh(tmp[idx], rho[idx + 1], xmu)[2]
        ) / (dtmp * tmp[idx])

        # pderfht = np.where(pderfht > 0, 0, pderfht)
    if DISPLAY:
        print("pderfht", pderfht)
    fht_rho = (
        calc_fh(tmp[idx], rho[idx + 1], xmu)[1] - calc_fh(tmp[idx], rho[idx], xmu)[1]
    )
    fion_rho = (
        calc_fh(tmp[idx], rho[idx + 1], xmu)[2] - calc_fh(tmp[idx], rho[idx], xmu)[2]
    )

    # fht_rho = np.where(fht_rho > 0, 0, fht_rho)
    cur_ap = (
        t_n
        * r[idx + 1][1] ** 2
        * 2
        / (rho_res[0] + rho_res[1])
        / deltar_res[1]
        / deltam[0]
    )
    a_j = 0
    b_j = (
        +R / xmu[0] / (gamma[0] - 1)
        + R / xmu[0] * rho_res[0] * coef_inv_rho[0]
        + 4 * coef_base * cur_ap * tmp_three[0]
        - xi_d * pderfht[0] * NA / xmu[0]
        + xi_h * pderfion[0] * NA / xmu[0]
    )
    c_j = coef_base * cur_ap * 4 * tmp_three[1]

    r_j = (
        -R / xmu[0] * (tmp[idx][0] * (rho[idx][0] + rho_res[0])) * coef_inv_rho[0]
        + efromq[0]
        + coef_base * cur_ap * (tmp_four[1] - tmp_four[0])
        + xi_d * fht_rho[0] * NA / xmu[0]
        - xi_h * fion_rho[0] * NA / xmu[0]
    )
    d[0] = c_j / (b_j)
    f[0] = (r_j) / (b_j)
    a[0] = a_j
    b[0] = b_j
    c[0] = c_j
    r_ls[0] = r_j
    if DISPLAY:
        print("1st", -R / xmu * (tmp[idx] * (rho[idx] + rho_res)) * coef_inv_rho)
        print("efromQ", efromq)
        print("dis", +xi_d * fht_rho * NA / xmu)

    for j in range(1, tmp[idx].shape[0] - 1):
        cur_am = (
            t_n
            * r[idx + 1][j] ** 2
            * 2
            / (rho_res[j - 1] + rho_res[j])
            / deltar_res[j + 1]
            / deltam[j]
        )
        cur_ap = (
            t_n
            * r[idx + 1][j + 1] ** 2
            * 2
            / (rho_res[j] + rho_res[j + 1])
            / deltar_res[j + 1]
            / deltam[j]
        )

        a_j = coef_base * cur_am * 4 * tmp_three[j - 1]
        b_j = (
            +R / xmu[j] / (gamma[j] - 1)
            + R / xmu[j] * rho_res[j] * coef_inv_rho[j]
            + 4 * coef_base * cur_ap * tmp_three[j]
            + 4 * coef_base * cur_am * tmp_three[j]
            - xi_d * pderfht[j] * NA / xmu[j]
            + xi_h * pderfion[j] * NA / xmu[j]
        )
        c_j = coef_base * cur_ap * 4 * tmp_three[j + 1]

        r_j = (
            -R / xmu[j] * (tmp[idx][j] * (rho[idx][j] + rho_res[j])) * coef_inv_rho[j]
            + efromq[j]
            + coef_base * cur_ap * (tmp_four[j + 1] - tmp_four[j])
            + coef_base * cur_am * (tmp_four[j - 1] - tmp_four[j])
            + xi_d * fht_rho[j] * NA / xmu[j]
            - xi_h * fion_rho[j] * NA / xmu[j]
        )
        a[j] = a_j
        b[j] = b_j
        c[j] = c_j
        r_ls[j] = r_j
        d[j] = c_j / (b_j - a_j * d[j - 1])
        f[j] = (r_j + a_j * f[j - 1]) / (b_j - a_j * d[j - 1])
    if DISPLAY:
        print("d:", d)
        print("f:", f)
    for j in reversed(range(tmp[idx].shape[0])):
        if j == tmp[idx].shape[0] - 1:
            deltatmp[j] = 0 * d[j] + f[j]
        else:
            deltatmp[j] = deltatmp[j + 1] * d[j] + f[j]
    if DISPLAY:
        for j in range(10):
            if deltatmp[j] < deltatmp[j + 1]:
                print(idx)
                print("Oops")
                print(deltatmp)
                print("a", a)
                print("b", b)
                print("c", c)
                print("r", r_ls)
                print("d", d)
                print("f", f)
                print("pderfht", pderfht)
                print("fht_rho", fht_rho)
                break
    tmp_res = tmp[idx] + deltatmp
    tmp = np.vstack((tmp, tmp_res))

    e_res = tmp_res * R / xmu / (gamma - 1)

    e = np.vstack((e, e_res))
    p_res = (gamma - 1) * rho_res * e_res
    p = np.vstack((p, p_res))

    if DISPLAY:
        print("delta_tmp:", deltatmp)
        print("tmp:", tmp_res)
        print("e:", e_res)
        print("p", p_res)

    fh_res, fht_res, fion_res = calc_fh(tmp_res, rho_res, xmu)
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
    if DISPLAY:
        print("deltam", deltam)
    p = np.zeros([3, GRID])
    Q = np.zeros([2, GRID])
    rho = vstack_n(deltam / ((4 / 3) * np.pi * (np.diff(np.power(r[2], 3)))), 3)
    tmp = np.ones([3, GRID]) * 10
    e = vstack_n(tmp[-1] * R / AVG / 0.4, 3)
    fh = np.zeros([3, GRID])
    fht = np.ones([3, GRID]) / 2
    fion = np.zeros([3, GRID])

    # main loop
    counter = 2
    idx = counter
    cur_t = 0.0
    cur_rho = np.max(np.floor(np.log10(rho[0])))
    skip = 0
    while counter < MAXSTEP:
        t, t_h, deltat = calc_t(idx, v, r, t, t_h, deltat, tmp[idx])
        r_l, p_l = calc_lambda(idx, v, r, p, t_h, r_l, p_l)
        r_h = calc_half(idx, r, r_h)
        v, r, rho, p, tmp, Q, e, fh, fht, fion = next(
            idx,
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

        cur_t += t_h[idx]
        if counter <= 10:
            idx += 1
        elif skip > 0:
            skip -= 1
            idx += 1
        elif (
            counter % 500 == 0
            or np.abs(np.max(np.floor(np.log10(rho[idx]))) - cur_rho) > 0.1
        ):

            cur_rho = np.max(np.floor(np.log10(rho[idx])))
            print("counter:", counter)
            print("cur_t:{:.8}".format(cur_t))
            print("core tmp", tmp[idx][0])
            print(calc_gamma(fh[idx], fht[idx], fion[idx])[0])
            skip = 5
            idx += 1

        else:
            t = np.delete(t, -3, 0)
            t_h = np.delete(t_h, -3, 0)
            deltat = np.delete(deltat, -3, 0)
            v = np.delete(v, -3, 0)
            r = np.delete(r, -3, 0)
            rho = np.delete(rho, -3, 0)
            p = np.delete(p, -3, 0)
            tmp = np.delete(tmp, -3, 0)
            r_h = np.delete(r_h, -3, 0)
            r_l = np.delete(r_l, -3, 0)
            p_l = np.delete(p_l, -3, 0)
            Q = np.delete(Q, -3, 0)
            e = np.delete(e, -3, 0)
            fh = np.delete(fh, -3, 0)
            fht = np.delete(fht, -3, 0)
            fion = np.delete(fion, -3, 0)
        if counter % 3000 == 0:
            save_with_ionization(
                base_dir, idx, v, r, rho, p, tmp, r_h, t, Q, e, fh, fht, fion
            )
        counter += 1

    save_with_ionization(base_dir, idx, v, r, rho, p, tmp, r_h, t, Q, e, fh, fht, fion)


if __name__ == "__main__":
    if True:
        main()
    else:
        with redirect_stdout(open(os.devnull, "w")):
            main()

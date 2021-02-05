import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from calc_operator import calc_fh, calc_gamma
from conditions import AVG


def search_fh():

    rho_ls = []
    tmp_ls_low = []
    tmp_ls_high = []

    # グリッドサーチ
    for rho_index in range(-10, -1):
        rho_res = 10 ** rho_index
        tmp_ls = np.linspace(100, 10000, 1000)
        best_low = 0.0
        best_tmp_low = 100
        best_high = 0.0
        best_tmp_high = 100
        for item in tmp_ls:
            fh_low = 2 * (0.5 - calc_fh(item, rho_res, 2.15)[1])
            fh_high = 2 * (0.5 - calc_fh(item, rho_res, 1.4)[1])

            if np.abs(fh_low - 0.1) < np.abs(best_low - 0.1):
                best_low = fh_low
                best_tmp_low = item
            if np.abs(fh_high - 0.9) < np.abs(best_high - 0.9):
                best_high = fh_high
                best_tmp_high = item
        print("index:", rho_index)
        print(best_low)
        print(best_high)
        rho_ls.append(rho_res)
        tmp_ls_low.append(best_tmp_low)
        tmp_ls_high.append(best_tmp_high)
    rho_ls = np.log10(np.array(rho_ls))
    tmp_ls_low = np.log10(np.array(tmp_ls_low))
    tmp_ls_high = np.log10(np.array(tmp_ls_high))
    return rho_ls, tmp_ls_low, tmp_ls_high


def main():
    sns.set_theme()
    rho_ls, tmp_ls_low, tmp_ls_high = search_fh()
    figure = plt.figure()
    plt.plot(rho_ls, tmp_ls_low, label="f_h = 0.1")
    plt.plot(rho_ls, tmp_ls_high, label="f_h = 0.9")
    plt.xlabel("log10rho cgs")
    plt.ylabel("log10T cgs")
    plt.legend()
    plt.savefig("fh_2d.png")


if __name__ == "__main__":
    main()
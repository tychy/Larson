import matplotlib.pyplot as plt
import numpy as np
import json
import os
import seaborn as sns
from calc_operator import calc_fh, calc_gamma
from conditions import AVG

from file_operator import read_json, read_index
from conditions import GRID


def main():
    sns.set_theme()
    config = read_json()
    idx = read_index("data/" + config["plot_tag"])
    if config["use_custom"]:
        idx = config["plot_step"]

    r_h = np.load(
        "data/" + config["plot_tag"] + "/step_r_h.npy",
        allow_pickle=True,
    )
    r = np.load(
        "data/" + config["plot_tag"] + "/step_r.npy",
        allow_pickle=True,
    )

    rho = np.load(
        "data/" + config["plot_tag"] + "/step_rho.npy",
        allow_pickle=True,
    )
    t = np.load(
        "data/" + config["plot_tag"] + "/step_t.npy",
        allow_pickle=True,
    )

    tmp = np.load(
        "data/" + config["plot_tag"] + "/step_tmp.npy",
        allow_pickle=True,
    )

    e = np.load(
        "data/" + config["plot_tag"] + "/step_e.npy",
        allow_pickle=True,
    )

    i = 2
    rho_ls = []
    tmp_ls = []
    while i < idx:
        if i % 10 == 0:
            rho_ls.append(np.log10(rho[i][1]))
            tmp_ls.append(np.log10(tmp[i][1]))
        i += 1
    figure = plt.figure()
    plt.plot(rho_ls, tmp_ls)
    plt.xlabel("log10rho cgs")
    plt.ylabel("log10T cgs")
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
            if np.abs(calc_fh(item, rho_res, AVG)[0] - 0.1) < np.abs(best_low - 0.1):
                best_low = calc_fh(item, rho_res, AVG)[0]
                best_tmp_low = item
            if np.abs(calc_fh(item, rho_res, AVG)[0] - 0.9) < np.abs(best_high - 0.9):
                best_high = calc_fh(item, rho_res, AVG)[0]
                best_tmp_high = item
        print(best_low)
        print(best_high)
        rho_ls.append(rho_res)
        tmp_ls_low.append(best_tmp_low)
        tmp_ls_high.append(best_tmp_high)
    rho_ls = np.log10(np.array(rho_ls))
    tmp_ls_low = np.log10(np.array(tmp_ls_low))
    tmp_ls_high = np.log10(np.array(tmp_ls_high))

    plt.plot(rho_ls, tmp_ls_low, linestyle="dashed", label="f_h = 0.1")
    plt.plot(rho_ls, tmp_ls_high, linestyle="dashed", label="f_h = 0.9")
    plt.legend()
    os.makedirs("results/" + config["plot_tag"], exist_ok=True)
    plt.savefig("results/" + config["plot_tag"] + "/core.png")
    i = 2
    rho_ls = []
    e_ls = []
    while i < idx:
        if i % 10 == 0:
            rho_ls.append(np.log10(rho[i][1]))
            e_ls.append(np.log10(e[i][1]))
        i += 1
    figure = plt.figure()
    plt.plot(rho_ls, e_ls)
    plt.xlabel("log10 rho")
    plt.ylabel("log10 E")
    plt.legend()
    os.makedirs("results/" + config["plot_tag"], exist_ok=True)
    plt.savefig("results/" + config["plot_tag"] + "/core_energy.png")


if __name__ == "__main__":
    main()

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from file_operator import read_json, read_index
from conditions import GRID, AVG
from plot_fh_2d import search_fh, search_fion


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
    # plot TMP
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
    plt.xlabel(r"$\log_{10}$" + r"$\rho$")
    plt.ylabel(r"$\log_{10}$" + "T")
    rho_fh, tmp_fh_low, tmp_fh_high = search_fh()
    rho_fion, tmp_fion_low, tmp_fion_high = search_fion()
    plt.plot(rho_fh, tmp_fh_low, linestyle="dashed", label=r"$f_H$ = 0.1")
    plt.plot(rho_fh, tmp_fh_high, linestyle="dashed", label=r"$f_H$ = 0.9")
    plt.plot(rho_fion, tmp_fion_low, linestyle="dotted", label=r"$f_{H+}$= 0.1")
    plt.plot(rho_fion, tmp_fion_high, linestyle="dotted", label=r"$f_{H+}$= 0.5")
    plt.legend()
    plt.tight_layout()
    os.makedirs("results/" + config["plot_tag"], exist_ok=True)
    plt.savefig("results/" + config["plot_tag"] + "/core.png")

    # plot Energy
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

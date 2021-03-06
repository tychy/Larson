import matplotlib.pyplot as plt
import numpy as np
import json
import os
import seaborn as sns

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

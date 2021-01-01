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

    r_h = np.load(
        "data/" + config["plot_tag"] + "/step_{}_r_h.npy".format(idx),
        allow_pickle=True,
    )
    r = np.load(
        "data/" + config["plot_tag"] + "/step_{}_r.npy".format(idx),
        allow_pickle=True,
    )

    rho = np.load(
        "data/" + config["plot_tag"] + "/step_{}_rho.npy".format(idx),
        allow_pickle=True,
    )
    t = np.load(
        "data/" + config["plot_tag"] + "/step_{}_t.npy".format(idx),
        allow_pickle=True,
    )

    tmp = np.load(
        "data/" + config["plot_tag"] + "/step_{}_tmp.npy".format(idx),
        allow_pickle=True,
    )

    i = 2
    rho_ls = []
    tmp_ls = []
    while i < idx:
        if i % 10 == 0:
            rho_ls.append(np.log10(rho[i][10]))
            tmp_ls.append(np.log10(tmp[i][10]))
        i += 1
    print(rho_ls)
    print(tmp_ls)
    plt.plot(rho_ls, tmp_ls)
    plt.xlabel("log10rho cgs")
    plt.ylabel("log10T cgs")
    plt.legend()
    os.makedirs("results/" + config["plot_tag"], exist_ok=True)
    plt.savefig("results/" + config["plot_tag"] + "/core.png")


if __name__ == "__main__":
    main()

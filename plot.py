import matplotlib.pyplot as plt
import numpy as np
import json
import os
import seaborn as sns

from file_operator import read_json
from conditions import GRID


def main():
    sns.set_theme()
    config = read_json()
    r_h = np.load(
        "data/" + config["plot_tag"] + "/step_{}_r_h.npy".format(config["plot_step"]),
        allow_pickle=True,
    )
    r = np.load(
        "data/" + config["plot_tag"] + "/step_{}_r.npy".format(config["plot_step"]),
        allow_pickle=True,
    )

    rho = np.load(
        "data/" + config["plot_tag"] + "/step_{}_rho.npy".format(config["plot_step"]),
        allow_pickle=True,
    )
    t = np.load(
        "data/" + config["plot_tag"] + "/step_{}_t.npy".format(config["plot_step"]),
        allow_pickle=True,
    )
    idx = int(config["plot_step"])
    cur_rho = np.max(np.floor(np.log10(rho[0])))
    print(cur_rho)
    i = 10
    while i < idx:
        if np.max(np.log10(rho[i])) >= cur_rho + 1:
            print(r_h[i].shape)
            print(rho[i].shape)
            plt.plot(
                np.log10(r_h[i][: GRID - 1]),
                np.log10((rho[i][: GRID - 1])),
                label="{:.5f} * 10^13s".format(t[int(i)] / 10 ** 13),
            )
            cur_rho = np.max(np.floor(np.log10(rho[i])))
        i += 1
    plt.xlabel("log10r cgs")
    plt.ylabel("log10rho cgs")
    plt.legend()
    os.makedirs("results/" + config["plot_tag"], exist_ok=True)
    plt.savefig("results/" + config["plot_tag"] + "/out.png")


if __name__ == "__main__":
    main()

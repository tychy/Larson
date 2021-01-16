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

    fh = np.load(
        "data/" + config["plot_tag"] + "/step_fh.npy",
        allow_pickle=True,
    )

    figure = plt.figure()
    cur_rho = np.max(np.floor(np.log10(rho[0])))

    i = 10
    prev = i
    plt.plot(
        np.log10(r_h[4]),
        fh[4],
        label="{:.5f} * 10^13s".format(t[int(4)] / 10 ** 13),
    )

    while i < idx:
        if np.abs(np.max(np.log10(rho[i])) - cur_rho) >= 1 or i - prev >= 10000:
            if not np.all(np.abs(fh[prev] - fh[i]) < 0.000001):
                plt.plot(
                    np.log10(r_h[i]),
                    fh[i],
                    label="{:.5f} * 10^13s".format(t[int(i)] / 10 ** 13),
                )
                cur_rho = np.max(np.floor(np.log10(rho[i])))
            prev = i
        i += 1

    plt.ylim(-0.1, 1.2)

    plt.xlabel("log10r")
    plt.ylabel("f")
    plt.legend()
    os.makedirs("results/" + config["plot_tag"], exist_ok=True)
    plt.savefig("results/" + config["plot_tag"] + "/f_r.png")


if __name__ == "__main__":
    main()

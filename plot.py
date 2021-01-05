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
        "data/" + config["plot_tag"] + "/step_{}_r_h.npy".format(idx),
        allow_pickle=True,
    )
    r = np.load(
        "data/" + config["plot_tag"] + "/step_{}_r.npy".format(idx),
        allow_pickle=True,
    )
    v = np.load(
        "data/" + config["plot_tag"] + "/step_{}_v.npy".format(idx),
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

    figure = plt.figure()
    cur_rho = np.max(np.floor(np.log10(rho[0])))
    i = 10
    prev = i
    plt.plot(
        np.log10(r_h[4][: GRID - 1]),
        np.log10((rho[4][: GRID - 1])),
        label="{:.5f} * 10^13s".format(t[int(4)] / 10 ** 13),
    )

    while i < idx:
        if np.abs(np.max(np.log10(rho[i])) - cur_rho) >= 1 or i - prev >= 500:
            plt.plot(
                np.log10(r_h[i][: GRID - 1]),
                np.log10((rho[i][: GRID - 1])),
                label="{:.5f} * 10^13s".format(t[int(i)] / 10 ** 13),
            )
            cur_rho = np.max(np.floor(np.log10(rho[i])))
            prev = i
        i += 1

    plt.xlabel("log10r")
    plt.ylabel("log10rho")
    plt.legend()
    os.makedirs("results/" + config["plot_tag"], exist_ok=True)
    plt.savefig("results/" + config["plot_tag"] + "/rho_r.png")

    figure = plt.figure()
    cur_rho = np.max(np.floor(np.log10(rho[0])))
    i = 10
    prev = i
    plt.plot(
        np.log10(r_h[4][: GRID - 1]),
        v[4][: GRID - 1],
        label="{:.5f} * 10^13s".format(t[int(4)] / 10 ** 13),
    )

    while i < idx:
        if np.abs(np.max(np.log10(rho[i])) - cur_rho) >= 1 or i - prev >= 500:
            plt.plot(
                np.log10(r_h[i][: GRID - 1]),
                v[i][: GRID - 1],
                label="{:.5f} * 10^13s".format(t[int(i)] / 10 ** 13),
            )
            cur_rho = np.max(np.floor(np.log10(rho[i])))
            prev = i
        i += 1

    plt.xlabel("log10r")
    plt.ylabel("v")
    plt.legend()
    os.makedirs("results/" + config["plot_tag"], exist_ok=True)
    plt.savefig("results/" + config["plot_tag"] + "/v_r.png")

    figure = plt.figure()
    cur_tmp = np.max(np.log10(tmp[0]))
    i = 10
    prev = i
    plt.plot(
        np.log10(r_h[4][: GRID - 1]),
        np.log10((tmp[4][: GRID - 1])),
        label="{:.5f} * 10^13s".format(t[int(4)] / 10 ** 13),
    )

    while i < idx:
        if np.abs(np.max(np.log10(tmp[i])) - cur_tmp) >= 0.5 or i - prev >= 500:
            plt.plot(
                np.log10(r_h[i][: GRID - 1]),
                np.log10((tmp[i][: GRID - 1])),
                label="{:.5f} * 10^13s".format(t[int(i)] / 10 ** 13),
            )
            cur_tmp = np.max(np.log10(tmp[i]))
            print(cur_tmp)
            prev = i

        i += 1
    plt.xlabel("log10 r")
    plt.ylabel("log10 T")
    plt.legend()
    os.makedirs("results/" + config["plot_tag"], exist_ok=True)
    plt.savefig("results/" + config["plot_tag"] + "/t_r.png")


if __name__ == "__main__":
    main()

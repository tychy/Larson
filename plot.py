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
    v = np.load(
        "data/" + config["plot_tag"] + "/step_v.npy",
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

    figure = plt.figure()
    cur_rho = np.max(np.floor(np.log10(rho[0])))
    i = 10
    prev = i
    plt.plot(
        np.log10(r_h[4][: GRID - 1]),
        np.log10((rho[4][: GRID - 1])),
        label="{:.6f}".format(t[int(4)] / 10 ** 12),
    )
    while i < idx:
        if np.abs(np.max(np.log10(rho[i])) - cur_rho) >= 2 or i - prev >= max(
            10000, prev
        ):
            plt.plot(
                np.log10(r_h[i][: GRID - 1]),
                np.log10((rho[i][: GRID - 1])),
                label="{:.6f}".format(t[int(i)] / 10 ** 12),
            )
            cur_rho = np.max(np.floor(np.log10(rho[i])))
            if cur_rho > -4:
                break
            prev = i
        i += 1

    plt.xlabel(r"$\log_{10}r$")
    plt.ylabel(r"$\log_{10}\rho$")
    plt.legend()
    plt.tight_layout()
    os.makedirs("results/" + config["plot_tag"], exist_ok=True)
    plt.savefig("results/" + config["plot_tag"] + "/rho_r.png")

    figure = plt.figure()
    cur_rho = np.max(np.floor(np.log10(rho[0])))
    i = 10
    prev = i
    while i < idx:
        if np.abs(np.max(np.log10(rho[i])) - cur_rho) >= 2:
            plt.plot(
                np.log10(r_h[i]),
                np.abs(v[i][:GRID]),
                label="{:.6f}".format(t[int(i)] / 10 ** 12),
            )
            cur_rho = np.max(np.floor(np.log10(rho[i])))
            print(cur_rho)
            if cur_rho > -3.5:
                break
            prev = i
        i += 1

    plt.xlabel(r"$\log_{10}r$")
    plt.ylabel(r"$v$")
    plt.legend()
    plt.tight_layout()
    os.makedirs("results/" + config["plot_tag"], exist_ok=True)
    plt.savefig("results/" + config["plot_tag"] + "/v_r.png")

    figure = plt.figure()
    cur_tmp = np.max(np.log10(tmp[0]))
    i = 10
    prev = i
    plt.plot(
        np.log10(r_h[4][: GRID - 1]),
        np.log10((tmp[4][: GRID - 1])),
        label="{:.5f}".format(t[int(4)] / 10 ** 12),
    )

    while i < idx:
        if np.abs(np.max(np.log10(tmp[i])) - cur_tmp) >= 0.5 or i - prev >= max(
            2500, prev
        ):
            plt.plot(
                np.log10(r_h[i][: GRID - 1]),
                np.log10((tmp[i][: GRID - 1])),
                label="{:.5f} ".format(t[int(i)] / 10 ** 12),
            )
            cur_tmp = np.max(np.log10(tmp[i]))
            print(cur_tmp)
            prev = i

        i += 1
    plt.xlabel(r"$\log_{10}r$")
    plt.ylabel(r"$\log_{10}T$")

    plt.legend()
    plt.tight_layout()
    os.makedirs("results/" + config["plot_tag"], exist_ok=True)
    plt.savefig("results/" + config["plot_tag"] + "/t_r.png")


if __name__ == "__main__":
    main()

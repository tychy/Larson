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

    fht = np.load(
        "data/" + config["plot_tag"] + "/step_fht.npy",
        allow_pickle=True,
    )
    fion = np.load(
        "data/" + config["plot_tag"] + "/step_fion.npy",
        allow_pickle=True,
    )

    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    cur_fh = np.max(fh[0])
    i = 10
    prev = i

    while i < idx:
        if np.abs(np.max(fh[i]) - cur_fh) >= 0.2 or i - prev >= 10000:
            if not np.all(np.abs(fh[prev] - fh[i]) < 0.000001):
                label = "{:.5f} * 10^13s,core={:.5f}K".format(
                    t[i] / 10 ** 13, tmp[i][10]
                )
                ax1.plot(
                    np.log10(r_h[i]),
                    fh[i],
                    label=label,
                )
                ax2.plot(
                    np.log10(r_h[i]),
                    fht[i],
                    label=label,
                )
                ax3.plot(
                    np.log10(r_h[i]),
                    fion[i],
                    label=label,
                )
                ax4.plot(
                    np.log10(r_h[i]),
                    np.log10(tmp[i]),
                    label=label,
                )

                cur_fh = np.max(fh[i])
            prev = i
        i += 1
    ax1.plot(
        np.log10(r_h[i - 10]),
        fh[i - 10],
        label="{:.5f} * 10^13s,core={:.5f}K".format(
            t[i - 10] / 10 ** 13, tmp[i - 10][10]
        ),
    )
    ax2.plot(
        np.log10(r_h[i - 10]),
        fht[i - 10],
        label="{:.5f} * 10^13s,core={:.5f}K".format(
            t[i - 10] / 10 ** 13, tmp[i - 10][10]
        ),
    )
    ax3.plot(
        np.log10(r_h[i - 10]),
        fion[i - 10],
        label="{:.5f} * 10^13s,core={:.5f}K".format(
            t[i - 10] / 10 ** 13, tmp[i - 10][10]
        ),
    )

    ax4.plot(
        np.log10(r_h[i - 10]),
        np.log10(tmp[i - 10]),
        label="{:.5f} * 10^13s,core={:.5f}K".format(
            t[i - 10] / 10 ** 13, tmp[i - 10][10]
        ),
    )

    ax1.set_ylim(-0.1, 1.2)
    ax1.set_xlabel("log10r")
    ax1.set_ylabel("H")

    ax2.set_ylim(-0.1, 1.2)
    ax2.set_xlabel("log10r")
    ax2.set_ylabel("H2")

    ax3.set_ylim(-0.1, 1.2)
    ax3.set_xlabel("log10r")
    ax3.set_ylabel("H+")

    ax4.set_xlabel("log10r")
    ax4.set_ylabel("TMP")
    plt.legend()
    fig.tight_layout()
    os.makedirs("results/" + config["plot_tag"], exist_ok=True)
    plt.savefig("results/" + config["plot_tag"] + "/f_r.png")


if __name__ == "__main__":
    main()

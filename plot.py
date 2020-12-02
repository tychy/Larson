import matplotlib.pyplot as plt
import numpy as np
import json
import os

from file_operator import read_json


def main():
    json_open = read_json()
    r_h = np.load(
        "data/" + json_open["plot_tag"] + "/step_{}_r_h.npy".format(json_open["step"])
    )
    rho = np.load(
        "data/" + json_open["plot_tag"] + "/step_{}_rho.npy".format(json_open["step"])
    )
    t = np.load(
        "data/" + json_open["plot_tag"] + "/step_{}_t.npy".format(json_open["step"])
    )
    idx = int(json_open["step"])
    for i in np.linspace(0, idx, 5):
        plt.plot(
            np.log10(r_h[int(i)]),
            np.log10((rho[int(i)])),
            label="{} * 10^13s".format(t[int(i)] / 10 ** 13),
        )
    plt.xlabel("log10r cgs")
    plt.ylabel("log10rho cgs")
    plt.legend()
    os.makedirs("results/" + json_open["plot_tag"], exist_ok=True)
    plt.savefig("results/" + json_open["plot_tag"] + "/out.png")


if __name__ == "__main__":
    main()

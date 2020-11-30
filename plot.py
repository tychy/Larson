import matplotlib.pyplot as plt
import numpy as np


def main():
    r_h = np.load("data/r_h.npy")
    rho = np.load("data/rho.npy")
    t = np.load("data/t.npy")

    if True:
        f = lambda x: np.log(x)
    else:
        f = lambda x: x
    for i in [100, 2000, 3000, 5000, 7000, 10000, 11000]:
        plt.plot(f(r_h[i]), f(rho[i]), label="{}".format(t[i]))
        plt.legend()
        plt.savefig("results/noQ{:.8}.png".format(t[i]))


if __name__ == "__main__":
    main()

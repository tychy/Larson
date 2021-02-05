import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from calc_operator import calc_fh, calc_gamma
from conditions import AVG

sns.set_theme()

na = 6 * 10 ** 23

n = 10 ** 15
rho_res = 2 / na * n
coef = 0.0001
tmp_ls = np.linspace(500, 10000, 100)
fh_ls = []
fht_ls = []
fion_ls = []
gamma_ls = []
for item in tmp_ls:
    fh, fht, fion = calc_fh(item, rho_res, AVG)
    fh_ls.append(fh)
    fht_ls.append(fht)
    fion_ls.append(fion)

plt.plot()

plt.plot(tmp_ls, fh_ls, label=r"$f_H$")
plt.plot(tmp_ls, fht_ls, label=r"$f_{H^2}$")
plt.plot(tmp_ls, fion_ls, label=r"$f_{H+}$")
plt.legend()
plt.savefig("fh.png")
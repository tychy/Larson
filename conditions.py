import numpy as np

M_SUN_LOG = 33
# R_SUN = 7 * np.power(10, 10)
TMP_init = 10  # or 5 or 30
AU = 1.4598 * np.power(10, 13)  # cm
GRID = 200
T_ORDER = np.power(10, 13)
DT = 0.001  # * 10^13 s
T_END = 100
M_cc = 1  # cloud core
G = 6.67259 * np.power(10, -8 - 3 * np.log10(AU) + M_SUN_LOG + 2 * np.log10(T_ORDER))
R_cc = 10000  # AU 1000 to 30000
R_LOG = np.log10(8.3) + 7 - M_SUN_LOG - 2 * np.log10(AU) + 2 * np.log(T_ORDER)

AVG = 2.4
KQ = 2
CFL_CONST = 0.3

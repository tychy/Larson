import numpy as np

M_SUN = 2 * np.power(10, 33)
R_SUN = 7 * np.power(10, 10)
TMP_init = 10  # or 5 or 30
AU = 1.4598 * np.power(10, 13)  # cm
GRID = 200
DT = 0.001  # * 10^13 s
T_ORDER = np.power(10, 13)
T_END = 2 * T_ORDER

M_cc = 1 * M_SUN  # cloud core
G = 6.67259 / np.power(10, 8)
Rho_init = 10 ** -17  # gcm-3 from 10-17to 10-20
R_cc = 10000 * AU  # AU 1000 to 30000

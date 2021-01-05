from file_operator import read_json

config = read_json()
TMP_init = config["TMP"]
# or 5 or 30
AU = 1.4598 * 10 ** 13  # cm
GRID = config["GRID"]
T_ORDER = 10 ** 13
T_END = config["T_END"] * T_ORDER
M_cc = config["M"] * 1.99 * 10 ** 33  # cloud core
G = 6.67259 * 10 ** (-8)
R_cc = config["radius"] * AU  # 1.63 * 10 ** 17  #   #  AU 1000 to 30000
R = 8 * 10 ** 7

AVG = 2.4
KQ = config["KQ"]
CFL_CONST = config["CFL"]
kb = 1.38 * 10 ** (-16)
xi_h = 13.4
m_e = 1
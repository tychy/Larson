from file_operator import read_json

config = read_json()
# or 5 or 30
AU = 1.4598 * 10 ** 13  # cm
GRID = config["GRID"]
T_ORDER = 10 ** 13
T_END = config["T_END"] * T_ORDER
M_cc = config["M"] * 1.99 * 10 ** 33  # cloud core
G = 6.67259 * 10 ** (-8)

if config["isLarson"]:
    TMP_INIT = 10
    R_CC = 1.63 * 10 ** 17
else:
    TMP_INIT = config["TMP"]
    R_CC = config["radius"] * AU  #  AU 1000 to 30000

AVG = 2.4
R = 8 * 10 ** 7 / AVG

KQ = config["KQ"]
CFL_CONST = config["CFL"]
planck = 6.63 * 10 ** (-27)
kb = 1.38 * 10 ** (-16)
xi_h = 2.18 * 10 ** (-11)
xi_d = 2.18 * 10 ** (-11) * 4.48 / 13.6

m_e = 9.11 * 10 ** (-28)
m_p = 1.67 * 10 ** (-24)
Kapper = 0.15  # cm^2/g
SB = 5.67 * 10 ** (-8)  # stefanboltzmann

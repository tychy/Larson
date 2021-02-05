from file_operator import read_json

config = read_json()
# or 5 or 30
AU = 1.4598 * 10 ** 13  # cm
GRID = config["GRID"]
T_ORDER = 10 ** 13
MAX_STEP = config["MAX_STEP"]
M_cc = config["M"] * 1.99 * 10 ** 33  # cloud core
G = 6.67259 * 10 ** (-8)

if config["isLarson"]:
    R_CC = 1.63 * 10 ** 17
else:
    R_CC = config["radius"] * AU  #  AU 1000 to 30000

TMP_INIT = config["TMP"]
DISPLAY = config["Display"]
AVG = 2.4
R = 8 * 10 ** 7

KQ = config["KQ"]
CFL_CONST = config["CFL"]
planck = 6.63 * 10 ** (-27)
kb = 1.38 * 10 ** (-16)
eV = 1.6 * 10 ** (-12)
xi_h = 13.6 * eV
xi_d = 4.48 * eV

m_e = 9.11 * 10 ** (-28)
m_p = 1.67 * 10 ** (-24)
Kapper = 0.15  # cm^2/g
SB = 5.67 * 10 ** (-5)  # stefanboltzmann
NA = 6 * 10 ** 23

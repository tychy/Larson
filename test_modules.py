from conditions import R_cc, GRID, T_ORDER, AU, G, M_SUN_LOG
from utils import CFL, vstack_n, get_cs, r_init, m_init
import numpy as np
import unittest


class TestModules(unittest.TestCase):
    def test_vstack_n(self):
        r = vstack_n(np.arange(0, R_cc + GRID, R_cc / GRID), 3)
        self.assertTrue(r.shape[0] == 3)

    def test_CFL(self):
        r = vstack_n(np.arange(0, R_cc + GRID, R_cc / GRID), 3)
        dt = CFL(r[r.shape[0] - 1])
        print("10K", CFL(r[r.shape[0] - 1], 10))
        print("100K", CFL(r[r.shape[0] - 1], 100))
        self.assertTrue(dt > 0)

    def test_get_cs(self):
        cs = get_cs(10)
        self.assertTrue(cs == 30000 * T_ORDER / AU)
        cs = get_cs(1000)
        self.assertTrue(cs == 300000 * T_ORDER / AU)

    def test_r_init(self):
        r = r_init()
        self.assertTrue(r.shape[0] == GRID + 1)

    def test_m_init(self):
        m = m_init()
        self.assertTrue(m.shape[0] == GRID + 1)

    def test_G(self):
        print(G)

    def test_rho(self):
        print(M_SUN_LOG - 3 * np.log10(AU) - 3 * np.log10(R_cc))


if __name__ == "__main__":
    unittest.main()
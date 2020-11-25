from conditions import R_cc, GRID, T_ORDER
from utils import CFL, vstack_n, get_cs
import numpy as np
import unittest


class TestModules(unittest.TestCase):
    def test_vstack_n(self):
        r = vstack_n(np.arange(0, R_cc + GRID, R_cc / GRID), 3)
        self.assertTrue(r.shape[0] == 3)

    def test_CFL(self):
        r = vstack_n(np.arange(0, R_cc + GRID, R_cc / GRID), 3)
        dt = CFL(r[r.shape[0] - 1])
        self.assertTrue(dt == 0.01)

    def test_get_cs(self):
        cs = get_cs(10)
        self.assertTrue(cs == 30000)
        cs = get_cs(1000)
        self.assertTrue(cs == 300000)


if __name__ == "__main__":
    unittest.main()
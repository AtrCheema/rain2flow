
import os
import site

package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
site.addsitedir(package_path)

import unittest

import numpy as np

from rain2flow import hbv

import numpy as np


rng = np.random.default_rng(313)

params = dict(
    BETA = 2,
    #CET = (0, 6),
    FC = 500,
    K0 = 0.5,
    K1 = 0.1,
    K2 = 0.01,
    LP = 0.05,
    MAXBAS = 3,
    PERC = 5,
    UZL = 50,
    PCORR = 1.5,
    TT = 5,
    CFMAX = 1,
    SFCF = 1.5,
    CFR=0.05,
    CWH = 0.1,
    )

class TestBasic(unittest.TestCase):
    def test_basic(self):
        pcp = rng.random(10_000).reshape(-1, 1)
        evap = rng.random(10_000).reshape(-1, 1)
        temp = rng.random(10_000).reshape(-1, 1)

        out = hbv(pcp, evap, temp, {k: np.array(v) for k, v in params.items()})

        assert out.shape == pcp.shape

        return



class TestVattholma(unittest.TestCase):
    def test_vattholma(self):
        pass

class TestNorrsjon(unittest.TestCase):
    def test_norrsjon(self):
        pass



if __name__ == '__main__':
    unittest.main()
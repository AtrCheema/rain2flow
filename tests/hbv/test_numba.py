
import os
import site

package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
site.addsitedir(package_path)

import time
import numpy as np
import pandas as pd
import unittest

from rain2flow._hbv._main import hbv
from rain2flow._hbv._numba import hbv_nb

rng = np.random.default_rng(42)

P = rng.random(1000) + 1
ETpot = rng.random(1000)
Temp = rng.random(1000) + 32


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


def compare_nb_with_native():
    # compare numba version with non-numba version
    start = time.time()
    qsim = hbv(P, Temp, ETpot, parameters=params, routing=True)
    print('HBV time:', round(time.time() - start, 3))

    start = time.time()
    qsim_n = hbv_nb(P, Temp, ETpot, parameters=params, routing=True)
    print('HBV_N time:', round(time.time() - start, 3))

    np.testing.assert_array_almost_equal(
        qsim.reshape(-1,), qsim_n, 4)

    initial = [1000.0,
    1.73457470574975,
    0.04971679124510588,
    0.3853698670864105,
    1.7197040617465973,
    1.7350587397813797,
    1.7521704137325287,
    1.7836166620254517,
    1734.5747057497501,
    0.049691926631775185]

    # todo
    # np.testing.assert_array_almost_equal(
    #     pd.Series(qsim).describe().tolist() + [qsim.sum(), qsim.std()],
    #     initial, 4)

    np.testing.assert_array_almost_equal(
        pd.Series(qsim_n).describe().tolist() + [qsim_n.sum(), qsim_n.std()],
        initial, 4)
    
    return


def check_speed():

    P = rng.random(10_000)
    ETpot = rng.random(10_000)
    Temp = rng.random(10_000)


    start = time.time()
    for _ in range(100):
        _ = hbv(P, Temp, ETpot, parameters=params)
    print('HBV time:', time.time() - start)

    start = time.time()
    for _ in range(100):
        _ = hbv_nb(P, Temp, ETpot, parameters=params)
    print('HBV_N time:', time.time() - start)

    return


compare_nb_with_native()

# check_speed()
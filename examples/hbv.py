"""
====================
HBV Model Example
====================
"""

import os
import site

package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
print(package_path)
site.addsitedir(package_path)

import numpy as np
import pandas as pd
from SeqMetrics import RegressionMetrics

from rain2flow import hbv

# %%
data = pd.read_csv('data.csv', 
                   comment='#', parse_dates=['date'], index_col='date')

# %%
parameters = {
    "BETA": 2.254272145204868,
    "CFMAX": 2.634558463081481,
    "CFR": 0.05316989970760903,
    "CWH": 0.06320640559590592,
    "FC": 938.4151865045613,
    "K0": 0.5533421892716448,
    "K1": 0.25566454560982704,
    "K2": 0.18448187501298327,
    "LP": 0.9976204042760504,
    "MAXBAS": 4.399887139522522,
    "PCORR": 1.991169069794055,
    "PERC": 5.256982141549575,
    "SFCF": 1.9656441421578075,
    "TT": 1.0041306252448585,
    "UZL": 36.983252412038695
}

sim = hbv(
    data['pcp'].values,
    data['temp'].values,
    data['pet'].values,
    parameters=parameters)

# %%

metrics = RegressionMetrics(data['Qobs'].values, sim.flatten())

for metric in ['kge', 'nse', 'pbias']:
    print(f"{metric} : {round(getattr(metrics, metric)(), 4)}")
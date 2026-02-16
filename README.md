[![Documentation Status](https://readthedocs.org/projects/rain2flow/badge/?version=latest)](https://rain2flow.readthedocs.io/en/latest/?badge=latest)

# rain2flow
A collection of conceptual lumped hydrological models for rainfall-runoff modeling in numpy

Currently following models are implemented.
- hbv

For usage help check examples.


## How to Install

You can use github link for install rain2flow.

    python -m pip install git+https://github.com/AtrCheema/rain2flow.git

go to folder where repository is downloaded

    python setup.py install


## How to Use
Check [this](https://rain2flow.readthedocs.io/en/latest/auto_examples/hbv.html) notebook for online run example

```python

import pandas as pd

from rain2flow import hbv

# read the file containing pcp, temp and pet columns
data = pd.read_csv('data.csv', comment='#', parse_dates=['date'], index_col='date')

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

```
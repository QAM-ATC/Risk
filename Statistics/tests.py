" PUT ALL THE STATS TESTS HERE, no need for class"
from statsmodels.tsa.stattools import adfuller
import pandas as pd
from typing import Union

def stationaryTest(series):
    """Conducts ADF summary test on series

    Parameters
    ----------
    series : pd series

    Returns
    -------
    ADF summary
    """

    X = series.values
    result = adfuller(X)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


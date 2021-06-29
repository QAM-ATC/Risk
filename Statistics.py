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

def annualizeVol(r: Union[pd.DataFrame,pd.Series], periodsPerYear: int) -> pd.Series:
    """Annualizes the volatility of a given set of returns

    Parameters
    ----------
    r : Union[pd.DataFrame,pd.Series]
        DataFrame of historical prices for each ticker, with column name as name of ticker and index as timestamps
    periodsPerYear : int
        Number of periods per year for the given data

    Returns
    -------
    pd.Series
        Annualized volatility for each ticker, with the ticker as index
    """
    return r.std()*(periodsPerYear**0.5)
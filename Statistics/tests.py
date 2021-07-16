"""Statistical tests"""
from statsmodels.tsa.stattools import adfuller,grangercausalitytests,acf,pacf
from typing import Union
import pandas as pd
import numpy as np

def stationaryTestADF(series: pd.Series,verbose: bool = True, stationaritySignifiance: float = 0.05) -> tuple:
    """Runs the Augmented Dickey-Fuller test on the series, with the null of non-stationarity
    Note: series is only for one security

    Parameters
    ----------
    series : pd.Series
        time series data
    verbose : bool, optional
        True if the ADF statistic, p-value and critical values are to be printed, by default True
    stationaritySignificance : float, optional
        The level of signifiance at which stationarity is checked, by default 0.05 (5%)

    Returns
    -------
    tuple
        Returns the relevant values in the format (p-value, ADF statistic, stationaryBool)
    """
    X = series.values
    result = adfuller(X)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    if result[1]<=stationaritySignifiance:
        stationaryBool = True # null is rejected and series is stationary
    else:
        stationaryBool = False # null cannot be rejected and series isn't stationary

    return result[1],result[0],stationaryBool

def grangerCausalityTest(series: pd.DataFrame, maxLags: Union[int,list], addConst: bool = True, verbose: bool = True) -> dict:
    """Performs the Granger Causality Test for the given series
    Note: pd.DataFrame should contain two columns
    Note: series data must be stationary, difference before passing if needed

    Parameters
    ----------
    series : pd.DataFrame
        data for testing whether the time series in the second column Granger causes the time series in the first column (missing values not supported)
    maxLags : int
        If an integer, computes the test for all lags up to maxlag. If a list, computes the tests only for the lags in maxlag
    addConst : bool
        Add a constant to the model, by default True
    verbose : bool, optional
        True if debugging information is to be printed, by default True

    Returns
    -------
    dict
        All test results, dictionary keys are the number of lags. For each lag the values are a tuple, 
            First element: a dictionary with test statistic, p-values, degrees of freedom, keys: 'lrtest', 'params_ftest', 'ssr_chi2test', 'ssr_ftest'
            Second element: the OLS estimation results for the restricted model, the unrestricted model and the restriction (contrast) matrix for the parameter f_test
        For example: to get p-value for ssr_ftest for ith lag: res[i][0]['ssr_ftest'][1]
        
    """

    res = grangercausalitytests(series, maxlag=maxLags,addconst=addConst,verbose=verbose)
    return res

def ACF(series: pd.Series, adjusted: bool=False, nLags: int=None, qStat: bool=False, fft: bool=None, alpha: float=None, missing: str='none') -> Union[np.ndarray,tuple]:
    """Calculates the ACF, and optionally the confidence intervals, Ljung-Box Q-Statistic, and its associated p-values for a given series
    Check documentation here: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html#statsmodels.tsa.stattools.acf
    Note: series is only for one security

    Parameters
    ----------
    series : pd.Series
        time series data
    adjusted : bool, optional
        If True, then denominators for autocovariance are n-k, otherwise n, by default False
    nLags : int, optional
        Number of lags to return autocorrelation for, by default None
    qStat : bool, optional
        If True, returns the Ljung-Box q statistic for each autocorrelation coefficient, by default False
    fft : bool, optional
        If True, computes the ACF via FFT, by default None
    alpha : float, optional
        If a number is given, the confidence intervals for the given level are returned, by default None
    missing : str, optional
        A string in [“none”, “raise”, “conservative”, “drop”] specifying how the NaNs are to be treated, by default 'none'

    Returns
    -------
    Union[np.ndarray,tuple]
        Returns the autocorrelation function of type np.ndarray, and 
            Confidence intervals for the ACF, if alpha is not None, of type np.ndarray
            The Ljung-Box Q-Statistic, if qStat is True, of type np.ndarray
            The p-values associated with the Q-statistics, if qStat is True, of type np.ndarray
    """
    return acf(x=series, adjusted=adjusted, nlags=nLags, qstat=qStat, fft=fft, alpha=alpha, missing=missing)

def PACF(series: pd.Series, nLags: int=None, method: str='ywadjusted', alpha: float=None) -> Union[np.ndarray,tuple]:
    """Calculates the PACF, and optionally the confidence intervals, for the returns of a given series
    Documentation: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.pacf.html#statsmodels.tsa.stattools.pacf
    Note: series is only for one security

    Parameters
    ----------
    series : pd.Series
        time series data
    nLags : int, optional
        The largest lag for which the PACF is returned, by default None
    method : str, optional
        Specifies which method for the calculations to use, full list in documentation, by default 'ywadjusted'
    alpha : float, optional
        If a number is given, the confidence intervals for the given level are returned, by default None

    Returns
    -------
    Union[np.ndarray,tuple]
        Partial autocorrelations, nlags elements, including lag zero, of type np.ndarray and
            Confidence intervals for the PACF if alpha is not None, of type np.ndarray
    """
    return pacf(x=series,nlags=nLags,method=method,alpha=alpha)

def hurstExponent(series: pd.Series, maxLags: int) -> float:
    """Returns the Hurst Exponent value for a given time series
    Source: https://towardsdatascience.com/introduction-to-the-hurst-exponent-with-code-in-python-4da0414ca52e

    Parameters
    ----------
    series : pd.Series
        time series
    maxLags : int
        maximum number of lags

    Returns
    -------
    float
        Hurst Exponent
    """
    lags = range(2, maxLags)

    # variances of the lagged differences
    tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]

    # calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(lags), np.log(tau), 1)

    return reg[0]
"Put summary function here that prints or returns a dataframe"
import pandas as pd
from typing import Union
import empyrical
import numpy as np
from scipy.stats import skew, kurtosis, skewtest, kurtosistest

__all__ = [
    'calculate_skewness',
    'calculate_kurtosis',
    'is_stable',
    'maximum_drawdown',
    'cumulative_returns'
]

def calculate_skewness(price: Union[pd.DataFrame,pd.Series], test: bool = False, **kwargs) -> Union[float,pd.Series]:
    """Calculates the skewness for a given set of prices

    Parameters
    ----------
    price : Union[pd.DataFrame,pd.Series]
        historical prices of a given security

    Returns
    -------
    Union[float,pd.Series]
        skewness for a given set of prices
    """
    if test:
        result = skewtest(price, **kwargs)
        return result

    return skew(price)


def calculate_kurtosis(price: Union[pd.Series,pd.DataFrame], test: bool = False, **kwargs) -> Union[float,pd.Series]:
    """Calculates the kurtosis for a given set of prices

    Parameters
    ----------
    price : Union[pd.DataFrame,pd.Series]
        historical prices of a given security

    Returns
    -------
    Union[float,pd.Series]
        kurtosis for a given set of prices
    """

    if test:

        result = kurtosistest(price, **kwargs)
        return result

    return kurtosis(price)

def is_stable(price: pd.Series) -> float:
    """Calculates stability for a given set of prices

    Parameters
    ----------
    price : pd.Series
       historical prices of a given security

    Returns
    -------
    float
       stability for a given set of prices
    """
    if isinstance(price, pd.DataFrame):
        return price.apply(is_stable)

    returns = price.pct_change().dropna()
    stability = empyrical.stats.stability_of_timeseries(returns)
    return stability

def maximum_drawdown(price: pd.Series) -> float:
    """Calculates maximum drawdown for a given set of prices

    Parameters
    ----------
    price : pd.Series
        historical prices of a given security

    Returns
    -------
    float
        maximum drawdown for a given set of prices
    """

    maximum_drawdown.drawdowns = (price / price.cummax()) - 1
    maximumDrawdown = (maximum_drawdown.drawdowns).min()

    return maximumDrawdown

def cumulative_returns(price: Union[pd.DataFrame, pd.Series]) -> float:
    """Calculates cumulative returns for a given set of prices

    Parameters
    ----------
    price : Union[pd.DataFrame, pd.Series]
        historical prices of a given security

    Returns
    -------
    float
        cumulative returns for a given set of prices
    """
    returns = price.pct_change().dropna()
    cumReturns = empyrical.stats.cum_returns(returns)

    return cumReturns

def alpha(price: pd.Series, marketReturn: float, riskFreeRate: float = 0.0, periodsPerYear: Union[float, int] = 252) -> float:
    """Calculates annualised alpha for a given set of prices, risk free rate and benchmark return (market return in CAPM)

    Parameters
    ----------
    price : pd.Series
        historical prices of a given security
    riskFreeRate : float
        given constant risk free rate throughout the period
    marketReturn : float
        daily noncumulative benchmark return throughout the period
    periodsPerYear : Union[float, int]
        periodicity of the returns data for purposes of annualising

    Returns
    -------
    float
        annualised alpha for a given set of prices, risk free rate and benchmark return (market return in CAPM)
    """

    raise NotImplementedError("Will do it later :P")
    if isinstance(price, pd.DataFrame):
        return price.apply(alpha, args = (marketReturn , riskFreeRate, periodsPerYear))

    returns = price.pct_change().dropna()
    a = empyrical.stats.alpha(returns, factor_returns = marketReturn, risk_free = riskFreeRate, annualization = periodsPerYear)

    return a

def beta(price: pd.Series, riskFreeRate: float, marketReturn: float, periodsPerYear: Union[float, int]) -> float:
    """Calculates annualised beta for a given set of prices, risk free rate and benchmark return (market return in CAPM)

    Parameters
    ----------
    price : pd.Series
        historical prices of a given security
    riskFreeRate : float
        given constant risk free rate throughout the period
    marketReturn : float
        daily noncumulative benchmark return throughout the period
    periodsPerYear : Union[float, int]
        periodicity of the returns data for purposes of annualising

    Returns
    -------
    float
        annualised beta for a given set of prices, risk free rate and benchmark return (market return in CAPM)
    """
    #TO DO: take dataframe as price input
    raise NotImplementedError("Will do it later :P")
    r = price.pct_change().dropna()
    b = empyrical.stats.beta(r, factor_returns = marketReturn, risk_free = riskFreeRate, annualisation = periodsPerYear)
    return b



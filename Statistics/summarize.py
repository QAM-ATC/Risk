"Put summary function here that prints or returns a dataframe"
import pandas as pd
from typing import Union
import empyrical

def skewness(price: Union[pd.DataFrame,pd.Series]) -> Union[float,pd.Series]:
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
    r = price.diff().dropna()
    deviation = r - r.mean()
    sigma = r.std()
    num = (deviation**3).mean()
    return num/(sigma**3)


def kurtosis(price: Union[pd.Series,pd.DataFrame]) -> Union[float,pd.Series]:
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
    r = price.diff().dropna()
    deviation = r - r.mean()
    sigma = r.std()
    num = (deviation**4).mean()
    return num/(sigma**4)

def stability(price: pd.Series) -> float:
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
    #TO DO: take dataframe as price input
    r = price.diff().dropna()
    stability = empyrical.stats.stability_of_timeseries(r)
    return stability

def maxDrawdown(price: pd.Series) -> float:
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
    #TO DO: take dataframe as price input
    r = price.diff().dropna()
    mdd = empyrical.stats.max_drawdown(r)
    return mdd

def cumulativeReturns(price: Union[pd.DataFrame, pd.Series]) -> float:
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
    r = price.diff().dropna()
    cumReturns = empyrical.stats.cum_returns(r)
    return cumReturns

def alpha(price: pd.Series, riskFreeRate: float, marketReturn: float, periodsPerYear: Union[float, int]) -> float:
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
    #TO DO: take dataframe as price input
    r = price.diff().dropna()
    a = empyrical.stats.alpha(r, factor_returns = marketReturn, risk_free = riskFreeRate, annualisation = periodsPerYear)
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
    r = price.diff().dropna()
    b = empyrical.stats.beta(r, factor_returns = marketReturn, risk_free = riskFreeRate, annualisation = periodsPerYear)
    return b



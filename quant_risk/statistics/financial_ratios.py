"Put all financial ratios here, no need for class I think"

from quant_risk.statistics.annualize import annualised_returns, annualised_volatility
import empyrical
import pandas as pd
from typing import Union
from quant_risk.statistics.statistics import maximum_drawdown

__all__ = [
    'sharpe_ratio',
    'calmar_ratio',
    'omega_ratio',
    'sortino_ratio',
    'tail_ratio'
]


def sharpe_ratio(price: Union[pd.DataFrame, pd.Series], riskFreeRate: float = 0.0, periodsPerYear: Union[float, int] = 252)-> float:
    """ Calculates annualised sharpe ratio for given set of prices and risk free rate

    Parameters
    ----------
    price : Union[pd.DataFrame, pd.Series]
        historical prices of a given security
    riskFreeRate : float
        given constant risk free rate throughout the period
    periodsPerYear : Union[float, int]
        periodicity of the returns data for purposes of annualising

    Returns
    -------
    float
        annualised sharpe ratio
    """
    returns = price.pct_change().dropna()
    rfPerPeriod = (1 + riskFreeRate) ** (1 / periodsPerYear) - 1
    excessReturn = returns - rfPerPeriod

    annualiseExcessReturn = annualised_returns(excessReturn, periodsPerYear)
    annualiseVol = annualised_volatility(returns, periodsPerYear)

    return annualiseExcessReturn / annualiseVol

def calmar_ratio(price: Union[pd.DataFrame, pd.Series], periodsPerYear: Union[float, int] = 252, riskFreeRate: float = 0.0)-> float:
    """Calculates annualised calmar ratio for given set of prices and risk free rate

    Parameters
    ----------
    price : Union[pd.DataFrame, pd.Series]
        historical prices of a given security
    periodsPerYear : Union[float, int]
        periodicity of the returns data for purposes of annualising

    Returns
    -------
    float
        annualised calmar ratio
    """

    returns = price.pct_change().dropna()
    rfPerPeriod = (1 + riskFreeRate) ** (1 / periodsPerYear) - 1
    excessReturn = returns - rfPerPeriod

    annualiseExcessReturn = annualised_returns(excessReturn, periodsPerYear)
    calmar = annualiseExcessReturn / maximum_drawdown(price)
    # calmar = empyrical.stats.calmar_ratio(returns=returns, annualization=periodsPerYear)

    return calmar

def omega_ratio(price: Union[pd.DataFrame, pd.Series], riskFreeRate: float = 0.0, periodsPerYear: Union[float, int] = 252)-> float:
    """Calculates annualised omega ratio for given set of prices and risk free rate

    Parameters
    ----------
    price : Union[pd.DataFrame, pd.Series]
       historical prices of a given security
    riskFreeRate : float
        given constant risk free rate throughout the period
    periodsPerYear : Union[float, int]
         periodicity of the returns data for purposes of annualising

    Returns
    -------
    float
        annualised omega ratio
    """
    if isinstance(price, pd.DataFrame):

        return price.apply(omega_ratio, axis=0)

    returns = price.pct_change().dropna()
    omega = empyrical.stats.omega_ratio(returns, risk_free = riskFreeRate, annualization = periodsPerYear)

    return omega

def sortino_ratio(price: Union[pd.DataFrame, pd.Series], periodsPerYear: Union[float, int] = 252, reqReturn: float = 0) -> float:
    """Calculates annualised sortino ratio for given set of prices and risk free rate

    Parameters
    ----------
    price : Union[pd.DataFrame, pd.Series]
       historical prices of a given security
    periodsPerYear : Union[float, int]
        periodicity of the returns data for purposes of annualising
    reqReturn : float, optional
        the minimum acceptable return by investors, by default 0

    Returns
    -------
    float
        annualised sortino ratio
    """

    if isinstance(price, pd.DataFrame):

        return price.apply(sortino_ratio, axis=0)

    returns = price.pct_change().dropna()
    sortino = empyrical.stats.sortino_ratio(returns, annualization = periodsPerYear, required_return = reqReturn)

    return sortino

def tail_ratio(price: Union[pd.DataFrame, pd.Series]) -> float:
    """Calculates annualised tail ratio for given set of prices and risk free rate

    Parameters
    ----------
    price : Union[pd.DataFrame, pd.Series]
       historical prices of a given security

    Returns
    -------
    float
        annualised tail ratio
    """
    if isinstance(price, pd.DataFrame):

        return price.apply(tail_ratio, axis=0)

    returns = price.pct_change().dropna()
    tail = empyrical.stats.tail_ratio(returns)

    return tail



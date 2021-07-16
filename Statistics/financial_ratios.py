"Put all financial ratios here, no need for class I think"

from annualize import annualiseReturns, annualiseVolatility
import empyrical
import pandas as pd
from typing import Union


def sharpeRatio(price: Union[pd.DataFrame, pd.Series], riskFreeRate: float, periodsPerYear: Union[float, int])-> float:
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
    r = price.diff().dropna()
    rfPerPeriod = (1+riskFreeRate)**(1/periodsPerYear)-1
    excessReturn = r - rfPerPeriod
    annualiseExcessReturn = annualiseReturns(excessReturn, periodsPerYear)
    annualiseVol = annualiseVolatility(r,periodsPerYear)
    return annualiseExcessReturn/annualiseVol

def calmarRatio(price: Union[pd.DataFrame, pd.Series], periodsPerYear: Union[float, int])-> float:
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
    r = price.diff().dropna()
    calmar = empyrical.stats.calmar_ratio(r, annualisation = periodsPerYear)
    return calmar

def omegaRatio(price: Union[pd.DataFrame, pd.Series], riskFreeRate: float, periodsPerYear: Union[float, int])-> float:
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
    r = price.diff().dropna()
    omega = empyrical.stats.omega_ratio(r, risk_free = riskFreeRate, annualisation = periodsPerYear)
    return omega

def sortinoRatio(price: Union[pd.DataFrame, pd.Series], periodsPerYear: Union[float, int], reqReturn: float = 0) -> float:
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
    r = price.diff().dropna()
    sortino = empyrical.stats.sortino_ratio(r, annualisation = periodsPerYear, required_return = reqReturn)
    return sortino

def tailRatio(price: Union[pd.DataFrame, pd.Series]) -> float:
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
    r = price.diff().dropna()
    tail = empyrical.stats.tail_ratio(r)
    return tail



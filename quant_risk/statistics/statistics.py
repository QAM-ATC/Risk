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
    'cumulative_returns',
    'elton_gruber_covariance',
    'covariance_shrinkage',
    'risk_contribution',
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

def elton_gruber_covariance(price: pd.DataFrame, **kwargs):
    """This function estimates the covariance matrix by assuming an implicit structure as defined by the
    Elton-Gruber Constant Correlation model.

    Parameters
    ----------
    price : pd.DataFrame
        Historical prices of a given security

    Returns
    -------
    pd.DataFrame
        Returns a covariance matrix
    """

    returns = price.pct_change().dropna()
    rhos = returns.corr()

    n = rhos.shape[0]
    rhoBar = (rhos.values.sum() - n) / (n(n - 1))
    constantCorrelation = np.full_like(rhos, rhoBar)
    np.fill_diagonal(constantCorrelation, 1.)
    standardDev = returns.std()

    result = pd.DataFrame(constantCorrelation * np.outer(standardDev, standardDev), index=returns.columns, columns=returns.columns)

    return result

def covariance_shrinkage(price: pd.DataFrame, delta: float = 0.5, **kwargs):
    """This function computes the covariance matrix using the Ledoit-Wolf covariance shrinkage method
    taking a linear combination of the Constant Correlation matrix, acting as our prior and the Sample covariance matrix. The posterior covariance matrix is then computed.

    Parameters
    ----------
    price : pd.DataFrame
        Historical prices of a given security
    delta : float, optional
        Constant by which to weigh the priori matrix, by default 0.5

    Returns
    -------
    pd.DataFrame
        Returns a covariance matrix
    """

    returns = price.pct_change().dropna()
    sampleCovariance = returns.cov()
    priorCovariance = elton_gruber_covariance(price, **kwargs)

    result = delta * priorCovariance + (1 - delta) * sampleCovariance

    return result

def risk_contribution(portfolioWeights: Union[np.array, pd.DataFrame], covarianceMatrix: pd.DataFrame):
    """This function computes the contributions to the risk/variance of the constituents of a portfolio, given
    a set of portfolio weights and a covariance matrix

    Parameters
    ----------
    portfolioWeights : Union[np.array, pd.DataFrame]
        weights of our assets in our portfolio
    covarianceMatrix : pd.DataFrame
        the covariance matrix of our assets computed by any method

    Returns
    -------
    pd.DataFrame
        Returns the risk contribution of each asset
    """

    portfolioVariance = (portfolioWeights.T @ covarianceMatrix @ portfolioWeights)**0.5
    marginalContribution = covarianceMatrix @ portfolioWeights
    riskContribution = np.multiply(marginalContribution, portfolioWeights.T) / portfolioVariance

    return riskContribution

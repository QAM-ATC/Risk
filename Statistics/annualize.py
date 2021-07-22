" This file implements different functions for annualising volatility and returns from a given dataframe of returns"

import pandas as pd

def annualised_returns(returns: pd.DataFrame, periodsPerYear: int = 252):
    """This function returns the annualised returns of a given dataframe of returns.
    If the freq of the data is not daily, the annualisation factor must be specified.
    The function returns nan if the value computed is too small

    Parameters
    ----------
    returns : pd.DataFrame
        dataframe of returns
    periodsPerYear : int, optional
        freq of returns in a year, by default 252

    Returns
    -------
    Annualised Returns
        Returns the annualised return for each column in the dataframe
    """

    compoundGrowth = (1 + returns).prod()
    nobs = returns.shape[0]
    return compoundGrowth ** (periodsPerYear/nobs) - 1

def annualised_volatility(returns: pd.DataFrame, periodsPerYear: int = 252):
    """This function returns the annualised volatility of a given dataframe of returns.
    If the freq of the data is not daily, the annualisation factor must be specified

    Parameters
    ----------
    returns : pd.DataFrame
        dataframe of returns
    periodsPerYear : int, optional
        freq of returns in a year, by default 252

    Returns
    -------
    Annualised Returns
        Returns the annualised volatility for each column in the dataframe
    """

    return returns.std() * (periodsPerYear**0.5)



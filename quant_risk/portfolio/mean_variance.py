""" This module implements classes for various portfolio optimization methods."""

import numpy as np
import pandas as pd
from typing import OrderedDict, Union
import pypfopt
from pypfopt import expected_returns, risk_models

__all__ = [
    'MeanVariance'
]

class MeanVariance:
    """Constructor to instantiate the class based on the input parameters.

    Parameters
    ----------
    historicalPrices : pd.DataFrame
        DataFrame of historical prices for each ticker, with column name as name of ticker and index as timestamps
    tickers : list, optional
        List of tickers of the assets in the portfolio, by default None
    frequency: int, optional
        Frequency of the data passed, default is daily, i.e., 252 days
    bounds : Union[tuple,list]
        Minimum and maximum weight of each asset or a single pair if all weights are identical, (-1,1) if shorting is allowed, by default (0,1)
    riskFreeRate : float, optional
        Risk free rate, by default None
    solver : str, optional
        Name of solver, by default None. List of solvers: cp.installed_solvers()
    solverOptions : dict, optional
        Parameters for the given solver in the format {parameter:value}, by default None
    verbose : bool, optional
        Whether performance and debugging information should be printed, by default False
    """
    def __init__(self, historicalPrices: pd.DataFrame, frequency: int=252, bounds: Union[tuple,list] = (0,1), riskFreeRate: float = None,
    solver: str = None, solverOptions: dict = None, verbose: bool = False):
        """Constructor to instantiate the class based on the input parameters.

        Parameters
        ----------
        historicalPrices : pd.DataFrame
            DataFrame of historical prices for each ticker, with column name as name of ticker and index as timestamps
        tickers : list, optional
            List of tickers of the assets in the portfolio, by default None
        frequency: int, optional
            Frequency of the data passed, default is daily, i.e., 252 days
        bounds : Union[tuple,list]
            Minimum and maximum weight of each asset or a single pair if all weights are identical, (-1,1) if shorting is allowed, by default (0,1)
        riskFreeRate : float, optional
            Risk free rate, by default None
        solver : str, optional
            Name of solver, by default None. List of solvers: cp.installed_solvers()
        solverOptions : dict, optional
            Parameters for the given solver in the format {parameter:value}, by default None
        verbose : bool, optional
            Whether performance and debugging information should be printed, by default False
        """
        expectedReturns = expected_returns.mean_historical_return(historicalPrices,frequency=frequency)
        covarianceMatrix = risk_models.CovarianceShrinkage(historicalPrices).ledoit_wolf()
        self.historicalPrices = historicalPrices
        self.expectedReturns=expectedReturns
        self.covarianceMatrix = covarianceMatrix
        self.portfolio = pypfopt.EfficientFrontier(expectedReturns, covarianceMatrix, bounds, solver, verbose, solverOptions)

        if riskFreeRate is None:

            self.riskFreeRate = 0

        else:

            self.riskFreeRate = riskFreeRate

        self.weights = None

    def fit(self, method: str = 'max_sharpe', **kwargs) -> dict:
        """Optimize the portfolio by maxizing the Sharpe Ratio, and return the tickers and their respective weights.

        Parameters
        ----------
        method : str, optional
            Different methods by which one can maximise the portfolio.
            Please have a look at the following link for the available methods that are available for optimisation : https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html

            #TODO: We can always add more objectives to the solver so that we can get a better estimate of our weights.
            #  We can take some lower or upper bounds from the investment team as an input and use that as a contraint in our optimization
            by default 'max_sharpe'

        Returns
        -------
        dict
            Returns a dictionary with format {ticker:weight}
        """
        if method not in dir(self.portfolio):
            raise ValueError(f"The Chosen method '{method}'' is not a valid optimisation method. Please have a look at the documentation and try again.")

        self.weights = eval(f"dict(self.portfolio.{method}(**kwargs))")

        return self.weights

    def stats(self, verbose: bool = True) -> tuple:
        """Generate the expected annual return, annual volatility and Sharpe Ratio of the portfolio.

        Parameters
        ----------
        verbose : bool, optional
            Print the statistics, by default True

        Returns
        -------
        tuple
            Calculated statistics in the format (expected annual return, annual volatility, Sharpe Ratio)
        """

        stat = self.portfolio.portfolio_performance(verbose=verbose,risk_free_rate=self.riskFreeRate)

        return stat

    def getRiskFreeRate(self) -> float:
        """Returns the risk free rate

        Returns
        -------
        float
            Risk free rate
        """
        return round(self.riskFreeRate,2)

    def getHistoricalPrices(self) -> pd.DataFrame:
        """Returns the historical prices

        Returns
        -------
        pd.DataFrame
            DataFrame of historical prices for each ticker, with column name as name of ticker and index as timestamps
        """
        return self.historicalPrices

    def getExpectedReturns(self) -> pd.DataFrame:
        """Returns the expected returns

        Returns
        -------
        pd.DataFrame
            DataFrame of expected returns, with index as ticker names
        """
        return self.expectedReturns

    def getCovarianceMatrix(self) -> pd.DataFrame:
        """Returns the historical prices

        Returns
        -------
        pd.DataFrame
            DataFrame of covariance between tickers
        """
        return self.covarianceMatrix



""" This module implements classes for various portfolio optimization methods."""

import numpy as np
import pandas as pd
from typing import OrderedDict, Union
import pypfopt
from pypfopt import expected_returns, risk_models
from Miscellaneous import FetchData

class MeanVariance:

    def __init__(self,historicalPrices: pd.DataFrame,tickers: list=None,bounds: Union[tuple,list]=(0,1),riskFreeRate: float=None,
    solver: str=None, solverOptions: dict=None,verbose: bool=False):
        """Constructor to instantiate the class based on the input parameters.

        Parameters
        ----------
        historicalPrices : pd.DataFrame
            DataFrame of historical prices for each ticker, with column name as name of ticker and index as timestamps
        tickers : list, optional
            List of tickers of the assets in the portfolio, by default None
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
        expectedReturns = expected_returns.mean_historical_return(historicalPrices)
        covarianceMatrix = risk_models.CovarianceShrinkage(historicalPrices).ledoit_wolf()
        self.portfolio=pypfopt.EfficientFrontier(expectedReturns,covarianceMatrix,bounds,solver,verbose,solverOptions)
        if riskFreeRate is None:
            # TODO: implement risk-free rate for same time period as returns; implement dynamic rf-rate rather than static
            self.riskFreeRate=FetchData().risk_free_rate().mean().values[0]
        else:
            self.riskFreeRate=riskFreeRate
        self.weights=None

    def fit(self)->dict:
        """Optimize the portfolio by maxizing the Sharpe Ratio, and return the tickers and their respective weights.

        Returns
        -------
        dict
            Returns a dictionary with format {ticker:weight}
        """ 
        self.weights=dict(self.portfolio.max_sharpe(self.riskFreeRate))
        return self.weights

    def stats(self,verbose: bool=True)->tuple:
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
        stat=self.portfolio.portfolio_performance(verbose=verbose,risk_free_rate=self.riskFreeRate)
        return stat

class CPPI:
    ...
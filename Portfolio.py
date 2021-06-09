""" This module implements classes for various portfolio optimization methods."""

import numpy as np
import pandas as pd
from typing import OrderedDict, Union
import pypfopt
from pypfopt import expected_returns, risk_models
from Miscellaneous import FetchData

class MeanVariance:

    def __init__(self,historicalReturns: pd.DataFrame,tickers: list=None,bounds: Union[tuple,list]=(0,1),
    solver: str=None, solverOptions: dict=None,verbose: bool=False):
        """Constructor to instantiate the class based on the input parameters.

        Parameters
        ----------
        historicalReturns : pd.DataFrame
            DataFrame of historical returns for each ticker, with column name as name of ticker and index as timestamps
        tickers : list, optional
            List of tickers of the assets in the portfolio, by default None
        bounds : Union[tuple,list]
            Minimum and maximum weight of each asset or a single pair if all weights are identical, (-1,1) if shorting is allowed, by default (0,1)
        solver : str, optional
            Name of solver, by default None. List of solvers: cp.installed_solvers()
        solverOptions : dict, optional
            Parameters for the given solver in the format {parameter:value}, by default None
        verbose : bool, optional
            Whether performance and debugging information should be printed, by default False
        """
        expectedReturns = expected_returns.mean_historical_return(historicalReturns)
        covarianceMatrix = risk_models.CovarianceShrinkage(historicalReturns).ledoit_wolf()
        self.portfolio=pypfopt.EfficientFrontier(expectedReturns,covarianceMatrix,bounds,solver,verbose,solverOptions)
        self.weights=None

    def fit(self,riskFreeRate: float=None)->OrderedDict:
        """Optimize the portfolio by maxizing the Sharpe Ratio, and return the tickers and their respective weights.

        Parameters
        ----------
        riskFreeRate : float, optional
            Risk free rate, by default None

        Returns
        -------
        OrderedDict
            Returns a dictionary with format {ticker:weight}
        """
        if riskFreeRate is None:
            # TODO: implement risk-free rate for same time period as returns; implement dynamic rf-rate rather than static
            riskFreeRate=FetchData().risk_free_rate().mean().values[0]
        self.weights=self.portfolio.max_sharpe(riskFreeRate)
        return self.weights

    def stats():
        # Get statistics for your portfolio here
        # import from the statistics file or use portfolioopt
        ...

class CPPI:
    ...


ret=pd.DataFrame({"a":[1,2,3,4,5],"b":[5,6,7,8,9],"c":[2,4,6,8,10],"time":["2021-01-01","2021-01-02","2021-01-03","2021-01-04","2021-01-05"]})
ret=ret.set_index('time')
print(ret)
pf = MeanVariance(ret)
print(pf.portfolio.__dict__)
weights=pf.fit()
print(weights)
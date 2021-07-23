""" Implements the regime signal model"""

import numpy as np
import pandas as pd
from typing import OrderedDict, Union
from portfolio.portfolio import MeanVariance
from dateutil.relativedelta import relativedelta
from statistics.summarize import print_summary

__all__ = [
    'RegimeSignalModel'
]

class RegimeSignalModel():

    def __init__(self, regimeSignals: pd.Series, historicalPrices: pd.DataFrame, frequency: int=252, bounds: Union[tuple,list] = (0,1), riskFreeRate: float = None,
    solver: str = None, solverOptions: dict = None, verbose: bool = False, constraint: bool = True,
    LOOKBACKMONTHS: int = 3, CUSTOM_CEILING_RISK: float = .15):
        """Constructor to instantiate the class based on the input parameters.

        Parameters
        ----------
        regimeSignals: pd.Series
            Series of integers representing the regime signal, i.e. -1, 0, +1, with the index as timestamps
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
        constraint: bool
            True if you want to be invested in all tickers, will set minimum weight to 1/n**2 where n is number of tickers, else False
        """
        self.LOOKBACK_MONTHS = LOOKBACKMONTHS
        self.CUSTOM_CEILING_RISK = CUSTOM_CEILING_RISK

        # TODO: historicalPrices is assumed to be daily data, and regimeSignals monthly
        self.regimeSignals = regimeSignals

        # TODO: constraint can either be True or False, modify to allow for user-defined constraint
        self.numberOfTickers = len(historicalPrices.columns)

        if constraint:

            n = self.numberOfTickers
            bounds = (1 / (n**2), 1)

        # Create a list of portfolios to backtest
        self.portfolios = list()

        # ASSUMPTION: The regime signals index is LOOKBACK_MONTHS ahead of our historical Prices time series
        end = list(regimeSignals.index)
        start = []
        for date in list(regimeSignals.index):
            start.append(date + relativedelta(months=-self.LOOKBACK_MONTHS))

        self.dates = list(zip(start, end))

        for start, end in self.dates:
            # Create a portfolio of last N months' worth of data
            self.portfolios.append(MeanVariance(historicalPrices.loc[start:end], frequency, bounds, riskFreeRate, solver, solverOptions, verbose))

        self.regimeWeights = None
        self.historicalPrices = historicalPrices

    def get_weights(self,verbose: bool=False) -> dict:
        """Get the average weights for each regime type.

        Parameters
        ----------
        verbose: bool, optional
            Print the performance and debugging information, default False

        Returns
        -------
        dict
            A dictionary with the average regime weights for each regime, of form {regimeType:setOfWeights}
        """

        self.weightsList = {}
        self.weightsByTime = []

        for regimeType in self.regimeSignals.value_counts().index.tolist():
            self.weightsList[regimeType] = []

        # For each portfolio in the list, look at regime and optimize portfolio based on required methods
        for idx, regime in enumerate(self.regimeSignals):

            print("=============================================")

            if regime == -1:

                if verbose:

                    print("Max Sharpe Optimisation")

                    print("\n Training dates",
                        self.portfolios[idx].getHistoricalPrices().index[0],
                        self.portfolios[idx].getHistoricalPrices().index[-1])

                    print("\n Regime Signal dates", self.regimeSignals.index[idx])

                    print("\n Risk-free rate",
                        self.portfolios[idx].getRiskFreeRate())


                # TODO: fails if all expected returns are negative
                riskFreeRate=self.portfolios[idx].getRiskFreeRate()
                weights = self.portfolios[idx].fit(method='max_sharpe',risk_free_rate=riskFreeRate)

            elif regime == 1:
                if verbose:

                    print("Minimum Volatility Optimisation")

                    print("\n Training dates",
                        self.portfolios[idx].getHistoricalPrices().index[0],
                        self.portfolios[idx].getHistoricalPrices().index[-1])

                    print("\n Regime Signal dates", self.regimeSignals.index[idx])

                    print("\n Risk-free rate",
                        self.portfolios[idx].getRiskFreeRate())

                weights = self.portfolios[idx].fit(method='min_volatility')

            elif regime == 0:
                if verbose:

                    print(f"Custom: Maximum {self.CUSTOM_CEILING_RISK * 100}% volatility")

                    print("\n Training dates",
                        self.portfolios[idx].getHistoricalPrices().index[0],
                        self.portfolios[idx].getHistoricalPrices().index[-1])

                    print("\n Regime Signal dates", self.regimeSignals.index[idx])

                    print("\n Risk-free rate",
                        self.portfolios[idx].getRiskFreeRate())

                try:
                    weights = self.portfolios[idx].fit(method='efficient_risk',target_volatility=self.CUSTOM_CEILING_RISK)

                except:
                    weights = self.portfolios[idx].fit(method='min_volatility')

            if verbose:

                print("\n", weights, "\n")

            self.weightsList[regime].append(weights)
            self.weightsByTime.append(weights)

        self.regimeWeights = {}

        for regimeType in list(self.weightsList.keys()):
            self.regimeWeights[regimeType] = pd.DataFrame([ticker for ticker in self.weightsList[regimeType]]).mean().to_dict()

        self.weightsByTime = pd.DataFrame.from_dict(self.weightsByTime)
        self.weightsByTime.index = self.regimeSignals.index


    def get_portfolio(self, verbose: bool = True):
        """Computes the portfolio value from the weights matrix calculated in get_weights function.
        If Verbose: prints out the summary statistics of the portfolio

        Parameters
        ----------
        verbose : bool, optional
            prints out the portfolio statistics, by default True

        Returns
        -------
        DataFrame
            Returns a pandas dataframe of the Portfolio indexed by date
        """

        temp = pd.DataFrame(index=pd.date_range(start=self.weightsByTime.index[0],
                                            end=self.weightsByTime.index[-1], freq='B'))

        weightsByTime = pd.concat([temp, self.weightsByTime], axis=1, join='outer').ffill().resample('B').asfreq()
        self.historicalPrices = pd.concat([self.historicalPrices, temp], axis=1, join='outer').ffill()

        portfolio = pd.DataFrame((self.historicalPrices.loc[weightsByTime.index.tolist(), :] * weightsByTime).sum(axis = 1))
        portfolio.columns = ['Portfolio Value']

        if verbose:
            print(print_summary(portfolio))

        return portfolio

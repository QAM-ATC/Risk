""" Implements the regime signal model"""

import numpy as np
import pandas as pd
from typing import OrderedDict, Union
import pypfopt
from copy import deepcopy
from pypfopt import expected_returns, risk_models
from Miscellaneous import FetchData
from Portfolio import MeanVariance

class regimeSignalModel():

    def __init__(self, regimeSignals: pd.Series, historicalPrices: pd.DataFrame, tickers: list = None, frequency: int=252, bounds: Union[tuple,list] = (0,1), riskFreeRate: float = None,
    solver: str = None, solverOptions: dict = None, verbose: bool = False, constraint: bool = True):
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
        
        # TODO: historicalPrices is assumed to be daily data, and regimeSignals monthly

        self.regimeSignals=regimeSignals

        # TODO: constraint can either be True or False, modify to allow for user-defined constraint
        self.numberOfTickers = len(historicalPrices.columns)
        if constraint==True:
            n=self.numberOfTickers
            bounds=(1/(n**2),1)
        
        # Create a list of portfolios to backtest
        portfolios=list()

        # Look back at last 3 months, TODO: last n months
        numberOfMonths=len(historicalPrices.resample('1M'))
        yr=historicalPrices.index[0].year
        month=historicalPrices.index[0].month+2
        month=month%12
        for i in range(3,numberOfMonths+1):
            if month==12:
                month=1
                yr+=1
            else:
                month+=1

            yearStart=yr
            yearEnd=yr
            monthStart=month-3
            if monthStart<=0:
                monthStart+=12
                yearStart=yr-1
            
            monthEnd=month-1
            if monthEnd==0:
                monthEnd=12
                yearEnd-=1

            stampStart=str(yearStart)+'-'+str(monthStart)
            stampEnd=str(yearEnd)+"-"+str(monthEnd)

            # Create a portfolio of last 3 months' worth of data
            portfolios.append(MeanVariance(historicalPrices.loc[stampStart:stampEnd],tickers,frequency,bounds,riskFreeRate,solver,solverOptions,verbose))
        
        self.portfolios=portfolios
        
        self.regimeWeights=None

    def getWeights(self,verbose: bool=False) -> dict:
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

        weightsList=dict()
        weightsList[1]=list()
        weightsList[-1]=list()
        weightsList[0]=list()

        # For each portfolio in the list, look at regime and optimize portfolio based on required methods
        for idx,regime in enumerate(self.regimeSignals):
            if regime==-1:
                if verbose:
                    print("max sharpe")
                    print("Training dates",
                        self.portfolios[idx].getHistoricalPrices().index[0],
                        self.portfolios[idx].getHistoricalPrices().index[-1],"\nRegime Signal dates",
                        self.regimeSignals.index[idx],"\nRisk-free rate",
                        self.portfolios[idx].getRiskFreeRate())
                    print(self.portfolios[idx].getExpectedReturns())
                    print(self.portfolios[idx].getCovarianceMatrix())
                
                # TODO: fails if all expected returns are negative
                riskFreeRate=self.portfolios[idx].getRiskFreeRate()
                wts=self.portfolios[idx].fit(method='max_sharpe',risk_free_rate=riskFreeRate)
            
            elif regime==1:
                if verbose:
                    print("min vol")
                    print("Training dates",
                        self.portfolios[idx].getHistoricalPrices().index[0],
                        self.portfolios[idx].getHistoricalPrices().index[-1],"\nRegime Signal dates",
                        self.regimeSignals.index[idx],"\nRisk-free rate",
                        self.portfolios[idx].getRiskFreeRate())
                    print(self.portfolios[idx].getExpectedReturns())
                    print(self.portfolios[idx].getCovarianceMatrix())
                wts=self.portfolios[idx].fit(method='min_volatility')
            
            elif regime==0:
                print("custom: maximum 15% volatility")
                print("Training dates",
                        self.portfolios[idx].getHistoricalPrices().index[0],
                        self.portfolios[idx].getHistoricalPrices().index[-1],"\nRegime Signal dates",
                        self.regimeSignals.index[idx],"\nRisk-free rate",
                        self.portfolios[idx].getRiskFreeRate())
                print(self.portfolios[idx].getExpectedReturns())
                print(self.portfolios[idx].getCovarianceMatrix())
        
                # TODO: currently target vol is 15%, which is the upper bound for the medium-vol regime; sometimes too low
                wts=self.portfolios[idx].fit(method='efficient_risk',target_volatility=0.15)
            
            if verbose:
                print()
                print(wts)
                print()
                print()
                print()
            weightsList[regime].append(wts)
        
        self.regimeWeights=dict()
        self.regimeWeights[1]=dict(pd.DataFrame([i for i in weightsList[1]]).mean())
        self.regimeWeights[-1]=dict(pd.DataFrame([i for i in weightsList[-1]]).mean())
        self.regimeWeights[0]=dict(pd.DataFrame([i for i in weightsList[0]]).mean())

        return self.regimeWeights









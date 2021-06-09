# Implements functions like annualisation of returns/volatility
# Get data for a example/test set / set up quandl api

import pypfopt
import numpy as np
import matplotlib.axes as ax
import matplotlib.pyplot as plt
import pandas as pd
import quandl
import datetime as dt

class Plot:

    def weights(weights:dict, ax:ax=None, plot:bool=False, **kwargs) -> ax:
        """The function plots the weights that are outputted by the
        PyPortfolioOpt optimizer.

        Parameters
        ----------
        weights : dict
            A dictionary of weights with the format {ticker:Weight}
        ax : matplotlib.axes, optional
            An ax object to plot upon, by default: None
        plot : bool, optional
            Whether to plot the figure or just to return the ax object, by default: False

        Returns
        -------
        matplotlib.axes
            Returns a matplotlib.axes object
        """

        # Calling the plot_weights function from PyPortfolioOpt
        ax = pypfopt.plotting.plot_weights(weights=weights, ax=ax, **kwargs)

        if plot:
            plt.show()

        return ax

    def efficient_frontier(optimizer:pypfopt.EfficientFrontier, efficientParameter:str='return',
                            efficentParameterRange:np.array=None, points:int=100, ax:ax=None,
                            showAssets=True, plot:bool=False, complex:bool=True, **kwargs) -> ax:
        """The function computes and plots the Efficient Frontier on an Efficient Frontier object
         instantiated from the PyPortolioOpt package

        Parameters
        ----------
        optimizer : pypfopt.EfficientFrontier
            An instantiated optimizer object before it is fitted
        efficientParameter : str, optional
            One of ("utility", "risk", "return")
            Whether to use a range over utility, risk, or return, by default 'return'
        efficientParameterRange : np.array, optional
            The range of parameter values for efficientParameter, if None,
            it automatically computes a range from min->max return , by default None
        points : int, optional
            The number of points to plot. This is overridden if an efficientParameterRange is provided
            explicitly , by default 100
        ax : ax, optional
            An ax object to plot upon, by default None
        showAssets : bool, optional
            Whether we should plot the asset risk/returns also, by default True
        plot : bool, optional
            Whether to plot the figure while calling the function or not, by default False
        complex : bool, optional
            Whether to plot a more comprehensive plot with suboptimal portfolios coloured by sharpe ratios.
            Note: this requires that the returns(mu) and the covariance matrix(S) also be provided in kwargs , by default True

        Returns
        -------
        ax
            Returns an ax object
        """

        fig, ax = plt.subplots()
        ax = pypfopt.plotting.plot_efficient_frontier(opt=optimizer, ef_param=efficientParameter,
                                                     ef_param_range=efficentParameterRange, points=points,
                                                     ax=ax, show_assets=showAssets, **kwargs)

        if complex:

            try:
                mu = kwargs['mu']
                S = kwargs['S']
            # To make sure that the complex plotting does not throw an Error
            except: raise NameError("Values of Return or Covariance not found, cannot plot complex plot. Please try again.")

            # Find the tangency portfolio
            optimizer.max_sharpe()
            tangentReturns, tangentStd, _ = optimizer.portfolio_performance()
            ax.scatter(tangentStd, tangentReturns, marker="*", s=100, c="r", label="Max Sharpe")

            # Generate random portfolios
            n_samples = 10000
            sampleWeights = np.random.dirichlet(np.ones(len(mu)), n_samples)
            sampleReturns = sampleWeights.dot(mu)
            sampleStd = np.sqrt(np.diag(sampleWeights @ S @ sampleWeights.T))
            sharpes = sampleReturns / sampleStd
            ax.scatter(sampleStd, sampleReturns, marker=".", c=sharpes, cmap="viridis_r")

            # Output
            ax.set_title("Efficient Frontier with random portfolios")
            ax.legend()
            plt.tight_layout()

        # If the figure is to be plotted
        if plot:
            plt.show()

        return ax

    def covariance_heatmap(covarianceMatrix:pd.DataFrame, showAssets:bool=True, plot:bool=False, **kwargs) -> ax:
        """The function returns a matplotlib axes object and computes the heatmap for the Covariance matrix
        of a given set of assets

        Parameters
        ----------
        covarianceMatrix : pd.Dataframe
            The Covariance matrix of the set of assets
        showAssets : bool, optional
            Whether to use the tickers as labels or not
            (not recommended for large set of assets), by default True

        Returns
        -------
        ax
            Retuns the matplotlib.axes object
        """

        ax = pypfopt.plotting.plot_covariance(cov_matrix=covarianceMatrix, plot_correlation=False,
                                             show_tickers=showAssets, **kwargs)

        if plot:
            plt.show()

        return ax

    def correlation_heatmap(correlationMatrix:pd.DataFrame, showAssets:bool=True, plot:bool=False, **kwargs) -> ax:
        """The function returns a matplotlib axes object and computes the heatmap for the correlationMatrix
        of a given set of assets

        Parameters
        ----------
        correlationMatrix : pd.Dataframe
            The Correlation Matrix of the set of assets
        showAssets : bool, optional
            Whether to use the tickers as labels or not
            (not recommended for large set of assets), by default True

        Returns
        -------
        ax
            Retuns the matplotlib.axes object
        """

        # I've directly asked the user for a correlation matrix rather than asking for a covariance matrix first and then converting
        #  it into a correlation matrix in PyPortfolioOpt
        ax = pypfopt.plotting.plot_covariance(cov_matrix=correlationMatrix, plot_correlation=False, show_tickers=showAssets, **kwargs)

        if plot:
            plt.show()

        return ax

class FetchData:

    def __init__(self):

        # For the first time
        API_KEY = "YOUR_KEY_HERE"
        quandl.save_key(API_KEY)

        # After the key has been added to your environment already
        quandl.read_key()

    # Gets test datasets from the quandl api
    def test_set(self, startDate:str=None, endDate:str=None, ticker:str="AAPL", **kwargs) -> pd.DataFrame:
        """Test sets which are called from Quandl each time.
        The function currently calls the given ticker close prices from the WIKI/PRICES database from Quandl.
        If no startDate or endDate is provided, the function returns the trailing twelve months (TTM) close prices for the ticker

        Parameters
        ----------
        startDate : str, optional
            Incase the user wants to supply a startDate to call data from a specific time period
            The format is "YYYY-MM-DD", by default None
        endDate : str, optional
            Incase the user wants to supply a endDate to call data from a specific time period
            The format is "YYYY-MM-DD", by default None
        ticker : str, optional
            The test set ticker dataset that is called.
            Incase, the called ticker is not available in the WIKI/PRICES database,
            the function throws an error, by default "AAPL"

        Returns
        -------
        pd.DataFrame
            Returns a pandas dataframe object consisting of the called data for the ticker
        """

        # Both start and end dates must be provided else the call reverts to the default set of
        # endDate as today and startDate as a year back
        if not startDate or endDate:
            endDate = dt.datetime.today().strftime(format="%Y-%m-%d")
            startDate = (dt.datetime.today() - dt.timedelta(years=1)).strftime(format="%Y-%m-%d")

        try:
            # The standard database that we want to use for our test cases
            database = "WIKI/PRICES"
            # Filtering the database by columns to only return the ticker, date, and close price for the dates greater than
            # or equal to the startDate and less than and equal to the endDate
            data = quandl.get_table(database, qopts = { 'columns': ['ticker', 'date', 'close'] },
                                     ticker = [ticker], date = { 'gte': startDate, 'lte': endDate })

        except: raise ImportError("Unable to Import test data, please try again.")

        else:

            print(f"...Data for {ticker} from {startDate} to {endDate} loaded successfully")

            return data

    # Gets actual data from the quandl api
    def get():
        ...
import pypfopt
import numpy as np
import matplotlib.axes as ax
import matplotlib.pyplot as plt
import pandas as pd
from typing import Union
from pypfopt import plotting

def weights(weights: dict, ax: ax = None, plot: bool = False, **kwargs) -> ax:
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

    # Type checking for weights
    if not isinstance(weights, dict):
        raise ValueError("Weights are required to be in a dictionary of the format {ticker:weight}")

    fig, ax = plt.subplots(figsize=(10, 6))
    # Calling the plot_weights function from PyPortfolioOpt
    ax = plotting.plot_weights(weights=weights, ax=ax, **kwargs)
    plt.title("Weights by Tickers in Optimised Portfolio", size=15)

    if plot:
        plt.show()

    return ax

def efficient_frontier(optimizer: pypfopt.EfficientFrontier, efficientParameter: str = 'return',
                        efficentParameterRange:Union[np.array,list]=None, points:int=100, ax:ax=None,
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

    fig, ax = plt.subplots(figsize=(12, 8))
    # TODO: Some error occuring in plotting
    ax = plotting.plot_efficient_frontier(opt=optimizer, ef_param=efficientParameter,
                                                    ef_param_range=efficentParameterRange, points=points,
                                                    ax=ax, show_assets=showAssets)

    if complex:

        expectedReturns = pypfopt.expected_returns.mean_historical_return(**kwargs)
        covarianceMatrix = pypfopt.risk_models.CovarianceShrinkage(**kwargs).ledoit_wolf()
        # Find the tangency portfolio
        optimizer.max_sharpe()
        tangentReturns, tangentStd, _ = optimizer.portfolio_performance()
        ax.scatter(tangentStd, tangentReturns, marker="*", s=100, c="r", label="Max Sharpe")

        # Generate random portfolios
        n_samples = 10000
        sampleWeights = np.random.dirichlet(np.ones(len(expectedReturns)), n_samples)
        sampleReturns = sampleWeights.dot(expectedReturns)
        sampleStd = np.sqrt(np.diag(sampleWeights @ covarianceMatrix @ sampleWeights.T))
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

def covariance_heatmap(assetDataframe:pd.DataFrame, showAssets:bool=True, plot:bool=False, **kwargs) -> ax:

    """The function returns a matplotlib axes object and computes the heatmap for the Covariance matrix
    of a given set of assets

    Parameters
    ----------
    assetDataframe : pd.Dataframe
        The dataframe containing the historical data for our tickers
    showAssets : bool, optional
        Whether to use the tickers as labels or not
        (not recommended for large set of assets), by default True

    Returns
    -------
    ax
        Retuns the matplotlib.axes object
    """
    covarianceMatrix = assetDataframe.cov()
    ax = plotting.plot_covariance(cov_matrix=covarianceMatrix, plot_correlation=False,
                                            show_tickers=showAssets, **kwargs)
    plt.title("Correlation Matrix of Price data\n", size=15)

    if plot:
        plt.show()

    return ax

def correlation_heatmap(assetDataframe:pd.DataFrame, showAssets:bool=True, plot:bool=False, **kwargs) -> ax:

    """The function returns a matplotlib axes object and computes the heatmap for the correlationMatrix
    of a given set of assets

    Parameters
    ----------
    assetDataframe : pd.Dataframe
        The dataframe containing the historical data for our tickers
    showAssets : bool, optional
        Whether to use the tickers as labels or not
        (not recommended for large set of assets), by default True

    Returns
    -------
    ax
        Retuns the matplotlib.axes object
    """

    correlationMatrix = assetDataframe.corr()

    ax = plotting.plot_covariance(cov_matrix=correlationMatrix, plot_correlation=False, show_tickers=showAssets, **kwargs)
    plt.title("Correlation Matrix of Price data\n", size=15)

    if plot:
        plt.show()

    return ax
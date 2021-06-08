# Implements functions like annualisation of returns/volatility
# Get data for a example/test set
import pypfopt
import numpy as np
import matplotlib.axes as ax
import matplotlib.pyplot as plt
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

    def efficient_frontier(optimizer:pypfopt.EfficientFrontier, efficientParameter:str='return', efficentParameterRange:np.array=None, points:int=100, ax:ax=None, showAssets=True, plot:bool=False, **kwargs) -> ax:
        """The function computes and plots the Efficient Frontier on an Efficient Frontier object instantiated from the PyPortolioOpt package

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

        Returns
        -------
        ax
            Returns an ax object
        """

        ax = pypfopt.plotting.plot_efficient_frontier(opt=optimizer, ef_param=efficientParameter, ef_param_range=efficentParameterRange, points=points,
        ax=ax, show_assets=showAssets, showfig=plot, **kwargs)

        return ax
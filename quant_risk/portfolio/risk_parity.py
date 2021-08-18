"This module has functions related to risk parity and risk contributions"

from quant_risk.statistics.statistics import risk_contribution
import numpy as np
from scipy.optimize import minimize
import pandas as pd

__all__ = [
    'target_risk_contribution',
    'mean_square_deviation',
    'risk_parity_portfolio'
]

def target_risk_contribution(targetRisk: np.array, covarianceMatrix: pd.DataFrame, bounds: tuple = (0, 1)):
    """This function computes the portfolio weights of each of our assets given a target risk contribution
    and the  covariance matrix by minimising the MSE between target and optimised risk contribution

    Parameters
    ----------
    targetRisk : np.array
        The risk contributions we want for each asset
    covarianceMatrix : pd.DataFrame
        The covariance matrix of our asset returns computed by any method
    bounds : tuple, optional
        The bound that each of our weights will follow, by default (0, 1)

    Returns
    -------
    np.array
        Returns the portfolio weights of the desired portfolio
    """

    numberOfAssets = covarianceMatrix.shape[0]
    initialGuess = np.repeat(1/numberOfAssets, numberOfAssets)
    bounds = (bounds,) * numberOfAssets

    weightsConstraint = {
                        'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
                        }

    def mean_square_deviation(weights: np.array, targetRisk: np.array, covarianceMatrix: pd.DataFrame):
        """This function computes the mean square deviation of the risk contributions of the constituents of
        our portfolio with respect to the targetRisk that we want

        Parameters
        ----------
        weights : np.array
            The portfolio weights of our assets
        targetRisk : np.array
            The risk contributions we want for each asset
        covarianceMatrix : pd.DataFrame
            The covariance matrix of our asset returns computed by any method

        Returns
        -------
        float
            Returns the mean square error between our target and current risk contributions
        """

        riskContributions = risk_contribution(weights, covarianceMatrix)
        result = ((riskContributions - targetRisk) ** 2).sum()

        return result

    weights = minimize(mean_square_deviation,
                        initialGuess,
                        args=(targetRisk, covarianceMatrix),
                        method = 'SLSQP',
                        options = {'disp': True},
                        constraints = (weightsConstraint),
                        bounds = bounds
                        ).x

    return weights

def risk_parity_portfolio(covarianceMatrix: pd.DataFrame, bounds: tuple = (0, 1)):
    """Returns the weights of the portfolio that equalizes the contributions of the constituents
    based on the given covariance matrix

    Parameters
    ----------
    covarianceMatrix : pd.DataFrame
        The covariance matrix of our asset returns computed by any method
    bounds : tuple
        The bound that each of our weights will follow, by default (0, 1)

    Returns
    -------
    np.array
        Returns the portfolio weights of the desired portfolio
    """

    numberOfAssets = covarianceMatrix.shape[0]
    targetRisk = np.repeat(1 / numberOfAssets, numberOfAssets)

    weights = target_risk_contribution(targetRisk, covarianceMatrix, bounds)

    return weights
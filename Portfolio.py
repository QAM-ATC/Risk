""" This module implements classes for various portfolio optimization methods."""

import numpy as np
import pandas as pd
from typing import Union
import pypfopt
import cvxpy as cp


class MeanVariance:

    def __init__(self,expectedReturns: Union[np.ndarray,pd.Series,list],covarianceMatrix: Union[np.ndarray,pd.DataFrame],
    tickers: list=None,bounds: Union[tuple,list]=(0,1),solver: str=None, solverOptions: dict=None,verbose: bool=False):
        """Constructor to instantiate the class based on the input parameters.

        Parameters
        ----------
        expectedReturns : Union[np.ndarray,pd.Series,list]
            Expected returns for each asset
        covarianceMatrix : Union[np.ndarray,pd.DataFrame]
            A 2-D matrix with the covariance of returns for each asset, must be positive semidefinite
        tickers : list, optional
            List of tickers of the assets in the portfolio, by default None
        bounds : Union[tuple,list]
            Minimum and maximum weight of each asset or a single pair if all weights are identical, (-1,1) if shorting is allowed, by default (0,1)
        solver : str, optional
            Name of solver, by default None. List of solvers: cvxpy.installed_solvers()
        solverOptions : dict, optional
            Parameters for the given solver in the format {parameter:value}, by default None
        verbose : bool, optional
            Whether performance and debugging information should be printed, by default False
        """
        # Inputs
        numberOfAssets=len(tickers)
        self.numberOfAssets = numberOfAssets
        
        self.covarianceMatrix = pypfopt.EfficientFrontier._validate_cov_matrix(covarianceMatrix)
        self.expectedReturns = pypfopt.EfficientFrontier._validate_expected_returns(expectedReturns)
        
        if tickers is None:
            if isinstance(expectedReturns, pd.Series):
                tickers = list(expectedReturns.index)
            elif isinstance(covarianceMatrix, pd.DataFrame):
                tickers = list(covarianceMatrix.columns)
            else:  # use integer labels
                tickers = list(range(len(expectedReturns)))
        self.tickers=tickers

        if expectedReturns is not None and covarianceMatrix is not None:
            if covarianceMatrix.shape != (len(expectedReturns), len(expectedReturns)):
                raise ValueError("Covariance matrix does not match expected returns")

        self._w = cp.Variable(numberOfAssets)
        self._objective = None
        self._additionalObjectives = []
        self._constraints = []
        self._lowerBounds = None
        self._upperBounds = None
        self._mapBoundsToConstraints(bounds)

        self._opt = None
        self._solver = solver
        self._verbose = verbose
        self._solverOptions = solverOptions if solverOptions else {}

        self.riskFreeRate = None
        
        # Outputs
        self.weights=None

    def _mapBoundsToConstraints(self,bounds: Union[tuple,list]):
        """Convert input bounds into the appropriate format and add them to the constraint list

        Parameters
        ----------
        bounds : Union[tuple,list]
            Minimum and maximum weight of each asset or a single pair if all weights are identical
        """
         # If it is a collection with the right length, assume they are all bounds.
        if len(bounds) == self.numberOfAssets and not isinstance(
            bounds[0], (float, int)
        ):
            bounds = np.array(bounds, dtype=np.float)
            self._lowerBounds = np.nan_to_num(bounds[:, 0], nan=-np.inf)
            self._upperBounds = np.nan_to_num(bounds[:, 1], nan=np.inf)
        else:
            # Otherwise this must be a pair.
            if len(bounds) != 2 or not isinstance(bounds, (tuple, list)):
                raise TypeError(
                    "bounds must be a pair (lower bound, upper bound) OR a collection of bounds for each asset"
                )
            lower, upper = bounds

            # Replace None values with the appropriate +/- 1
            if np.isscalar(lower) or lower is None:
                lower = -1 if lower is None else lower
                self._lowerBounds = np.array([lower] * self.numberOfAssets)
                upper = 1 if upper is None else upper
                self._upperBounds = np.array([upper] * self.numberOfAssets)
            else:
                self._lowerBounds = np.nan_to_num(lower, nan=-1)
                self._upperBounds = np.nan_to_num(upper, nan=1)

        self._constraints.append(self._w >= self._lowerBounds)
        self._constraints.append(self._w <= self._upperBounds)
        
    def _maximizeSharpeRatio(self,riskFreeRate: float=None):
        """Constructions the tangency portfolio by maximizing Sharpe Ratio, and sets the weights accordingly.

        Parameters
        ----------
        riskFreeRate : float, optional
            Risk free rate, by default None
        """
        if not isinstance(riskFreeRate, (int, float)):
            raise ValueError("riskFreeRate should be numeric")
        if riskFreeRate is None:
            riskFreeRate=getRiskFreeRate()

        self._riskFreeRate = riskFreeRate

        # max_sharpe requires us to make a variable transformation.
        # Here we treat w as the transformed variable.
        self._objective = cp.quad_form(self._w, self.cov_matrix)
        k = cp.Variable()

        # Note: objectives are not scaled by k. Hence there are subtle differences
        # between how these objectives work for max_sharpe vs min_volatility
        if len(self._additional_objectives) > 0:
            warnings.warn(
                "max_sharpe transforms the optimization problem so additional objectives may not work as expected."
            )
        for obj in self._additional_objectives:
            self._objective += obj

        new_constraints = []
        # Must rebuild the constraints
        for constr in self._constraints:
            if isinstance(constr, cp.constraints.nonpos.Inequality):
                # Either the first or second item is the expression
                if isinstance(
                    constr.args[0], cp.expressions.constants.constant.Constant
                ):
                    new_constraints.append(constr.args[1] >= constr.args[0] * k)
                else:
                    new_constraints.append(constr.args[0] <= constr.args[1] * k)
            elif isinstance(constr, cp.constraints.zero.Equality):
                new_constraints.append(constr.args[0] == constr.args[1] * k)
            else:
                raise TypeError(
                    "Please check that your constraints are in a suitable format"
                )

        # Transformed max_sharpe convex problem:
        self._constraints = [
            (self.expected_returns - risk_free_rate).T @ self._w == 1,
            cp.sum(self._w) == k,
            k >= 0,
        ] + new_constraints

        self._solve_cvxpy_opt_problem()
        # Inverse-transform
        self.weights = (self._w.value / k.value).round(16) + 0.0
        return self._make_output_weights()

    def fit(self):
        # utilise pyportfoliopt to create our weights
        # return as a dataframe/series with tickers as index
        ...
    def stats():
        # Get statistics for your portfolio here
        # import from the statistics file or use portfolioopt
        ...


class CPPI:
    ...

pypfopt.base_optimizer.BaseConvexOptimizer._map_bounds_to_constraints((0,1))
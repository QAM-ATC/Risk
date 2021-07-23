from statsmodels.tsa.arima.model import ARIMA
from typing import Union
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

__all__ = [
    'auto_arima'
]

# AUTO ARIMA
def auto_arima(endogenousSeries: Union[pd.Series, np.array], exogenousSeries: Union[pd.DataFrame, np.array],
                 pRange: int = 5, dRange: int = 1, qRange: int = 5, metric: str = 'BIC',  **kwargs):
    """This function implements an auto-arima model by utilising a grid search over the parameter ranges for the
    autoregressive, differencing, moving average parameters for each model. Each model is then evaluated based on the
    specifed metric and the model with the lowest metric statistic is chosen as the best model. The order params for
    the best model are saved and another model is fitted with those params.

    Parameters
    ----------
    endogenousSeries : Union[pd.Series, np.array]
        The endogenous variable for our ARIMA model
    exogenousSeries : Union[pd.DataFrame, np.array]
        The exogeneous variables for our ARIMA model
    pRange : int, optional
        The maximum value of the autogressive component till where we want to search, by default 5
    dRange : int, optional
        The maximum value of the differencing/integrated order component till where we want to search, by default 1
    qRange : int, optional
        The maximum value of the moving average component till where we want to search, by default 5
    metric : str, optional
        The metric by which we want to search and choose our model, by default 'BIC'

    Returns
    -------
    Fitted ARIMA Result
        Returns a fitted arima model with the best chosen order of components

    Raises
    ------
    RuntimeWarning
        If the model fails to converge on any order, a RuntimeWarning is engaged
    """

    auto_arima.bestParams = None
    auto_arima.bestMetric = np.inf

    for d in range(dRange+1):
        for p in range(pRange+1):
            for q in range(qRange+1):

                order = (p, d, q)

                try:
                    results = ARIMA(endog=endogenousSeries, exog=exogenousSeries, order=order, **kwargs).fit()

                except:
                    raise RuntimeWarning(f"Model failed to converge on order {order}")

                if results.info_criteria(metric) < auto_arima.bestMetric:
                    auto_arima.bestMetric = results.info_criteria(metric)
                    auto_arima.bestParams = order

                else: continue

    results = ARIMA(endog=endogenousSeries, exog=exogenousSeries, order=auto_arima.bestParams, **kwargs).fit()

    return results
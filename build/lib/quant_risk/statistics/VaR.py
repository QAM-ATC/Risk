"Value at risk functions here/ no need for class"

import pandas as pd
import empyrical
import numpy as np

__all__ = [
    'conditional_value_at_risk',
    'value_at_risk'
]

def conditional_value_at_risk(price: pd.Series, threshold: float = 0.05) -> float:
    """Calculates Conditional Value at Risk for given price series

    Parameters
    ----------
    price : pd.Series
        historical prices of a given security

    Returns
    -------
    float
        Conditional Value at Risk (VaR value) for given price
    """

    returns = price.pct_change().dropna()

    cVar = np.mean(returns[returns < value_at_risk(price)])

    return cVar

def value_at_risk(price: pd.Series, threshold: float = 0.05) -> float:
    """Calculates Value at Risk for given price series

    Parameters
    ----------
    price : pd.Series
        historical prices of a given security

    Returns
    -------
    float
        Value at Risk (VaR value) for given price
    """

    if isinstance(price, pd.DataFrame):

        return price.apply(value_at_risk)

    returns = price.pct_change().dropna()
    var = empyrical.stats.value_at_risk(returns, threshold)

    return var

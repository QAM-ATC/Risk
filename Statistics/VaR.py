"Value at risk functions here/ no need for class"

import pandas as pd
import empyrical
import numpy as np

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

    returns = price.diff().dropna()

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

        results = {}

        for col in price.columns:
            results[col] = conditional_value_at_risk(price[col], threshold)

        return results

    returns = price.diff().dropna()
    var = empyrical.stats.value_at_risk(returns, threshold)

    return var

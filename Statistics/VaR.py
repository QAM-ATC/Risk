"Value at risk functions here/ no need for class"

import pandas as pd
import empyrical

def conditionalVar(price: pd.Series) -> float:
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
    r = price.diff().dropna()
    cVar = empyrical.stats.conditional_value_at_risk(r)
    return cVar

def var(price: pd.Series) -> float:
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
    r = price.diff().dropna()
    var = empyrical.stats.value_at_risk(r)
    return var

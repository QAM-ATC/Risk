"Put summary function here that prints or returns a dataframe"
import pandas as pd
from typing import Union

def skewness(price: Union[pd.DataFrame,pd.Series]) -> Union[float,pd.Series]:
    """Calculates the skewness for a given set of prices

    Parameters
    ----------
    price : Union[pd.DataFrame,pd.Series]
        historical prices of a given security

    Returns
    -------
    Union[float,pd.Series]
        skewness for a given set of prices
    """
    r = price.diff().dropna()
    deviation = r - r.mean()
    sigma = r.std()
    num = (deviation**3).mean()
    return num/(sigma**3)


def kurtosis(price: Union[pd.Series,pd.DataFrame]) -> Union[float,pd.Series]:
    """Calculates the kurtosis for a given set of prices

    Parameters
    ----------
    price : Union[pd.DataFrame,pd.Series]
        historical prices of a given security

    Returns
    -------
    Union[float,pd.Series]
        kurtosis for a given set of prices
    """
    r = price.diff().dropna()
    deviation = r - r.mean()
    sigma = r.std()
    num = (deviation**4).mean()
    return num/(sigma**4)


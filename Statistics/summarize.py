"Put summary function here that pprints or returns a dataframe"

def skewness(price):
    """
    Computes skewness of series or dataframe
    
    Parameters
    ----------
    price : series / dataframe of asset price
        
    Returns
    -------
    float or series
    
    """
    r = price.diff().dropna()
    deviation = r - r.mean()
    sigma = r.std()
    num = (deviation**3).mean()
    return num/(sigma**3)


def kurtosis(price):
    """ Computes kurtosis of series or dataframe

    Parameters
    ----------
    price : series / dataframe of asset price

    Returns
    -------
    float or series

    """
    r = price.diff().dropna()
    deviation = r - r.mean()
    sigma = r.std()
    num = (deviation**4).mean()
    return num/(sigma**4)


from statsmodels.tsa.stattools import adfuller

def stationaryTest(series):
    """Conducts ADF summary test on series

    Parameters
    ----------
    series : pd series

    Returns
    -------
    ADF summary
    """
    
    X = series.values
    result = adfuller(X)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
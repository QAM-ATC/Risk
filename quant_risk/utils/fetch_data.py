import pandas as pd
import quandl
import datetime as dt
from typing import Union

__all__ = [
    'test_set',
    'risk_free_rate'
]

# Gets test datasets from the quandl api
def test_set(startDate: str = None, endDate: str = None, ticker: Union[str, list] = "AAPL", **kwargs) -> pd.DataFrame:
    """Test sets which are called from Quandl each time.
    The function currently calls the given ticker close prices from the WIKI/PRICES database from Quandl.
    If no startDate or endDate is provided, the function returns the trailing twelve months (TTM) close prices for the ticker

    Parameters
    ----------
    startDate : str, optional
        Incase the user wants to supply a startDate to call data from a specific time period
        The format is "YYYY-MM-DD", by default None
    endDate : str, optional
        Incase the user wants to supply a endDate to call data from a specific time period
        The format is "YYYY-MM-DD", by default None
    ticker : str, optional
        The test set ticker dataset that is called.
        Incase, the called ticker is not available in the WIKI/PRICES database,
        the function throws an error, by default "AAPL"

    Returns
    -------
    pd.DataFrame
        Returns a pandas dataframe object consisting of the called data for the ticker
    """
    # Incase the ticker provided is a single string rather than a list of tickers

    if isinstance(ticker, str):
        ticker = [ticker]

    # Both start and end dates must be provided else the call reverts to the default set of
    # endDate as today and startDate as a year back

    if not isinstance(startDate, str) or not isinstance(endDate, str):
        endDate = dt.datetime.today().strftime(format="%Y-%m-%d")
        startDate = (dt.datetime.today() - dt.timedelta(days=365)).strftime(format="%Y-%m-%d")

    try:
        # The standard database that we want to use for our test cases
        # Please note: the database does not have data beyond 2018-03-27, it will be swapped out in future versions
        database = "WIKI/PRICES"
        # Filtering the database by columns to only return the ticker, date, and close price for the dates greater than
        # or equal to the startDate and less than and equal to the endDate
        data = quandl.get_table(database, qopts = { 'columns': ['ticker', 'date', 'close'] },
                                    ticker = ticker, date = { 'gte': startDate, 'lte': endDate })
        data = data.pivot(index='date', columns='ticker', values='close')

    except: raise ImportError("Unable to Import test data, please try again.")

    else:

        print(f"...Data for {ticker} from {startDate} to {endDate} loaded successfully")

    return data

def risk_free_rate(startDate: str = None, endDate: str = None, **kwargs) -> pd.DataFrame:
    """The function returns the riskFreeRate for a given start and end date from Quandl.
    For now, the riskFreeRate is defined as the 3 Month US Treasury Bill Rate which is accessible
    through the database: "USTREASURY/YIELD.1"

    Parameters
    ----------
    startDate : str, optional
        Incase the user wants to supply a startDate to call data from a specific time period
        The format is "YYYY-MM-DD", by default None
    endDate : str, optional
        Incase the user wants to supply a endDate to call data from a specific time period
        The format is "YYYY-MM-DD", by default None

    Returns
    -------
    pd.DataFrame
        Returns a pandas dataframe object consisting of the called data for the riskFreeRate
    """


    # Both start and end dates must be provided else the call reverts to the default set of
    # endDate as today and startDate as a year back
    if not isinstance(startDate, str) or not isinstance(endDate, str):
        endDate = dt.datetime.today().strftime(format="%Y-%m-%d")
        startDate = (dt.datetime.today() - dt.timedelta(days=365)).strftime(format="%Y-%m-%d")

    try:
        # The standard database that we want to use for our test cases
        database = "USTREASURY/YIELD.3"

        data = quandl.get(database, start_date = startDate, end_date  = endDate)
        data.columns = ['riskFreeRate']

    except: raise ImportError("Unable to Import test data, please try again.")

    else:
        print(f"...Data for {database} from {startDate} to {endDate} loaded successfully")
        return data
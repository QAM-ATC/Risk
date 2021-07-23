from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from typing import Union
import pandas as pd

__all__ = [
    'regress'
]

def regress(endogenousSeries: pd.Series, exogenousSeries: Union[pd.Series, pd.DataFrame], method: str = 'OLS', **kwargs):
    """This function implements regression for a given set of endogeneous and exogeneous variables.
    Note: summary() function is not available for any method except 'OLS'

    Parameters
    ----------
    endogenousSeries : pd.Series
        Endogenous series for our regression
    exogenousSeries : Union[pd.Series, pd.DataFrame]
        Exogenous covariates for our regression
    method : str, optional
        Type of regression to be conducted
        Possible inputs include:
        1. OLS
        2. Ridge
        3. Lasso
        , by default 'OLS'

    Returns
    -------
    RegressionResults
        Returns a fitted instance of the regression model

    Raises
    ------
    NameError
        Incase an invalid method is selected, a NameError is raised
    """

    exogenousSeries = sm.add_constant(exogenousSeries)
    model = OLS(endog=endogenousSeries, exog=exogenousSeries, **kwargs)

    if method == 'OLS':
        results = model.fit()

    elif method == 'Ridge':
        results = model.fit_regularized(alpha=1, L1_wt=0, refit=True)

    elif method == 'Lasso':
        results = model.fit_regularized(alpha=1, L1_wt=1, refit=True)

    else:
        raise NameError(f"Invalid method '{method}'Please choose from 'OLS', 'Ridge', or 'Lasso'")

    return results
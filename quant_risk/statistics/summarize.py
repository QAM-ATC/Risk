from quant_risk.statistics import financial_ratios
from quant_risk.statistics import annualize
from quant_risk.statistics import stats
from quant_risk.statistics import VaR
import pandas as pd

__all__ = [
    'print_summary'
]

def print_summary(price: pd.Series, **kwargs):

    """
    This function returns a dataframe with the following characteristics:
    Financial Ratios:
    1. Sharpe ratio
    2. Sortino ratio
    3. Calmar ratio
    4. Omega ratio
    5. Tail Ratio

    Statistics:
    1. Skewness
    2. Kurtosis
    3. Stability
    4. Max Drawdown
    5. Cumulative returns

    Annualise:
    1. Returns
    2. Vol

    Value at Risk
    1. VaR
    2. cVaR
    """
    result = {}
    for ratio in financial_ratios.__all__ :
        result[ratio] = eval(f"financial_ratios.{ratio}(price, **kwargs)")

    returns = price.pct_change().dropna()

    for annual in annualize.__all__:
        result[annual] = eval(f"annualize.{annual}(returns, **kwargs)")

    for stat in stats.__all__[:-3]:
        result[stat] = eval(f"stats.{stat}(price, **kwargs)")

    for var in VaR.__all__:
        result[var] = eval(f"VaR.{var}(price, **kwargs)")

    result['cumulative_returns'] = result['cumulative_returns'].iloc[-1, :]

    cols = [
        'Sharpe ratio',
        'Calmar ratio',
        'Omega ratio',
        'Sortino ratio',
        'Tail Ratio',

        'Annualised Returns',
        'Annualised Volatility',

        'Skewness',
        'Kurtosis',
        'Stability',
        'Maximum Drawdown',
        'Cumulative Returns',

        'Value at Risk',
        'Conditional Value at Risk'
    ]

    result = pd.DataFrame.from_dict(result).T
    result.index = cols

    return result




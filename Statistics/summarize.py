from statistics import financial_ratios
from statistics import annualize
from statistics import statistics
from statistics import VaR
import pandas as pd

__all__ = [
    'print_summary'
]

def print_summary(price: pd.Series, **kwargs):

    # Fin ratios
    """
    1. Sharpe ratio
    2. Sortino ratio
    3. calmar ratio
    4. Omega ratio
    5. Tail Ratio

    Stats
    1. Skewness
    2. Kurtosis
    3. Stability
    4. Max Drawdown
    5. Cumulative returns

    Annualise
    1. Returns
    2. Vol

    VaR
    1. var
    2. cvar
    """

    result = {}
    for ratio in financial_ratios.__all__ :
        result[ratio] = eval(f"financial_ratios.{ratio}(price, **kwargs)")

    returns = price.pct_change().dropna()

    for annual in annualize.__all__:
        result[annual] = eval(f"annualize.{annual}(returns, **kwargs)")

    for stat in statistics.__all__:
        result[stat] = eval(f"statistics.{stat}(price, **kwargs)")

    for var in VaR.__all__:
        result[var] = eval(f"VaR.{var}(price, **kwargs)")

    result['cumulative_returns'] = result['cumulative_returns'].iloc[-1, :]

    return pd.DataFrame.from_dict(result).T




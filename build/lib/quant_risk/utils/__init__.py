import matplotlib.axes as ax
import matplotlib.pyplot as plt

plt.style.use('seaborn')
plt.rcParams['image.cmap'] = 'winter_r'
plt.rcParams['font.family'] = 'serif'

from quant_risk.utils import *

__all__ = [
    'backtest',
    'plot'
]


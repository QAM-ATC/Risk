import numpy as np
import pandas as pd
from typing import Union
import datetime as dt

class BacktestEngine:

    def __init__(self, dataframe):

        self.dataframe = dataframe
        self.portfolio = None

    def fit(self, start: Union[str, dt.datetime.timestamp()], end, weightsFunction, **kwargs):
        ...
        raise NotImplementedError("Will do it later")

    def summary(self):

        raise NotImplementedError("Will do it later")
        
        ...


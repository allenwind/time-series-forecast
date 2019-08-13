from functools import total_ordering

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

def time_series_train_test_split(x, y, train_size=0.7):
    pass

class LogisticTransfer:

    def __init__(self, C):
        self.C = C

    def fit_transform(self, series):
        return np.log(self.C/series - 1)

    def inverse_transform(self, series):
        return self.C / (1 + np.exp(series))

class SeriesMinMaxScaler:

    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit_transform(self, series):
        pass

    def inverse_transform(self, series):
        pass


@total_ordering
class Forecaster:

    def fit(self, x, y):
        pass

    def forecast(self, n_steps, interval):
        pass

    def predict(self, x):
        pass

    @property
    def error(self):
        pass
    
    def __eq__(self, other):
        return self.error == other.error

    def __gt__(self, other):
        return self.error > other.error


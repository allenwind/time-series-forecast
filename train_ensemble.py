import argparse

import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from tsforecast import train_val_split, SimpleScaler, eval_model, datasets
from tsforecast import find_time_series_degree, StationaryTransfer
from tsforecast import XGBForecaster
from tsforecast import time_series_move_lag, view_rolling_features, visualize_autocorrelation
from tsforecast.tsfeatures.utils import find_time_series_max_periodic

def test():
    fn = datasets[3]
    series = fn()
    size = find_time_series_max_periodic(series)
    size = 200

    # series = np.log(np.log(series))
    # series = np.diff(series)
    # visualize_autocorrelation(series, offset=1)
    # size = find_time_series_max_periodic(series)
    
    # k = find_time_series_degree(series)
    # sta = StationaryTransfer(k)
    # series = sta.fit_transform(series)

    scaler = SimpleScaler()
    series = scaler.fit_transform(series)

    s1, s2 = train_val_split(series, train_rate=0.7)

    m = XGBForecaster(size, booster="gbtree")
    m.fit(s1, epochs=300, batch_size=100, validation_series=s2)
    y_predict = m.predict(2*len(s2))
    y_predict = time_series_move_lag(y_predict, pad="mean")

    eval_model(m, s1, s2, y_predict, 20, fn.__name__, bound=True)

if __name__ == "__main__":
    test()

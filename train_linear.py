import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from tsforecast import train_val_split, SimpleScaler, eval_model, datasets
from tsforecast import find_time_series_degree, StationaryTransfer
from tsforecast import LinearForecaster
from tsforecast import time_series_move_lag

def test():
    fn = datasets[-1]
    series = fn()
    size = 200

    # k = find_time_series_degree(series)
    # sta = StationaryTransfer(k)
    # series = sta.fit_transform(series)

    scaler = SimpleScaler()
    series = scaler.fit_transform(series)
    s1, s2 = train_val_split(series, train_rate=0.7)

    m = LinearForecaster(size)
    m.fit(s1, epochs=300, batch_size=100, validation_series=s2)
    y_predict = m.predict(2*len(s2))
    y_predict = time_series_move_lag(y_predict, pad="mean")

    eval_model(m, s1, s2, y_predict, 20, fn.__name__, bound=True)

if __name__ == "__main__":
    test()

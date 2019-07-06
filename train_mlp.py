import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from tsforecast import train_val_split, SimpleScaler, eval_model, datasets
from tsforecast import find_time_series_degree, StationaryTransfer
from tsforecast import MLPForecaster
from tsforecast import time_series_move_lag

def test():
    fn = datasets[3]
    series = fn()
    size = 200

    scaler = SimpleScaler()
    series = scaler.fit_transform(series)
    s1, s2 = train_val_split(series, train_rate=0.7)

    m = MLPForecaster(size, with_norm=True)
    m.fit(s1, epochs=300, batch_size=100, validation_series=s2)
    y_predict = m.predict(2*len(s2))
    y_predict = time_series_move_lag(y_predict, pad="first")

    eval_model(m, s1, s2, y_predict, 20, fn.__name__, bound=False)

if __name__ == "__main__":
    test()

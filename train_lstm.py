import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from tsforecast import train_val_split, SimpleScaler, eval_model, datasets
from tsforecast import find_time_series_degree, StationaryTransfer
from tsforecast import LSTMForecaster

def test():
    fn = datasets[5]
    series = fn()
    size = 15

    scaler = SimpleScaler()
    series = scaler.fit_transform(series)
    s1, s2 = train_val_split(series, train_rate=0.7)

    m = LSTMForecaster(size)
    m.fit(s1, epochs=3, batch_size=1, validation_series=s2)
    y_predict = m.predict(2*len(s2))

    eval_model(m, s1, s2, y_predict, 20, fn.__name__)

if __name__ == "__main__":
    test()

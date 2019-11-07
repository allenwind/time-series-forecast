import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from tsforecast import train_val_split, SimpleScaler, eval_model, datasets
from tsforecast import find_time_series_degree, StationaryTransfer
from tsforecast import MLPForecaster

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

    eval_model(m, s1, s2, y_predict, 20, fn.__name__)

if __name__ == "__main__":
    test()

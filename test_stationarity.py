import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import *

from tsforecast import train_val_split, SimpleScaler, eval_model, datasets
from tsforecast import find_time_series_degree, StationaryTransfer
from tsforecast import XGBForecaster

def test():
    fn = datasets[5]
    series = fn()
    size = 200

    o = np.diff(np.log(series.copy()))
    p = PowerTransformer(method='yeo-johnson', standardize=False)
    # p = QuantileTransformer()
    series = p.fit_transform(series.reshape((-1, 1))).ravel()

    plt.subplot(311)
    plt.plot(o)
    plt.subplot(312)
    plt.plot(series)
    plt.subplot(313)
    plt.hist(series, bins=100)
    plt.show()

if __name__ == "__main__":
    test()

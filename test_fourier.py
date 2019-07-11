import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import *

from tsforecast import train_val_split, SimpleScaler, eval_model, datasets
from tsforecast import find_time_series_degree, StationaryTransfer
from tsforecast import XGBForecaster

def test():
    fn = datasets[3]
    series = fn(n=5)
    size = 200

    d = np.abs(np.fft.fft(series))

    plt.subplot(311)
    plt.plot(series)
    plt.subplot(312)
    plt.plot(d, "+")
    plt.axhline(d[0], color="r")
    plt.subplot(313)
    plt.hist(series, bins=100)
    plt.show()

if __name__ == "__main__":
    test()

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


for fn in reversed(datasets):
    series = fn()
    visualize_autocorrelation(series)


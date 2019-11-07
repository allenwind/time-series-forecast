import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tsforecast import TimeSeriesForecaster
from tsforecast import MLPModel
from tsforecast import plot_forecast
from tsutils.validation import time_series_train_test_split
from tsutils.scaler import SimpleScaler

df = pd.read_csv("dogfood-iops.csv")
series = df.iloc[:, 1].values

train_series, test_series = time_series_train_test_split(series, train_size=0.7)
scaler = SimpleScaler()
train_series = scaler.fit_transform(train_series)
test_series = scaler.transform(test_series)

window_size = 50
n_steps = len(test_series) + 50
model = MLPModel(window_size)
fr = TimeSeriesForecaster(model)
fr.fit(train_series, epochs=100, batch_size=50, validation_rate=0)
pred_series = fr.forecast(n_steps=n_steps)

plot_forecast(train_series, test_series, pred_series)
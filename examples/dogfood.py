import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tsforecast import TimeSeriesForecaster
from tsforecast import MLPModel
from tsforecast import plot_forecast
from tsforecast.validation import time_series_train_test_split
from tsforecast.scaler import SimpleScaler

df = pd.read_csv("monthly-sunspots.csv")
series = df.iloc[:, 1].values

train_series, test_series = time_series_train_test_split(series, train_size=0.7)
scaler = SimpleScaler()
train_series_t = scaler.fit_transform(train_series)

window_size = 60
n_steps = int(len(test_series) * 1.5)
model = MLPModel(window_size)
fr = TimeSeriesForecaster(model)
fr.fit(train_series_t, epochs=300, batch_size=50, validation_rate=0)
pred_series_t = fr.forecast(n_steps=n_steps)
pred_series = scaler.inverse_transform(pred_series_t)

plot_forecast(train_series, test_series, pred_series)
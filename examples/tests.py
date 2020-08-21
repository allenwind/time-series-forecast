import numpy as np
import matplotlib.pyplot as plt

from tsforecast import TimeSeriesForecaster
from tsforecast import MLPModel
from tsforecast import plot_forecast
from tsforecast.validation import time_series_train_test_split
from tsforecast.scaler import SimpleScaler

def _add_noise(y, add=True, multiply=False):
    u = np.max(y)
    if add:
        anoise = np.random.normal(0, u/20, size=len(y))
    else:
        anoise = 0
    
    if multiply:
        mnoise = np.random.normal(loc=0, scale=u/15, size=len(y))
    else:
        mnoise = 0
    return y * (1 + mnoise) + anoise

def multi_periodic_function(size=1000):
    x = np.linspace(0, 20*np.pi, size)
    y = np.log(x+1) + np.sqrt(x) + np.sin(x) + np.cos(2*x) + \
        1/3 * np.sin(6*x) + 1/4 * np.cos(10*x) + \
        1/5 * np.sin(15*x) + 1/5 * np.cos(14*x)
    return _add_noise(y, add=False, multiply=False)

series_size = 1000
window_size = 100
n_steps = 400
train_size = 0.8
# generate data
series = multi_periodic_function(size=series_size)

train_series, test_series = time_series_train_test_split(series, train_size)
scaler = SimpleScaler()
train_series_t = scaler.fit_transform(train_series)

model = MLPModel(window_size)
fr = TimeSeriesForecaster(model)
fr.fit(train_series_t, epochs=100, batch_size=50, validation_rate=0)
pred_series_t = fr.forecast(n_steps=n_steps)
pred_series = scaler.inverse_transform(pred_series_t) # 预测结果逆变换

plot_forecast(train_series, test_series, pred_series)
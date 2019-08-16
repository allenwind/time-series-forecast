import collections
import datetime
import glob

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


from .tsfeatures import extract_time_series_forecast_features
from .tsfeatures.utils import find_time_series_max_periodic
from .tsfeatures.autocorrelation import time_series_all_autocorrelation
from .errors import eval_errors

time_series_features_size = lambda x: len(extract_time_series_forecast_features(x))

# 时间序列处理有关的模块






def time_series_move_lag(series, lag=1, pad="first"):
    # 预测的滞后性，把预测结果往后移动 lag 个时间步，并使用 pad 进行填充
    # 理论上，滑动窗口的预测都有这个情况，可以理解为模型为了最优化，选择和
    # 最近一个时间步相近的取值。

    if pad == "first":
        v = series[0]
    elif pad == "last":
        v = series[0]
    elif pad == "mean":
        v = np.mean(series)

    values = [v] * lag
    values.extend(series.tolist())
    return np.array(values)

def time_series_exponential_decay(series):
    # wiki:
    # https://en.wikipedia.org/wiki/Exponential_decay
    pass

def find_best_window_size(series):
    # 确定最佳的滑动窗口大小
    # 该实现是经验方法, 经验方法
    # 也可以通过自相关的方式计算最佳值
    # 但是计算十分耗时
    # 此外, 还可以使用傅里叶方法
    # 详细见 fourier.py 模块
    return find_time_series_max_periodic(series)

def find_max_autocorrelation_lag(series):
    # 计算最大自相关系数的 lag
    # wiki:
    # https://en.wikipedia.org/wiki/Autocorrelation

    return find_time_series_max_periodic(series)



def timestamp2datetime(ts):
    return datetime.datetime.fromtimestamp(int(ts))

def timestamps2datetimes(mts):
    return [timestamp2datetime(ts) for ts in mts]


    
def rolling_time_series(series, window_size, slide_size):
    length = len(series)
    step = 0

    while step * slide_size + window_size < length:
        begin = step * slide_size
        end = begin + window_size
        step += 1
        yield series[begin:end]

def sliding_window(series, size, step=1):
    # 把滑动窗口直接转换成矩阵形式
    w = np.hstack([series[i:i+i-size or None:step] for i in range(size)])
    return w.reshape((-1, size))

def sliding_window_2d(X, size, step=1):
    # 把滑动窗口直接转换成矩阵形式
    n_features = X.shape[1]
    w = np.array([X[i:i+i-size or None:step, :] for i in range(size)])
    return w.reshape((-1, size))

_row = lambda x: x

def series2X(series, size, func=_row):
    # 把时间序列转换为滑动窗口形式
    X = np.array([series[i:i+size] for i in range(len(series)-size+1)])
    return np.apply_along_axis(func, 1, X)

def series2Xy(series, size, func=_row):
    # 把时间序列转换为单步带标注形式数据
    X = np.array([series[:-1][i:i+size] for i in range(len(series)-size)])
    y = np.array(series[size:])
    return np.apply_along_axis(func, 1, X), y

class E2ETransfer:

    # 端到端的数据转换, 包括:
    # 数据变换, 逆变换, 滤波
    # 离群点处理

    def fit(self, series):
        pass

    def fit_transform(self, series):
        pass

    def inverse_transform(self, series):
        pass

class SimpleScaler:

    EPS = 0.1
    
    def __init__(self, type="std"):
        if type == "std":
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler(feature_range=(self.EPS, 1+self.EPS))

    def fit(self, series):
        self.scaler.fit(series.reshape((-1, 1)))
    
    def fit_transform(self, series):
        return self.scaler.fit_transform(series.reshape((-1, 1))).ravel()
    
    def inverse_transform(self, series):
        return self.scaler.inverse_transform(series.reshape((-1, 1))).ravel()

class lazyproperty:
    
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value





class PowerTransfer:

    # 把时间序列变换为正太分布
    
    def __init__(self):
        pass

    def fit(self):
        return self

    def fit_transform(self, series):
        pass

    def inverse_transform(self, series):
        pass

class Rolling:

    # 时间序列预测的滑动窗口
    
    def __init__(self, size):
        self.size = size
        self.window = collections.deque(maxlen=size)
    
    def update(self, value):
        self.window.append(value)

    def updates(self, values):
        if isinstance(values, np.ndarray):
            values = values.tolist()
        self.window.extend(values)

    def slide(self, func=None):
        if func is None:
            return np.array(self.window)
        return func(np.array(self.window)) 

class FeaturesRolling(Rolling):

    # 时间序列预测中特征提取的滑动窗口

    def update_features(self, func):
        pass

    def _compute(self, arr):
        return extract_time_series_forecast_features(arr)

    def slide(self):
        return self._compute(super().slide())

    @lazyproperty
    def support_features_size(self):
        # 只计算一次
        dummy = np.random.uniform(size=self.size)
        return len(extract_time_series_forecast_features(dummy))

class TimeSeriesTransfer:

    # 把时间序列转换为带标注的训练数据
    
    def __init__(self, values):
        self.values = values
        self.size = len(values)

    def transform(self, window_size):
        X = []
        y = []

        for idx in range(self.size-window_size):
            X.append(self.values[idx:idx+window_size])
            y.append(self.values[idx+window_size])
        return np.array(X), np.array(y)

class FeaturesTimeSeriesTransfer(TimeSeriesTransfer):

    # 把时间序列转换为带特征和标注的训练数据

    def transform_features(self, window_size):
        X = []
        y = []
        for idx in range(self.size-window_size):
            raw = self.values[idx:idx+window_size]
            X.append(extract_time_series_forecast_features(raw))
            y.append(self.values[idx+window_size])
        return np.array(X), np.array(y)

import collections
import datetime
import glob
import io

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller

from .tsfeatures import extract_time_series_forecast_features
from .tsfeatures import find_time_series_max_periodic
from .tsfeatures.autocorrelation import time_series_all_autocorrelation
from .errors import eval_errors

time_series_features_size = lambda x: len(extract_time_series_forecast_features(x))

# 时间序列处理有关的模块

def check_time_series(series):
    if not isinstance(series, np.ndarray):
        raise ValueError("time series invalidation")

def check_white_noise(series):
    # 检测时间序列或残差是否为白噪声
    # wiki:
    # https://en.wikipedia.org/wiki/White_noise
    pass

def check_random_walk(series):
    # 检测时间序列是否为随机游走
    pass

def find_time_series_degree(series, threshold=0.05):
    # 计算时间序列的阶
    # wiki
    # https://en.wikipedia.org/wiki/Dickey%E2%80%93Fuller_test

    n = 0
    adf = adfuller(series)
    while adf[1] > threshold:
        n += 1
        series = np.diff(series)
        adf = adfuller(series)
    return n

def view_rolling_features(series, size):
    transfer = FeaturesTimeSeriesTransfer(series)
    X, y = transfer.transform_features(size)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    plt.subplot(211)
    plt.plot(series)
    plt.subplot(212)
    plt.imshow(X.T)
    plt.xticks([])
    plt.yticks([])
    plt.show()

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

def visualize_autocorrelation(series, offset=0):
    auto = np.array(time_series_all_autocorrelation(series))
    plt.subplot(211)
    plt.plot(series)
    plt.subplot(212)
    plt.plot(auto[offset:])
    plt.show()

def timestamp2datetime(ts):
    return datetime.datetime.fromtimestamp(int(ts))

def timestamps2datetimes(mts):
    return [timestamp2datetime(ts) for ts in mts]

def find_best_model(filepath="weights"):
    files = glob.glob("weights/*.hdf5".format(filepath))
    return min(files, key=lambda r: float(r.split("-")[-1].split(".hdf5")[0]))

def train_val_split(series, train_rate=0.7, test_rate=0.7):
    # 训练与检验集的分离
    # TODO 交叉检验

    idx1 = int(train_rate * len(series))
    idx2 = int(test_rate * len(series))
    s1 = series[:idx1+1]
    s2 = series[idx1:]
    return s1, s2
    
def rolling_time_series(series, window_size, slide_size):
    length = len(series)
    step = 0

    while step * slide_size + window_size < length:
        begin = step * slide_size
        end = begin + window_size
        step += 1
        yield series[begin:end]

def sliding_window(series, size, step):
    w = np.hstack([series[i:i+i-size or None:step] for i in range(size)])
    return w.reshape((-1, size))

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

    # 需要注意在神经网络训练时，如果feature range 在 (0,1) 区间
    # 可能导致无法收敛，因为计算 loss 时出现 inf.

    EPS = 0.1
    
    def __init__(self):
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

class Pipeline:
    
    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, series):
        return self

    def fit_transform(self, series):
        for e in self.estimators:
            series = e.fit_transform(series)
        return series

    def inverse_transform(self, series):
        for e in reversed(self.estimators):
            series = e.inverse_transform(series)
        return series

class StationaryTransfer:

    # 平稳时间序列与非平稳时间序列的转换
    # 确定序列是否平稳可以通过 ACF
    # 非平稳化为平稳序列可以通过差分方法
    # 从平稳序列还原为源序列需要保留每次
    # 差分的序列的首值.
    
    # TODO 整合自动定阶方法

    def __init__(self, k=1, log=False):
        self.k = k
        self.log = log
        self._init_values = []
    
    def fit_transform(self, series):
        if self.log:
            series = np.log(series)

        # 迭代地执行高阶差分
        k = self.k
        while k:
            self._init_values.append(series[0])
            series = np.diff(series)
            k -= 1
        return series

    def inverse_transform(self, series):
        # 迭代地还原高阶差分
        k = self.k
        while k:
            k -= 1
            values = [self._init_values[k]]
            #values.extend(series.tolist()) # 损失精度
            values = np.append(values, series)
            values = np.cumsum(values)
            series = values
        
        if self.log:
            values = np.exp(values)
        return values

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

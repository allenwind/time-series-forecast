import numpy as np

from ._dummy import _raw

__all__ = ["series2X", "series2Xy", "series2d2X", "series2d2Xy", "TimeSeriesLabelizer"]

def series2X(series, size, func=_raw):
    # 把时间序列转换为滑动窗口形式
    X = np.array([series[i:i+size] for i in range(len(series)-size+1)])
    return np.apply_along_axis(func, 1, X)

def series2Xy(series, size, func=_raw):
    # 把时间序列转换为单步带标注形式数据
    X = np.array([series[:-1][i:i+size] for i in range(len(series)-size)])
    y = np.array(series[size:])
    return np.apply_along_axis(func, 1, X), y

def series2d2X(series2d, size):
    # 把多维时间序列转换为滑动窗口形式
    return np.array([series2d[i:i+size,:] for i in range(len(series2d)-size+1)])

def series2d2Xy(series2d, size):
    # 把多维时间序列转换为单步带标注形式数据
    X = np.array([series2d[:-1][i:i+size,:] for i in range(len(series2d)-size)])
    y = np.array(series2d[size:,:])
    return X, y

class TimeSeriesLabelizer:

    #  把时序标签化

    def __init__(self, size, func=_raw):
        self.size = size
        self.func = func

    def fit(self, X):
        return self

    def fit_transform(self, X):
        if np.ndim(X) == 1:
            return series2Xy(X, self.size, self.func)
        else:
            return series2d2Xy(X, self.size)

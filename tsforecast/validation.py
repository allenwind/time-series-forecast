import numpy as np

# 由于时间序列的位置特征，导致交叉验证的思路不能 shuffle -> split
# 否则存在数据泄漏风险.
# 交叉验证思路：
# 1. labeling -> split
# 2. split -> labeling
# 第一种方法有重叠的时间步
# 第二种方法则是完全的分离

def train_test_split(X, y, train_size=0.7, test_size=0.3):
    idx = int(train_size * len(y))

    X_train = X[:idx, :]
    y_train = y[:idx]
    X_test = X[idx:, :]
    y_test = y[idx:]
    return (X_train, y_train), (X_test, y_test)

def train_val_test_split(X, y, train_size=0.7, val_size=0.1, test_size=0.2):
    t_idx = int(train_size * len(y))
    v_idx = int((train_size+val_size) * len(y))

    X_train = X[:t_idx, :]
    y_train = y[:t_idx]
    X_val = X[t_idx:v_idx, :]
    y_val = y[t_idx:v_idx]
    X_test = X[v_idx:, :]
    y_test = y[v_idx:]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def time_series_train_test_split(series, train_size=0.7, test_size=0.3):
    idx = int(train_size * len(series))

    series_train = series[:idx]
    series_test = series[idx:]
    return series_train, series_test

def time_series_train_val_test_split(series, train_size=0.7, val_size=0.1, test_size=0.2):
    t_idx = int(train_size * len(series))
    v_idx = int((train_size+val_size) * len(series))

    series_train = series[:t_idx]
    series_val = series[t_idx:v_idx]
    series_test = series[v_idx:]
    return series_train, series_val, series_test


import argparse
import glob
import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy import signal
from dateutil import parser as dparser

from .utils import Rolling

# np.random.seed(2**32 - 26)

# 用于测试模型效果的数据集, 包括
# 真实数据和模拟数据. 模拟数据包括
# 如下成分:
# 线性增长 非线性增长 周期 多周期 复杂周期 噪声 伪随机数模拟(logistic chaos)

def _add_noise(y, add=True, multiply=False):
    # 给时序添加噪声
    # 包括: 加成和乘成噪声

    u = np.max(y)
    if add:
        anoise = np.random.normal(0, u/15, size=len(y))
    else:
        anoise = 0
    
    if multiply:
        mnoise = np.random.normal(loc=0, scale=u/15, size=len(y))
    else:
        mnoise = 0
    return y * (1 + mnoise) + anoise

def _add_anomaly(y, ans=10):
    size = len(y)
    u = np.max(y)
    l = np.min(y)

    a = np.zeros(size)
    for _ in range(ans):
        i = np.random.randint(low=0, high=size-1, size=1)
        v = random.choice([u, l])
        a[i] = 3 * v
    return y + a

def _add_trend(y, shape="s", add=True, multiply=False):
    # shape in ("s", "linear", "log", "exp", "square", "sqrt")
    max = np.max(y)
    min = np.min(y)
    if min < 0:
        min = 0.001
        max += min
    x = np.linspace(min, max, len(y))

    if shape == "linear":
        r = x
    if shape == "log":
        r = np.log(x)
    if shape == "square":
        r = np.square(x)
    if shape == "sqrt":
        r = np.sqrt(x)
    if shape == "exp":
        r = np.exp(x)
    if shape == "s":
        r = max * np.exp(x) / (1 + np.exp(x))
    
    if add:
        return y + r
    if multiply:
        return y * r

def trending_function(size=1000):
    x = np.linspace(0, 10, size)
    y = _add_trend(x, shape="sqrt")
    return _add_noise(y)
    
def non_linear_function(size=1000):
    # non-linear function
    # 周期 + 非线性增长

    x = np.linspace(0, 30*np.pi, size)
    y = 1 + x + np.sqrt(x) * np.random.randn(len(x)) + x ** np.sin(x) + np.cos(x)
    return y

def multi_periodic_function(size=1000):
    # multi periodic function
    # 多周期 + 非线性增长

    x = np.linspace(0, 20*np.pi, size)
    y = np.log(x+1) + np.sqrt(x) + np.sin(x) + np.cos(2*x) + 1/3 * np.sin(6*x) + 1/4 * np.cos(10*x) + \
        1/5 * np.sin(15*x) + 1/5 * np.cos(14*x)
    return _add_noise(y, add=True, multiply=False)

def random_fourier_series(size=1000, n=50):
    # random fourier series
    # 随机傅里叶序列, 用于模拟复杂的多周期时序数据
    # 傅里叶级数的系数是随机生成的, 服从均匀分布
    # 更多的相关理论可以参考
    # wiki: https://en.wikipedia.org/wiki/Lacunary_function#Lacunary_trigonometric_series
    # wiki: https://en.wikipedia.org/wiki/Gaussian_process#Continuity
    # 返回的序列中包括加成和乘成噪声

    x = np.linspace(0, 10*np.pi, size)
    an = np.random.normal(0, 1, n)
    bn = np.random.normal(0, 1, n)

    s = 0
    y = 0
    for a, b in zip(an, bn):
        s += 1
        y += a * np.cos(s*x) + b * np.sin(s*x)
    return _add_noise(y, add=False, multiply=False)

def random_fourier_series_with_change_phase(size=1000, n=50):
    pass

def chaos_series(size=1000):
    # chaos r = 3.88
    # 生成混沌序列, 这种方法可以生成伪随机数, 主要用于测试神经网络能否把
    # 非线性的递推关系学习下来. 相关理论可参考
    # wiki: https://en.wikipedia.org/wiki/Logistic_map#Behavior_dependent_on_r

    r = 3.88
    x0 = 0.8989
    y = []
    for i in range(size):
        x = r * x0 * (1-x0)
        y.append(x)
        x0 = x
    return np.array(y)

def repeat_random_series(size=1000):
    s1 = np.random.normal(loc=0, scale=1, size=25)
    s2 = np.random.uniform(low=-2, high=0, size=25)
    s3 = np.random.uniform(low=-1, high=1, size=25)
    s4 = np.random.normal(loc=1, scale=2, size=25)
    series = []
    series.extend(s1)
    series.extend(s2)
    series.extend(s4)
    series.extend(s3)
    series = np.array(series)
    series = np.tile(series, size//len(series))
    return _add_trend(_add_noise(series, add=True), shape="log")

def autoregression_series(size=1000, p=5):
    ws = np.random.normal(size=p) # 生成随机权重
    v0 = np.random.uniform(size=p) # 初始化序列
    r = Rolling(size=p)
    r.updates(v0)

    values = v0.tolist()
    for _ in range(size-p):
        s = r.slide()
        v = np.sum(np.array(s) * ws)
        values.append(v)
        r.update(v)
    return np.array(values)

def airline_passengers(size=1000, resample=False):
    # 真实数据, 国际航空航班数据
    # airline-passengers
    # 由于数据只有一百多个, 对于神经网络训练来说并不够
    # 因此, 采用数字信号处理中的傅里叶方法进行重抽样以扩大数据集

    df = pd.read_csv('asset/internet/example_air_passengers.csv', usecols=[1], engine='python')
    values = df.values.ravel()
    if resample:
        return signal.resample(values, size)[:-10]
    return values

def dogfood_cpu(size=1000):
    # dogfood 集群 CPU 使用情况

    df = pd.read_csv("asset/dogfood/dogfood_cluster_cpu_total_hz.csv")
    values = df.values[:, 1:]
    scaler = MinMaxScaler((0,1))
    values = scaler.fit_transform(values).ravel()
    return values

def dogfood_memory(size=1000):
    # dogfood 集群 memory 使用情况

    df = pd.read_csv("asset/dogfood/dogfood_cluster_memory_used_bytes.csv")
    values = df.values[:, 1:]
    scaler = MinMaxScaler((0,1))
    values = scaler.fit_transform(values).ravel()
    return values

def dogfood_disk(size=1000):
    # dogfood 集群 disk 使用情况

    df = pd.read_csv("asset/dogfood/dogfood_zbs_cluster_provisioned_data_space_bytes.csv")
    values = df.values[:, 1:]
    scaler = MinMaxScaler((0,1))
    values = scaler.fit_transform(values).ravel()
    return values

def dogfood_vm(size=1000):
    # dogfood 中某 VM CPU 使用情况

    df = pd.read_csv("asset/dogfood/csv/dogfood_vm.csv")
    values = df.values[:, 1:]
    scaler = MinMaxScaler((0,1))
    values = scaler.fit_transform(values).ravel()
    return values

def monthly_sunspots(size=1000):
    # 真实数据 monthly sunspots

    df = pd.read_csv("asset/internet/monthly-sunspots.csv")
    values = df.values[:, 1:]
    scaler = MinMaxScaler((0,1))
    values = scaler.fit_transform(values).ravel()
    return values[-size:]

def to_dataframe(series):
    size = len(series)
    columns = ["ds", "y"]
    ds = pd.date_range("2010-01-01", periods=size, freq="H") # use fake date
    return pd.DataFrame(data=[(i,j) for i,j in zip(ds, series)], columns=columns)

def load_data_by_keyword(kw):
    files = glob.glob("**/*.csv", recursive=True)
    for file in files:
        if kw in file:
            return pd.read_csv(file)
    raise ValueError(kw)

def list_datasets():
    ds = [trending_function, # 0
          non_linear_function, # 1
          multi_periodic_function, # 2 
          random_fourier_series, # 3
          chaos_series, # 4
          airline_passengers, # 5
          monthly_sunspots, # 6
          dogfood_vm,
          dogfood_cpu,
          dogfood_disk,
          dogfood_memory,
          repeat_random_series,
          autoregression_series,
          ] 
    return ds

def show_all_funcs():
    plt.figure(figsize=(16, 8))
    datasets =list_datasets()

    size = len(datasets)
    for i in range(size):
        idx = i + 1
        fig = plt.subplot(size, 1, idx)
        
        func = datasets[i]
        fig.plot(func(), label=func.__name__)

        plt.xticks([])
        plt.yticks([])
        box = fig.get_position()
        fig.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        plt.legend(loc="upper left", bbox_to_anchor=(1, 0.5))

    plt.suptitle("time series datasets")
    plt.show()

def view_csv_data(path):
    if os.path.isdir(path):
        files = glob.glob("{}/*.csv".format(path))
    else:
        files = [path]

    for file in files:
        df = pd.read_csv(file)
        ts = df.values[:, 0]
        ts = [dparser.parse(t) for t in ts]
        vs = df.values[:, 1]
        plt.plot(ts, vs, label=file)
        # plt.legend(loc="lower right")
        plt.title(file)
        plt.show()

def view_autocorrelation(series, max_lag=20):
    for lag in range(1, max_lag+1):
        plt.subplot(4, 5, lag)
        sx = series[:-lag*10]
        sy = series[lag*10:]
        plt.plot(sx, sy, "+")
    plt.show()

datasets = list_datasets()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=".")
    args = parser.parse_args()
    view_csv_data(args.path)

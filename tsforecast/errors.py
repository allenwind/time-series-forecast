import numpy as np

# 误差估计, 需要预测本身是无偏预测

def compute_mape_error(y_true, y_fore):
    # 计算 MAPE 误差
    # 计算可能会出现 inf
    return 100 * np.mean(np.abs(y_true-y_fore) / y_true)

def eval_errors(y_val, y_pred):
    y_val_size = len(y_val)
    res = y_pred[:y_val_size] - y_val

    mean = np.mean(res)
    std = np.std(res)

    # 2 sigma 区间
    y2u = y_pred + mean + 2 * std
    y2l = y_pred + mean - 2 * std

    # 3 sigma 区间
    y3u = y_pred + mean + 3 * std
    y3l = y_pred + mean - 3 * std

    return (y2u, y2l), (y3u, y3l)

def find_anomaly(y_val, y_pred, k=3):
    y_val_size = len(y_val)
    res = y_pred[:y_val_size] - y_val

    mean = np.mean(res)
    std = np.std(res)

    # k sigma 区间
    y2u = y_pred + mean + k * std
    y2l = y_pred + mean - k * std

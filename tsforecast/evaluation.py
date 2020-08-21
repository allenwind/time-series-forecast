import matplotlib.pyplot as plt
import numpy as np

def eval_errors(y_true, y_pred):
    res = y_pred[:len(y_true)] - y_true
    mean = np.mean(res)
    std = np.std(res)

    # 2 sigma 区间
    y2u = y_pred + mean + 2 * std
    y2l = y_pred + mean - 2 * std

    # 3 sigma 区间
    y3u = y_pred + mean + 3 * std
    y3l = y_pred + mean - 3 * std
    return (y2u, y2l), (y3u, y3l)

def plot_forecast(series, val_series, pred_series):
    size = len(series)
    vsize = len(val_series)
    n_steps = len(pred_series)
    idx = np.arange(size+n_steps)

    plt.figure(figsize=(16,8))
    fig = plt.subplot(111)
    fig.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    fig.axvline(idx[size], linestyle="dotted")
    fig.plot(idx[:size], series, label="train series", color="blue", alpha=0.8)
    fig.plot(idx[size:size+vsize], val_series, label="test series", color="green", alpha=0.8)
    fig.plot(idx[size:], pred_series, label="forecast series", color="red")

    (y2u, y2l), (y3u, y3l) = eval_errors(val_series, pred_series)
    fig.fill_between(idx[size:], y2l, y2u, color="#0072B2", alpha=0.5, label="2-sigma")
    fig.fill_between(idx[size:], y3l, y3u, color="#0072B2", alpha=0.2, label="3-sigma")

    box = fig.get_position()
    fig.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    fig.legend(loc="upper left", bbox_to_anchor=(1, 0.5))
    plt.show()


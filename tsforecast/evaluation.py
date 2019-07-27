import matplotlib.pyplot as plt
import numpy as np

from .errors import eval_errors, compute_mape_error

# 模型的评估和预测可视化

def eval_model(model, series, val_series, forecast_series, size, dataset="", bound=True):
    # TODO
    # draw res
    # hist unbias

    series_size = len(series)
    val_series_size = len(val_series)
    forecast_series_size = len(forecast_series)
    size = series_size + max(val_series_size, forecast_series_size)
    idx = np.arange(size)

    mape  = compute_mape_error(val_series, series[:val_series_size])

    plt.figure(figsize=(16,8))

    fig = plt.subplot(211)
    fig.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    fig.axvline(idx[series_size], linestyle="dotted")

    fig.plot(idx[:series_size], series, label="train series", color="blue", alpha=0.8)
    fig.plot(idx[series_size:series_size+val_series_size], val_series, label="validate series", color="green", alpha=0.8)
    
    forecast_idx = idx[series_size:series_size+forecast_series_size]
    fig.plot(forecast_idx, forecast_series, label="forecast series", color="red")

    if bound:
        (y2u, y2l), (y3u, y3l) = eval_errors(val_series, forecast_series)
        fig.fill_between(forecast_idx, y2l, y2u, color="#0072B2", alpha=0.5, label="2-sigma")
        fig.fill_between(forecast_idx, y3l, y3u, color="#0072B2", alpha=0.2, label="3-sigma")

    box = fig.get_position()
    fig.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    fig.legend(loc="upper left", bbox_to_anchor=(1, 0.5))

    if hasattr(model, "history"):
        fig = plt.subplot(212)
        fig.plot(model.history.history["loss"], color="blue", label="train loss")
        fig.plot(model.history.history["val_loss"], color="red", label="val loss")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        box = fig.get_position()
        fig.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        fig.legend(loc="upper left", bbox_to_anchor=(1, 0.5))

    elif hasattr(model, "evals_result"):
        fig = plt.subplot(212)
        results = model.evals_result()
        plt.plot(results["validation_0"]["error"], "b", label="train")
        plt.plot(results["validation_1"]["error"], "r", label="valid")
        plt.legend(loc="best")
        plt.title("xgboost learning curve")
        if show:
            plt.show()
    else:
        fig = plt.subplot(212)
        plt.text(0.5, 0.5, "no history found", size=20, ha="left", va="center", alpha=0.5)

    if dataset:
        plt.suptitle("dataset:{} mape:{:.2f}".format(dataset, mape))
    else:
        plt.suptitle("eval model")
    plt.show()



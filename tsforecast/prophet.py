import pandas
from fbprophet import Prophet
import matplotlib.pyplot as plt

from .dataset import datasets, to_dataframe, load_data_by_keyword
from .base import Forecaster

# Facebook 开源的时间序列建模预测工具 Prophet
# 它可以把握数据中的整体趋势. 但在复杂的时序中,
# 对于 end-to-end 的神经网络来说, Prophet 并
# 没有多大的优势. 当然, 我们也会把它整合到 Combination

# 尝试用 Propeht 学习时序中的整体趋势, 接着用神经网络
# 学习残差

# TODO 改写成 sklearn API

class ProphetForecaster(Forecaster, Prophet):
    pass

def test_fbprophet():
    ds = [load_data_by_keyword("retail")]
    for df in ds:
        #series = fn()
        #df = to_dataframe(series)

        #m = Prophet(changepoint_prior_scale=0.01, weekly_seasonality=False)
        #m.add_seasonality(name='hourly', period=100, fourier_order=5)
        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=2000) # freq="H"
        forecast = m.predict(future)
        m.plot(forecast)
        # plt.title(fn.__name__)
        plt.show()

if __name__ == "__main__":
    test_fbprophet()

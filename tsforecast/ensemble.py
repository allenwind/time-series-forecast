from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from xgboost import plot_importance
from xgboost.callback import early_stop
import matplotlib.pyplot as plt

from .linear import LinearForecaster
from .utils import FeaturesRolling

# 使用集成模型+特征提取预测平稳时序
# 然后把平稳时序逆变换为非平稳时序(原时序数据)

class BoostingLinearForecaster(LinearForecaster):

    def __init__(self, size):
        self.size = size
        self.roller = FeaturesRolling(size)
        self.model = AdaBoostRegressor(LinearRegression())

class BaggingLinearForecaster(LinearForecaster):

    def __init__(self, size):
        self.size = size
        self.roller = FeaturesRolling(size)
        self.model = BaggingRegressor(LinearRegression())

class XGBForecaster(LinearForecaster):

    # 目前这个模型在使用 booster="gbtree" 只能对平稳序列进行建模
    # 因为 tree 类模型无法进行"外推". 但可以使用 booster="gblinear".
    # 但是, 预测能力大打折扣. 解决方法有两种:
    # 1. 时间序列转化为平稳后, 再建模预测, 然后对预测结果变换到原序列.
    # 2. 使用 time series decomposition, 把趋势类分解出来.

    # 非平稳时序的类型:
    # 1. scale 变
    # 2. location 变
    # 3. scale location 同时变

    # TODO
    # 支持 Trend stationarity 时序预测

    def __init__(self, size, booster="gbtree"):
        self.size = size
        self.roller = FeaturesRolling(size)
        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=10,
            objective='reg:squarederror',
            booster=booster,
            n_jobs=-1,
        )

    def fit(self, series, epochs=None, batch_size=None, validation_series=None):
        transfer = FeaturesTimeSeriesTransfer(series)

        # 初始化预测窗口
        self._init_roller(*transfer.transform(self.size))

        X, y = transfer.transform_features(self.size)
        callbacks = [early_stop(stopping_rounds=20)]
        self.model.fit(X, y, callbacks=callbacks)

    def plot_features(self):
        plot_importance(self.model)
        plt.show()

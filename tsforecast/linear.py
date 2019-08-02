import functools

import numpy as np
import scipy.optimize as optimize
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from keras.metrics import mean_squared_error
from keras.layers import Input, Dense
from keras.models import Model
from keras.losses import mean_absolute_percentage_error
from keras.regularizers import l2

from .utils import FeaturesRolling, FeaturesTimeSeriesTransfer, Rolling
from .base import Forecaster
from .mlp import MLPForecaster

# 以特征为基础的线性模型

class RoughLinearForecaster(Forecaster):

    # 比较粗暴的线性模型, 但能够直接学习到时序中的增长模式
    # 理论上是一种基于时间外推的预测方法

    def __init__(self, n_steps=None):
        self.n_steps = n_steps
        self.model = LinearRegression()

    def fit(self, series, epochs=None, batch_size=None, validation_series=None):
        X = np.arange(len(series)).reshape((-1, 1))
        y = series
        self._last = len(y)
        self.model.fit(X, y)

    def predict(self, forecast_size, post_fit=False):
        X = np.arange(self._last, self._last + forecast_size).reshape((-1, 1))
        return self.model.predict(X).ravel()

class LogLinearForecaster(Forecaster):

    def __init__(self, n_steps=None):
        self.n_steps = n_steps
        self.model = LinearRegression()

    def fit(self, series, epochs=None, batch_size=None, validation_series=None):
        X = np.arange(len(series)).reshape((-1, 1))
        y = np.log(series)
        self._last = len(y)
        self.model.fit(X, y)

    def predict(self, forecast_size, post_fit=False):
        X = np.arange(self._last, self._last + forecast_size).reshape((-1, 1))
        return np.exp(self.model.predict(X).ravel())


def _logistic_growth(t, k, m, C):
    # $$ g ( t ) = \frac { C } { 1 + \exp ( - k ( t - m ) ) } $$
    return C / (1 + np.exp(-k * (t - m)))

class LogisticForecaster(Forecaster):
    
    def __init__(self, C, n_steps=None):
        self.func = functools.partial(_logistic_growth, C=C)
    
    def fit(self, series, epochs=None, batch_size=None, validation_series=None):
        ts = np.array(1, len(series)+1)
        self._last = len(series) + 1
        params, _ = optimize.curve_fit(self.func, ts, series)
        self.params = params

    def predict(self, forecast_size, post_fit=False):
        ts = np.arange(self._last, self._last+forecast_size)
        y_pred = self.func(ts, *self.params)
        return y_pred

class LinearForecaster(Forecaster):

    # 添加人工特征的线性模型, 默认使用 L2 正则化避免过拟合

    def __init__(self, size, l1=None, l2=0.5):
        self.size = size
        if l1 is None and l2 is None:
            # not regularization
            self.model = LinearRegression()
        elif l1 is None and l2 is not None:
            # only L2 regularization
            self.model = Ridge(l2)
        elif l1 is not None and l2 is None:
            # only L1 regularization
            self.model =  Lasso(l1)
        else:
            # L1, L2 regularization
            self.model = ElasticNet(l2, l1)

        self.roller = FeaturesRolling(size)

    def fit(self, series, epochs=None, batch_size=None, validation_series=None):
        transfer = FeaturesTimeSeriesTransfer(series)
        self._init_roller(*transfer.transform(self.size))
        X, y = transfer.transform_features(self.size)
        self.model.fit(X, y)

    def predict(self, forecast_size, post_fit=False):
        y_predict = []
        for i in range(forecast_size):
            X = self.roller.slide().reshape((-1, self.roller.support_features_size))
            y = self.model.predict(X)

            if post_fit:
                self.model.fit(X, y)
            
            y = y.ravel()[0]
            y_predict.append(y)
            self.roller.update(y)

        return np.array(y_predict)

    def score(self, val_series):
        predict_series = self.predict(len(val_series))
        return mean_squared_error(val_series, predict_series)

    def _init_roller(self, X, y):
        self.roller.updates(X[-1])
        self.roller.update(y[-1])

class KLinearForecaster(MLPForecaster):
    
    # Keras 实现的线性模型

    def __init__(self, size):
        inputs = Input(shape=(size,))
        outputs = Dense(1, kernel_regularizer=l2(0.01))(inputs)
        model = Model(inputs, outputs)
        model.compile(optimizer="adam", loss=mean_absolute_percentage_error)
        
        self.size = size
        self.model = model
        self.roller = Rolling(size)

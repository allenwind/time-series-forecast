import numpy as np
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mean_absolute_percentage_error

from .utils import Rolling, TimeSeriesTransfer
from .base import Forecaster
from .advance import SaveBestModelOnMemory

# 循环神经网络时序预测
# 此外，可以通过堆叠或双向来增加
# 神经网络的容量

class LSTMForecaster(Forecaster):

    def __init__(self, n_steps, n_features=1, units=20):
        # 因为是单维度时间序列预测，所以 n_features = 1
        self.n_steps = size
        self.n_features = n_features
        self.units = units

        # 构造监督的 LSTM 神经网络
        inputs = Input(shape=(self.n_steps, self.n_features))
        x = LSTM(self.units=20, stateful=False)(inputs)
        outputs = Dense(1)(x)
        model = Model(inputs, outputs)
        model.compile(optimizer="adam", loss=mean_absolute_percentage_error)

        self.model = model
        self.roller = Rolling(self.n_steps)

    def fit(self, series, epochs, batch_size, validation_series=None, save_best=True):
        transfer = TimeSeriesTransfer(series)
        X, y = transfer.transform(self.n_steps)

        # 初始化预测的滑动窗口
        self._init_roller(X, y)

        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        if validation_series is not None:
            transfer = TimeSeriesTransfer(validation_series)
            X_val, y_val = transfer.transform(self.n_steps)
            X_val = np.reshape(X_val, (-1, self.n_steps, 1))
            validation_data = (X_val, y_val)
        else:
            validation_data = None

        if save_best:
            fn = SaveBestModelOnMemory()
            callbacks = [fn]
        else:
            callbacks = []

        self.history = self.model.fit(X, y, 
                                      epochs=epochs, 
                                      batch_size=batch_size, 
                                      validation_data=validation_data,
                                      callbacks=callbacks,
                                      shuffle=False)

    def predict(self, forecast_size, post_fit=False):
        y_predict = []
        for i in range(forecast_size):
            X = self.roller.slide().reshape((-1, self.n_steps, 1))
            y = self.model.predict(X)

            if post_fit:
                self.model.fit(X, y)
            
            y = y.ravel()[0]
            y_predict.append(y)
            self.roller.update(y)

        return np.array(y_predict)

    def scores(self, val_series):
        predict_series = self.predict(len(val_series))
        return mean_absolute_percentage_error(val_series, predict_series)

    def _init_roller(self, X, y):
        self.roller.updates(X[-1])
        self.roller.update(y[-1])

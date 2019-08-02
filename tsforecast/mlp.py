import numpy as np
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Layer
from keras.models import Model
from keras.losses import mean_absolute_percentage_error

from .utils import Rolling, TimeSeriesTransfer
from .base import Forecaster
from .advance import adam
from .advance import SpectralNormalization, WeightEMA, gelu
from .advance import SaveBestModelOnMemory, calculate_batch_size

# 前馈神经网络时序预测

class MLPForecaster(Forecaster):

    def __init__(self, size, with_norm=True, ema_up=False):
        inputs = Input(shape=(size,))
        if with_norm:
            x = SpectralNormalization(Dense(units=2*size-1, 
                                            activation=gelu,
                                            kernel_initializer="random_normal",
                                            bias_initializer="random_normal"))(inputs)
        else:
            x = Dense(units=2*size-1, 
                      activation=gelu,
                      kernel_initializer="random_normal",
                      bias_initializer="random_normal")(inputs)

        outputs = Dense(1, activation="relu")(x)
        model = Model(inputs, outputs)
        model.compile(optimizer=adam, loss=mean_absolute_percentage_error)

        self.size = size
        self.model = model
        self.ema_up = ema_up

        if self.ema_up:
            self.ema = WeightEMA(self.model)
            self.ema.inject()
        self.roller = Rolling(size)

    def fit(self, series, epochs, batch_size, validation_series=None, save_best=True):
        if self.ema_up:
            self.ema.reset_old_weights()

        self.save_best = save_best
        if save_best:
            # 把最佳权重存放到内存中
            fn = SaveBestModelOnMemory()
            callbacks = [fn]
        else:
            callbacks = []

        if not batch_size:
            batch_size = calculate_batch_size(series, vseries, self.size)

        transfer = TimeSeriesTransfer(series)
        X, y = transfer.transform(self.size)
        
        if validation_series is not None:
            transfer = TimeSeriesTransfer(validation_series)
            X_val, y_val = transfer.transform(self.size)
            validation_data = (X_val, y_val)
        else:
            validation_data = None

        self.history = self.model.fit(X, y, 
                                      epochs=epochs, 
                                      batch_size=batch_size, 
                                      validation_data=validation_data,
                                      callbacks=callbacks,
                                      shuffle=True)
        # 初始化预测的滑动窗口
        self._init_roller(X, y)
    
    def predict(self, forecast_size, post_fit=False):
        if self.ema_up:
            self.ema.apply_ema_weights()

        y_predict = []
        for i in range(forecast_size):
            X = self.roller.slide().reshape((-1, self.size))
            y = self.model.predict(X)

            if post_fit:
                self.model.fit(X, y, epochs=1)
            
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

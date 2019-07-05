import numpy as np
from keras.layers import Input, Dense
from keras.layers import Layer, Lambda
from keras.layers import BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.losses import mean_absolute_percentage_error
from keras import backend as K

from .utils import Rolling, TimeSeriesTransfer, find_best_model
from .base import Forecaster
from .advance import SpectralNormalization, WeightEMA, gelu
from .advance import adam, symmetric_mean_absolute_percentage_error
from .advance import SaveBestModelOnMemory

# 前馈神经网络时序预测

class MLPForecaster(Forecaster):

    def __init__(self, size, with_norm=True):
        inputs = Input(shape=(size,))
        x = BatchNormalization()(inputs)
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

        # x = Dense(size, activation="relu")(x)
        outputs = Dense(1, activation="relu")(x)
        model = Model(inputs, outputs)
        model.compile(optimizer=adam, loss=mean_absolute_percentage_error)

        self.size = size
        self.model = model
        # self.ema = WeightEMA(self.model)
        # self.ema.inject()
        self.roller = Rolling(size)

    def fit(self, series, epochs, batch_size, validation_series=None, save_best=True):
        # self.ema.reset_old_weights()

        self.save_best = save_best
        if save_best:
            filepath = "weights/{epoch:02d}-{val_loss:.2f}.hdf5"
            fn = ModelCheckpoint(filepath, save_best_only=True)
            # fn = SaveBestModelOnMemory()
            callbacks = [fn]
        else:
            callbacks = []

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

        self._init_roller(X, y)
    
    def predict(self, forecast_size, post_fit=False):
        # self.ema.apply_ema_weights()

        if self.save_best:
            self.model.load_weights(find_best_model("."))
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

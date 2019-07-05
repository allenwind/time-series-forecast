import numpy as np
from keras.layers import Input, Dense, LSTM
from keras.layers import Bidirectional, Dropout
from keras.models import Model
from keras.losses import mean_absolute_percentage_error

from .utils import Rolling, TimeSeriesTransfer
from .base import Forecaster
from .advance import SpectralNormalization

# 循环神经网络时序预测
# TODO change (n_steps, n_features)

class LSTMForecaster(Forecaster):

    def __init__(self, size, n_features=1):
        # input_shape (batch_size, timesteps, ndims)
        n_features = 1
        self.n_steps = size
        self.n_features = n_features

        inputs = Input(shape=(self.n_steps, self.n_features))
        x = LSTM(size)(inputs)
        outputs = Dense(1)(x)
        model = Model(inputs, outputs)
        model.compile(optimizer="adam", loss=mean_absolute_percentage_error)

        self.model = model
        self.roller = Rolling(self.n_steps)

    def fit(self, series, epochs, batch_size, validation_series=None):
        # validation_series only use as validation data set
        transfer = TimeSeriesTransfer(series)
        X, y = transfer.transform(self.n_steps)
        self._init_roller(X, y)

        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        if validation_series is not None:
            transfer = TimeSeriesTransfer(validation_series)
            X_val, y_val = transfer.transform(self.n_steps)
            X_val = np.reshape(X_val, (-1, self.n_steps, 1))
            validation_data = (X_val, y_val)
        else:
            validation_data = None

        self.history = self.model.fit(X, y, 
                                      epochs=epochs, 
                                      batch_size=batch_size, 
                                      validation_data=validation_data,
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

class StackedLSTMForecaster(LSTMForecaster):
    def __init__(self, size):
        n_features = 1
        self.n_steps = size
        self.n_features = n_features
        inputs = Input(shape=(self.n_steps, self.n_features))
        x = LSTM(size, dropout=0.5, return_sequences=True)(inputs)
        x = LSTM(size)(x)
        outputs = Dense(1)(x)
        model = Model(inputs, outputs)
        model.compile(optimizer="adam", loss=mean_absolute_percentage_error)

        self.size = size
        self.model = model
        self.roller = Rolling(size)

class BiLSTMForecaster(LSTMForecaster):
    def __init__(self, size):
        n_features = 1
        self.n_steps = size
        self.n_features = n_features
        inputs = Input(shape=(self.n_steps, self.n_features))
        x = Bidirectional(LSTM(size, activation="relu"))(inputs)
        outputs = Dense(1, activation="relu")(x)
        model = Model(inputs, outputs)
        model.compile(optimizer="adam", loss=mean_absolute_percentage_error)

        self.size = size
        self.model = model
        self.roller = Rolling(size)

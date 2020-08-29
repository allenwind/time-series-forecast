from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

from .base import ModelBase

# 内置预测模型

class MLPModel(ModelBase):

    # 两隐层的 MLP 模型

    def __init__(self, window_size):
        self._window_size = window_size
        self.build()

    def build(self):
        inputs = Input(shape=(self.window_size,))
        x = Dense(2*self.window_size+1, activation="relu")(inputs)
        x = Dense(self.window_size, activation="relu")(x)
        x = Dense(self.window_size, activation="relu")(x)
        outputs = Dense(1)(x)
        self.model = Model(inputs, outputs)
        self.model.compile(optimizer="adam", loss="mse")

    def fit(self, X, y, epochs, batch_size, validation_rate):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        y = self.model.predict(X)
        return y.ravel()[0]

class LinearModel(ModelBase):

    def __init__(self, window_size):
        self._window_size = window_size
        self.build()

    def build(self):
        inputs = Input(shape=(self.window_size,))
        x = Dense(self.window_size)(inputs)
        outputs = Dense(1)(x)
        self.model = Model(inputs, outputs)
        self.model.compile(optimizer="adam", loss="mse")

    def fit(self, X, y, epochs, batch_size, validation_rate=None):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        y = self.model.predict(X)
        return y.ravel()[0]

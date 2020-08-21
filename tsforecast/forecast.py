import numpy as np

from .rolling import TimeSeriesRolling
from .labeling import TimeSeriesLabelizer
from .base import ForecasterBase

class TimeSeriesForecaster(ForecasterBase):

    def __init__(self, model):
        self.model = model
        self.window_size = model.window_size
        self.roller = TimeSeriesRolling(self.window_size)
        self.labelizer = TimeSeriesLabelizer(self.window_size)

    def fit(self, series, epochs, batch_size, validation_rate):
        self.reset_state()
        if len(series) < self.window_size+1:
            raise ValueError()

        X, y = self.labelizer.fit_transform(series)
        self.model.fit(X, y,
                        epochs=epochs, 
                        batch_size=batch_size, 
                        validation_rate=validation_rate)
        # 初始化预测的滑动窗口
        self.roller.init(series[-self.window_size:])

    def forecast(self, n_steps):
        y_pred = np.zeros(n_steps)
        for i in range(n_steps):
            # 获取当前窗口
            X = self.roller.transform()
            X = np.expand_dims(X, 0)
            # 预测当前时间步取值
            y = self.model.predict(X)
            y_pred[i] = y
            # 更新滑动窗口
            self.roller.fit(y)
        return y_pred

    def reset_state(self):
        self.roller.reset()

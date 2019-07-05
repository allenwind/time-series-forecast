import numpy as np
from keras.layers import Input, Dense
from keras.layers import Flatten, LSTM
from keras.layers import TimeDistributed, Reshape
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from keras.losses import mean_squared_error, mean_absolute_percentage_error

from .utils import Rolling, TimeSeriesTransfer
from .base import Forecaster
from .lstm import LSTMForecaster
from .parallel import ParallelFitting

"""doc
http://www.oxford-man.ox.ac.uk/sites/default/files/events/combination_Sofie.pdf
"""

class BestForecaster(Forecaster):
    # 集成多模型的 topk 策略
    # 这种策略比传统的 ensemble 更为直接

    def __init__(self, size, topk, n_jobs, *models):
        self.size = size
        self.topk = topk
        self.n_jobs = n_jobs
        self.models = models

    def find_topk_models(self):
        pass

class CombiningForecaster(BestForecaster):
    # see markdown:
    # 集成预测的数学证明与算法设计

    # support model set:
    # Linear, MLP, CNN, LSTM, StackedLSTM, BiLSTM etc.

    def __init__(self, size, topk, n_jos, *models):
        self.parallel = ParallelFitting(self, n_jobs, *models)
        self.size = size
        self.topk = topk
        self.n_jobs = n_jos
        self.models = models
        self._topk_models = []

    def fit(self, series, epochs, batch_size, validation_series):
        if validation_series is None:
            raise ValueError("need validation series to compute weights of models")
        self.validation_series = validation_series
        self.parallel.execute_fit(series, epochs, batch_size, validation_series)

    def predict(self, forecast_size, post_fit=False):
        parr = self.parallel.execute_predict(forecast_size, post_fit)
        mapping = []
        for series, model in zip(parr, self.models):
            sigma = np.var(series-self.validation_series)
            mapping.append((series, sigma, model))

        series_with_weights = self._find_topk_forecast(mapping)
        return self._compute_combiantion_forecast(series_with_weights)

    def _find_topk_forecast(self, mapping):
        pass

    def _compute_combiantion_forecast(series, series_with_weights):
        pass

    @property
    def topk_models(self):
        return self._topk_models


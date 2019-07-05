from multiprocessing import cpu_count

from joblib import Parallel, delayed
import numpy as np

from .base import BaseForecaster

class ParallelFitting(BaseForecaster):

    # 并行地训练多个 Forecaster
    # https://joblib.readthedocs.io/en/latest/

    def __init__(self, size, n_jobs, *models):
        self.size = size
        if n_jobs == 0:
            n_jobs = cpu_count()
        self.n_jobs = n_jobs
        self.models = models
        self._init_models()

    def execute_fit(self, series, epochs, batch_size, validation_series=None):
        Parallel(self.n_jobs)(delayed(model)(series, epochs, batch_size, validation_series) \
            for model in self.models)

    def execute_predict(self, forecast_size, post_fit=False):
        results = Parallel(self.n_jobs)(delayed(model)(forecast_size, post_fit) \
            for model in self.models)
        return np.array(results)
        
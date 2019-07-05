import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA

from .base import Forecaster
from .utils import find_time_series_degree

# ARIMA 模块

def find_difference_degree(series, threshold=0.05):
    return find_time_series_degree(series, threshold=threshold)

def find_optimal_pq_with_bic(timeseries, d=1, p_max=None, q_max=None):
    """find parameter p,q of ARIMA(p,d,q) with
    Bayesian information criterion from 
    time series.

    wiki: https://en.wikipedia.org/wiki/Bayesian_information_criterion

    other choice is Akaike information criterion
    wiki: https://en.wikipedia.org/wiki/Akaike_information_criterion
    """

    if p_max is None:
        p_max = int(len(timeseries)/50) + 1
    if q_max is None:
        q_max = int(len(timeseries)/50) + 1
    matrix = []
    for p in range(p_max):
        row = []
        for q in range(q_max):
            try:
                row.append(ARIMA(timeseries, (p, d, q)).fit(disp=0).bic)
            except:
                row.append(None)
        matrix.append(row)

    matrix_df = pd.DataFrame(matrix)
    p, q = matrix_df.stack().idxmin()
    return p, q

class ARIMAForecaster(Forecaster):

    # ARIMA 模型
    # 目前 statsmodel 模块不支持 mape metric

    def __init__(self, size):
        self.size = size

    def fit(self, series):
        params = sm.tsa.arma_order_select_ic(series, max_ar=self.size, max_ma=0,ic=['bic'],trend='nc')
        p = params.bic_min_order[0]
        q = params.bic_min_order[1]
        self.model = ARIMA(series, order=(p,1,q))
        self.model.fit(disp=-1, method="mle")

    def predict(self, n_steps):
        y_pred = self.model.forecast(n_steps)
        return y_pred[0]

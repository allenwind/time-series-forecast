from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

__all__ = ["SimpleScaler", "PowerTransfer", "SeriesMinMaxScaler", "SeriesStandardScaler"]

class SimpleScaler:
    
    def __init__(self, type="std", eps=0.1):
        self.eps = eps
        if type == "std":
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler(feature_range=(self.eps, 1+self.eps))

    def fit(self, series):
        self.scaler.fit(series.reshape((-1, 1)))

    def transform(self, series):
        return self.scaler.transform(series.reshape((-1, 1))).ravel()
    
    def fit_transform(self, series):
        return self.scaler.fit_transform(series.reshape((-1, 1))).ravel()
    
    def inverse_transform(self, series):
        return self.scaler.inverse_transform(series.reshape((-1, 1))).ravel()

class PowerTransfer:

    # 把时间序列变换为 gaussian-like 分布
    
    def __init__(self):
        pass

    def fit(self):
        return self

    def fit_transform(self, series):
        pass

    def inverse_transform(self, series):
        pass

class SeriesMinMaxScaler:
    
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit_transform(self, series):
        pass

    def inverse_transform(self, series):
        pass

class SeriesStandardScaler:

    def __init__(self):
        pass

    def fit_transform(self, series):
        pass

    def inverse_transform(self, series):
        pass

import os

from keras.utils import plot_model
from keras.models import Model
import matplotlib.pyplot as plt

class BaseForecaster:

    # 预测模型基础接口
    
    def fit(self, series, epochs, batch_size, validation_series=None):
        pass

    def predict(self, forecast_size, post_fit=False):
        pass

    def _init_roller(self, X, y):
        pass

    def score(self, val_series):
        pass

class Forecaster(BaseForecaster):

    # 预测模型通用函数

    def plot_model(self, path=None, show=True):
        if path is None:
            path = "{}.png".format(str(self))

        if isinstance(self.model, Model):
            plot_model(self.model, to_file=path, show_shapes=True, show_layer_names=True)
        
        if show and os.path.exists(path):
            plt.imread(path)
            plt.imshow()

    def __repr__(self):
        return "<{}>".format(self.__class__.__name__)

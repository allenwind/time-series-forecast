import argparse
import warnings

from .utils import find_longest_period
from .utils import E2ETransfer

__all__ = []

__author__ = "zhiwen@smartx.com"

def auto_hpyerparams():
    pass

class SMTXForecaster:

    """end-to-end 时序预测"""

    def __init__(self):
        pass

    def fit(self, df):
        """
        :param df: pd.DataFrame
        """
        pass

    def preidct(self, n_steps):
        # (ts, y_pred), (y_pred_2l, y_pred_2u), (y_pred_3l, y_pred_3u)

    def plot(self):
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="auto", help="machine learning model")
    parser.add_argument("--dataset", type=str, help="dataset for training and forecast")
    parser.add_argument("--query", type=str, default=None, help="metric from octopus for training and forecast")
    parser.add_argument("--retry", type=int, default=1, help="retry training count")
    parser.add_argument("--n_steps", type=float, default=1.0)
    parser.add_argument("--save_weights", type=str, default=None)
    parser.add_argument("--load_weights", type=str, default=None)
    args = parser.parse_args()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()

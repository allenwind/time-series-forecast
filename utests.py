import numpy as np
import matplotlib.pyplot as plt

from features import extract_time_series_features

def test_stationary_transfer():
    # 测试差分还原效果

    for fn in datasets:
        series = fn()
        s = StationaryTransfer(k=1)
        st = s.fit_transform(series)
        so = s.inverse_transform(st)

        plt.subplot(211)
        plt.plot(series, color="blue", ls="-.", label="native")
        plt.plot(so, color="red", ls="--", label="inverse")
        plt.legend(loc="lower right")
        plt.subplot(212)
        plt.plot(st, label="diff")
        plt.legend(loc="lower right")
        plt.suptitle(fn.__name__)
        plt.show()

def test_features():
    _ = extract_time_series_features(100)

if __name__ == "__main__":
    test_features
    test_stationary_transfer()
import numpy as np

def build_cycle_atrix(series):
    # 创建循环矩阵
    return np.array([np.roll(series, i) for i in range(len(series))])

def vec_to_onehot(v):
    # argmax -> onehot
    r = np.zeros_like(v)
    r[np.argmax(v)] = 1
    return r

def loc_to_onehot(v, size):
    # loc -> onehot
    r = np.zeros(size)
    r[v] = 1
    return r

def onehot_to_loc(v):
    return np.argmax(v)

class TokenizerBase:

    # 连续信号的离散化

    def fit(self, series):
        pass

    def transform(self, series):
        # 变换为离散信号
        # 要求使用 one-hot 形式表示
        pass

    def inverse_transform(self, X):
        # 还原为模拟信号
        pass

    def sample_func(self, series):
        # 采样函数
        pass

    def inverse_sample_func(self, idx):
        # 逆采样函数
        pass

    def scores_f(self, s1, s2):
        # 采样误差函数
        pass

class SimpleTokenizer(TokenizerBase):

    # 连续信号的离散化

    def __init__(self, size, bin_range=(0, 1)):
        self.size = size
        self.r_min = bin_range[0]
        self.r_max = bin_range[1]
        self.bins = np.linspace(self.r_min, self.r_max, self.size)
        self.onehot = np.eye(self.size)

    def fit(self, series):
        return self

    def transform(self, series):
        # 从模拟信号中采样
        bs = self.sample_func(series)
        return self.onehot[bs]

    def inverse_transform(self, X):
        # 还原模拟信号
        # 从 one-hot 中还原模拟信号
        locs = np.where(np.not_equal(X, 0))[1]
        return self.inverse_sample_func(locs)

    def sample_func(self, series):
        # 采样函数
        # 目前采样的方法是等间隔采样
        return np.digitize(series, self.bins) - 1

    def inverse_sample_func(self, locs):
        # 逆采样函数
        series = self.bins[locs]
        return series

    def scores_f(self, s1, s2):
        return np.sqrt(np.sum(np.square(s1-s2))) / len(s1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    b = SimpleTokenizer(size=100, bin_range=(-1, 1))
    x = np.linspace(0, 2*np.pi, 1000)
    series = np.random.uniform(-1, 1, len(x))
    tseries = b.transform(series)
    plt.imshow(tseries)
    plt.show()
    series2 = b.inverse_transform(tseries)

    s = b.scores_f(series, series2)
    print(s)

    plt.plot(series)
    plt.plot(series2)
    plt.plot(series-series2, color="red")
    plt.show()

from .tokenizer import SimpleTokenizer

class BeamSearchForecaster:

    # 寻找一条定长路径，使其观察概率最大化
    # conditional beam search
    # 如何在组合爆炸中寻找最优观察概率的方法

    def __init__(self, model, topk, size, bin_range=(0, 1)):
        self.model = model
        self.topk = topk
        self.window_size = model.window_size
        self.tokenizer = SimpleTokenizer(size, bin_range)
        # topk 路径
        self.paths = [[]] * topk
        # topk 路径的观察概率
        # shape = (topk, 1)
        self.scores = np.ones((topk, 1)) / topk
        self.roller = TimeSeriesRolling2D(self.window_size)
        self.labelizer = TimeSeriesLabelizer(self.window_size)

    def search(self, n_steps):
        # 解码步骤
        # (1) 生成 topk 个最有可能的解
        # (2) 接下来每一步，从 topk*V 中 选择最后可能 topk 个解
        for i in range(self.n_steps):
            X = self.roller.transform()
            # shape = (window_size, bin_size)
            X = np.expand_dims(X, 0)
            # (1, n)
            y_proba = self.model.predict(X) 
            # how to zeros?
            y_proba = np.log(y_proba) 
            # 利用广播 (topk, n)
            # np.kron(a, b)
            S = self.scores + y_proba 
            scores = self._find_topk(S)
            self.scores = scores

    def _find_topk(self, S):
        ix, iy = np.unravel_index(np.argsort(S, axis=None), S.shape)
        scores = []
        for i, j in zip(ix[-self.topk:], iy[-self.topk:]):
            scores.append(S[i][j])
            self.paths[i].append(j)
        return scores

    def fit(self, series, epocs, batch_size, validation_rate):
        pass

    def forecast(self, n_steps):
        pass

    def to_topk_series(self):
        pass

    def to_best_series(self):
        return self.binnizer()
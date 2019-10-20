class Forecaster:

    # 滑动窗口预测的实现

    def __init__(self, model):
        self.state = 0
        self.model = model

    def fit(self, series):
        pass

    def predict_next_step(self):
        pass

    def predict_n_steps(self, n_steps):
        pass

    def reset_state(self):
        pass
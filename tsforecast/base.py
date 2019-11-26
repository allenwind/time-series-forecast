__all__ = ["ModelBase", "ForecasterBase"]


class ForecasterBase:

    """
    1. 关注如何完成多步预测
    2. 关注对模型的控制，如训练、预测、监控指标
    3. 处理状态信息
    """

    def fit(self, series, epochs, batch_size, validation_rate):
        pass

    def forecast(self, n_steps):
        pass

    def reset_state(self):
        pass

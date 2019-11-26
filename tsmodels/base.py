class ModelBase:

    """
    1. 关注场景本身，根据场景定义模型，如果是机器学习模型，则关注特征函数
    2. 关注如何训练与优化模型的实现
    3. 无状态，确定的输入，确定的输出，不随时间、操作变化
    4. 模型持久化
    """

    def fit(self, X, y, epochs=None, batch_size=None, validation_rate=0):
        # validation_rate
        # 输入的数据中，取部分作为验证集合，通常用在 callback 中
        pass

    def predict(self, X):
        # 输出下一个时间步的取值
        pass

    def reset(self):
        # 清空模型权重
        pass

    @property
    def window_size(self):
        # 返回滑动窗口的大小
        # window_size 作为一个超参数，像 batch_size 一样关乎模型的预测效果
        return self._window_size

class ProbaModelBase:
    """
    概率化预测模型
    """

    def predict(self, X):
        # 输出下一个时间步取值的概率分布
        pass